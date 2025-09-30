import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import (
    VideoTransformerBase,
    webrtc_streamer,
    RTCConfiguration,
    WebRtcMode,
)

# Thresholds for classification
CLEAN_THRESH = 0.02
KINDA_THRESH = 0.08

# RTC config for WebRTC (STUN server)
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# ---------- Helper: find biggest 4-point contour ----------
def find_paper_contour(contours):
    max_area = 0
    best_cnt = None
    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        area = cv2.contourArea(cnt)
        if len(approx) == 4 and area > max_area:
            best_cnt = approx
            max_area = area
    return best_cnt


# ---------- Video Transformer ----------
class PaperDetector(VideoTransformerBase):
    def __init__(self):
        self.last_label = None
        self.last_ratio = 0.0
        self.last_debug = None

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)

        cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        pts = find_paper_contour(cnts)

        if pts is None:
            # --- Fallback: brightness-based detection ---
            mean_val = np.mean(gray)
            if mean_val > 200:
                label = "clean"
            elif mean_val > 150:
                label = "kinda"
            else:
                label = "nope"
            ratio = mean_val / 255.0

            debug_img = img.copy()
            cv2.putText(debug_img, "Fallback detection", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            self.last_debug = debug_img

        else:
            # --- Warp paper region ---
            pts = pts.reshape(4, 2)
            rect = np.zeros((4, 2), dtype="float32")
            s = pts.sum(axis=1)
            rect[0] = pts[np.argmin(s)]
            rect[2] = pts[np.argmax(s)]
            diff = np.diff(pts, axis=1)
            rect[1] = pts[np.argmin(diff)]
            rect[3] = pts[np.argmax(diff)]

            (tl, tr, br, bl) = rect
            width = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
            height = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))

            dst = np.array(
                [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
                dtype="float32",
            )
            M = cv2.getPerspectiveTransform(rect, dst)
            warp = cv2.warpPerspective(gray, M, (width, height))

            th = cv2.adaptiveThreshold(
                warp, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 10
            )
            mark_ratio = np.sum(th > 0) / float(th.size)

            if mark_ratio < CLEAN_THRESH:
                label = "clean"
            elif mark_ratio < KINDA_THRESH:
                label = "kinda"
            else:
                label = "nope"
            ratio = mark_ratio

            self.last_debug = th

            # Draw detected contour on webcam
            debug_img = img.copy()
            cv2.drawContours(debug_img, [pts.astype(int)], -1, (0, 255, 0), 2)
            cv2.putText(debug_img, f"{label} ({ratio:.2%})", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            img = debug_img

        # Save last results
        self.last_label = label
        self.last_ratio = ratio

        return img


# ---------- Streamlit UI ----------
st.title("📄 Blank Paper Detection Demo")

webrtc_ctx = webrtc_streamer(
    key="paper-detector",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    video_transformer_factory=PaperDetector,
    media_stream_constraints={"video": True, "audio": False},
    async_transform=True,
)

if webrtc_ctx.video_transformer:
    vt = webrtc_ctx.video_transformer
    label = vt.last_label if vt.last_label is not None else "No paper yet"
    ratio = vt.last_ratio

    if label == "clean":
        headline = "I'm clean bruhh..."
        cssdiv = "clean"
        color = "#00ffc2"
    elif label == "kinda":
        headline = "Hmm kinda.."
        cssdiv = "kinda"
        color = "#ffd166"
    elif label == "nope":
        headline = "Nuh uh"
        cssdiv = "nope"
        color = "#ff3b6f"
    else:
        headline = "Waiting for paper..."
        cssdiv = ""
        color = "#cfcfcf"

    st.markdown(
        f"""
        <div class="result {cssdiv}">
          <div class="pulse" style="color:{color}">{headline}</div>
          <div class="hint">Mark ratio: <b>{ratio:.4%}</b></div>
          <div class="hint">Thresholds: clean < {CLEAN_THRESH:.3%}, kinda < {KINDA_THRESH:.3%}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.checkbox("Show debug warped/threshold image"):
        if vt.last_debug is not None:
            if vt.last_debug.ndim == 2:  # grayscale
                st.image(vt.last_debug, caption="Debug image", channels="GRAY")
            else:
                st.image(vt.last_debug, caption="Debug image", channels="BGR")
        else:
            st.info("No debug image available yet.")
else:
    st.info("Start the webcam above to begin live detection.")

