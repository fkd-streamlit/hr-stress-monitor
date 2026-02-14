import time
from collections import deque
import json

import av
import cv2
import numpy as np
import streamlit as st
from scipy.signal import butter, filtfilt, find_peaks
from scipy.fft import fft
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase


# =============================
# Signal processing
# =============================
def bandpass_filter(signal, fs, low=0.8, high=2.5):
    if fs < 5:
        return signal
    nyq = 0.5 * fs
    low = max(low, 0.01)
    high = min(high, nyq - 0.01)
    if low >= high:
        return signal
    b, a = butter(3, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, signal)


def compute_rr_intervals(peaks, fs):
    if len(peaks) < 2:
        return np.array([])
    return np.diff(peaks) / fs


def compute_time_domain(rr):
    if len(rr) < 2:
        return 0.0, 0.0
    sdnn = float(np.std(rr))
    rmssd = float(np.sqrt(np.mean(np.diff(rr) ** 2)))
    return sdnn, rmssd


def compute_frequency_domain(rr, fs=4):
    # rr: seconds between beats
    if len(rr) < 4:
        return 0.0, 0.0, 0.0

    rr_interp = np.interp(
        np.linspace(0, len(rr), len(rr) * fs),
        np.arange(len(rr)),
        rr,
    )

    spectrum = np.abs(fft(rr_interp)) ** 2
    freqs = np.fft.fftfreq(len(rr_interp), 1 / fs)

    lf = float(np.sum(spectrum[(freqs >= 0.04) & (freqs < 0.15)]))
    hf = float(np.sum(spectrum[(freqs >= 0.15) & (freqs < 0.4)]))
    ratio = float(lf / hf) if hf > 0 else 0.0
    return lf, hf, ratio


def compute_stress_level(lf_hf, rmssd, hr):
    # very simple heuristic (prototype)
    lf_norm = min(lf_hf / 4.0, 1.0)
    rmssd_norm = min(rmssd / 0.1, 1.0)
    hr_norm = min(max((hr - 60) / 60, 0), 1.0)

    score = 0.5 * lf_norm + 0.3 * (1 - rmssd_norm) + 0.2 * hr_norm
    level = int(score * 11) + 1
    return max(1, min(level, 12))


# =============================
# Video Processor
# =============================
class StressVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.g_values = deque(maxlen=1500)  # a bit more buffer
        self.t_values = deque(maxlen=1500)

        self.last_metrics = {
            "hr": None,
            "stress": None,
            "lfhf": None,
            "rmssd": None,
            "sdnn": None,
            "fs": None,
            "rr_n": 0,
        }

    def recv(self, frame: av.VideoFrame):
        img = frame.to_ndarray(format="bgr24")

        now = time.time()
        g_mean = float(np.mean(img[:, :, 1]))  # green mean

        self.g_values.append(g_mean)
        self.t_values.append(now)

        # Estimate FPS
        fs = 0.0
        if len(self.t_values) > 20:
            dt = self.t_values[-1] - self.t_values[0]
            fs = (len(self.t_values) - 1) / dt if dt > 0 else 0.0

        if fs > 8:  # a bit stricter for stability
            n_win = int(20 * fs)
            if len(self.g_values) >= n_win:
                sig = np.array(list(self.g_values)[-n_win:], dtype=np.float64)
                filtered = bandpass_filter(sig, fs)

                # peak distance: at least 0.45s between peaks (~133 bpm max)
                min_dist = max(int(fs * 0.45), 1)
                peaks, _ = find_peaks(filtered, distance=min_dist)
                rr = compute_rr_intervals(peaks, fs)

                if len(rr) >= 5 and 0.3 < np.mean(rr) < 1.5:
                    hr = float(60.0 / np.mean(rr))
                    sdnn, rmssd = compute_time_domain(rr)
                    _, _, lfhf = compute_frequency_domain(rr)
                    stress = compute_stress_level(lfhf, rmssd, hr)

                    self.last_metrics.update({
                        "hr": hr,
                        "stress": int(stress),
                        "lfhf": float(lfhf),
                        "rmssd": float(rmssd),
                        "sdnn": float(sdnn),
                        "fs": float(fs),
                        "rr_n": int(len(rr)),
                    })

        # Overlay
        m = self.last_metrics
        if m["stress"] is not None and m["hr"] is not None:
            color = (0, int(max(0, 255 - m["stress"] * 20)), int(min(255, m["stress"] * 20)))

            cv2.putText(img, f"FPS(est): {m['fs']:.1f}",
                        (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255), 2)

            cv2.putText(img, f"HR: {m['hr']:.1f} bpm (RR n={m['rr_n']})",
                        (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                        (255, 255, 255), 2)

            cv2.putText(img, f"LF/HF: {m['lfhf']:.2f}  RMSSD: {m['rmssd']:.3f}",
                        (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                        (255, 255, 255), 2)

            cv2.putText(img, f"Stress Level: {m['stress']}/12",
                        (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        color, 3)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# =============================
# Streamlit UI
# =============================
st.set_page_config(page_title="HR Stress Monitor", layout="wide")
st.title("🫀 HR Stress Monitor")
st.caption("※ 医療用途ではありません（research prototype）")


RTC_CONFIGURATION = {
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
}

ctx = webrtc_streamer(
    key="stress",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    video_processor_factory=StressVideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

st.markdown("---")

# Safe guards (ctx can exist before media starts)
if ctx and ctx.state and ctx.state.playing and ctx.video_processor:
    m = ctx.video_processor.last_metrics

    c1, c2, c3 = st.columns(3)
    c1.metric("HR", "-" if m["hr"] is None else f"{m['hr']:.1f} bpm")
    c2.metric("Stress", "-" if m["stress"] is None else f"{m['stress']}/12")
    c3.metric("RR count", f"{m.get('rr_n', 0)}")

    # Export only when fully ready
    if m["hr"] is not None and m["stress"] is not None:
        output = {
            "ts": time.time(),
            "hr_bpm": float(m["hr"]),
            "stress_level_12": int(m["stress"]),
            "lfhf": float(m["lfhf"]) if m["lfhf"] is not None else 0.0,
            "rmssd": float(m["rmssd"]) if m["rmssd"] is not None else 0.0,
            "sdnn": float(m["sdnn"]) if m["sdnn"] is not None else 0.0,
        }

        st.download_button(
            "⬇️ Download QUBO Input JSON",
            data=json.dumps(output, ensure_ascii=False, indent=2),
            file_name="qubo_input.json",
            mime="application/json",
        )
else:
    st.info("▶ カメラを開始すると、HR / Stress が表示されます（明るい環境で、顔をなるべく動かさないのがコツ）。")
