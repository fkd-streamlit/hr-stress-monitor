import time
import json
from collections import deque

import numpy as np
import streamlit as st
from scipy.signal import butter, filtfilt, find_peaks
from scipy.fft import fft

import av
import cv2

from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase


# =============================
# Signal processing
# =============================
def bandpass_filter(signal, fs, low=0.8, high=2.5):
    if fs < 5:
        return signal
    nyq = 0.5 * fs
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
    if len(rr) < 4:
        return 0.0, 0.0, 0.0

    rr = np.asarray(rr, dtype=float)
    x = np.linspace(0, len(rr) - 1, len(rr) * fs)
    rr_interp = np.interp(x, np.arange(len(rr)), rr)

    spectrum = np.abs(fft(rr_interp)) ** 2
    freqs = np.fft.fftfreq(len(rr_interp), 1 / fs)

    lf = float(np.sum(spectrum[(freqs >= 0.04) & (freqs < 0.15)]))
    hf = float(np.sum(spectrum[(freqs >= 0.15) & (freqs < 0.4)]))
    ratio = float(lf / hf) if hf != 0 else 0.0
    return lf, hf, ratio


def compute_stress_level(lf_hf, rmssd, hr):
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
        self.g_values = deque(maxlen=1200)
        self.t_values = deque(maxlen=1200)

        self.last_metrics = {
            "hr": None,
            "stress": None,
            "lfhf": None,
            "rmssd": None,
            "sdnn": None,
            "fs": None,
        }

    def recv(self, frame: av.VideoFrame):
        img = frame.to_ndarray(format="bgr24")

        now = time.time()
        g_mean = float(np.mean(img[:, :, 1]))  # green channel mean

        self.g_values.append(g_mean)
        self.t_values.append(now)

        # estimate sampling rate
        fs = 0.0
        if len(self.t_values) > 10:
            dt = self.t_values[-1] - self.t_values[0]
            fs = (len(self.t_values) - 1) / dt if dt > 0 else 0.0

        # compute metrics on a rolling 20s window
        if fs > 5:
            n_win = int(20 * fs)
            if len(self.g_values) >= n_win:
                sig = np.asarray(list(self.g_values)[-n_win:], dtype=float)
                filtered = bandpass_filter(sig, fs)

                peaks, _ = find_peaks(filtered, distance=max(int(fs / 2), 1))
                rr = compute_rr_intervals(peaks, fs)

                if len(rr) > 3:
                    hr = float(60.0 / np.mean(rr))
                    sdnn, rmssd = compute_time_domain(rr)
                    _, _, lfhf = compute_frequency_domain(rr)
                    stress = compute_stress_level(lfhf, rmssd, hr)

                    self.last_metrics.update(
                        {"hr": hr, "stress": stress, "lfhf": lfhf, "rmssd": rmssd, "sdnn": sdnn, "fs": fs}
                    )

        # overlay
        m = self.last_metrics
        if m["stress"] is not None:
            color = (0, max(0, 255 - m["stress"] * 20), min(255, m["stress"] * 20))

            cv2.putText(img, f"HR: {m['hr']:.1f} bpm", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            cv2.putText(img, f"Stress Level: {m['stress']}/12", (20, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# =============================
# Streamlit UI
# =============================
st.set_page_config(page_title="HR Stress Monitor (Webcam)", layout="wide")
st.title("🫀 HR Stress Monitor (Webcam)")
st.caption("※ 医療用途ではありません（research prototype）")

RTC_CONFIGURATION = {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}

ctx = webrtc_streamer(
    key="stress",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    video_processor_factory=StressVideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

st.markdown("---")

if ctx.video_processor:
    m = ctx.video_processor.last_metrics
    st.metric("HR", "-" if m["hr"] is None else f"{m['hr']:.1f} bpm")
    st.metric("Stress", "-" if m["stress"] is None else f"{m['stress']}/12")

    if m["hr"] is not None:
        output = {
            "ts": time.time(),
            "hr_bpm": float(m["hr"]),
            "stress_level_12": int(m["stress"]),
            "lfhf": float(m["lfhf"]) if m["lfhf"] is not None else None,
            "rmssd": float(m["rmssd"]) if m["rmssd"] is not None else None,
            "sdnn": float(m["sdnn"]) if m["sdnn"] is not None else None,
        }

        st.download_button(
            "⬇️ Download QUBO Input JSON",
            data=json.dumps(output, indent=2, ensure_ascii=False),
            file_name="qubo_input.json",
            mime="application/json",
        )
else:
    st.info("左上の▶️で開始。カメラ許可が必要です。明るい環境で顔をなるべく動かさないのがコツ。")
