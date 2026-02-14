import io
import json
import time
from collections import deque

import cv2
import numpy as np
import streamlit as st
from scipy.signal import butter, filtfilt, find_peaks
from scipy.fft import fft


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
    lf_norm = min(lf_hf / 4.0, 1.0)
    rmssd_norm = min(rmssd / 0.1, 1.0)
    hr_norm = min(max((hr - 60) / 60, 0), 1.0)
    score = 0.5 * lf_norm + 0.3 * (1 - rmssd_norm) + 0.2 * hr_norm
    level = int(score * 11) + 1
    return max(1, min(level, 12))


# =============================
# Video analysis
# =============================
def analyze_video(file_bytes: bytes, window_sec=20, low=0.8, high=2.5, roi_mode="center"):
    # OpenCV VideoCapture can read from a temp file. Streamlit gives bytes.
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tf:
        tf.write(file_bytes)
        tmp_path = tf.name

    cap = cv2.VideoCapture(tmp_path)
    if not cap.isOpened():
        return None, "動画を読み込めませんでした（形式が非対応の可能性）"

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1:
        # fallback
        fps = 30.0

    n_win = int(window_sec * fps)

    g = deque(maxlen=n_win)
    ts = deque(maxlen=n_win)

    # Read frames
    start_time = time.time()
    frame_count = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_count += 1

        h, w = frame.shape[:2]
        if roi_mode == "center":
            # center ROI
            x1, y1 = int(w*0.35), int(h*0.25)
            x2, y2 = int(w*0.65), int(h*0.55)
            roi = frame[y1:y2, x1:x2]
        else:
            roi = frame

        g_mean = float(np.mean(roi[:, :, 1]))
        g.append(g_mean)
        ts.append(frame_count / fps)

    cap.release()

    if len(g) < n_win:
        return None, f"動画が短すぎます。{window_sec}秒以上の動画をアップしてください。"

    sig = np.array(list(g), dtype=np.float64)
    filtered = bandpass_filter(sig, fps, low=low, high=high)

    min_dist = max(int(fps * 0.45), 1)
    peaks, _ = find_peaks(filtered, distance=min_dist)

    rr = compute_rr_intervals(peaks, fps)
    if len(rr) < 5 or not (0.3 < np.mean(rr) < 1.5):
        return None, "脈波ピークが安定して検出できませんでした（明るさ/顔の動き/ROIを調整してください）"

    hr = float(60.0 / np.mean(rr))
    sdnn, rmssd = compute_time_domain(rr)
    _, _, lfhf = compute_frequency_domain(rr)
    stress = int(compute_stress_level(lfhf, rmssd, hr))

    result = {
        "hr_bpm": hr,
        "stress_level_12": stress,
        "lfhf": float(lfhf),
        "rmssd": float(rmssd),
        "sdnn": float(sdnn),
        "fps_est": float(fps),
        "rr_n": int(len(rr)),
        "window_sec": int(window_sec),
    }
    return result, None


# =============================
# UI
# =============================
st.set_page_config(page_title="HR Stress Monitor (Upload)", layout="wide")
st.title("🫀 HR Stress Monitor (Cloud-safe)")
st.caption("※ 医療用途ではありません（research prototype）")
st.info("Streamlit Cloud では WebRTC カメラが起動しない環境があるため、公開版は「動画アップロード解析」にしています。")

with st.sidebar:
    st.header("Settings")
    window_sec = st.slider("解析ウィンドウ(秒)", 10, 40, 20, 1)
    low = st.slider("Bandpass low (Hz)", 0.5, 1.5, 0.8, 0.05)
    high = st.slider("Bandpass high (Hz)", 1.5, 3.5, 2.5, 0.05)
    roi_mode = st.selectbox("ROI", ["center", "full"], index=0)
    st.caption("コツ：明るい環境・顔を大きく・動かさない・20秒以上の動画")

uploaded = st.file_uploader("顔が映った動画（mp4 推奨）をアップロード", type=["mp4", "mov", "m4v", "webm"])

if uploaded:
    st.video(uploaded)

    if st.button("解析する"):
        with st.spinner("解析中..."):
            result, err = analyze_video(uploaded.getvalue(), window_sec=window_sec, low=low, high=high, roi_mode=roi_mode)

        if err:
            st.error(err)
        else:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("HR", f"{result['hr_bpm']:.1f} bpm")
            c2.metric("Stress", f"{result['stress_level_12']}/12")
            c3.metric("LF/HF", f"{result['lfhf']:.2f}")
            c4.metric("RMSSD", f"{result['rmssd']:.3f}")

            qubo = {
                "ts": time.time(),
                "hr_bpm": float(result["hr_bpm"]),
                "stress_level_12": int(result["stress_level_12"]),
                "lfhf": float(result["lfhf"]),
                "rmssd": float(result["rmssd"]),
                "sdnn": float(result["sdnn"]),
            }

            st.download_button(
                "⬇️ Download QUBO Input JSON",
                data=json.dumps(qubo, ensure_ascii=False, indent=2),
                file_name="qubo_input.json",
                mime="application/json",
            )

else:
    st.warning("まず動画ファイルをアップしてください。")
