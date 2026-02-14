# app.py
# =========================================
# Stress Monitor 12-Level (Local Prototype)
# Streamlit + streamlit-webrtc
# =========================================
# 起動:
#   streamlit run app.py
#
# 注意:
# - 医療用途ではありません（デモ/研究用途）
# - 照明、顔の動き、カメラ品質で結果が大きく変動します
# =========================================

import time
from collections import deque

import av
import cv2
import numpy as np
import streamlit as st
from scipy.signal import butter, filtfilt, find_peaks
from scipy.fft import fft
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase


# =============================
# Signal Processing
# =============================
def bandpass_filter(signal, fs, low=0.8, high=2.5):
    """Butterworth bandpass filter"""
    signal = np.asarray(signal, dtype=np.float64)

    # fs が低い/不正の場合はそのまま返す
    if fs is None or fs < 5:
        return signal

    nyq = 0.5 * fs
    low_n = max(low / nyq, 1e-6)
    high_n = min(high / nyq, 0.999999)

    if low_n >= high_n:
        return signal

    b, a = butter(3, [low_n, high_n], btype="band")
    try:
        return filtfilt(b, a, signal)
    except Exception:
        # 異常時は生信号で継続
        return signal


def compute_rr_intervals(peaks, fs):
    """RR intervals in seconds from peak indices"""
    if peaks is None or len(peaks) < 2 or fs <= 0:
        return np.array([], dtype=np.float64)
    return np.diff(peaks) / float(fs)


def compute_time_domain(rr):
    """SDNN, RMSSD (seconds)"""
    rr = np.asarray(rr, dtype=np.float64)
    if len(rr) < 2:
        return 0.0, 0.0
    sdnn = float(np.std(rr))
    rmssd = float(np.sqrt(np.mean(np.diff(rr) ** 2))) if len(rr) >= 3 else 0.0
    return sdnn, rmssd


def compute_frequency_domain(rr, fs=4):
    """
    超簡易版（後で Welch / Lomb-Scargle に差し替え推奨）
    rr: RR intervals (sec)
    """
    rr = np.asarray(rr, dtype=np.float64)
    if len(rr) < 4:
        return 0.0, 0.0, 0.0

    # 補間して等間隔化
    x = np.arange(len(rr), dtype=np.float64)
    xi = np.linspace(0, len(rr) - 1, len(rr) * fs)
    rr_interp = np.interp(xi, x, rr)

    spectrum = np.abs(fft(rr_interp)) ** 2
    freqs = np.fft.fftfreq(len(rr_interp), d=1 / fs)

    lf = float(np.sum(spectrum[(freqs >= 0.04) & (freqs < 0.15)]))
    hf = float(np.sum(spectrum[(freqs >= 0.15) & (freqs < 0.40)]))
    ratio = float(lf / hf) if hf > 0 else 0.0
    return lf, hf, ratio


def compute_stress_level(lf_hf, rmssd, hr):
    """12段階ストレス推定（経験則）"""
    # 正規化（経験的レンジ）
    lf_norm = min(max(lf_hf / 4.0, 0.0), 1.0)            # LF/HF: 0..4
    rmssd_norm = min(max(rmssd / 0.1, 0.0), 1.0)         # RMSSD: 0..0.1 sec
    hr_norm = min(max((hr - 60.0) / 60.0, 0.0), 1.0)     # HR: 60..120 bpm

    stress_score = (0.5 * lf_norm) + (0.3 * (1.0 - rmssd_norm)) + (0.2 * hr_norm)

    level = int(stress_score * 11) + 1
    return max(1, min(level, 12))


# =============================
# ROI Utilities
# =============================
def detect_face_roi(frame_bgr, face_cascade):
    """Detect face and return forehead-like ROI (x1,y1,x2,y2)"""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
    )
    if len(faces) == 0:
        return None

    # 一番大きい顔を採用
    x, y, w, h = sorted(faces, key=lambda r: r[2] * r[3], reverse=True)[0]

    # 額あたり（顔の上部）
    fx1 = x + int(w * 0.20)
    fx2 = x + int(w * 0.80)
    fy1 = y + int(h * 0.08)
    fy2 = y + int(h * 0.35)

    fx1 = max(0, fx1)
    fy1 = max(0, fy1)
    fx2 = min(frame_bgr.shape[1], fx2)
    fy2 = min(frame_bgr.shape[0], fy2)

    if fx2 - fx1 < 10 or fy2 - fy1 < 10:
        return None

    return (fx1, fy1, fx2, fy2)


# =============================
# WebRTC Video Processor
# =============================
class StressVideoProcessor(VideoProcessorBase):
    def __init__(self):
        # 40秒@30fps相当を保持（十分余裕）
        self.g_values = deque(maxlen=1400)
        self.t_values = deque(maxlen=1400)

        self.last_metrics = {
            "hr": None,
            "sdnn": None,
            "rmssd": None,
            "lfhf": None,
            "stress": None,
            "fs": None,
            "rr_count": 0,
        }

        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

        # default params
        self.use_face_roi = True
        self.window_sec = 20
        self.low = 0.8
        self.high = 2.5

    def set_params(self, use_face_roi, window_sec, low, high):
        self.use_face_roi = bool(use_face_roi)
        self.window_sec = int(window_sec)
        self.low = float(low)
        self.high = float(high)

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        now = time.time()

        # ROI決定
        roi = None
        if self.use_face_roi:
            roi = detect_face_roi(img, self.face_cascade)

        if roi is None:
            g_mean = float(np.mean(img[:, :, 1]))
        else:
            x1, y1, x2, y2 = roi
            g_mean = float(np.mean(img[y1:y2, x1:x2, 1]))
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        self.g_values.append(g_mean)
        self.t_values.append(now)

        # timestampベースでfps推定
        fs = None
        if len(self.t_values) >= 10:
            dt = self.t_values[-1] - self.t_values[0]
            fs = (len(self.t_values) - 1) / dt if dt > 0 else None

        # 解析
        if fs is not None and fs > 5:
            n_win = int(self.window_sec * fs)
            if len(self.g_values) >= n_win:
                sig = np.array(list(self.g_values)[-n_win:], dtype=np.float64)

                filtered = bandpass_filter(sig, fs, low=self.low, high=self.high)

                # ピーク検出（距離は0.5秒相当）
                min_dist = max(int(fs / 2), 1)
                peaks, _ = find_peaks(filtered, distance=min_dist)

                rr = compute_rr_intervals(peaks, fs)
                if len(rr) > 3:
                    hr = float(60.0 / np.mean(rr))
                    sdnn, rmssd = compute_time_domain(rr)
                    _, _, lfhf = compute_frequency_domain(rr)
                    stress = compute_stress_level(lfhf, rmssd, hr)

                    self.last_metrics.update(
                        {
                            "hr": hr,
                            "sdnn": sdnn,
                            "rmssd": rmssd,
                            "lfhf": lfhf,
                            "stress": stress,
                            "fs": float(fs),
                            "rr_count": int(len(rr)),
                        }
                    )
                else:
                    self.last_metrics.update({"fs": float(fs), "rr_count": int(len(rr))})

        # 画面描画
        m = self.last_metrics
        if m["stress"] is not None:
            stress = int(m["stress"])
            color = (0, max(0, 255 - stress * 20), min(255, stress * 20))

            cv2.putText(
                img,
                f"FPS(est): {m['fs']:.1f}" if m["fs"] else "FPS(est): -",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                img,
                f"HR: {m['hr']:.1f} bpm",
                (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.85,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                img,
                f"LF/HF: {m['lfhf']:.2f}  RMSSD: {m['rmssd']:.3f}  (RR n={m['rr_count']})",
                (20, 95),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                img,
                f"Stress Level: {stress}/12",
                (20, 135),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.05,
                color,
                3,
            )
        else:
            cv2.putText(
                img,
                "Collecting... (keep face steady & well-lit)",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )
            if fs:
                cv2.putText(
                    img,
                    f"FPS(est): {fs:.1f}",
                    (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# =============================
# Streamlit UI
# =============================
st.set_page_config(page_title="Stress Monitor (Local)", page_icon="🫀", layout="wide")
st.title("🫀 Stress Monitor 12-Level (Local Prototype)")
st.caption("※医療用途ではありません。照明・姿勢・カメラ品質で結果が大きく変動します。")

with st.sidebar:
    st.header("Settings")
    use_face_roi = st.checkbox("Face ROI（推奨）", value=True)
    window_sec = st.slider("解析ウィンドウ（秒）", 10, 30, 20, 1)
    low = st.slider("Bandpass low (Hz)", 0.5, 1.2, 0.8, 0.05)
    high = st.slider("Bandpass high (Hz)", 2.0, 3.5, 2.5, 0.1)
    st.markdown("---")
    st.write("コツ：顔を明るく、動かさず、カメラから30〜60cm程度。")


# webrtc_streamer は「別スレッドで factory を呼ぶ」ので、session_state依存は避ける
params = {"use_face_roi": use_face_roi, "window_sec": window_sec, "low": low, "high": high}


def make_processor():
    p = StressVideoProcessor()
    p.set_params(params["use_face_roi"], params["window_sec"], params["low"], params["high"])
    return p


ctx = webrtc_streamer(
    key="stress-monitor",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=make_processor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)
# ===== ここから追加（ctx の後ろ）=====
if ctx.video_processor:
    m = ctx.video_processor.last_metrics
    hr = m.get("hr")
    stress = m.get("stress")

    # 取り出した値を確認（表示）
    st.write("Extracted:", {"hr": hr, "stress": stress})

    # 後工程（量子神託）に渡しやすい形にまとめる
    qubo_input = {
        "ts": time.time(),
        "hr_bpm": None if hr is None else float(hr),
        "stress_level_12": None if stress is None else int(stress),
        "lfhf": None if m.get("lfhf") is None else float(m.get("lfhf")),
        "rmssd": None if m.get("rmssd") is None else float(m.get("rmssd")),
        "sdnn": None if m.get("sdnn") is None else float(m.get("sdnn")),
        "fps_est": None if m.get("fs") is None else float(m.get("fs")),
    }

    # JSONとしてダウンロード（量子神託側の入力に使える）
    import json
    st.download_button(
        "⬇️ QUBO入力JSONをダウンロード",
        data=json.dumps(qubo_input, ensure_ascii=False, indent=2),
        file_name="qubo_input.json",
        mime="application/json",
    )
else:
    st.info("カメラ開始後、少し待つとHR/Stressが出ます。")
# ===== ここまで追加 =====

st.markdown("---")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Status / Metrics")
    if ctx.video_processor:
        m = ctx.video_processor.last_metrics
        st.metric("HR (bpm)", "-" if m["hr"] is None else f"{m['hr']:.1f}")
        st.metric("Stress (1-12)", "-" if m["stress"] is None else f"{int(m['stress'])}/12")
        st.write(
            {
                "FPS(est)": None if m["fs"] is None else round(m["fs"], 1),
                "LF/HF": m["lfhf"],
                "RMSSD": m["rmssd"],
                "SDNN": m["sdnn"],
                "RR count": m["rr_count"],
            }
        )
    else:
        st.info("カメラ開始後、数秒待ってください（初期化中）。")

with col2:
    st.subheader("Raw signal (Green mean)")
    if ctx.video_processor:
        g = list(ctx.video_processor.g_values)[-300:]
        if len(g) > 10:
            st.line_chart(g, height=220)
        else:
            st.info("信号を収集中です。顔を明るくして数秒待ってください。")
    else:
        st.info("カメラ開始後、数秒待ってください。")

st.markdown("---")
st.caption("終了するには、ブラウザタブを閉じるか、ターミナルで Ctrl+C を押してください。")
