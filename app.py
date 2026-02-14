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


#
