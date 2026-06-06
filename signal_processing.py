"""
signal_processing.py
────────────────────
Bandpass filter, baseline correction, segmentation, SQC.
Identical to training pipeline.
"""

import numpy as np
from scipy.signal      import butter, filtfilt, find_peaks
from scipy.fft         import rfft, rfftfreq
from scipy.interpolate import CubicSpline

# ── Config ────────────────────────────────────────────────────────────────────
FS            = 100
WIN_S         = 10
STRIDE_S      = 10
BANDPASS_LOW  = 0.3
BANDPASS_HIGH = 7.0
KAPPA_THRESH  = 50.0
SQI_THRESH    = 0.20


def detect_peaks(sig: np.ndarray, fs: int = FS) -> np.ndarray:
    prom     = 0.05 * (sig.max() - sig.min())
    peaks, _ = find_peaks(sig, distance=int(0.4 * fs), prominence=prom)
    return peaks


def bandpass_filter(signal: np.ndarray, fs: int = FS) -> np.ndarray:
    nyq  = fs / 2
    b, a = butter(4, [BANDPASS_LOW / nyq, BANDPASS_HIGH / nyq], btype="band")
    return filtfilt(b, a, signal)


def remove_baseline(signal: np.ndarray, fs: int = FS) -> np.ndarray:
    peaks = detect_peaks(signal, fs)
    if len(peaks) < 3:
        return signal
    valleys = [(peaks[i] + peaks[i + 1]) // 2 for i in range(len(peaks) - 1)]
    valleys  = [0] + valleys + [len(signal) - 1]
    xs = np.array(valleys)
    ys = signal[xs]
    cs = CubicSpline(xs, ys)
    return signal - cs(np.arange(len(signal)))


def full_filter(signal: np.ndarray, fs: int = FS) -> np.ndarray:
    return remove_baseline(bandpass_filter(signal, fs), fs)


def segment_signal(ir: np.ndarray, red: np.ndarray,
                   fs: int = FS,
                   win_s: int = WIN_S,
                   stride_s: int = STRIDE_S) -> list:
    wl   = int(win_s * fs)
    sl   = int(stride_s * fs)
    segs = []
    i, sid = 0, 0
    sno  = np.arange(len(ir), dtype=float)
    while i + wl <= len(ir):
        segs.append({
            "seg_id": sid,
            "ir"    : ir[i:i + wl],
            "red"   : red[i:i + wl],
            "time"  : sno[i:i + wl] / fs,
        })
        sid += 1
        i   += sl
    return segs


def check_sqc(seg: dict, fs: int = FS) -> bool:
    sig   = seg["ir"]
    pgram = np.abs(rfft(sig)) ** 2
    kappa = pgram.max() / (pgram.mean() + 1e-9)
    peaks = detect_peaks(sig, fs)
    if len(peaks) < 3:
        sqi = 1.0
    else:
        rr  = np.diff(peaks)
        sqi = rr.std() / (rr.mean() + 1e-9)
    return (kappa >= KAPPA_THRESH) and (sqi <= SQI_THRESH)
