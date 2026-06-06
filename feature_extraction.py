"""
feature_extraction.py
──────────────────────
Group A (morphological + FFT), Group C (MFCC), Group D (statistical).
Identical to training pipeline.
"""

import numpy as np
from scipy.fft   import rfft, rfftfreq
from scipy.stats import skew, kurtosis
from signal_processing import detect_peaks, FS

MIN_CYCLES  = 2
MFCC_COEFFS = 13
MFCC_FRAMES = 16


# ── Group A helpers ────────────────────────────────────────────────────────────
def find_cycle_onset(sig: np.ndarray, peak_idx: int, fs: int = FS) -> int:
    start  = max(0, peak_idx - int(0.6 * fs))
    region = sig[start:peak_idx]
    return start + int(np.argmin(region)) if len(region) >= 2 else start


def extract_one_cycle(sig: np.ndarray, fs: int,
                      p1: int, p2: int) -> dict | None:
    eps   = 1e-9
    onset = find_cycle_onset(sig, p1, fs)
    base  = sig[onset]
    x     = sig[p1] - base
    if x <= 0:
        return None
    tpi = (p2 - onset) / fs
    t1  = (p1 - onset) / fs
    if tpi <= 0 or t1 <= 0:
        return None

    half  = base + x / 2
    rise  = sig[onset:p1]
    cross = np.where(rise >= half)[0]
    t_half = (cross[0] / fs) if len(cross) > 0 else t1 / 2

    asc  = np.maximum(sig[onset:p1] - base, 0)
    desc = np.maximum(sig[p1:p2]   - base, 0)
    s1   = np.trapezoid(asc)  / fs
    s2   = np.trapezoid(desc) / fs

    d1  = np.gradient(sig)
    d1a = d1[onset:p1]
    d1d = d1[p1:p2]
    a1l = int(np.argmax(d1a))
    ta1 = a1l / fs
    zc  = np.where(np.diff(np.sign(d1a[a1l:])) < 0)[0]
    tb1 = ((a1l + zc[0]) / fs) if len(zc) > 0 else t1
    tf1 = (len(d1a) + int(np.argmin(d1d))) / fs

    d2  = np.gradient(d1)
    d2s = d2[onset:p2]
    sl  = len(d2s)
    q1  = max(1, sl // 4)
    a2l = int(np.argmax(d2s[:q1]))
    a2v = d2s[a2l]
    ta2 = a2l / fs
    b2r = d2s[a2l:a2l + sl // 3]
    b2v = b2r[int(np.argmin(b2r))] if len(b2r) > 0 else a2v
    tb2 = (a2l + int(np.argmin(b2r))) / fs if len(b2r) > 0 else ta2
    d2h = d2s[sl // 2:]
    e2v = d2h[int(np.argmin(d2h))] if len(d2h) > 0 else b2v

    return {
        "x": x, "tpi": tpi, "t1": t1, "t_half": t_half,
        "s1": s1, "s2": s2, "A2_A1": s2 / (s1 + eps),
        "kup": x / (t1 + eps), "kdown": x / (tpi - t1 + eps),
        "t1_tpi": t1 / tpi, "t_half_tpi": t_half / tpi,
        "ta1": ta1, "tb1": tb1, "tf1": tf1,
        "ta1_tpi": ta1 / tpi, "tb1_tpi": tb1 / tpi,
        "tf1_tpi": tf1 / tpi,
        "ta1tb1_tpi": (ta1 + tb1) / tpi,
        "b2_a2": b2v / (a2v + eps),
        "e2_a2": e2v / (a2v + eps),
        "b2e2_a2": (b2v + e2v) / (a2v + eps),
        "ta2": ta2, "tb2": tb2,
        "ta2_tpi": ta2 / tpi, "tb2_tpi": tb2 / tpi,
        "ta1ta2_tpi": (ta1 + ta2) / tpi,
        "tb1tb2_tpi": (tb1 + tb2) / tpi,
    }


def extract_fft(sig: np.ndarray, fs: int = FS) -> dict:
    N     = len(sig)
    freqs = rfftfreq(N, 1 / fs)
    mags  = np.abs(rfft(sig))
    m     = (freqs >= 0.5) & (freqs <= 4.0)
    if not m.any():
        return {k: np.nan for k in
                ["fbase", "sbase", "f2nd", "s2nd", "f3rd", "s3rd"]}
    bi  = int(np.argmax(mags[m]))
    fb  = freqs[m][bi]
    sb  = mags[m][bi]
    m2  = (freqs >= 1.8 * fb) & (freqs <= 2.2 * fb)
    m3  = (freqs >= 2.8 * fb) & (freqs <= 3.2 * fb)
    return {
        "fbase": fb, "sbase": sb,
        "f2nd": freqs[m2][int(np.argmax(mags[m2]))] if m2.any() else fb * 2,
        "s2nd": mags[m2].max() if m2.any() else 0.0,
        "f3rd": freqs[m3][int(np.argmax(mags[m3]))] if m3.any() else fb * 3,
        "s3rd": mags[m3].max() if m3.any() else 0.0,
    }


def extract_group_A(seg: dict, ch: str = "ir",
                    fs: int = FS) -> dict | None:
    sig   = seg[ch]
    peaks = detect_peaks(sig, fs)
    if len(peaks) < MIN_CYCLES + 1:
        return None
    cfs = [extract_one_cycle(sig, fs, peaks[i], peaks[i + 1])
           for i in range(len(peaks) - 1)]
    cfs = [f for f in cfs if f is not None]
    if len(cfs) < MIN_CYCLES:
        return None
    avg = {k: float(np.nanmean([c[k] for c in cfs])) for k in cfs[0]}
    avg.update(extract_fft(sig, fs))
    return {f"{ch}_A_{k}": v for k, v in avg.items()}


# ── Group C — MFCC ────────────────────────────────────────────────────────────
def extract_mfcc(sig: np.ndarray, fs: int = FS,
                 nc: int = MFCC_COEFFS,
                 nf: int = MFCC_FRAMES) -> dict:
    N    = len(sig)
    fl   = N // nf
    hop  = int(fl * 0.4)
    nfilt = 26
    fmax  = 50.0

    hz2mel = lambda h: 1127 * np.log1p(h / 700)
    mel2hz = lambda m: 700 * (np.expm1(m / 1127))
    mpts   = np.linspace(hz2mel(0), hz2mel(fmax), nfilt + 2)
    hpts   = mel2hz(mpts)
    bins   = np.clip(np.floor((fl + 1) * hpts / fs).astype(int), 0, fl // 2)
    fbank  = np.zeros((nfilt, fl // 2 + 1))
    for m in range(1, nfilt + 1):
        for k in range(bins[m - 1], bins[m]):
            if bins[m] != bins[m - 1]:
                fbank[m - 1, k] = (k - bins[m - 1]) / (bins[m] - bins[m - 1])
        for k in range(bins[m], bins[m + 1]):
            if bins[m + 1] != bins[m]:
                fbank[m - 1, k] = (bins[m + 1] - k) / (bins[m + 1] - bins[m])

    ceps = []
    s = 0
    while s + fl <= N:
        frame = sig[s:s + fl] * np.hamming(fl)
        sp    = np.maximum(np.abs(rfft(frame, n=fl))[:fl // 2 + 1], 1e-10)
        le    = np.log(fbank @ sp + 1e-10)
        ceps.append([np.sum(le * np.cos(
            np.pi * i / nfilt * (np.arange(nfilt) + 0.5)))
            for i in range(nc)])
        s += hop

    avg = np.nanmean(ceps, axis=0) if ceps else np.zeros(nc)
    return {f"mfcc_{i}": float(avg[i]) for i in range(nc)}


def extract_group_C(seg: dict, ch: str = "ir", fs: int = FS) -> dict:
    return {f"{ch}_C_{k}": v for k, v in extract_mfcc(seg[ch], fs).items()}


# ── Group D — Statistical ─────────────────────────────────────────────────────
def extract_group_D(seg: dict, ch: str = "ir", fs: int = FS) -> dict:
    sig = seg[ch]
    N   = len(sig)
    return {
        f"{ch}_D_mean": float(sig.mean()),
        f"{ch}_D_std" : float(sig.std()),
        f"{ch}_D_skew": float(skew(sig)),
        f"{ch}_D_kurt": float(kurtosis(sig)),
        f"{ch}_D_rms" : float(np.sqrt(np.mean(sig ** 2))),
        f"{ch}_D_ptp" : float(sig.max() - sig.min()),
        f"{ch}_D_zcr" : float(
            np.sum(np.abs(np.diff(np.sign(sig)))) / (2 * N)),
    }


# ── Full feature name order ────────────────────────────────────────────────────
def get_full_feat_names() -> list:
    """Returns ordered list of all feature names — must match training."""
    dummy_ir  = np.sin(np.linspace(0, 6 * np.pi, 1000)) * 500 + 100
    dummy_red = np.sin(np.linspace(0, 6 * np.pi, 1000)) * 300 + 50
    dummy_seg = {"ir": dummy_ir, "red": dummy_red,
                 "time": np.arange(1000) / FS}
    fA_ir  = extract_group_A(dummy_seg, "ir")  or {}
    fA_red = extract_group_A(dummy_seg, "red") or {}
    fC_ir  = extract_group_C(dummy_seg, "ir")
    fC_red = extract_group_C(dummy_seg, "red")
    fD_ir  = extract_group_D(dummy_seg, "ir")
    fD_red = extract_group_D(dummy_seg, "red")
    return (list(fA_ir) + list(fA_red) +
            list(fC_ir) + list(fC_red) +
            list(fD_ir) + list(fD_red) +
            ["meta_hr_avg", "meta_age", "meta_height", "meta_weight"])
