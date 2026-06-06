"""
predict.py
──────────
Full inference pipeline:
  raw IR/Red arrays
    → bandpass filter + baseline correction
    → segment (10s windows)
    → skip first 2 + last segment
    → SQC check per segment
    → extract Group A + C + D features
    → select GA-identified features (height/weight excluded)
    → SVR predict per segment
    → average → return result
"""

import numpy as np
import pickle
import os

from signal_processing  import full_filter, segment_signal, check_sqc
from feature_extraction import (extract_group_A, extract_group_C,
                                 extract_group_D, get_full_feat_names)

# ── Load model once at startup ────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "svr_final_model.pkl")

_model_pkg  = None
_full_names = None


def _load_model():
    """Load model pickle once and cache it."""
    global _model_pkg, _full_names
    if _model_pkg is None:
        with open(MODEL_PATH, "rb") as f:
            _model_pkg = pickle.load(f)
        _full_names = get_full_feat_names()
        print(f"[predict] Model loaded")
        print(f"  Features  : {len(_model_pkg['feature_names'])}")
        print(f"  Kernel    : {_model_pkg.get('kernel', 'rbf')}")
        print(f"  C         : {_model_pkg.get('C', 10)}")
        print(f"  LOPO RMSE : {_model_pkg.get('lopo_rmse', 'N/A')}")
    return _model_pkg, _full_names


def clarke_zone(pred: float) -> str:
    """
    Classify glucose level into clinical category.
    (No reference glucose available from device)
    """
    if pred < 70:
        return "Low (Hypoglycemic)"
    elif pred < 100:
        return "Normal"
    elif pred < 126:
        return "Pre-diabetic"
    else:
        return "Diabetic"


def run_pipeline(
    ir_raw : list,
    red_raw: list,
    hr_avg : float,
    fs     : int   = 100,
) -> dict:
    """
    Full inference pipeline — raw signal to glucose prediction.

    Parameters
    ----------
    ir_raw  : Raw IR channel samples (list of ints/floats)
    red_raw : Raw Red channel samples
    hr_avg  : Average heart rate (bpm) computed from HR_Valid samples
    age     : Patient age in years (optional — set 0 if unknown)
    fs      : Sampling frequency Hz (default 100)

    Returns
    -------
    dict with keys:
        status, glucose, zone, std, n_segments_used,
        seg_predictions, total_segments,
        skipped_sqc, skipped_cycle, signal_duration_s
    """
    pkg, full_names = _load_model()
    model      = pkg["model"]
    scaler     = pkg["scaler"]
    feat_names = pkg["feature_names"]   # GA-selected feature names

    ir  = np.array(ir_raw,  dtype=np.float64)
    red = np.array(red_raw, dtype=np.float64)

    # ── Validate ──────────────────────────────────────────────────────────────
    if len(ir) < fs * 10:
        return {
            "status" : "error",
            "message": f"Signal too short: {len(ir)} samples, "
                       f"need >= {fs * 10} (10 seconds minimum)",
        }

    # ── Step 1: Filter ────────────────────────────────────────────────────────
    ir_f  = full_filter(ir,  fs)
    red_f = full_filter(red, fs)

    # ── Step 2: Segment ───────────────────────────────────────────────────────
    segs       = segment_signal(ir_f, red_f, fs=fs)
    total_segs = len(segs)

    if total_segs < 4:
        return {
            "status" : "error",
            "message": f"Too few segments: {total_segs}. "
                       f"Need >= 4 (signal needs to be at least 40 seconds)",
        }

    # ── Step 3: Extract features (skip first 2 + last segment) ───────────────
    valid_feats   = []
    skipped_sqc   = 0
    skipped_cycle = 0

    for seg in segs:
        sid = seg["seg_id"]

        # Skip first 2 segments — sensor settling artifacts
        if sid < 2:
            continue

        # Skip last segment — signal end artifacts
        if sid == total_segs - 1:
            continue

        # Signal quality check
        if not check_sqc(seg, fs):
            skipped_sqc += 1
            continue

        # Extract Group A — morphological + FFT
        fA_ir  = extract_group_A(seg, "ir",  fs)
        fA_red = extract_group_A(seg, "red", fs)
        if fA_ir is None or fA_red is None:
            skipped_cycle += 1
            continue

        # Build feature dict
        feat = {}
        feat.update(fA_ir)
        feat.update(fA_red)
        feat.update(extract_group_C(seg, "ir",  fs))   # MFCC
        feat.update(extract_group_C(seg, "red", fs))
        feat.update(extract_group_D(seg, "ir",  fs))   # Statistical
        feat.update(extract_group_D(seg, "red", fs))

        # Metadata — only hr_avg and age used by model
        # height and weight set to 0 (excluded from GA features)
        feat["meta_hr_avg"] = hr_avg
        feat["meta_age"]    = 0.0    # age not used by model
        feat["meta_height"] = 0.0    # not used by model
        feat["meta_weight"] = 0.0    # not used by model

        # Build full feature vector (must match training order)
        vec = np.array(
            [feat.get(n, np.nan) for n in full_names],
            dtype=np.float32
        )
        valid_feats.append(vec)

    # ── Check valid segments ───────────────────────────────────────────────────
    if not valid_feats:
        return {
            "status" : "error",
            "message": (
                f"No valid segments after quality check. "
                f"Total={total_segs}, "
                f"skipped_sqc={skipped_sqc}, "
                f"skipped_cycle={skipped_cycle}. "
                f"Check signal quality and finger placement."
            ),
        }

    # ── Step 4: Select GA features ────────────────────────────────────────────
    X_all = np.nan_to_num(np.array(valid_feats), nan=0.0)

    # Select only GA-identified features in correct order
    X_sel = np.array([
        [row[full_names.index(n)] for n in feat_names]
        for row in X_all
    ], dtype=np.float32)

    # ── Step 5: Scale + predict ───────────────────────────────────────────────
    X_sc      = scaler.transform(X_sel)
    seg_preds = model.predict(X_sc).tolist()

    # ── Step 6: Average predictions ───────────────────────────────────────────
    glucose = float(np.mean(seg_preds))
    std     = float(np.std(seg_preds))
    zone    = clarke_zone(glucose)

    return {
        "status"           : "success",
        "glucose"          : round(glucose, 2),
        "zone"             : zone,
        "std"              : round(std, 2),
        "n_segments_used"  : len(seg_preds),
        "seg_predictions"  : [round(p, 2) for p in seg_preds],
        "total_segments"   : total_segs,
        "skipped_sqc"      : skipped_sqc,
        "skipped_cycle"    : skipped_cycle,
        "signal_duration_s": round(len(ir) / fs, 1),
    }