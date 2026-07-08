"""
predict.py — with full logging
"""

import numpy as np
import pickle, os, logging, time

from signal_processing  import full_filter, segment_signal, check_sqc
from feature_extraction import (extract_group_A, extract_group_C,
                                 extract_group_D, get_full_feat_names)

logger     = logging.getLogger("glucose_api")
MODEL_PATH = os.path.join(os.path.dirname(__file__),
                          "model", "svr_final_model.pkl")

_model_pkg  = None
_full_names = None


def _load_model():
    global _model_pkg, _full_names
    if _model_pkg is None:
        logger.info(f"Loading model from {MODEL_PATH}")
        with open(MODEL_PATH, "rb") as f:
            _model_pkg = pickle.load(f)
        _full_names = get_full_feat_names()
        logger.info(f"Model loaded — "
                    f"type={_model_pkg.get('model_type','SVR')} "
                    f"features={len(_model_pkg['feature_names'])} "
                    f"kernel={_model_pkg.get('kernel','rbf')} "
                    f"C={_model_pkg.get('C',10)}")
    return _model_pkg, _full_names


def run_pipeline(ir_raw, red_raw, age=0.0, hr=0.0, fs=100) -> dict:

    pkg, full_names = _load_model()
    model      = pkg["model"]
    scaler     = pkg["scaler"]
    feat_names = pkg["feature_names"]

    ir  = np.array(ir_raw,  dtype=np.float64)
    red = np.array(red_raw, dtype=np.float64)

    logger.info(f"Signal received — IR:{len(ir)} "
                f"Red:{len(red)} duration:{len(ir)/fs:.1f}s hr:{hr}")

    if len(ir) < fs * 10:
        logger.warning(f"Signal too short: {len(ir)} samples")
        return {"status":"error",
                "message":f"Signal too short: {len(ir)} samples"}

    # ── Filter ────────────────────────────────────────────────────
    logger.info("Applying bandpass filter + baseline correction...")
    t0    = time.time()
    ir_f  = full_filter(ir,  fs)
    red_f = full_filter(red, fs)
    logger.info(f"Filtering done in {time.time()-t0:.2f}s")

    # ── Segment ───────────────────────────────────────────────────
    segs       = segment_signal(ir_f, red_f, fs=fs)
    total_segs = len(segs)
    logger.info(f"Segmented into {total_segs} segments")

    if total_segs < 4:
        logger.warning(f"Too few segments: {total_segs}")
        return {"status":"error",
                "message":f"Too few segments: {total_segs}, need >=4"}

    # ── Extract features ──────────────────────────────────────────
    valid_feats   = []
    skipped_sqc   = 0
    skipped_cycle = 0

    for seg in segs:
        sid = seg["seg_id"]

        if sid < 2:
            logger.info(f"Seg {sid} skipped (first 2 settling)")
            continue
        if sid == total_segs - 1:
            logger.info(f"Seg {sid} skipped (last segment)")
            continue

        if not check_sqc(seg, fs):
            skipped_sqc += 1
            logger.info(f"Seg {sid} failed SQC")
            continue

        fA_ir  = extract_group_A(seg, "ir",  fs)
        fA_red = extract_group_A(seg, "red", fs)
        if fA_ir is None or fA_red is None:
            skipped_cycle += 1
            logger.info(f"Seg {sid} failed cycle extraction")
            continue

        feat = {}
        feat.update(fA_ir)
        feat.update(fA_red)
        feat.update(extract_group_C(seg, "ir",  fs))
        feat.update(extract_group_C(seg, "red", fs))
        feat.update(extract_group_D(seg, "ir",  fs))
        feat.update(extract_group_D(seg, "red", fs))
        feat["meta_hr_avg"] = hr
        feat["meta_age"]    = age
        feat["meta_height"] = 0.0
        feat["meta_weight"] = 0.0

        vec = np.array([feat.get(n, np.nan) for n in full_names],
                       dtype=np.float32)
        valid_feats.append(vec)
        logger.info(f"Seg {sid} features extracted OK")

    logger.info(f"Valid segments: {len(valid_feats)} | "
                f"skipped_sqc: {skipped_sqc} | "
                f"skipped_cycle: {skipped_cycle}")

    if not valid_feats:
        logger.error("No valid segments after all checks!")
        return {
            "status" : "error",
            "message": (f"No valid segments. "
                        f"total={total_segs} "
                        f"skipped_sqc={skipped_sqc} "
                        f"skipped_cycle={skipped_cycle}"),
        }

    # ── Select GA features ────────────────────────────────────────
    logger.info(f"Selecting {len(feat_names)} GA features...")
    X_all = np.nan_to_num(np.array(valid_feats), nan=0.0)
    X_sel = np.array([
        [row[full_names.index(n)] for n in feat_names]
        for row in X_all
    ], dtype=np.float32)
    logger.info(f"Feature matrix shape: {X_sel.shape}")

    # ── Scale + predict ───────────────────────────────────────────
    logger.info("Scaling and predicting...")
    X_sc      = scaler.transform(X_sel)
    seg_preds = model.predict(X_sc).tolist()
    glucose   = float(np.mean(seg_preds))
    std       = float(np.std(seg_preds))

    logger.info(f"Prediction: glucose={glucose:.2f} "
                f"std={std:.2f} "
                f"segments={len(seg_preds)} "
                f"preds={[round(p,2) for p in seg_preds]}")

    return {
        "status"           : "success",
        "glucose"          : round(glucose, 2),
        "std"              : round(std, 2),
        "n_segments_used"  : len(seg_preds),
        "seg_predictions"  : [round(p, 2) for p in seg_preds],
        "total_segments"   : total_segs,
        "skipped_sqc"      : skipped_sqc,
        "skipped_cycle"    : skipped_cycle,
        "signal_duration_s": round(len(ir) / fs, 1),
    }