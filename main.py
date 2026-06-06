"""
main.py
───────
FastAPI application — glucose prediction from PPG signals.
Hosted on Render (Docker).
"""

from fastapi                  import FastAPI, HTTPException
from fastapi.middleware.cors  import CORSMiddleware
from pydantic                 import BaseModel, Field
from typing                   import List, Optional
import os
import uvicorn

from predict import run_pipeline

# ── App setup ─────────────────────────────────────────────────────
app = FastAPI(
    title       = "PPG Glucose Prediction API",
    description = "Non-invasive glucose estimation from MAX30102 PPG signals",
    version     = "1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# ── Request / Response schemas ────────────────────────────────────
class PPGRequest(BaseModel):
    ir  : List[float] = Field(...,
            description="IR channel samples — any length ≥1000 (10s min)")
    red : List[float] = Field(...,
            description="Red channel samples — same length as IR")
    age : Optional[float] = Field(0.0,
            description="Patient age in years (optional)")
    # hr_avg, height, weight removed — not used by model

class PPGResponse(BaseModel):
    status            : str
    glucose           : Optional[float]      = None
    zone              : Optional[str]        = None
    std               : Optional[float]      = None
    n_segments_used   : Optional[int]        = None
    seg_predictions   : Optional[List[float]]= None
    total_segments    : Optional[int]        = None
    skipped_sqc       : Optional[int]        = None
    skipped_cycle     : Optional[int]        = None
    signal_duration_s : Optional[float]      = None
    message           : Optional[str]        = None

# ── Routes ────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "service"  : "PPG Glucose Prediction API",
        "version"  : "1.0.0",
        "status"   : "running",
        "endpoints": {
            "POST /predict": "Predict glucose from PPG signal",
            "GET  /health" : "Health check",
            "GET  /info"   : "Model info",
        }
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/info")
def info():
    """Returns model configuration info."""
    try:
        from predict import _load_model
        pkg, _ = _load_model()
        return {
            "model_type"      : pkg.get("model_type",       "SVR"),
            "kernel"          : pkg.get("kernel",            "rbf"),
            "C"               : pkg.get("C",                 10),
            "epsilon"         : pkg.get("epsilon",           0.1),
            "n_features"      : pkg.get("n_features",        0),
            "feature_names"   : pkg.get("feature_names",     []),
            "lopo_mae"        : pkg.get("lopo_mae",          None),
            "lopo_rmse"       : pkg.get("lopo_rmse",         None),
            "lopo_r2"         : pkg.get("lopo_r2",           None),
            "lopo_mard"       : pkg.get("lopo_mard",         None),
            "n_train_patients": pkg.get("n_train_patients",  None),
            "n_train_segments": pkg.get("n_train_segments",  None),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict", response_model=PPGResponse)
def predict(req: PPGRequest):
    """
    Predict blood glucose from PPG signals.

    Accepts any signal length between 10s (1000 samples) and
    120s (12000 samples) at 100Hz.

    Processing pipeline:
      - Bandpass filter (0.3-7Hz) + cubic spline baseline correction
      - 10s non-overlapping segmentation
      - Skip first 2 and last segment
      - SQC per segment (Fisher Kappa + SQI)
      - Extract Group A + C + D features
      - Select GA-identified features
      - SVR predict per segment → mean prediction
    """
    # ── Validate IR length ────────────────────────────────────────
    if len(req.ir) < 1000:
        raise HTTPException(
            status_code=422,
            detail=(f"IR array too short: {len(req.ir)} samples. "
                    f"Minimum 1000 samples (10 seconds at 100Hz)."))

    if len(req.ir) > 12000:
        raise HTTPException(
            status_code=422,
            detail=(f"IR array too long: {len(req.ir)} samples. "
                    f"Maximum 12000 samples (120 seconds at 100Hz)."))

    # ── Validate IR and Red same length ───────────────────────────
    if len(req.ir) != len(req.red):
        raise HTTPException(
            status_code=422,
            detail=(f"IR length ({len(req.ir)}) and "
                    f"Red length ({len(req.red)}) must be equal."))

    # ── Run pipeline ──────────────────────────────────────────────
    try:
        result = run_pipeline(
            ir_raw = req.ir,
            red_raw= req.red,
            age    = req.age or 0.0,
        )
        return PPGResponse(**result)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}")


# ── Entry point ───────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)