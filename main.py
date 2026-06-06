"""
main.py — with full request/response logging
"""

from fastapi                  import FastAPI, HTTPException, Request
from fastapi.middleware.cors  import CORSMiddleware
from pydantic                 import BaseModel, Field
from typing                   import List, Optional
import os, time, logging, uvicorn

from predict import run_pipeline

# ── Logging setup ─────────────────────────────────────────────────
logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("glucose_api")

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

# ── Request logging middleware ─────────────────────────────────────
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    logger.info(f"→ {request.method} {request.url.path}")
    response = await call_next(request)
    elapsed  = time.time() - start
    logger.info(f"← {request.method} {request.url.path} "
                f"status={response.status_code} "
                f"time={elapsed:.2f}s")
    return response

# ── Schemas ───────────────────────────────────────────────────────
class PPGRequest(BaseModel):
    ir  : List[float] = Field(...,
            description="IR channel samples")
    red : List[float] = Field(...,
            description="Red channel samples")
    age : Optional[float] = Field(0.0,
            description="Patient age (optional)")

class PPGResponse(BaseModel):
    status            : str
    glucose           : Optional[float]       = None
    zone              : Optional[str]         = None
    std               : Optional[float]       = None
    n_segments_used   : Optional[int]         = None
    seg_predictions   : Optional[List[float]] = None
    total_segments    : Optional[int]         = None
    skipped_sqc       : Optional[int]         = None
    skipped_cycle     : Optional[int]         = None
    signal_duration_s : Optional[float]       = None
    message           : Optional[str]         = None

# ── Routes ────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"service":"PPG Glucose Prediction API",
            "version":"1.0.0","status":"running"}

@app.get("/health")
def health():
    logger.info("Health check called")
    return {"status": "healthy"}

@app.get("/info")
def info():
    try:
        from predict import _load_model
        pkg, _ = _load_model()
        return {
            "model_type"      : pkg.get("model_type",      "SVR"),
            "kernel"          : pkg.get("kernel",           "rbf"),
            "C"               : pkg.get("C",                10),
            "epsilon"         : pkg.get("epsilon",          0.1),
            "n_features"      : pkg.get("n_features",       0),
            "feature_names"   : pkg.get("feature_names",    []),
            "lopo_mae"        : pkg.get("lopo_mae",         None),
            "lopo_rmse"       : pkg.get("lopo_rmse",        None),
            "lopo_r2"         : pkg.get("lopo_r2",          None),
            "lopo_mard"       : pkg.get("lopo_mard",        None),
            "n_train_patients": pkg.get("n_train_patients", None),
            "n_train_segments": pkg.get("n_train_segments", None),
        }
    except Exception as e:
        logger.error(f"Info error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict", response_model=PPGResponse)
def predict(req: PPGRequest):
    logger.info(f"Predict called — IR:{len(req.ir)} "
                f"Red:{len(req.red)} age:{req.age}")

    # ── Validate ──────────────────────────────────────────────────
    if len(req.ir) < 1000:
        logger.warning(f"IR too short: {len(req.ir)}")
        raise HTTPException(status_code=422,
            detail=f"IR too short: {len(req.ir)}, need >=1000")

    if len(req.ir) > 12000:
        logger.warning(f"IR too long: {len(req.ir)}")
        raise HTTPException(status_code=422,
            detail=f"IR too long: {len(req.ir)}, max 12000")

    if len(req.ir) != len(req.red):
        logger.warning(f"Length mismatch IR:{len(req.ir)} Red:{len(req.red)}")
        raise HTTPException(status_code=422,
            detail=f"IR ({len(req.ir)}) and Red ({len(req.red)}) must match")

    # ── Run pipeline ──────────────────────────────────────────────
    try:
        logger.info("Starting pipeline...")
        t0     = time.time()
        result = run_pipeline(
            ir_raw = req.ir,
            red_raw= req.red,
            age    = req.age or 0.0,
        )
        elapsed = time.time() - t0
        logger.info(f"Pipeline done in {elapsed:.2f}s — "
                    f"status={result['status']} "
                    f"glucose={result.get('glucose','N/A')} "
                    f"segments={result.get('n_segments_used','N/A')}")
        return PPGResponse(**result)

    except Exception as e:
        logger.error(f"Pipeline exception: {type(e).__name__}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500,
            detail=f"Prediction failed: {str(e)}")

# ── Entry point ───────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)