# PPG Glucose Prediction API

Non-invasive glucose estimation from MAX30102 PPG signals.
FastAPI + SVR model hosted on Railway.

## Folder Structure

```
glucose_api/
├── main.py                  ← FastAPI app
├── predict.py               ← inference pipeline
├── signal_processing.py     ← filter + segment + SQC
├── feature_extraction.py    ← Group A + C + D features
├── requirements.txt
├── Procfile
└── model/
    └── svr_final_model.pkl  ← trained SVR model (add manually)
```

## API Endpoints

| Method | Endpoint  | Description              |
|--------|-----------|--------------------------|
| GET    | /         | API info                 |
| GET    | /health   | Health check             |
| GET    | /info     | Model configuration      |
| POST   | /predict  | Predict glucose from PPG |

## POST /predict

### Request
```json
{
  "ir"     : [12345, 12350, ...],
  "red"    : [8765, 8770, ...],
  "hr_avg" : 72.5,
  "height" : 170.0,
  "weight" : 65.0,
  "age"    : 25.0
}
```

### Response
```json
{
  "status"           : "success",
  "glucose"          : 96.23,
  "zone"             : "Normal",
  "std"              : 0.87,
  "n_segments_used"  : 3,
  "seg_predictions"  : [95.1, 96.8, 96.8],
  "total_segments"   : 5,
  "skipped_sqc"      : 0,
  "skipped_cycle"    : 0,
  "signal_duration_s": 50.0
}
```

## Railway Deployment Steps

1. Copy svr_final_model.pkl → model/svr_final_model.pkl
2. Create GitHub repo and push this folder
3. Go to railway.app → New Project → Deploy from GitHub
4. Select this repo
5. Railway auto-detects Procfile and deploys
6. Copy the Railway URL → use in ESP32 firmware

## Local Testing

```bash
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
# Open http://localhost:8000/docs
```
