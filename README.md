# FastAPI Model Server

A production-grade REST API wrapper around a trained XGBoost model. Exposes predictions, metadata, and monitoring endpoints with auto-generated interactive documentation.

## Quick Start

```bash
# Install dependencies
uv sync

# Generate test model (if you don't have one)
python create_test_model.py

# Start server
uv run uvicorn app:app --port 9000
```

Then visit:
- **Interactive API docs:** http://localhost:9000/docs
- **Alternative docs:** http://localhost:9000/redoc
- **Landing page:** http://localhost:9000/

## What You Get

### 8 Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/` | HTML page explaining the API |
| GET | `/health` | Health status (for load balancers) |
| GET | `/model` | Model metadata, features, ranges |
| GET | `/model/features` | Feature importance ranking |
| GET | `/patients/sample` | Random valid input example |
| POST | `/predictions` | Single prediction (201 Created) |
| POST | `/predictions/batch` | Batch predictions, up to 50 records |
| GET | `/metrics` | Request count, uptime, high-risk % |

### Data Validation

Input is validated by Pydantic models before reaching the model:
```python
class PatientInput(BaseModel):
    age: str              # e.g. "[60-70)"
    num_medications: int
    num_lab_procedures: int
    num_procedures: int
    number_diagnoses: int
    time_in_hospital: int
    insulin: str          # "No" | "Steady" | "Yes"
    change: str           # "No" | "Yes"
    diabetesMed: str      # "No" | "Yes"
```

### Responses

Single prediction:
```json
{
  "risk_level": "LOW RISK",
  "readmission_probability": 0.01,
  "model_version": "v1.0.0-xgboost-full-pipeline",
  "timestamp": "2026-05-26T13:45:34.823757"
}
```

Batch response (example with 2 patients):
```json
{
  "results": [
    { "risk_level": "LOW RISK", "readmission_probability": 0.01, ... },
    { "risk_level": "HIGH RISK", "readmission_probability": 0.72, ... }
  ],
  "total": 2,
  "high_risk_count": 1,
  "high_risk_percentage": 50.0
}
```

## Model File

The API loads a pickled sklearn/XGBoost pipeline:

```
../data/readmission_pipeline.joblib
```

### If you don't have a model file

Generate a test model locally:
```bash
python create_test_model.py
```

This creates a valid but untrained model for testing.

### If you have a trained model

Place it at `../data/readmission_pipeline.joblib` (relative to this folder).

## Example Requests

### Single prediction (curl)
```bash
curl -X POST http://localhost:9000/predictions \
  -H "Content-Type: application/json" \
  -d '{
    "age": "[60-70)",
    "num_medications": 16,
    "num_lab_procedures": 41,
    "num_procedures": 6,
    "number_diagnoses": 9,
    "time_in_hospital": 3,
    "insulin": "Yes",
    "change": "No",
    "diabetesMed": "Yes"
  }'
```

### Batch prediction (Python)
```python
import requests

response = requests.post(
    "http://localhost:9000/predictions/batch",
    json={
        "patients": [
            {
                "age": "[60-70)",
                "num_medications": 16,
                "num_lab_procedures": 41,
                "num_procedures": 6,
                "number_diagnoses": 9,
                "time_in_hospital": 3,
                "insulin": "Yes",
                "change": "No",
                "diabetesMed": "Yes",
            },
            # ... more patients ...
        ]
    }
)

result = response.json()
print(f"Total: {result['total']}")
print(f"High-risk: {result['high_risk_count']} ({result['high_risk_percentage']:.1f}%)")
```

### Get feature importance
```bash
curl http://localhost:9000/model/features | jq .
```

### Health check
```bash
curl http://localhost:9000/health | jq .
```

## Code Structure

### SOLID Principles

- **ModelService** — Loads pipeline, handles inference, extracts feature importances
- **PredictionService** — Business logic, age mapping, risk thresholding, stats
- **Pydantic models** — Input/output validation, auto-generated docs
- **Dependency injection** — Services injected via FastAPI `Depends()`
- **Abstract base** — `ABCModel` allows swapping model implementations

### Adding Features

To add authentication:
```python
from fastapi.security import HTTPBearer

security = HTTPBearer()

@app.post("/predictions")
async def predict(
    patient: PatientInput,
    credentials: HTTPAuthCredentials = Depends(security),
    service: PredictionService = Depends(get_prediction_service),
):
    # Verify token here
    return service.predict_single(patient)
```

To add request logging:
```python
@app.middleware("http")
async def log_request(request: Request, call_next):
    start = datetime.now()
    response = await call_next(request)
    duration = (datetime.now() - start).total_seconds()
    print(f"{request.method} {request.url.path} {response.status_code} {duration:.2f}s")
    return response
```

## Deployment

This code works on any platform. No changes needed.

### Local
```bash
uv run uvicorn app:app --port 9000
```

### Production (with reload disabled)
```bash
uv run uvicorn app:app --host 0.0.0.0 --port 9000 --workers 4
```

### Docker
```dockerfile
FROM python:3.13-slim
WORKDIR /app
COPY . .
RUN pip install -e .
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "9000"]
```

### Cloud
- **Vercel:** `vercel deploy`
- **Railway:** Connect GitHub repo, auto-deploys
- **AWS Lambda:** `zappa init && zappa deploy prod`
- **Google Cloud Run:** `gcloud run deploy --source .`

## Testing

### Run demo client
```bash
uv run python demo_client.py
```

Shows 4 interaction patterns:
1. Direct Python (no HTTP)
2. HTTP single request
3. HTTP batch request
4. curl command

### Run pytest
```bash
uv pip install pytest pytest-asyncio httpx
pytest
```

## Files

- `app.py` — FastAPI application
- `demo_client.py` — Example client showing all interaction patterns
- `create_test_model.py` — Generate test model if needed
- `pyproject.toml` — Dependencies and metadata
- `uv.lock` — Locked dependency versions
- `.github/workflows/test.yml` — CI/CD pipeline

## Troubleshooting

**Model file not found**
```
FileNotFoundError: ../data/readmission_pipeline.joblib
```
→ Run `python create_test_model.py` to generate a test model

**Port already in use**
```bash
uv run uvicorn app:app --port 9001
```

**ModuleNotFoundError: fastapi**
```bash
uv sync
```

**Server won't start on Windows**
→ Use `python -m uvicorn app:app --port 9000` instead

## Performance

- Single prediction: ~10ms (model-dependent)
- Batch (50 patients): ~200ms
- Startup: ~2s (model loading)

Add caching for frequently accessed endpoints:
```python
from functools import lru_cache

@lru_cache(maxsize=100)
def get_feature_importances():
    return model.get_feature_importances()
```

## What's NOT Included

This is intentionally minimal:

- ❌ Database (store predictions)
- ❌ Authentication (API keys)
- ❌ Rate limiting
- ❌ Real monitoring (Prometheus)
- ❌ Logging
- ❌ Caching

Add these as needed for your use case.

## License

MIT
