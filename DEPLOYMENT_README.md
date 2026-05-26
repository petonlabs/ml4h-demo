# 🏥 Clinical ML Deployment API — Standalone Edition

**Ready-to-ship FastAPI for the Week 8 XGBoost readmission prediction model.**

This is a **self-contained, production-ready API** that wraps a trained XGBoost model and exposes it as a REST service. Perfect for:

- ✅ Student learning (real deployment patterns)
- ✅ Clinical prototypes (running locally)
- ✅ Cloud deployment (AWS, GCP, Azure, Vercel)
- ✅ GitHub sharing (one folder, no external dependencies)

---

## Quick Start (2 minutes)

### Option 1: Using `uv` (recommended)

```bash
# Clone or download this folder, then:
cd api_deployment

# Install
uv sync

# Start server
uv run uvicorn app:app --port 9000

# In another terminal, run demo
uv run python demo_client.py
```

### Option 2: Using `pip`

```bash
# Clone or download this folder, then:
cd api_deployment

# Install
pip install -r requirements.txt

# Start server
python -m uvicorn app:app --port 9000

# In another terminal, run demo
python demo_client.py
```

### Option 3: Using `pip` with setup.py

```bash
cd api_deployment
pip install -e .
python -m uvicorn app:app --port 9000
```

---

## What You Get

### 📁 Files

- **`app.py`** — FastAPI application (26 KB, well-commented)
- **`demo_client.py`** — Interactive demo (9.3 KB, 4 scenarios)
- **`README.md`** — Student guide (complete API reference)
- **`QUICK_START.txt`** — Quick reference card
- **`pyproject.toml`** — uv project configuration
- **`uv.lock`** — Dependency lock (reproducible builds)

### 🔌 API Endpoints

| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/` | HTML landing page (what is an API?) |
| GET | `/health` | Health check (is model ready?) |
| GET | `/model` | Model metadata |
| GET | `/model/features` | Feature importance ranking |
| GET | `/patients/sample` | Random sample patient |
| POST | `/predictions` | Single prediction (201 Created) |
| POST | `/predictions/batch` | Batch predictions (up to 50) |
| GET | `/metrics` | Monitoring metrics |

### 📚 Documentation

- **Interactive:** http://localhost:9000/docs (Swagger UI)
- **Formatted:** http://localhost:9000/redoc (ReDoc)
- **Educational:** http://localhost:9000/ (landing page)

---

## Example Requests

### Single Prediction (curl)

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

### Batch Prediction (Python)

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
print(f"High-risk: {result['high_risk_count']}")
```

### Check if Model is Ready

```bash
curl http://localhost:9000/health | jq .
```

---

## Important: The Model File

This API loads a **pre-trained model** from disk. You need:

### Path (relative to app.py)

```
api_deployment/
├── app.py
└── ../data/readmission_pipeline.joblib   ← Model must be here
```

### Two Options

**Option A: Local Model (if you have the training notebook)**

From the main project:
```bash
# Run the training notebook first
python week_eight/notebooks/week8_mon_full_pipeline.ipynb

# This saves: week_eight/data/readmission_pipeline.joblib
# The API will find it automatically
```

**Option B: Standalone (generate a dummy model for testing)**

```bash
# Create a dummy model for testing (won't be clinically valid)
python create_test_model.py
```

---

## Code Architecture (SOLID Principles)

Even though it's one file, production patterns are visible:

### Services (Single Responsibility)
```python
class ModelService(ABCModel):
    """Load & inference"""
    def predict(self, data: pd.DataFrame) -> dict
    def get_feature_importances(self) -> dict

class PredictionService:
    """Business logic"""
    def predict_single(self, patient: PatientInput) -> PredictionOutput
    def predict_batch(self, batch: BatchRequest) -> BatchResponse
```

### Data Models (Interface Segregation)
```python
class PatientInput(BaseModel):      # What clinicians send
class PredictionOutput(BaseModel):  # What API returns
class BatchRequest(BaseModel):      # Batch input contract
class BatchResponse(BaseModel):     # Batch output contract
```

### Dependency Injection (Inversion of Control)
```python
@app.post("/predictions")
async def predict_single(
    patient: PatientInput,
    service: PredictionService = Depends(get_prediction_service),
):
    return service.predict_single(patient)
```

---

## Deployment Options

This code works on any platform. Choose one:

### Local (Development)
```bash
uv run uvicorn app:app --reload --port 9000
```

### Vercel (Easiest)
```bash
npm install -g vercel
vercel deploy
```

### Railway
```bash
# Connect your GitHub repo, auto-deploys
```

### AWS Lambda
```bash
pip install zappa
zappa init
zappa deploy prod
```

### Google Cloud Run
```bash
gcloud run deploy clinic-api --source . --platform managed
```

### Docker
```bash
# Create a Dockerfile
docker build -t clinic-api .
docker run -p 9000:9000 clinic-api
```

---

## Troubleshooting

### "FileNotFoundError: readmission_pipeline.joblib"

The model file doesn't exist. Two solutions:

**A) Create it locally:**
```bash
python week_eight/notebooks/week8_mon_full_pipeline.ipynb
# Then copy: cp week_eight/data/readmission_pipeline.joblib week_eight/data/
```

**B) For GitHub, include a placeholder:**
```bash
# Create a dummy model for CI/testing
python -c "import pickle; pickle.dump({'dummy': True}, open('week_eight/data/readmission_pipeline.joblib', 'wb'))"
```

### "ModuleNotFoundError: fastapi"

Install dependencies:
```bash
uv sync    # if using uv
# or
pip install -r requirements.txt
```

### "Port 9000 already in use"

Use a different port:
```bash
uv run uvicorn app:app --port 9001
```

### "Address already in use" (even after killing)

Wait a moment for the OS to release the port, then retry.

---

## Testing

### Run the demo client

```bash
# With server running, in another terminal:
uv run python demo_client.py
```

### Or test with curl

```bash
# Health check
curl http://localhost:9000/health

# Sample prediction
curl -X POST http://localhost:9000/predictions \
  -H "Content-Type: application/json" \
  -d @sample.json
```

### Or test with Python

```python
import requests

# Single prediction
resp = requests.post(
    "http://localhost:9000/predictions",
    json={"age": "[60-70)", "num_medications": 16, ...}
)
print(resp.json())
```

---

## Extending It

### Add Authentication

```python
from fastapi.security import HTTPBearer, HTTPAuthCredentials
from fastapi import Depends, HTTPException

security = HTTPBearer()

async def verify_token(credentials: HTTPAuthCredentials = Depends(security)):
    if credentials.credentials != "your-secret-token":
        raise HTTPException(status_code=403)
    return credentials.credentials

@app.post("/predictions")
async def predict_single(
    patient: PatientInput,
    token: str = Depends(verify_token),
    service: PredictionService = Depends(get_prediction_service),
):
    return service.predict_single(patient)
```

### Add Database Storage

```python
from sqlalchemy import create_engine, Column, DateTime, String, Float
from sqlalchemy.orm import declarative_base

Base = declarative_base()

class PredictionRecord(Base):
    __tablename__ = "predictions"
    id = Column(String, primary_key=True)
    patient_id = Column(String)
    risk_level = Column(String)
    probability = Column(Float)
    timestamp = Column(DateTime)

engine = create_engine("sqlite:///predictions.db")
Base.metadata.create_all(engine)
```

### Add Rate Limiting

```bash
pip install slowapi
```

```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.post("/predictions")
@limiter.limit("100/minute")
async def predict_single(...):
    pass
```

---

## Production Checklist

Before deploying clinically:

- [ ] Authenticate users (OAuth2, API keys)
- [ ] Store predictions in database (audit trail)
- [ ] Use HTTPS (encryption)
- [ ] Add rate limiting (abuse prevention)
- [ ] Monitor uptime (Prometheus, DataDog)
- [ ] Log all requests (ELK stack, CloudWatch)
- [ ] Set up alerting (model drift, errors)
- [ ] Document governance (Kenya DPIA)
- [ ] Test load (k6, locust)
- [ ] Set up CI/CD (GitHub Actions, GitLab CI)

---

## File Structure

```
api_deployment/
├── app.py                      # FastAPI server
├── demo_client.py              # Interactive demo
├── README.md                   # This file
├── QUICK_START.txt             # Quick reference
├── requirements.txt            # pip dependencies
├── pyproject.toml              # uv + setuptools config
├── setup.py                    # setuptools entry point
│
└── week_eight/data/            (required, but outside this folder)
    └── readmission_pipeline.joblib
```

---

## What's NOT Included (Intentionally)

This is **teaching code**, not production-grade. Missing:

- ❌ Database (store predictions)
- ❌ Authentication (who can access?)
- ❌ Rate limiting (prevent abuse)
- ❌ Encryption (HTTPS)
- ❌ Load balancing (multiple replicas)
- ❌ Real monitoring (Prometheus)
- ❌ DPIA docs (Kenya compliance)
- ❌ Role-based access (admin vs clinician)

**You can ADD these as exercises!**

---

## Learning Path

1. **Understand:** Read `README.md`
2. **Run:** `uv run uvicorn app:app --port 9000`
3. **Explore:** Visit http://localhost:9000/docs
4. **Demo:** `uv run python demo_client.py`
5. **Extend:** Add a feature to `app.py`
6. **Deploy:** Push to Vercel/AWS/GCP

---

## References

- **FastAPI docs:** https://fastapi.tiangolo.com
- **Uvicorn:** https://www.uvicorn.org
- **Pydantic:** https://docs.pydantic.dev
- **RESTful design:** https://restfulapi.net
- **SOLID principles:** https://en.wikipedia.org/wiki/SOLID

---

## License

MIT — feel free to use, modify, deploy.

---

## Questions?

This is a **teaching API**, designed for the Week 8 "Last Mile" curriculum in the Clinical ML course (Kenya, 2026).

- **For students:** See `README.md`
- **For instructors:** See the main project's `DEPLOYMENT_API_GUIDE.md`

---

**Made with ❤️ for clinicians in East Africa.**

*Week 8: From Notebook to Production — The Last Mile*
