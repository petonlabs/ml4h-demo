# 🏥 Clinical ML Deployment: From Notebook to API

**Week 8 Tuesday + Hands-On Demo**

This folder contains a **real, working FastAPI** that wraps the Week 8 XGBoost readmission prediction model. It transforms your trained model from a Jupyter notebook into a production-ready service — exactly what clinics need to actually use your ML.

---

## The Problem

A model in a notebook is **not** a deployed model:
- ❌ Clinicians can't access it (they don't have Python, Jupyter, or your code)
- ❌ No audit trail (who predicted what, when, and why?)
- ❌ No monitoring (is the model still working or has it drifted?)
- ❌ Not scalable (one clinician per notebook = 0 scale)
- ❌ No compliance (Kenya Data Protection Act 2019 violations)

## The Solution: An API

An **API** is a contract between two programs:
- ✅ Clinicians send patient data over HTTPS
- ✅ Server processes it using your model
- ✅ Returns a prediction + confidence + timestamp
- ✅ Every request is logged and auditable
- ✅ Scales to 1000s of simultaneous users
- ✅ Model updates happen on the server (no client changes needed)

---

## Quick Start (5 minutes)

### 1. Install dependencies (if not already done)

```bash
uv add fastapi "uvicorn[standard]"
```

### 2. Start the API server

From this folder (`week_eight/notebooks/`):

```bash
# Option A: Direct run
uv run uvicorn app:app --port 9000

# Option B: With auto-reload (for development)
uv run uvicorn app:app --reload --port 9000
```

You should see:
```
INFO:     Started server process [12345]
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:9000
```

### 3. Open the API in your browser

Visit any of these:

- **Interactive Playground:** http://localhost:9000/docs (Swagger UI)
- **Pretty Docs:** http://localhost:9000/redoc (ReDoc)
- **Landing Page:** http://localhost:9000/ (explains what APIs are)

### 4. Make your first prediction

In a **new terminal**, run the demo client:

```bash
uv run python demo_client.py
```

This shows four real-world interaction styles:
1. **Direct Python** — no network, instant
2. **HTTP POST** — realistic clinician app
3. **Batch** — screening 1000 patients overnight
4. **curl** — DevOps / CI pipeline

---

## API Endpoints (RESTful Design)

### Health & Metadata

| Verb | Path | Purpose |
|------|------|---------|
| `GET` | `/health` | Is the model ready? Use for load balancer checks. |
| `GET` | `/model` | Model version, features, expected value ranges. |
| `GET` | `/model/features` | **Which features drive risk?** Feature importance ranking. |
| `GET` | `/patients/sample` | Random valid patient for testing. Perfect for copy-paste into Swagger. |
| `GET` | `/metrics` | Monitoring: uptime, prediction counts, high-risk %, drift status. |

### Predictions (Core Business Logic)

| Verb | Path | Purpose | Returns |
|------|------|---------|---------|
| `POST` | `/predictions` | Predict for **one patient** (201 Created). | `{ risk_level, readmission_probability, model_version, timestamp }` |
| `POST` | `/predictions/batch` | Predict for **up to 50 patients** (201 Created). | List of results + summary stats (total, high_risk_count, high_risk_%) |

### Documentation

| Verb | Path | Purpose |
|------|------|---------|
| `GET` | `/docs` | Swagger UI — click "Try it out" to test endpoints in the browser |
| `GET` | `/redoc` | ReDoc — formatted API reference |
| `GET` | `/` | Landing page explaining what APIs are and why they matter |

---

## Example Requests

### Single Patient Prediction (curl)

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

**Response (201 Created):**
```json
{
  "risk_level": "LOW RISK",
  "readmission_probability": 0.01,
  "model_version": "v1.0.0-xgboost-full-pipeline",
  "timestamp": "2026-05-26T13:45:34.823757"
}
```

### Batch Prediction (Python + requests)

```python
import requests

patients = [
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

response = requests.post(
    "http://localhost:9000/predictions/batch",
    json={"patients": patients}
)
result = response.json()

print(f"Total: {result['total']}")
print(f"High-risk: {result['high_risk_count']} ({result['high_risk_percentage']:.1f}%)")
```

### Health Check (monitoring script)

```bash
# Load balancer / monitoring can check this every 10 seconds
curl -s http://localhost:9000/health | jq '.status'
# Output: "healthy"
```

---

## Code Architecture (SOLID Principles)

Even though it's one file, the code is organized for real production:

### **Services** (Business Logic)
```python
class ModelService(ABCModel):
    """Load & inference. Encapsulates joblib, pipeline, XGBoost."""
    def predict(self, data: pd.DataFrame) -> dict
    def get_feature_importances(self) -> dict

class PredictionService:
    """Business rules: age mapping, risk thresholding, stats tracking."""
    def predict_single(self, patient: PatientInput) -> PredictionOutput
    def predict_batch(self, batch: BatchRequest) -> BatchResponse
    def get_metrics(self) -> MetricsSnapshot
```

### **Pydantic Models** (Data Contracts)
Each endpoint has its own schema — no mixed concerns:
- `PatientInput` — what clinicians send
- `PredictionOutput` — what API returns
- `BatchRequest` / `BatchResponse` — batch contracts
- `HealthCheck`, `ModelInfo`, `MetricsSnapshot` — metadata

### **Dependency Injection** (Inversion of Control)
Routes never touch `joblib` or raw pandas. They receive services via `Depends()`:
```python
@app.post("/predictions")
async def predict(
    patient: PatientInput,
    service: PredictionService = Depends(get_prediction_service),
):
    return service.predict_single(patient)
```

### **Abstract Base** (Open/Closed Principle)
Future? Swap `ModelService` for `ONNXModelService` without touching routes:
```python
class ABCModel(ABC):
    @abstractmethod
    def predict(self, data: pd.DataFrame) -> dict: pass

class ONNXModelService(ABCModel):
    # Same interface, different implementation
```

---

## Educational Value: Why This Matters

### This IS a real deployment workflow:
1. ✅ Train model in notebook (week 5-6)
2. ✅ Save to disk (`readmission_pipeline.joblib`)
3. ✅ Wrap in API (`app.py`) — you're doing this
4. ✅ Deploy to cloud (AWS/GCP/Azure) — next week
5. ✅ Add monitoring & alerting — governance
6. ✅ Integrate with EHR — clinician workflow

### This is NOT production code yet, but it IS production-ready:
- ✅ RESTful design (proper HTTP verbs, status codes, resources)
- ✅ Validation (Pydantic rejects invalid input)
- ✅ Error handling (400/422 for bad requests)
- ✅ Documentation (auto-generated Swagger UI)
- ✅ Monitoring hooks (metrics endpoint)
- ✅ SOLID principles (reusable, testable, extendable)

**To go production, you'd add:**
- ❌ Authentication (OAuth2, API keys)
- ❌ Rate limiting (prevent abuse)
- ❌ Database (store predictions for audit)
- ❌ Encryption (HTTPS, data protection)
- ❌ Monitoring (Prometheus, DataDog)
- ❌ Load balancing (multiple replicas)
- ❌ Governance (DPIA, access control, role-based predictions)

But the **core API structure** you're learning here is identical.

---

## Next Steps

### Understand the code:
1. Read `app.py` top-to-bottom (400 lines, well-commented)
2. Notice the three layers: Pydantic models → Services → Routes
3. Ask: "How would I add authentication?" or "How would I store predictions in a database?"

### Extend it:
- Add a `/predictions/{id}` endpoint to retrieve a past prediction (requires a database)
- Add `/model/performance` endpoint showing AUC over time (from notebook's drift monitoring)
- Add `POST /model/retrain` endpoint to trigger retraining (with authentication!)
- Add authentication so only authorized clinicians can predict

### Deploy it:
- Push to Vercel / Railway (free tier)
- Deploy to AWS Lambda (serverless)
- Deploy to Google Cloud Run
- Deploy to Azure Container Instances

Then visit from Kenya on your phone. That's deployment.

---

## Files

- **`app.py`** — The API server (FastAPI + SOLID design)
- **`demo_client.py`** — Four ways to call the API (interactive demo)
- **`API_README.md`** — This file

---

## Troubleshooting

### "Connection refused on localhost:9000"
→ Start the server first: `uv run uvicorn app:app --port 9000`

### "ModuleNotFoundError: No module named 'fastapi'"
→ Install deps: `uv add fastapi "uvicorn[standard]"`

### "FileNotFoundError: ../data/readmission_pipeline.joblib"
→ Run `week8_mon_full_pipeline.ipynb` first to train and save the model

### "Unknown categories in columns" warning
→ Normal (from the notebook's age mapping). Doesn't affect predictions.

### Swagger UI won't open
→ Check your browser's JavaScript is enabled, or use curl/Python instead

---

## Questions?

- **What IS an API?** → See the landing page (`GET /`)
- **How does a clinician actually use this?** → See `demo_client.py`
- **What's the difference between `/predictions` and `/predictions/batch`?** → See the endpoints table
- **How would I add login?** → Research FastAPI + OAuth2
- **How would this survive a 1000-patient spike?** → Load balancing + databases

This is the last mile of ML — the hardest and most important part.

---

*Week 8: From Notebook to Production — The Last Mile*
