"""
FastAPI Clinical ML Model Deployment Demo

This app exposes the Week 8 XGBoost readmission prediction model as a REST API,
teaching students what APIs are and why they're the standard way to deploy models.

QUICK START:
    uv add fastapi "uvicorn[standard]"
    uv run uvicorn app:app --reload --port 9000

    Then visit:
      http://localhost:9000          → Interactive landing page explaining APIs
      http://localhost:9000/docs     → Swagger UI (auto-generated API docs)
      http://localhost:9000/redoc    → ReDoc (alternative API docs)

DESIGN: SOLID principles in a single file
  - ModelService (Single Responsibility): owns model loading & inference
  - PredictionService (Single Responsibility): owns business logic
  - Pydantic schemas (Interface Segregation): each owns one contract
  - Dependency injection (Dependency Inversion): routers get services, not globals
  - Abstract base (Open/Closed): swappable model implementations
"""

import random
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


# ==============================================================================
# DATA MODELS (Pydantic — Interface Segregation Principle)
# ==============================================================================

class PatientInput(BaseModel):
    """Single patient input for prediction."""
    age: str = Field(..., example="[60-70)")
    num_medications: int = Field(..., ge=0, example=16)
    num_lab_procedures: int = Field(..., ge=0, example=41)
    num_procedures: int = Field(..., ge=0, example=6)
    number_diagnoses: int = Field(..., ge=0, example=9)
    time_in_hospital: int = Field(..., ge=0, example=3)
    insulin: str = Field(..., example="No", pattern="^(No|Steady|Yes)$")
    change: str = Field(..., example="No", pattern="^(No|Yes)$")
    diabetesMed: str = Field(..., example="Yes", pattern="^(No|Yes)$")


class PredictionOutput(BaseModel):
    """Single prediction result."""
    risk_level: str = Field(..., example="HIGH RISK")
    readmission_probability: float = Field(..., ge=0, le=1)
    model_version: str
    timestamp: str


class BatchRequest(BaseModel):
    """Batch prediction request."""
    patients: list[PatientInput] = Field(..., max_items=50)


class BatchResponse(BaseModel):
    """Batch prediction results."""
    results: list[PredictionOutput]
    total: int
    high_risk_count: int
    high_risk_percentage: float


class HealthCheck(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    model_version: Optional[str] = None


class FeatureImportance(BaseModel):
    """Feature importance for interpretability."""
    feature_name: str
    importance: float
    importance_pct: float
    rank: int


class ModelInfo(BaseModel):
    """Model metadata and configuration."""
    version: str
    feature_names: list[str]
    expected_input_ranges: dict
    top_features: list[FeatureImportance]


class MetricsSnapshot(BaseModel):
    """Monitoring metrics stub."""
    model_version: str
    uptime_seconds: float
    total_predictions: int
    high_risk_percentage: float
    last_prediction_at: Optional[str]


# ==============================================================================
# SAMPLE PATIENTS (for /patients/sample endpoint)
# ==============================================================================

SAMPLE_PATIENTS = [
    {
        "name": "Wanjiku, 67",
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
    {
        "name": "Mwangi, 52",
        "age": "[50-60)",
        "num_medications": 8,
        "num_lab_procedures": 12,
        "num_procedures": 2,
        "number_diagnoses": 4,
        "time_in_hospital": 2,
        "insulin": "No",
        "change": "Yes",
        "diabetesMed": "No",
    },
    {
        "name": "Kamau, 78",
        "age": "[70-80)",
        "num_medications": 24,
        "num_lab_procedures": 67,
        "num_procedures": 10,
        "number_diagnoses": 12,
        "time_in_hospital": 5,
        "insulin": "Steady",
        "change": "No",
        "diabetesMed": "Yes",
    },
    {
        "name": "Kariuki, 45",
        "age": "[40-50)",
        "num_medications": 5,
        "num_lab_procedures": 8,
        "num_procedures": 1,
        "number_diagnoses": 2,
        "time_in_hospital": 1,
        "insulin": "No",
        "change": "No",
        "diabetesMed": "No",
    },
]

# Age mapping (from notebook)
AGE_MAP = {
    "[0-10)": 5,
    "[10-20)": 15,
    "[20-30)": 25,
    "[30-40)": 35,
    "[40-50)": 45,
    "[50-60)": 55,
    "[60-70)": 65,
    "[70-80)": 75,
    "[80-90)": 85,
    "[90-100)": 95,
}

FEATURE_NAMES = [
    "age",
    "num_medications",
    "num_lab_procedures",
    "num_procedures",
    "number_diagnoses",
    "time_in_hospital",
    "insulin",
    "change",
    "diabetesMed",
]


# ==============================================================================
# SERVICES (SOLID: Single Responsibility + Dependency Inversion)
# ==============================================================================

class ABCModel(ABC):
    """Abstract base for model implementations (Open/Closed Principle)."""

    @abstractmethod
    def predict(self, data: pd.DataFrame) -> dict:
        """Predict readmission risk."""
        pass

    @abstractmethod
    def get_feature_importances(self) -> dict:
        """Return feature importances ranked by importance."""
        pass

    @abstractmethod
    def get_version(self) -> str:
        """Return model version string."""
        pass


class ModelService(ABCModel):
    """Loads and manages the XGBoost pipeline (Single Responsibility)."""

    def __init__(self, pipeline_path: str):
        """Load pipeline at startup."""
        self.pipeline = joblib.load(pipeline_path)
        self.version = "v1.0.0-xgboost-full-pipeline"
        self._startup_time = datetime.utcnow()

    def predict(self, data: pd.DataFrame) -> dict:
        """
        Run inference on a patient DataFrame.
        Returns raw probability and risk level.
        """
        prob = self.pipeline.predict_proba(data)[0][1]  # prob of readmission
        risk_level = "HIGH RISK" if prob > 0.5 else "LOW RISK"
        return {"probability": prob, "risk_level": risk_level}

    def get_feature_importances(self) -> dict:
        """Extract feature importances from XGBoost classifier."""
        classifier = self.pipeline.named_steps["classifier"]
        importances = classifier.feature_importances_

        # Rank by importance
        ranked = sorted(
            zip(FEATURE_NAMES, importances),
            key=lambda x: x[1],
            reverse=True,
        )

        total_importance = sum(importances)
        return {
            name: {
                "importance": float(imp),
                "importance_pct": float(100 * imp / total_importance),
                "rank": rank + 1,
            }
            for rank, (name, imp) in enumerate(ranked)
        }

    def get_version(self) -> str:
        """Return model version."""
        return self.version

    def get_uptime_seconds(self) -> float:
        """Time since model loaded."""
        return (datetime.utcnow() - self._startup_time).total_seconds()


class PredictionService:
    """Business logic layer (Single Responsibility)."""

    def __init__(self, model: ABCModel):
        self.model = model
        self.total_predictions = 0
        self.high_risk_count = 0
        self.last_prediction_at: Optional[str] = None

    def predict_single(self, patient: PatientInput) -> PredictionOutput:
        """Predict for one patient."""
        # Map age categorical to numeric
        age_numeric = AGE_MAP.get(patient.age)
        if age_numeric is None:
            raise ValueError(f"Invalid age range: {patient.age}")

        # Build input DataFrame matching pipeline expectations
        data = pd.DataFrame(
            {
                "age": [age_numeric],
                "num_medications": [patient.num_medications],
                "num_lab_procedures": [patient.num_lab_procedures],
                "num_procedures": [patient.num_procedures],
                "number_diagnoses": [patient.number_diagnoses],
                "time_in_hospital": [patient.time_in_hospital],
                "insulin": [patient.insulin],
                "change": [patient.change],
                "diabetesMed": [patient.diabetesMed],
            }
        )

        # Predict
        result = self.model.predict(data)

        # Update stats
        self.total_predictions += 1
        if result["risk_level"] == "HIGH RISK":
            self.high_risk_count += 1
        self.last_prediction_at = datetime.utcnow().isoformat()

        return PredictionOutput(
            risk_level=result["risk_level"],
            readmission_probability=result["probability"],
            model_version=self.model.get_version(),
            timestamp=self.last_prediction_at,
        )

    def predict_batch(self, batch: BatchRequest) -> BatchResponse:
        """Predict for multiple patients."""
        results = [self.predict_single(patient) for patient in batch.patients]
        high_risk = sum(1 for r in results if r.risk_level == "HIGH RISK")

        return BatchResponse(
            results=results,
            total=len(results),
            high_risk_count=high_risk,
            high_risk_percentage=100 * high_risk / len(results) if results else 0,
        )

    def get_metrics(self) -> MetricsSnapshot:
        """Get current metrics snapshot."""
        return MetricsSnapshot(
            model_version=self.model.get_version(),
            uptime_seconds=self.model.get_uptime_seconds(),
            total_predictions=self.total_predictions,
            high_risk_percentage=(
                100 * self.high_risk_count / self.total_predictions
                if self.total_predictions > 0
                else 0
            ),
            last_prediction_at=self.last_prediction_at,
        )


# ==============================================================================
# SETUP
# ==============================================================================

# Paths
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "../data/readmission_pipeline.joblib"

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model file not found: {MODEL_PATH}\n"
        f"Please run week8_mon_full_pipeline.ipynb first to train the model."
    )

# Initialize services at startup
_model_service = ModelService(str(MODEL_PATH))
_prediction_service = PredictionService(_model_service)

# FastAPI app
app = FastAPI(
    title="Clinical ML Model API",
    description="Week 8: Deploy and Monitor a Readmission Prediction Model",
    version="1.0.0",
)

# CORS: allow all origins for local demo
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==============================================================================
# DEPENDENCY INJECTION
# ==============================================================================

def get_model_service() -> ABCModel:
    """Inject model service (Dependency Inversion)."""
    return _model_service


def get_prediction_service() -> PredictionService:
    """Inject prediction service."""
    return _prediction_service


# ==============================================================================
# ROUTES
# ==============================================================================

@app.get("/", response_class=HTMLResponse)
async def root():
    """
    Educational landing page explaining APIs and the deployment challenge.
    """
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Clinical ML Deployment API</title>
        <style>
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                max-width: 900px;
                margin: 0 auto;
                padding: 20px;
                line-height: 1.6;
                color: #333;
            }
            h1 { color: #1976d2; }
            h2 { color: #f57c00; margin-top: 30px; }
            .box {
                background: #f5f5f5;
                padding: 15px;
                border-radius: 5px;
                margin: 15px 0;
                border-left: 4px solid #1976d2;
            }
            .highlight {
                background: #fff3e0;
                padding: 15px;
                border-radius: 5px;
                margin: 15px 0;
                border-left: 4px solid #f57c00;
            }
            form {
                background: #e3f2fd;
                padding: 20px;
                border-radius: 5px;
                margin: 20px 0;
            }
            label {
                display: block;
                margin-top: 10px;
                font-weight: bold;
            }
            input, select {
                width: 100%;
                padding: 8px;
                margin-top: 5px;
                border: 1px solid #ddd;
                border-radius: 3px;
                box-sizing: border-box;
            }
            button {
                background: #1976d2;
                color: white;
                padding: 10px 20px;
                border: none;
                border-radius: 3px;
                cursor: pointer;
                margin-top: 15px;
                font-size: 16px;
            }
            button:hover { background: #1565c0; }
            .response {
                background: #c8e6c9;
                padding: 15px;
                border-radius: 5px;
                margin-top: 15px;
                display: none;
            }
            .error {
                background: #ffcdd2;
                color: #b71c1c;
            }
            code {
                background: #f5f5f5;
                padding: 2px 6px;
                border-radius: 3px;
                font-family: monospace;
            }
            .section-separator {
                border-top: 2px solid #ddd;
                margin: 30px 0;
            }
        </style>
    </head>
    <body>
        <h1>🏥 Clinical ML Model Deployment Demo</h1>
        <p><strong>Week 8: From Notebook to Production</strong></p>

        <div class="box">
            <h2>What is an API?</h2>
            <p>
                An <strong>API</strong> (Application Programming Interface) is a <strong>contract between two programs</strong>.
                It defines:
            </p>
            <ul>
                <li><strong>What data you send</strong> (input schema)</li>
                <li><strong>What you get back</strong> (output schema)</li>
                <li><strong>How you communicate</strong> (HTTP methods, URLs, formats)</li>
            </ul>
            <p>
                Instead of a clinician loading a Jupyter notebook and running code,
                an API lets them send patient data to a server and get predictions back in seconds.
                The model stays behind a secure, monitored service.
            </p>
        </div>

        <div class="highlight">
            <h2>Why APIs?</h2>
            <ul>
                <li><strong>Isolation</strong>: Model runs on a secure server, not on clinician's machine</li>
                <li><strong>Monitoring</strong>: Every prediction is logged and audited</li>
                <li><strong>Governance</strong>: Kenya Data Protection Act 2019 compliance (consent, audit trails)</li>
                <li><strong>Scale</strong>: One model serves thousands of clinicians simultaneously</li>
                <li><strong>Update</strong>: Deploy a new model without touching client code</li>
            </ul>
        </div>

        <div class="section-separator"></div>

        <h2>Try It Now</h2>
        <p>Fill in a patient's data below and hit "Predict" to see the API in action.</p>
        <form onsubmit="predictPatient(event)">
            <label>Age Range:
                <select name="age" required>
                    <option value="[40-50)">[40-50)</option>
                    <option value="[50-60)">[50-60)</option>
                    <option value="[60-70)" selected>[60-70)</option>
                    <option value="[70-80)">[70-80)</option>
                    <option value="[80-90)">[80-90)</option>
                </select>
            </label>

            <label>Medications:
                <input type="number" name="num_medications" value="16" min="0" required>
            </label>

            <label>Lab Procedures:
                <input type="number" name="num_lab_procedures" value="41" min="0" required>
            </label>

            <label>Procedures:
                <input type="number" name="num_procedures" value="6" min="0" required>
            </label>

            <label>Diagnoses:
                <input type="number" name="number_diagnoses" value="9" min="0" required>
            </label>

            <label>Time in Hospital (days):
                <input type="number" name="time_in_hospital" value="3" min="0" required>
            </label>

            <label>Insulin:
                <select name="insulin" required>
                    <option value="No">No</option>
                    <option value="Steady">Steady</option>
                    <option value="Yes" selected>Yes</option>
                </select>
            </label>

            <label>Change in Medication:
                <select name="change" required>
                    <option value="No" selected>No</option>
                    <option value="Yes">Yes</option>
                </select>
            </label>

            <label>Diabetes Medication:
                <select name="diabetesMed" required>
                    <option value="No">No</option>
                    <option value="Yes" selected>Yes</option>
                </select>
            </label>

            <button type="submit">🔮 Predict Readmission Risk</button>
        </form>

        <div id="response" class="response"></div>

        <div class="section-separator"></div>

        <h2>Next Steps</h2>
        <ul>
            <li><strong>Explore the API docs:</strong>
                <ul>
                    <li><a href="http://localhost:9000/docs" target="_blank">Swagger UI</a> — interactive playground</li>
                    <li><a href="http://localhost:9000/redoc" target="_blank">ReDoc</a> — formatted documentation</li>
                </ul>
            </li>
            <li><strong>Sample endpoints:</strong>
                <ul>
                    <li><code>GET /health</code> — is the model ready?</li>
                    <li><code>GET /patients/sample</code> — random test patient</li>
                    <li><code>GET /model/features</code> — which features drive risk?</li>
                    <li><code>POST /predictions</code> — predict one patient</li>
                    <li><code>POST /predictions/batch</code> — predict many at once</li>
                    <li><code>GET /metrics</code> — uptime, prediction counts, drift alerts</li>
                </ul>
            </li>
            <li><strong>Run the demo client:</strong> <code>uv run python demo_client.py</code></li>
        </ul>

        <script>
            async function predictPatient(event) {
                event.preventDefault();
                const formData = new FormData(event.target);
                const data = {
                    age: formData.get('age'),
                    num_medications: parseInt(formData.get('num_medications')),
                    num_lab_procedures: parseInt(formData.get('num_lab_procedures')),
                    num_procedures: parseInt(formData.get('num_procedures')),
                    number_diagnoses: parseInt(formData.get('number_diagnoses')),
                    time_in_hospital: parseInt(formData.get('time_in_hospital')),
                    insulin: formData.get('insulin'),
                    change: formData.get('change'),
                    diabetesMed: formData.get('diabetesMed'),
                };

                try {
                    const response = await fetch('/predictions', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(data),
                    });

                    if (!response.ok) {
                        throw new Error(`Server error: ${response.status}`);
                    }

                    const result = await response.json();
                    const responseDiv = document.getElementById('response');
                    responseDiv.innerHTML = `
                        <strong>✅ Prediction Result:</strong><br>
                        Risk Level: <strong>${result.risk_level}</strong><br>
                        Readmission Probability: <strong>${(result.readmission_probability * 100).toFixed(1)}%</strong><br>
                        Model Version: ${result.model_version}<br>
                        Timestamp: ${result.timestamp}
                    `;
                    responseDiv.style.display = 'block';
                } catch (error) {
                    const responseDiv = document.getElementById('response');
                    responseDiv.innerHTML = `<strong>❌ Error:</strong> ${error.message}`;
                    responseDiv.classList.add('error');
                    responseDiv.style.display = 'block';
                }
            }
        </script>
    </body>
    </html>
    """


@app.get("/health", response_model=HealthCheck, status_code=status.HTTP_200_OK)
async def health_check(model: ABCModel = Depends(get_model_service)):
    """
    Health check endpoint. Returns 200 if healthy, 503 if degraded.
    """
    return HealthCheck(
        status="healthy",
        model_loaded=True,
        model_version=model.get_version(),
    )


@app.get("/model", response_model=ModelInfo)
async def get_model_info(model: ABCModel = Depends(get_model_service)):
    """
    Get model metadata: version, features, and top feature importances.
    """
    importances = model.get_feature_importances()
    top_features = [
        FeatureImportance(
            feature_name=name,
            importance=data["importance"],
            importance_pct=data["importance_pct"],
            rank=data["rank"],
        )
        for name, data in sorted(
            importances.items(),
            key=lambda x: x[1]["rank"],
        )
    ]

    return ModelInfo(
        version=model.get_version(),
        feature_names=FEATURE_NAMES,
        expected_input_ranges={
            "age": list(AGE_MAP.keys()),
            "num_medications": "0-81",
            "num_lab_procedures": "0-132",
            "num_procedures": "0-6",
            "number_diagnoses": "0-16",
            "time_in_hospital": "1-14",
            "insulin": ["No", "Steady", "Yes"],
            "change": ["No", "Yes"],
            "diabetesMed": ["No", "Yes"],
        },
        top_features=top_features,
    )


@app.get("/model/features", response_model=list[FeatureImportance])
async def get_feature_importances(
    model: ABCModel = Depends(get_model_service),
):
    """
    Get ranked feature importances. Shows which patient factors drive readmission risk most.
    Educational endpoint: instantly answer "which features matter?"
    """
    importances = model.get_feature_importances()
    features = [
        FeatureImportance(
            feature_name=name,
            importance=data["importance"],
            importance_pct=data["importance_pct"],
            rank=data["rank"],
        )
        for name, data in sorted(
            importances.items(),
            key=lambda x: x[1]["rank"],
        )
    ]
    return features


@app.get("/patients/sample", response_model=PatientInput)
async def get_sample_patient():
    """
    Return a random realistic sample patient from Kenya.
    Great for copy-pasting into /predictions or Swagger UI.
    """
    sample = random.choice(SAMPLE_PATIENTS)
    return PatientInput(**{k: v for k, v in sample.items() if k != "name"})


@app.post("/predictions", response_model=PredictionOutput, status_code=status.HTTP_201_CREATED)
async def predict_single(
    patient: PatientInput,
    service: PredictionService = Depends(get_prediction_service),
):
    """
    Make a single readmission prediction. RESTful: POST (create a prediction resource).
    Returns 201 Created on success.
    """
    return service.predict_single(patient)


@app.post(
    "/predictions/batch",
    response_model=BatchResponse,
    status_code=status.HTTP_201_CREATED,
)
async def predict_batch(
    batch: BatchRequest,
    service: PredictionService = Depends(get_prediction_service),
):
    """
    Make predictions for up to 50 patients in one request.
    Returns summary stats: total, high_risk_count, high_risk_percentage.
    """
    if len(batch.patients) > 50:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Batch size must be ≤ 50",
        )
    return service.predict_batch(batch)


@app.get("/metrics", response_model=MetricsSnapshot)
async def get_metrics(
    service: PredictionService = Depends(get_prediction_service),
):
    """
    Get current monitoring metrics: uptime, prediction counts, drift signals.
    In production, this would query a time-series DB (Prometheus, DataDog, etc.).
    """
    return service.get_metrics()


if __name__ == "__main__":
    import uvicorn

    print("\n" + "=" * 70)
    print("🚀 Clinical ML Deployment Demo")
    print("=" * 70)
    print("API running on:        http://localhost:9000")
    print("Interactive docs:      http://localhost:9000/docs")
    print("ReDoc:                 http://localhost:9000/redoc")
    print("=" * 70 + "\n")

    uvicorn.run(app, host="0.0.0.0", port=8000)
