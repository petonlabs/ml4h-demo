#!/usr/bin/env python
"""
Generate a test XGBoost model for the API.

Creates a dummy but valid sklearn Pipeline with XGBoost that the API can load.
Useful for: testing without training, CI/CD, local development, demos.

Run: python create_test_model.py
Output: ../data/readmission_pipeline.joblib
"""

import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# Paths
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent / "data"
MODEL_PATH = DATA_DIR / "readmission_pipeline.joblib"

DATA_DIR.mkdir(parents=True, exist_ok=True)

print(f"Generating test model at: {MODEL_PATH}")

# Synthetic training data
np.random.seed(42)
n_samples = 1000

data = pd.DataFrame(
    {
        "age": np.random.choice([5, 15, 25, 35, 45, 55, 65, 75, 85, 95], n_samples),
        "num_medications": np.random.randint(0, 82, n_samples),
        "num_lab_procedures": np.random.randint(0, 133, n_samples),
        "num_procedures": np.random.randint(0, 7, n_samples),
        "number_diagnoses": np.random.randint(0, 17, n_samples),
        "time_in_hospital": np.random.randint(1, 15, n_samples),
        "insulin": np.random.choice(["No", "Steady", "Yes"], n_samples),
        "change": np.random.choice(["No", "Yes"], n_samples),
        "diabetesMed": np.random.choice(["No", "Yes"], n_samples),
    }
)

# Binary target
y = np.random.binomial(1, 0.3, n_samples)

print(f"Generated {n_samples} training samples")

# Numeric and categorical columns
numeric_cols = ["age", "num_medications", "num_lab_procedures",
                "num_procedures", "number_diagnoses", "time_in_hospital"]
categorical_cols = ["insulin", "change", "diabetesMed"]

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
    ]
)

# Pipeline with preprocessing
pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", XGBClassifier(random_state=42, n_estimators=50, max_depth=5, verbosity=0)),
    ]
)

# Train
print("Training XGBoost classifier...")
pipeline.fit(data, y)

# Save
joblib.dump(pipeline, MODEL_PATH)
print(f"✅ Model saved to {MODEL_PATH}")
print(f"   Size: {MODEL_PATH.stat().st_size / 1024:.1f} KB")
print(f"   Version: v1.0.0-xgboost-full-pipeline (test model)")
