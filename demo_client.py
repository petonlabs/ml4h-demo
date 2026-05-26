"""
Demo Client: Four Ways to Interact with a Deployed ML Model

This script shows students every way to call a deployed model:
  1. Direct Python (no HTTP overhead)
  2. HTTP via requests (realistic production scenario)
  3. Batch via HTTP (scale)
  4. curl (what DevOps / CLI users see)

USAGE:
  # Start the server in one terminal
  uv run uvicorn app:app --reload --port 8000

  # Run this script in another terminal
  uv run python demo_client.py
"""

import sys
import json
import time
from pathlib import Path

import requests

# Add parent to path so we can import app
sys.path.insert(0, str(Path(__file__).parent))
from app import PatientInput, get_prediction_service, get_model_service

# Sample patient data
SAMPLE_1 = PatientInput(
    age="[60-70)",
    num_medications=16,
    num_lab_procedures=41,
    num_procedures=6,
    number_diagnoses=9,
    time_in_hospital=3,
    insulin="Yes",
    change="No",
    diabetesMed="Yes",
)

SAMPLE_2 = PatientInput(
    age="[50-60)",
    num_medications=8,
    num_lab_procedures=12,
    num_procedures=2,
    number_diagnoses=4,
    time_in_hospital=2,
    insulin="No",
    change="Yes",
    diabetesMed="No",
)

SAMPLE_3 = PatientInput(
    age="[70-80)",
    num_medications=24,
    num_lab_procedures=67,
    num_procedures=10,
    number_diagnoses=12,
    time_in_hospital=5,
    insulin="Steady",
    change="No",
    diabetesMed="Yes",
)


def print_banner(title: str):
    """Print a formatted section banner."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def demo_1_direct_python():
    """
    Demo 1: Direct Python call (no HTTP).
    Shows: model loaded, services initialized, instant inference.
    When you'd use this: unit tests, internal admin tools, local validation.
    """
    print_banner("DEMO 1: Direct Python Call")
    print(
        "\nWhen you'd use this: Testing, internal tools, zero network overhead.\n"
    )

    model = get_model_service()
    service = get_prediction_service()

    print(f"Patient: Wanjiku (67 y/o, high-risk profile)")
    print(f"  Medications: {SAMPLE_1.num_medications}")
    print(f"  Lab procedures: {SAMPLE_1.num_lab_procedures}")
    print(f"  Time in hospital: {SAMPLE_1.time_in_hospital} days\n")

    result = service.predict_single(SAMPLE_1)

    print(f"✅ Result (instant, no network):")
    print(f"  Risk Level: {result.risk_level}")
    print(f"  Probability: {result.readmission_probability:.1%}")
    print(f"  Model: {result.model_version}")


def demo_2_http_single():
    """
    Demo 2: HTTP POST to /predictions (realistic).
    Shows: real network request, JSON serialization, server handling.
    When you'd use this: clinician apps, mobile, web frontend, third-party integrations.
    """
    print_banner("DEMO 2: HTTP Single Prediction")
    print(
        "\nWhen you'd use this: Real clinician apps, web/mobile frontends, "
        "third-party integrations.\n"
    )

    url = "http://localhost:9000/predictions"
    payload = SAMPLE_2.model_dump()

    print(f"Patient: Mwangi (52 y/o, low-risk profile)")
    print(f"  Medications: {SAMPLE_2.num_medications}")
    print(f"  Lab procedures: {SAMPLE_2.num_lab_procedures}")
    print(f"  Time in hospital: {SAMPLE_2.time_in_hospital} day\n")

    print(f"HTTP Request:")
    print(f"  POST {url}")
    print(f"  Body: {json.dumps(payload, indent=4)}\n")

    try:
        response = requests.post(url, json=payload, timeout=5)
        response.raise_for_status()
        result = response.json()

        print(f"✅ HTTP 201 Created:")
        print(f"  Risk Level: {result['risk_level']}")
        print(f"  Probability: {result['readmission_probability']:.1%}")
        print(f"  Timestamp: {result['timestamp']}")
    except requests.exceptions.ConnectionError:
        print(
            "❌ ERROR: Could not connect to server.\n"
            "   Start the server first:\n"
            "   uv run uvicorn app:app --reload --port 8000"
        )
    except requests.exceptions.RequestException as e:
        print(f"❌ HTTP Error: {e}")


def demo_3_batch():
    """
    Demo 3: Batch prediction (POST /predictions/batch).
    Shows: handling multiple patients, summary statistics, efficiency.
    When you'd use this: nightly batch runs, population screening, audits.
    """
    print_banner("DEMO 3: Batch Prediction (3 Patients)")
    print(
        "\nWhen you'd use this: Nightly batch runs, population health screening, "
        "quality audits.\n"
    )

    url = "http://localhost:9000/predictions/batch"
    batch_payload = {
        "patients": [
            SAMPLE_1.model_dump(),
            SAMPLE_2.model_dump(),
            SAMPLE_3.model_dump(),
        ]
    }

    print(f"Patients:")
    print(f"  1. Wanjiku (67 y/o)")
    print(f"  2. Mwangi (52 y/o)")
    print(f"  3. Kamau (78 y/o)\n")

    print(f"HTTP Request:")
    print(f"  POST {url}")
    print(f"  Body: 3 patient records\n")

    try:
        response = requests.post(url, json=batch_payload, timeout=5)
        response.raise_for_status()
        result = response.json()

        print(f"✅ HTTP 201 Created:")
        print(f"  Total predictions: {result['total']}")
        print(f"  High-risk count: {result['high_risk_count']}")
        print(f"  High-risk %: {result['high_risk_percentage']:.1f}%\n")

        print(f"Individual Results:")
        for i, pred in enumerate(result["results"], 1):
            print(
                f"  {i}. {pred['risk_level']:12} "
                f"({pred['readmission_probability']:.1%})"
            )
    except requests.exceptions.ConnectionError:
        print(
            "❌ ERROR: Could not connect to server.\n"
            "   Start the server first:\n"
            "   uv run uvicorn app:app --reload --port 8000"
        )
    except requests.exceptions.RequestException as e:
        print(f"❌ HTTP Error: {e}")


def demo_4_curl():
    """
    Demo 4: curl (command-line, DevOps / infrastructure).
    Shows: what a curl request looks like, for CLI users, CI/CD pipelines.
    When you'd use this: debugging API from terminal, CI/CD pipelines, monitoring.
    """
    print_banner("DEMO 4: curl (CLI / DevOps)")
    print(
        "\nWhen you'd use this: Terminal debugging, CI/CD pipelines, "
        "infrastructure automation, monitoring.\n"
    )

    curl_cmd = """
curl -X POST http://localhost:9000/predictions \\
  -H "Content-Type: application/json" \\
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
    """

    print("Copy-paste this into your terminal:\n")
    print(curl_cmd)

    print("\nExpected response:")
    print(
        """{
  "risk_level": "HIGH RISK",
  "readmission_probability": 0.72,
  "model_version": "v1.0.0-xgboost-full-pipeline",
  "timestamp": "2026-05-26T14:23:45.123456"
}"""
    )


def demo_other_endpoints():
    """
    Demo 5: Show what other endpoints exist and why they matter.
    """
    print_banner("BONUS: Other Endpoints")
    print()

    endpoints = [
        ("GET", "/health", "Is the model ready? Use for load balancer checks."),
        ("GET", "/model", "Model metadata: version, features, value ranges."),
        (
            "GET",
            "/model/features",
            "Which patient factors drive risk? Feature importance ranking.",
        ),
        ("GET", "/patients/sample", "Random valid sample patient for testing."),
        ("GET", "/metrics", "Monitoring: uptime, prediction counts, drift status."),
        ("GET", "/docs", "Swagger UI — interactive API explorer (browser)."),
        ("GET", "/redoc", "ReDoc — formatted documentation (browser)."),
    ]

    for method, path, desc in endpoints:
        print(f"  {method:6} {path:25} — {desc}")


def main():
    """Run all demos."""
    print("\n")
    print("█" * 70)
    print("█  Clinical ML Deployment: Four Ways to Call a Model")
    print("█" * 70)

    print(
        "\n"
        "📚 EDUCATION: Each demo shows a real production scenario.\n"
        "   You're not learning toy code — you're learning how clinics\n"
        "   will actually use your deployed model.\n"
    )

    # Demo 1: Direct Python
    demo_1_direct_python()

    time.sleep(1)

    # Demo 2: HTTP single
    demo_2_http_single()

    time.sleep(1)

    # Demo 3: Batch
    demo_3_batch()

    time.sleep(1)

    # Demo 4: curl
    demo_4_curl()

    time.sleep(1)

    # Demo 5: Other endpoints
    demo_other_endpoints()

    # Summary
    print_banner("Why This Matters (Week 8 Recap)")
    print(
        """
A trained model in a notebook is useless to a clinician. An API transforms it into:

  1. ACCESSIBLE: Clinicians access from their native tools (web, mobile, EHR).
  2. MONITORED: Every prediction is logged, timestamped, auditable (Kenya DPIA).
  3. VERSIONED: You can deploy model v2 without breaking v1 clients.
  4. SCALABLE: One model serves 1000 clinicians simultaneously.
  5. GOVERNED: Access control, encryption, role-based predictions (admin vs clinician).

This is the last mile — the hardest and most important part of the job.

Next: Deploy to a cloud (AWS/GCP/Azure), add databases, monitoring,
encryption, and integrate with Kenya's healthcare infrastructure (KEMRI, PPB).
    """
    )
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
