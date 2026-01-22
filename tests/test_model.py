import pandas as pd
import pytest
from pathlib import Path
from fastapi.testclient import TestClient

from src.model_loader import ModelPredictor
from src.main import app


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models/best_model_lightgbm.pkl"
TOP_FEATURES_PATH = BASE_DIR / "data/top_features.csv"
SEUIL_METIER = 0.35

client = TestClient(app)


def get_test_predictor():
    return ModelPredictor(
        model_path=MODEL_PATH,
        top_features_path=TOP_FEATURES_PATH,
        threshold=SEUIL_METIER
    )

def get_valid_payload():
    predictor = get_test_predictor()
    return {feat: 0.0 for feat in predictor.top_features}


def test_predict_proba_range():
    predictor = get_test_predictor()
    df = pd.DataFrame([get_valid_payload()])
    proba = predictor.predict_proba(df)[0]

    assert 0 <= proba <= 1

def test_predict_class_logic():
    predictor = get_test_predictor()
    df = pd.DataFrame([get_valid_payload()])
    proba = predictor.predict_proba(df)[0]
    classe = predictor.predict_class(df)[0]

    assert classe in [0, 1]
    assert (proba >= SEUIL_METIER) == (classe == 1)


def test_api_root_ok():
    response = client.get("/")
    assert response.status_code == 200

def test_api_predict_success():
    response = client.post("/predict", json=get_valid_payload())

    assert response.status_code == 200
    data = response.json()

    assert "proba" in data
    assert "classe" in data
    assert 0 <= data["proba"] <= 1
    assert data["classe"] in [0, 1]

def test_api_predict_missing_feature():
    payload = get_valid_payload()
    payload.pop(next(iter(payload)))  # supprime une feature

    response = client.post("/predict", json=payload)

    assert response.status_code == 422  # Pydantic bloque

