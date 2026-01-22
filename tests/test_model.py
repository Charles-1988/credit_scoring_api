import pytest
import pandas as pd
from fastapi.testclient import TestClient
from pathlib import Path
from src.model_loader import ModelPredictor
from src.main import app

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models/best_model_lightgbm.pkl"
TOP_FEATURES_PATH = BASE_DIR / "data/top_features.csv"
SEUIL_METIER = 0.09

client = TestClient(app)

def test_predictor_normal():
    predictor = ModelPredictor(MODEL_PATH, TOP_FEATURES_PATH, SEUIL_METIER)
    df = pd.DataFrame([{f: 0.0 for f in predictor.top_features}])
    proba = predictor.predict_proba(df)[0]
    classe = predictor.predict_class(df)[0]

    assert 0 <= proba <= 1
    assert classe in [0, 1]
    assert (proba >= SEUIL_METIER) == (classe == 1)


def test_predictor_missing_feature():
    predictor = ModelPredictor(MODEL_PATH, TOP_FEATURES_PATH, SEUIL_METIER)
    df = pd.DataFrame([{predictor.top_features[0]: 0.0}])  # seulement 1 feature
    with pytest.raises(KeyError):
        predictor.predict_proba(df)


def test_invalid_model_path():
    with pytest.raises(FileNotFoundError):
        ModelPredictor("modele_inexistant.pkl", TOP_FEATURES_PATH, SEUIL_METIER)

def test_invalid_features_path():
    with pytest.raises(FileNotFoundError):
        ModelPredictor(MODEL_PATH, "features_inexistantes.csv", SEUIL_METIER)


def test_api_get_clients():
    response = client.get("/clients")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, dict)


def test_api_predict_success():
    predictor = ModelPredictor(MODEL_PATH, TOP_FEATURES_PATH, SEUIL_METIER)
    payload = {feat: 0.0 for feat in predictor.top_features}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "proba" in data and "classe" in data
    assert 0 <= data["proba"] <= 1
    assert data["classe"] in [0, 1]


def test_api_predict_missing_feature():
    predictor = ModelPredictor(MODEL_PATH, TOP_FEATURES_PATH, SEUIL_METIER)
    payload = {predictor.top_features[0]: 0.0}  # seulement 1 feature
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "error" in data
