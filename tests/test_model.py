import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from src.model_loader import ModelPredictor

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models/best_model_lightgbm.pkl"
TOP_FEATURES_PATH = BASE_DIR / "data/top_features.csv"
SEUIL_METIER = 0.09

# Fixture pour instancier le modèle réel
@pytest.fixture
def predictor():
    return ModelPredictor(MODEL_PATH, TOP_FEATURES_PATH, SEUIL_METIER)

# Fixture pour un payload valide
@pytest.fixture
def valid_payload(predictor):
    return {feat: 0.0 for feat in predictor.top_features}

def test_predict_proba_classe(predictor, valid_payload):
    """Cas nominal : proba entre 0 et 1, classe = 0 ou 1"""
    df = pd.DataFrame([valid_payload])
    proba = predictor.predict_proba(df)[0]
    classe = predictor.predict_class(df)[0]
    assert 0 <= proba <= 1
    assert classe in [0, 1]

def test_feature_manquante(predictor):
    """Comportement si une feature est manquante : doit lever KeyError"""
    df = pd.DataFrame([{predictor.top_features[0]: 0.0}])  # seulement 1 feature
    with pytest.raises(KeyError):
        predictor.predict_proba(df)

def test_seuil_metier():
    """Cas limite : si la proba = threshold, la classe doit être 1"""
    # Crée un payload factice
    dummy_payload = {f"feat{i}": 0.0 for i in range(5)}
    
    # Dummy Model qui renvoie exactement le seuil
    class DummyModel:
        def predict_proba(self, X):
            # On renvoie [[1-threshold, threshold]] pour simuler la proba
            return np.array([[1-SEUIL_METIER, SEUIL_METIER]])
    
    # Instanciation du predictor factice
    predictor = ModelPredictor()
    predictor.model = DummyModel()
    predictor.top_features = list(dummy_payload.keys())
    predictor.threshold = SEUIL_METIER

    df = pd.DataFrame([dummy_payload])
    assert predictor.predict_class(df)[0] == 1


