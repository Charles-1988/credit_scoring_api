import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock
from src.model_loader import ModelPredictor
from src.main import predict_logic, credit_decision


def test_predict_proba_missing_feature():
    """Vérifie que predict_proba lève une KeyError si une feature obligatoire est absente."""
    predictor = ModelPredictor.__new__(ModelPredictor)
    predictor.top_features = ["a", "b"]
    predictor.model = Mock()
    df = pd.DataFrame([{"a": 1}])
    with pytest.raises(KeyError):
        predictor.predict_proba(df)

def test_predict_proba_calls_model():
    """Vérifie que predict_proba appelle le modèle et retourne la bonne probabilité."""
    predictor = ModelPredictor.__new__(ModelPredictor)
    predictor.top_features = ["a", "b"]
    mock_model = Mock()
    mock_model.predict_proba.return_value = np.array([[0.7, 0.3]])
    predictor.model = mock_model
    df = pd.DataFrame([{"a": 1, "b": 2}])
    proba = predictor.predict_proba(df)
    assert proba[0] == 0.3
    mock_model.predict_proba.assert_called_once()

def test_predict_class_threshold():
    """Vérifie que predict_class convertit correctement les probabilités en 0 ou 1 selon le seuil."""
    predictor = ModelPredictor.__new__(ModelPredictor)
    predictor.threshold = 0.5
    predictor.predict_proba = Mock(return_value=np.array([0.4, 0.6]))
    df = pd.DataFrame([{}, {}])
    classes = predictor.predict_class(df)
    assert list(classes) == [0, 1]

def test_get_shap_values_mocked():
    """Vérifie que get_shap_values retourne un array et fonctionne même si explainer est mocké."""
    predictor = ModelPredictor.__new__(ModelPredictor)
    predictor.top_features = ["a", "b"]
    predictor._prepare = Mock(return_value=pd.DataFrame([[1,2]], columns=["a","b"]))
    predictor.explainer = Mock()
    predictor.explainer.shap_values.return_value = [np.array([[0.1, -0.1]]), np.array([[0.2, 0.3]])]
    shap_vals = predictor.get_shap_values(pd.DataFrame([[1,2]], columns=["a","b"]))
    assert isinstance(shap_vals, np.ndarray)
    assert shap_vals.shape == (1, 2)

def test_invalid_model_path():
    """Vérifie qu'un chemin de modèle invalide lève bien un FileNotFoundError."""
    from pathlib import Path
    with pytest.raises(FileNotFoundError):
        ModelPredictor(model_path=Path("chemin/inexistant.pkl"))



def test_predict_logic_ok():
    """Vérifie que predict_logic retourne la probabilité et la classe correctement."""
    mock_predictor = Mock()
    mock_predictor.predict_proba.return_value = np.array([0.2])
    mock_predictor.predict_class.return_value = np.array([1])
    data = {"f1": 0.0, "f2": 1.0}
    result = predict_logic(data, mock_predictor)
    assert result["proba"] == 0.2
    assert result["classe"] == 1

def test_predict_logic_missing_feature():
    """Vérifie que predict_logic lève une KeyError si une feature est manquante."""
    mock_predictor = Mock()
    mock_predictor.predict_proba.side_effect = KeyError("Feature manquante")
    data = {"f1": 0.0}
    with pytest.raises(KeyError):
        predict_logic(data, mock_predictor)

def test_credit_decision():
    """Vérifie que credit_decision retourne 'refusé' pour 1 et 'accordé' pour 0."""
    assert credit_decision(1) == "refusé"
    assert credit_decision(0) == "accordé"

def test_new_client_logic():
    """Simule un nouveau client avec toutes les features à 0.0."""
    predictor = ModelPredictor.__new__(ModelPredictor)
    features_test = ["age", "revenu", "dette"]
    predictor.top_features = features_test
    predictor.threshold = 0.5
    predictor.predict_proba = Mock(return_value=np.array([0.3]))
    predictor.predict_class = Mock(return_value=np.array([0]))

    new_client = {feat: 0.0 for feat in features_test}
    result = predict_logic(new_client, predictor)

    assert "proba" in result
    assert "classe" in result
    assert result["proba"] == 0.3
    assert result["classe"] == 0





