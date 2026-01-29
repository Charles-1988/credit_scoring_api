import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock
from src.model_loader import ModelPredictor, predict_logic, credit_decision

def test_predict_proba_missing_feature():
    predictor = ModelPredictor.__new__(ModelPredictor)
    predictor.top_features = ["a", "b"]
    predictor.model = Mock()
    df = pd.DataFrame([{"a": 1}])
    with pytest.raises(KeyError):
        predictor.predict_proba(df)

def test_predict_proba_calls_model():
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
    predictor = ModelPredictor.__new__(ModelPredictor)
    predictor.threshold = 0.5
    predictor.predict_proba = Mock(return_value=np.array([0.4, 0.6]))
    df = pd.DataFrame([{}, {}])
    classes = predictor.predict_class(df)
    assert list(classes) == [0, 1]

def test_get_shap_values_mocked():
    predictor = ModelPredictor.__new__(ModelPredictor)
    predictor.top_features = ["a", "b"]
    predictor._prepare = Mock(return_value=pd.DataFrame([[1,2]], columns=["a","b"]))
    predictor.explainer = Mock()
    predictor.explainer.shap_values.return_value = [np.array([[0.1, -0.1]]), np.array([[0.2, 0.3]])]
    shap_vals = predictor.get_shap_values(pd.DataFrame([[1,2]], columns=["a","b"]))
    assert isinstance(shap_vals, np.ndarray)
    assert shap_vals.shape == (1, 2)

def test_predict_logic_ok():
    mock_predictor = Mock()
    mock_predictor.predict_proba.return_value = np.array([0.2])
    mock_predictor.predict_class.return_value = np.array([1])
    data = {"f1": 0.0, "f2": 1.0}
    result = predict_logic(data, mock_predictor)
    assert result["proba"] == 0.2
    assert result["classe"] == 1

def test_credit_decision():
    assert credit_decision(1) == "refusé"
    assert credit_decision(0) == "accordé"






