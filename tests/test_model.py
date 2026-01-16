import pandas as pd
from model_loader import predictor
def test_model_file_exists():
    """Vérifie que le modèle est chargé."""
    assert predictor.model is not None

def test_top_features_file_exists():
    """Vérifie que la liste des features n’est pas vide."""
    assert len(predictor.top_features) > 0

def test_predict_proba():
    """Teste que la prédiction de probabilité retourne des valeurs entre 0 et 1."""
    test_df = pd.DataFrame([{feat: 0 for feat in predictor.top_features}])
    probas = predictor.predict_proba(test_df)
    assert all(0 <= p <= 1 for p in probas)

def test_predict_class():
    """Teste que la prédiction de classe retourne 0 ou 1."""
    test_df = pd.DataFrame([{feat: 0 for feat in predictor.top_features}])
    classes = predictor.predict_class(test_df)
    assert all(c in [0, 1] for c in classes)


