import joblib
import pandas as pd
from pathlib import Path
import shap
import numpy as np

class ModelPredictor:
    def __init__(self, model_path=None, top_features_path=None, threshold=0.09):
        base_path = Path(__file__).parent.parent
        model_path = model_path or base_path / "models/best_model_lightgbm.pkl"
        features_path = top_features_path or base_path / "data/top_features.csv"

        self.model = joblib.load(model_path)
        self.top_features = pd.read_csv(features_path)["feature"].tolist()
        self.threshold = threshold
        self.explainer = None

    def _prepare(self, X: pd.DataFrame) -> pd.DataFrame:
        missing = set(self.top_features) - set(X.columns)
        if missing:
            raise KeyError(f"Feature(s) manquante(s) : {missing}")
        return X[self.top_features]

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X_prepared = self._prepare(X)
        return self.model.predict_proba(X_prepared)[:, 1]

    def predict_class(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X) >= self.threshold).astype(int)

    def get_shap_values(self, X: pd.DataFrame) -> np.ndarray:
        X_prepared = self._prepare(X)
        if self.explainer is None:
            model_to_explain = self.model
            if hasattr(self.model, 'named_steps') and 'clf' in self.model.named_steps:
                model_to_explain = self.model.named_steps['clf']
            self.explainer = shap.TreeExplainer(model_to_explain)
        shap_values = self.explainer.shap_values(X_prepared)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        return shap_values


def predict_logic(client_data: dict, predictor: ModelPredictor) -> dict:
    df = pd.DataFrame([client_data])
    proba = predictor.predict_proba(df)[0]
    classe = predictor.predict_class(df)[0]
    return {"proba": float(proba), "classe": int(classe)}

def credit_decision(classe: int) -> str:
    return "refusé" if classe == 1 else "accordé"


