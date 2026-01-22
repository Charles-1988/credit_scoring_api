import joblib
import pandas as pd
from pathlib import Path

class ModelPredictor:
    def __init__(self, model_path=None, top_features_path=None, threshold=0.09):
        base_path = Path(__file__).parent.parent

        # Chemins par défaut
        if model_path is None:
            model_path = base_path / "models/best_model_lightgbm.pkl"
        if top_features_path is None:
            top_features_path = base_path / "data/top_features.csv"

        # Charger le modèle et les features
        try:
            self.model = joblib.load(model_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Fichier modèle introuvable : {model_path}")

        try:
            self.top_features = pd.read_csv(top_features_path)["feature"].tolist()
        except FileNotFoundError:
            raise FileNotFoundError(f"Fichier top_features introuvable : {top_features_path}")

        self.threshold = threshold

    def predict_proba(self, X: pd.DataFrame):
        # Vérifie que toutes les features sont présentes
        missing = set(self.top_features) - set(X.columns)
        if missing:
            raise KeyError(f"Feature(s) manquante(s) : {missing}")
        X_sel = X[self.top_features]
        return self.model.predict_proba(X_sel)[:, 1]

    def predict_class(self, X: pd.DataFrame):
        return (self.predict_proba(X) >= self.threshold).astype(int)



