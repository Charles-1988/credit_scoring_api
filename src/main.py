from fastapi import FastAPI
from pydantic import create_model
import pandas as pd
from pathlib import Path
from src.model_loader import ModelPredictor

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models/best_model_lightgbm.pkl"
TOP_FEATURES_PATH = BASE_DIR / "data/top_features.csv"
SEUIL_METIER = 0.35


def get_predictor() -> ModelPredictor:
    """Retourne une instance du ModelPredictor"""
    return ModelPredictor(
        model_path=MODEL_PATH,
        top_features_path=TOP_FEATURES_PATH,
        threshold=SEUIL_METIER
    )


app = FastAPI(title="Credit Scoring API")

# Charger les clients
clients_file = BASE_DIR / "data" / "five_clients.csv"
clients_df = pd.read_csv(clients_file, index_col=0)
clients_df = clients_df.astype(float)  

@app.get("/")
def read_root():
    return {"message": "API Credit Scoring active"}

@app.get("/clients")
def get_clients():
    return clients_df.to_dict(orient="index")


predictor_temp = get_predictor()  
ClientData = create_model(
    "ClientData",
    **{feat: (float, ...) for feat in predictor_temp.top_features}
)
ClientData.__config__ = type("Config", (), {"extra": "forbid"})


@app.post("/predict")
def predict(data: ClientData):
    try:
        predictor = get_predictor()  
        df = pd.DataFrame([data.dict()])

        # Vérifier qu'on a toutes les features
        missing_cols = set(predictor.top_features) - set(df.columns)
        if missing_cols:
            return {"error": f"Features manquantes: {missing_cols}"}

        proba = predictor.predict_proba(df)[0]
        classe = predictor.predict_class(df)[0]
        return {"proba": float(proba), "classe": int(classe)}

    except KeyError as e:
        return {"error": f"Feature manquante: {str(e)}"}
    except Exception as e:
        return {"error": f"Erreur serveur: {str(e)}"}



