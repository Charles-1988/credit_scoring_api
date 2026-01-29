from fastapi import FastAPI
from pydantic import create_model
import pandas as pd
from pathlib import Path
from src.model_loader import ModelPredictor, predict_logic, credit_decision

BASE_DIR = Path(__file__).resolve().parent.parent

predictor = ModelPredictor(
    model_path=BASE_DIR / "models/best_model_lightgbm.pkl",
    top_features_path=BASE_DIR / "data/top_features.csv",
    threshold=0.09
)

clients_df = pd.read_csv(BASE_DIR / "data/five_clients.csv", index_col=0)

app = FastAPI(title="Credit Scoring API")

ClientData = create_model(
    "ClientData",
    **{feat: (float, ...) for feat in predictor.top_features}
)

@app.get("/")
def root():
    return {"status": "API Credit Scoring active"}

@app.get("/clients")
def get_clients():
    return clients_df.to_dict(orient="index")

@app.get("/features")
def get_features():
    return predictor.top_features

@app.post("/predict")
def predict(data: ClientData):
    client_data = data.dict()
    try:
        result = predict_logic(client_data, predictor)
    except KeyError as e:
        return {"error": str(e)}

    result["decision"] = credit_decision(result["classe"])
    return result

@app.post("/explain")
def explain(data: ClientData):
    client_data = data.dict()
    try:
        shap_vals = predictor.get_shap_values(pd.DataFrame([client_data]))[0]
    except KeyError as e:
        return {"error": str(e)}

    return {feat: float(val) for feat, val in zip(predictor.top_features, shap_vals)}








