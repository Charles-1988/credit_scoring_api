from fastapi import FastAPI
import pandas as pd
from src.model_loader import predictor

app = FastAPI(title="Credit Scoring API")

@app.get("/")
def read_root():
    return {"message": "API Credit Scoring active"}

@app.post("/predict")
def predict(data: dict):
    try:
        df = pd.DataFrame([data])
        df = df[predictor.top_features]  
        proba = predictor.predict_proba(df)[0]
        classe = predictor.predict_class(df)[0]
        return {"proba": float(proba), "classe": int(classe)}
    except Exception as e:
        return {"error": str(e)}

