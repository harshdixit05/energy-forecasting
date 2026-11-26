
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd

from src.predict import predict, config

app = FastAPI(title="Energy Forecasting API")


class PredictionRequest(BaseModel):
    features: dict[str, float]


@app.get("/")
def home():
    return {"message": "Energy Forecasting API is running"}


@app.post("/predict")
def make_prediction(request: PredictionRequest):
    input_data = pd.DataFrame([request.features])
    result = predict(input_data)

    return {
        "predicted_energy_kwh": round(float(result[0]), 2),
        "forecast_horizon": "6 hours"
    }
