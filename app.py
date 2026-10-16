import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# 1. INITIALIZE THE PRODUCTION FASTAPI ENGINE
app = FastAPI(title="Bike-Sharing Demand Forecaster API")

# Restrict who can communicate with your API over the web
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-frontend-website.com"], # Allow only your app frontend
    allow_credentials=True,
    allow_methods=["POST"], # Restrict to POST prediction requests only
    allow_headers=["*"],
)


# 2. LOAD THE MACHINE LEARNING BINARY FILE DIRECTLY
# Docker copies this file into the workspace, so it loads instantly in milliseconds
model = joblib.load("best_bike_sharing_model.joblib")

# Explicit list of columns required by the model
feature_cols = [
    "hr_sin", "hr_cos", "weathersit", "temp", "hum", "windspeed",
    "holiday", "workingday", "season", "cnt_lag_24h"
]

# 3. DEFINE INBOUND PAYLOAD DATA VALIDATION
class BikePredictionInput(BaseModel):
    hr: int
    weathersit: int
    temp: float
    hum: float
    windspeed: float
    holiday: int
    workingday: int
    season: int
    cnt_lag_24h: float

# 4. CORE ROUTE logic ENDPOINT (No TestClient allowed here)
@app.post("/predict")
def predict_bike_demand(payload: BikePredictionInput):
    try:
        # Convert incoming hour input into geometric circular features
        hr_sin = np.sin(2 * np.pi * payload.hr / 24.0)
        hr_cos = np.cos(2 * np.pi * payload.hr / 24.0)

        # Build a labeled DataFrame matching the model's expected column names
        live_data = pd.DataFrame(
            [[
                hr_sin, hr_cos, payload.weathersit, payload.temp,
                payload.hum, payload.windspeed, payload.holiday,
                payload.workingday, payload.season, payload.cnt_lag_24h
            ]],
            columns=feature_cols
        )

        # Pass the formatted data row into the model
        raw_prediction_array = model.predict(live_data)

        # Extract scalar value from array matrix to avoid numpy rounding errors
        scalar_prediction = raw_prediction_array[0]
        final_prediction = max(0, int(round(scalar_prediction)))

        return {"status": "success", "predicted_bike_count": final_prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 5. METADATA HEALTH CHECK ROUTE
@app.get("/")
def health_check():
    return {"status": "online", "message": "Bike-sharing API is active and loaded."}
