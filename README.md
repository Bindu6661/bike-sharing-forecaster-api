# 🚲 Bike-Sharing Hourly Demand Forecaster API

An enterprise-grade, production-ready machine learning microservice that predicts hourly bike rental demand. The pipeline is built using **FastAPI**, containerized via **Docker**, and deployed to the public web with automated **CI/CD** pipelines.

Live Public API Documentation: [https://onrender.com](https://onrender.com)

## 📊 Engineering & Model Performance
To prevent severe **Data Leakage**, underlying cheating column vectors (`casual` and `registered`) were completely audited and removed before training. Time metrics were engineered into geometric 2D circular wave features (`hr_sin`, `hr_cos`) to preserve midnight wrapping transitions.

We evaluated our predictive models chronologically on unknown 2012 test data against a human-logic naive 24-hour lag benchmark:

*   **24-Hour Lag Baseline MAE:** 77.62 rental bikes
*   **Linear Regression MAE:** 130.84 rental bikes
*   **Optimized Random Forest MAE:** **77.16 rental bikes** 🎉 *(Successfully broke the baseline barrier)*

## 🛠️ Infrastructure & Tech Stack
*   **Data Pipeline:** Python, Pandas, NumPy (Sine/Cosine cyclical time mapping)
*   **Machine Learning:** Scikit-Learn (Random Forest Regressor optimized with `max_features="sqrt"` to handle history tracking variance limits)
*   **API Layer:** FastAPI, Pydantic (Strict input structure payload validation)
*   **Containerization:** Docker Desktop & WSL 2 (Lightweight Linux system virtualization)
*   **Deployment:** GitHub Repository & Render Cloud Engine (Automated pipeline web hosting)

## 🚀 API Endpoint Schema Validation
### POST `/predict`
**Sample Inbound JSON Payload:**
```json
{
  "hr": 17,
  "weathersit": 1,
  "temp": 0.45,
  "hum": 0.50,
  "windspeed": 0.15,
  "holiday": 0,
  "workingday": 1,
  "season": 3,
  "cnt_lag_24h": 210.0
}
```

**Successful JSON Response Block:**
```json
{
  "status": "success",
  "predicted_bike_count": 353
}
```
