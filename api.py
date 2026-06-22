"""HTTP API for single-customer churn predictions.

Run:
    uvicorn api:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, Query
from pydantic import BaseModel, ConfigDict, Field

from churn_prediction import clean_data


MODEL_PATH = Path("models/churn_model.joblib")


class CustomerPayload(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    city: str = Field(..., alias="City")
    zip_code: int = Field(..., alias="Zip Code")
    latitude: float = Field(..., alias="Latitude")
    longitude: float = Field(..., alias="Longitude")
    gender: str = Field(..., alias="Gender")
    senior_citizen: str = Field(..., alias="Senior Citizen")
    partner: str = Field(..., alias="Partner")
    dependents: str = Field(..., alias="Dependents")
    tenure_months: int = Field(..., alias="Tenure Months")
    phone_service: str = Field(..., alias="Phone Service")
    multiple_lines: str = Field(..., alias="Multiple Lines")
    internet_service: str = Field(..., alias="Internet Service")
    online_security: str = Field(..., alias="Online Security")
    online_backup: str = Field(..., alias="Online Backup")
    device_protection: str = Field(..., alias="Device Protection")
    tech_support: str = Field(..., alias="Tech Support")
    streaming_tv: str = Field(..., alias="Streaming TV")
    streaming_movies: str = Field(..., alias="Streaming Movies")
    contract: str = Field(..., alias="Contract")
    paperless_billing: str = Field(..., alias="Paperless Billing")
    payment_method: str = Field(..., alias="Payment Method")
    monthly_charges: float = Field(..., alias="Monthly Charges")
    total_charges: float = Field(..., alias="Total Charges")


class PredictionResponse(BaseModel):
    churn_probability: float
    churn_prediction: str
    churn_prediction_value: int
    threshold: float


app = FastAPI(
    title="Customer Churn Prediction API",
    version="1.0.0",
)


@app.on_event("startup")
def load_model() -> None:
    app.state.model_package = joblib.load(MODEL_PATH)


@app.get("/")
def root() -> dict[str, str]:
    return {
        "service": "Customer Churn Prediction API",
        "health": "/health",
        "predict": "/predict",
        "docs": "/docs",
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(
    customer: CustomerPayload,
    threshold: float | None = Query(default=None, ge=0.0, le=1.0),
) -> PredictionResponse:
    model_package = app.state.model_package
    model = model_package["model"]
    features = model_package["features"]
    prediction_threshold = (
        float(threshold)
        if threshold is not None
        else float(model_package.get("threshold", 0.5))
    )

    customer_df = pd.DataFrame([customer.model_dump(by_alias=True)])
    customer_df = clean_data(customer_df).reindex(columns=features)

    probability = float(model.predict_proba(customer_df)[0, 1])
    prediction_value = int(probability >= prediction_threshold)

    return PredictionResponse(
        churn_probability=probability,
        churn_prediction="Yes" if prediction_value else "No",
        churn_prediction_value=prediction_value,
        threshold=prediction_threshold,
    )
