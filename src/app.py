"""FastAPI service wrapping the saved churn model.

Week 6, Day 1: basic /predict endpoint, plain JSON in/out.
Week 6, Day 2: strict Pydantic input validation, 422 on bad input.
"""

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from src.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from src.logging_config import get_logger
from src.paths import MODEL_PATH

log = get_logger(__name__)

app = FastAPI(title="Customer Churn Prediction API")
model = joblib.load(MODEL_PATH)
log.info(f"Loaded model from {MODEL_PATH}")

HIGH_RISK_THRESHOLD = 0.70
MEDIUM_RISK_THRESHOLD = 0.30


class ChurnPredictionRequest(BaseModel):
    tenant_id: str
    contract_value: float
    events_last_7d: int
    events_last_30d: int
    events_last_90d: int
    support_tickets_last_30d: int
    subscription_tier: str


class ChurnPredictionResponse(BaseModel):
    tenant_id: str
    churn_probability: float
    risk_tier: str


def get_risk_tier(probability: float) -> str:
    if probability >= HIGH_RISK_THRESHOLD:
        return "high"
    if probability >= MEDIUM_RISK_THRESHOLD:
        return "medium"
    return "low"


@app.post("/predict", response_model=ChurnPredictionResponse)
async def predict(payload: ChurnPredictionRequest) -> ChurnPredictionResponse:
    features = {name: getattr(payload, name) for name in NUMERIC_FEATURES + CATEGORICAL_FEATURES}
    X = pd.DataFrame([features])

    probability = float(model.predict_proba(X)[0, 1])

    return ChurnPredictionResponse(
        tenant_id=payload.tenant_id,
        churn_probability=round(probability, 4),
        risk_tier=get_risk_tier(probability),
    )
