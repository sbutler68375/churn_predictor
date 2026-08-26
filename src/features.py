"""Feature/target column names -- the single source of truth for which
columns feed the model. Kept dependency-free (no sklearn/xgboost) so that
lightweight consumers, especially src/app.py, don't pull in model-training
libraries just to know which fields a prediction request needs."""

NUMERIC_FEATURES = [
    "contract_value",
    "events_last_7d",
    "events_last_30d",
    "events_last_90d",
    "support_tickets_last_30d",
]
CATEGORICAL_FEATURES = ["subscription_tier"]
TARGET = "is_churned"
