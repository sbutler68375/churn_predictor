# Customer Churn Prediction System

An end-to-end churn-prediction system built on synthetic B2B SaaS data: a
PostgreSQL database with hand-designed churn behavior, SQL feature
engineering, a trained and explainable ML model, a live dashboard, and a
containerized prediction API with CI.

## Architecture

```
PostgreSQL (Docker)                 tenants, users, product_events
        |
        v
v_ml_churn_features (SQL view)      rolling 7d/30d/90d activity windows,
        |                           engagement velocity, is_churned label
        v
src/data_loader.py                  pandas DataFrame, env-var credentials
        |
        v
src/train.py + src/evaluate.py      preprocessing pipeline, 3-model
        |                           comparison, SHAP explainability
        v
models/churn_model.joblib           saved Random Forest pipeline
        |
        +----------------+----------------+
        v                                 v
src/app.py (FastAPI)              dashboard.py (Streamlit)
POST /predict                     Executive Overview / Model
-> churn_probability, risk_tier   Diagnostics / Account Deep-Dive
```

The API and the dashboard each load `models/churn_model.joblib`
independently -- neither depends on the other.

## Repo layout

```
init.sql                   Week 1: tenants/users/product_events schema
data_generator.py          Week 1: synthetic company/user generator
generate_events.py         Week 1: 90-day activity generator (healthy + at-risk cohorts)
feature_engineering.sql    Week 2: v_ml_churn_features (rolling windows, velocity, label)
src/
  data_loader.py           Week 3: Postgres -> pandas
  train.py                 Week 3: preprocessing pipeline, 3-model training
  evaluate.py               Week 4: metrics, cross-validation, SHAP, artifact export
  app.py                    Week 6: FastAPI /predict endpoint
  features.py, paths.py     shared constants (kept dependency-light for the API)
  logging_config.py         shared logging setup
dashboard.py                Week 5: Streamlit dashboard
models/                     saved model + held-out test predictions
assets/                     SHAP feature importance chart
tests/test_api.py           Week 6: API test suite
Dockerfile, requirements-api.txt   containerized API (slim, API-only deps)
.github/workflows/ci.yml    Week 6: GitHub Actions test pipeline
MODEL_METRICS.md            evaluation results and business rationale
```

## Setup

```bash
# 1. Start the database (Docker Desktop must be running)
docker start churn-postgres

# 2. Install dependencies
pip install -r requirements.txt

# 3. Copy DATABASE_INFO.txt's connection details into a local .env
#    (DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD -- see DATABASE_INFO.txt)

# 4. Run the dashboard
streamlit run dashboard.py

# 5. Run the API
uvicorn src.app:app --reload

# 6. Run the tests
pytest
```

## Model performance

Random Forest, 5-fold stratified cross-validation (full 100-company dataset):

| Metric | Score |
|---|---|
| Precision | 0.893 ± 0.137 |
| Recall | 0.833 ± 0.211 |
| ROC AUC | 0.993 ± 0.009 |

Full results for all three candidate models (Logistic Regression, Random
Forest, XGBoost), the business rationale for prioritizing recall, a label
leakage bug that was caught and fixed, and honest discussion of this
dataset's limitations (100 companies, synthetic data) are in
[`MODEL_METRICS.md`](MODEL_METRICS.md).

## API

### `POST /predict`

Request:

```json
{
  "tenant_id": "0aa66501-3e08-4f21-9cb5-277a89644d3a",
  "contract_value": 8357.40,
  "events_last_7d": 179,
  "events_last_30d": 1067,
  "events_last_90d": 4201,
  "support_tickets_last_30d": 162,
  "subscription_tier": "Starter"
}
```

Response:

```json
{
  "tenant_id": "0aa66501-3e08-4f21-9cb5-277a89644d3a",
  "churn_probability": 0.74,
  "risk_tier": "high"
}
```

Malformed or missing fields return `422` with a field-level error message
(Pydantic validation, before the request ever reaches the model).

## Running the API in Docker

```bash
docker build -t churn-api .
docker run -p 8000:8000 churn-api
```

The image installs from `requirements-api.txt` (FastAPI, pandas,
scikit-learn, etc. only) rather than the full project's `requirements.txt`
-- the API never needs SHAP, matplotlib, or Streamlit, so leaving them out
keeps the image smaller and avoids an unrelated Python-version conflict
(`shap` requires Python 3.12+; the API image uses `python:3.11-slim`).

## CI

`.github/workflows/ci.yml` installs dependencies and runs `pytest` on every
push. The test suite (`tests/test_api.py`) exercises the API through
FastAPI's `TestClient` and never touches Postgres, so CI doesn't need a
database service.

## Limitations

- **Small dataset.** 100 companies, ~19-20 at-risk. Metrics -- especially
  recall -- have real variance across cross-validation folds as a direct
  consequence of sample size, not model instability. See `MODEL_METRICS.md`
  for detail.
- **Synthetic, sharply-separated data.** Week 1's data generator injected a
  deliberate, fairly obvious behavioral break (activity collapse + support
  ticket spike) into the at-risk cohort. Real-world churn signals are
  usually noisier and more gradual. These results demonstrate that the
  pipeline correctly recovers a known, injected signal -- not a claim about
  real-world churn prediction accuracy.
