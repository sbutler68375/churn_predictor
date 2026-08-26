"""Shared file paths. Kept dependency-free (no shap/matplotlib/xgboost) so
that lightweight consumers -- especially src/app.py, which ships in the
Docker image -- don't pull in heavy explainability/training libraries just
to know where the model file lives."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "churn_model.joblib"
SHAP_CHART_PATH = PROJECT_ROOT / "assets" / "shap_feature_importance.png"
TEST_PREDICTIONS_PATH = PROJECT_ROOT / "models" / "test_predictions.csv"
