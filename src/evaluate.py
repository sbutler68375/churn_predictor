"""Model evaluation: real classification metrics instead of plain accuracy."""

from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.base import clone
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from src.logging_config import get_logger
from src.train import CATEGORICAL_FEATURES, MODELS, NUMERIC_FEATURES, RANDOM_STATE, build_preprocessing_pipeline

log = get_logger(__name__)

CV_SCORING = ["precision", "recall", "f1", "roc_auc"]
PROJECT_ROOT = Path(__file__).parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "churn_model.joblib"
SHAP_CHART_PATH = PROJECT_ROOT / "assets" / "shap_feature_importance.png"
TEST_PREDICTIONS_PATH = PROJECT_ROOT / "models" / "test_predictions.csv"


def evaluate_model(
    pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series
) -> dict[str, float]:
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    return {
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }


def evaluate_all_models(
    fitted_pipelines: dict[str, Pipeline], X_test: pd.DataFrame, y_test: pd.Series
) -> pd.DataFrame:
    results = {
        model_name: evaluate_model(pipeline, X_test, y_test)
        for model_name, pipeline in fitted_pipelines.items()
    }
    return pd.DataFrame(results).T


def cross_validate_models(
    X: pd.DataFrame, y: pd.Series, n_splits: int = 5
) -> pd.DataFrame:
    """5-fold stratified CV over the full dataset (not just the train split) --
    checks whether performance holds up consistently across different slices
    of data, since a single 80/20 split on 100 rows is a small sample."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    results = {}
    for model_name, model in MODELS.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessing_pipeline()),
                ("model", model),
            ]
        )
        scores = cross_validate(pipeline, X, y, cv=skf, scoring=CV_SCORING)

        row = {}
        for metric in CV_SCORING:
            row[f"{metric}_mean"] = scores[f"test_{metric}"].mean()
            row[f"{metric}_std"] = scores[f"test_{metric}"].std()
        results[model_name] = row
        log.info(f"Cross-validated {model_name} across {n_splits} folds.")

    return pd.DataFrame(results).T


def compute_shap_values(
    pipeline: Pipeline, X: pd.DataFrame
) -> tuple[np.ndarray, list[str]]:
    """SHAP values for the "at-risk" class (is_churned=1) from the fitted
    Random Forest model inside `pipeline`. TreeExplainer only works on
    tree-based models -- that's why Random Forest was chosen over Logistic
    Regression as the model carried forward for explainability."""
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]

    X_transformed = preprocessor.transform(X)
    feature_names = list(preprocessor.get_feature_names_out())

    explainer = shap.TreeExplainer(model)
    raw_shap_values = explainer.shap_values(X_transformed)

    # RandomForestClassifier is binary here; TreeExplainer returns values for
    # both classes. We only care about class 1 (is_churned=1, the "at-risk" class).
    if isinstance(raw_shap_values, list):
        shap_values = raw_shap_values[1]
    elif raw_shap_values.ndim == 3:
        shap_values = raw_shap_values[:, :, 1]
    else:
        shap_values = raw_shap_values

    return shap_values, feature_names


def compute_shap_explanation(pipeline: Pipeline, X: pd.DataFrame) -> shap.Explanation:
    """Like compute_shap_values, but returns a shap.Explanation object suitable
    for per-company plots (e.g. shap.plots.waterfall). Display values use each
    row's real, human-readable numbers (raw event counts, dollar amounts, 0/1
    tier flags) instead of the internally-scaled numbers SHAP computes on --
    a chart showing "events_last_7d = -0.42" would mean nothing to a reader."""
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]

    X_transformed = preprocessor.transform(X)
    feature_names = list(preprocessor.get_feature_names_out())

    explainer = shap.TreeExplainer(model)
    raw_shap_values = explainer.shap_values(X_transformed)
    raw_expected_value = explainer.expected_value

    if isinstance(raw_shap_values, list):
        shap_values = raw_shap_values[1]
        base_value = raw_expected_value[1]
    elif raw_shap_values.ndim == 3:
        shap_values = raw_shap_values[:, :, 1]
        base_value = raw_expected_value[1] if hasattr(raw_expected_value, "__len__") else raw_expected_value
    else:
        shap_values = raw_shap_values
        base_value = raw_expected_value

    # ColumnTransformer output is [numeric columns..., one-hot columns...] in that
    # order (matches build_preprocessing_pipeline's transformer list), so swapping
    # in raw numeric values for display only requires slicing at len(NUMERIC_FEATURES).
    raw_numeric = X[NUMERIC_FEATURES].to_numpy()
    onehot_part = X_transformed[:, len(NUMERIC_FEATURES):]
    display_data = np.hstack([raw_numeric, onehot_part])

    readable_names = [
        name.replace("numeric__", "").replace("categorical__", "").replace("_", " ")
        for name in feature_names
    ]

    return shap.Explanation(
        values=shap_values,
        base_values=np.full(len(X), base_value),
        data=display_data,
        feature_names=readable_names,
    )


def summarize_shap_values(shap_values: np.ndarray, feature_names: list[str]) -> pd.Series:
    """Mean absolute SHAP value per feature -- how much each feature moves the
    at-risk prediction on average, sorted highest-impact first."""
    mean_abs = np.abs(shap_values).mean(axis=0)
    return pd.Series(mean_abs, index=feature_names).sort_values(ascending=False)


def save_test_predictions(
    y_test: pd.Series,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    path: Path = TEST_PREDICTIONS_PATH,
) -> None:
    """Persist the held-out test-set truth/predictions so the dashboard's
    Model Diagnostics page can plot an ROC curve and confusion matrix
    instantly, without reconnecting to Postgres or retraining anything live."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"y_true": y_test.to_numpy(), "y_pred": y_pred, "y_proba": y_proba})
    df.to_csv(path, index=False)
    log.info(f"Saved {len(df)} test-set predictions to {path}")


def save_model(pipeline: Pipeline, path: Path = MODEL_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
    log.info(f"Saved model pipeline to {path}")


def save_shap_summary_chart(shap_summary: pd.Series, path: Path = SHAP_CHART_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    readable_names = [
        name.replace("numeric__", "").replace("categorical__", "").replace("_", " ")
        for name in shap_summary.index
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(readable_names[::-1], shap_summary.values[::-1], color="#2f6f8f")
    ax.set_xlabel("Mean |SHAP value| (impact on churn prediction)")
    ax.set_title("Global Feature Importance -- Random Forest Churn Model")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

    log.info(f"Saved SHAP summary chart to {path}")


if __name__ == "__main__":
    from src.data_loader import load_churn_features
    from src.train import TARGET, split_data, train_models

    data = load_churn_features()
    X_train, X_test, y_train, y_test = split_data(data)
    fitted_pipelines = train_models(X_train, y_train)

    results_df = evaluate_all_models(fitted_pipelines, X_test, y_test)
    log.info(f"Test-set metrics ({len(X_test)} rows):\n{results_df.round(3)}")

    X_full = data[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y_full = data[TARGET]
    cv_results_df = cross_validate_models(X_full, y_full, n_splits=5)
    log.info(
        f"5-fold CV results (mean +/- std, {len(X_full)} rows):\n"
        f"{cv_results_df.round(3).to_string()}"
    )

    rf_pipeline = fitted_pipelines["random_forest"]

    rf_y_pred = rf_pipeline.predict(X_test)
    rf_y_proba = rf_pipeline.predict_proba(X_test)[:, 1]
    save_test_predictions(y_test, rf_y_pred, rf_y_proba)

    shap_values, feature_names = compute_shap_values(rf_pipeline, X_test)
    log.info(f"SHAP values shape: {shap_values.shape} (rows x features)")

    shap_summary = summarize_shap_values(shap_values, feature_names)
    log.info(f"Mean |SHAP value| by feature, most impactful first:\n{shap_summary.round(4).to_string()}")

    # Ship a model trained on ALL available data (not just the 80-row train split) --
    # the train/test split and cross-validation above already did their job of honestly
    # measuring performance, so the final saved artifact can use every row we have.
    final_pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessing_pipeline()),
            ("model", clone(MODELS["random_forest"])),
        ]
    )
    final_pipeline.fit(X_full, y_full)
    log.info(f"Refit final Random Forest pipeline on all {len(X_full)} rows before saving.")

    save_model(final_pipeline)
    save_shap_summary_chart(shap_summary)
