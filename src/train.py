"""Preprocessing pipeline and model training/comparison."""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from src.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET
from src.logging_config import get_logger

log = get_logger(__name__)

RANDOM_STATE = 42

MODELS = {
    "logistic_regression": LogisticRegression(random_state=RANDOM_STATE),
    "random_forest": RandomForestClassifier(random_state=RANDOM_STATE),
    "xgboost": XGBClassifier(random_state=RANDOM_STATE, eval_metric="logloss"),
}


def build_preprocessing_pipeline() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )


def split_data(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X = data[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = data[TARGET]
    return train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )


def train_models(
    X_train: pd.DataFrame, y_train: pd.Series
) -> dict[str, Pipeline]:
    fitted_pipelines: dict[str, Pipeline] = {}

    for model_name, model in MODELS.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessing_pipeline()),
                ("model", model),
            ]
        )
        pipeline.fit(X_train, y_train)
        fitted_pipelines[model_name] = pipeline
        log.info(f"Trained {model_name} on {len(X_train)} rows.")

    return fitted_pipelines


if __name__ == "__main__":
    from src.data_loader import load_churn_features

    data = load_churn_features()
    X_train, X_test, y_train, y_test = split_data(data)

    log.info(
        f"Train: {len(X_train)} rows ({y_train.sum()} churned / "
        f"{len(y_train) - y_train.sum()} healthy)"
    )
    log.info(
        f"Test: {len(X_test)} rows ({y_test.sum()} churned / "
        f"{len(y_test) - y_test.sum()} healthy)"
    )

    fitted_pipelines = train_models(X_train, y_train)
    log.info(f"All models trained: {list(fitted_pipelines.keys())}")
