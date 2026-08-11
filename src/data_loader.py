"""Data ingestion layer: reads v_ml_churn_features (joined with a couple of
business columns from tenants) into a pandas DataFrame for model training."""

import os

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine

from src.logging_config import get_logger

log = get_logger(__name__)

QUERY = """
    SELECT
        f.tenant_id,
        f.company_name,
        t.subscription_tier,
        t.contract_value,
        f.events_last_7d,
        f.events_last_30d,
        f.events_last_90d,
        f.support_tickets_last_30d,
        f.engagement_velocity,
        f.is_churned
    FROM v_ml_churn_features f
    JOIN tenants t ON t.tenant_id = f.tenant_id;
"""


def build_connection_string() -> str:
    load_dotenv()
    host = os.environ["DB_HOST"]
    port = os.environ["DB_PORT"]
    name = os.environ["DB_NAME"]
    user = os.environ["DB_USER"]
    password = os.environ["DB_PASSWORD"]
    return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{name}"


def load_churn_features() -> pd.DataFrame:
    conn_string = build_connection_string()
    engine = create_engine(conn_string)

    log.info("Loading v_ml_churn_features (joined with tenants) from the database...")
    df = pd.read_sql(QUERY, engine)
    log.info(f"Loaded {len(df)} rows, {len(df.columns)} columns.")

    return df


if __name__ == "__main__":
    data = load_churn_features()
    log.info(f"is_churned distribution:\n{data['is_churned'].value_counts()}")
    print(data.dtypes)
    print(data.head())
