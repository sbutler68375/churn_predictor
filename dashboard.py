"""Customer Churn Prediction System -- Streamlit dashboard.

Week 5, Day 1: page layout and sidebar navigation skeleton.
Week 5, Day 2: Executive Overview KPI row.
Week 5, Day 3: Model Diagnostics page (SHAP chart, ROC curve, confusion matrix).
Week 5, Day 4: Account Deep-Dive company selector.
Week 5, Day 5: live risk score + local SHAP waterfall explanation.
"""

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import shap
import streamlit as st
from sklearn.metrics import auc, confusion_matrix, roc_curve

from src.data_loader import load_churn_features
from src.evaluate import MODEL_PATH, SHAP_CHART_PATH, TEST_PREDICTIONS_PATH, compute_shap_explanation
from src.train import CATEGORICAL_FEATURES, NUMERIC_FEATURES

st.set_page_config(
    page_title="Customer Churn Prediction System",
    page_icon="\U0001F4CA",
    layout="wide",
)

PAGES = ["Executive Overview", "Model Diagnostics", "Account Deep-Dive"]
HIGH_RISK_THRESHOLD = 0.70


@st.cache_resource
def get_model():
    return joblib.load(MODEL_PATH)


@st.cache_data(ttl=300)
def get_live_predictions() -> pd.DataFrame:
    """Pull current data from Postgres and score every company with the
    saved model. Cached for 5 minutes so navigating between pages doesn't
    re-hit the database and re-run predictions on every click."""
    data = load_churn_features()
    model = get_model()

    X = data[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    data["churn_probability"] = model.predict_proba(X)[:, 1]
    return data


@st.cache_resource
def get_deep_dive_data() -> tuple[pd.DataFrame, shap.Explanation]:
    """Predictions and their per-company SHAP explanations, computed together
    from one snapshot so the two stay row-aligned by position -- they're kept
    in a separate cache from get_live_predictions() (which refreshes every 5
    minutes) specifically to avoid the two ever drifting out of sync."""
    data = load_churn_features().reset_index(drop=True)
    model = get_model()

    X = data[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    data["churn_probability"] = model.predict_proba(X)[:, 1]
    explanation = compute_shap_explanation(model, X)
    return data, explanation


st.sidebar.title("Churn Prediction System")
page = st.sidebar.radio("Navigate", PAGES)

st.title(page)

if page == "Executive Overview":
    predictions = get_live_predictions()
    high_risk = predictions[predictions["churn_probability"] > HIGH_RISK_THRESHOLD]

    total_arr_at_risk = high_risk["contract_value"].sum()
    high_risk_count = len(high_risk)
    portfolio_health = 1 - predictions["churn_probability"].mean()

    col1, col2, col3 = st.columns(3)
    col1.metric("Total ARR at Risk", f"${total_arr_at_risk:,.0f}")
    col2.metric("High-Risk Tenants", f"{high_risk_count} / {len(predictions)}")
    col3.metric("Avg. Portfolio Health Score", f"{portfolio_health:.0%}")

    st.caption(f"Companies above {HIGH_RISK_THRESHOLD:.0%} predicted churn probability are counted as high-risk.")
elif page == "Model Diagnostics":
    st.subheader("Global Feature Importance (SHAP)")
    st.image(str(SHAP_CHART_PATH), use_container_width=True)
    st.caption("Mean absolute SHAP value per feature, computed on the held-out test set (Week 4).")

    test_predictions = pd.read_csv(TEST_PREDICTIONS_PATH)
    y_true = test_predictions["y_true"]
    y_pred = test_predictions["y_pred"]
    y_proba = test_predictions["y_proba"]

    roc_col, cm_col = st.columns(2)

    with roc_col:
        st.subheader("ROC Curve")
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC curve (AUC = {roc_auc:.2f})"))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random guess", line=dict(dash="dash")))
        fig.update_layout(
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            template="plotly_dark",
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

    with cm_col:
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_true, y_pred)
        labels = ["Healthy (0)", "At-Risk (1)"]

        fig = go.Figure(
            data=go.Heatmap(
                z=cm,
                x=[f"Predicted: {l}" for l in labels],
                y=[f"Actual: {l}" for l in labels],
                text=cm,
                texttemplate="%{text}",
                colorscale="Blues",
            )
        )
        fig.update_layout(template="plotly_dark", height=400)
        st.plotly_chart(fig, use_container_width=True)

    st.caption(f"Computed on the {len(test_predictions)}-row held-out test set from Week 4 (Random Forest model).")
elif page == "Account Deep-Dive":
    data, explanation = get_deep_dive_data()

    # company_name isn't guaranteed unique (two different tenants can be randomly
    # named the same thing by the Week 1 data generator), so the dropdown label
    # disambiguates duplicates with a short tenant_id suffix. label_to_position
    # maps back to data's row position, which is also the SHAP explanation's
    # row position -- the two were computed together in get_deep_dive_data().
    duplicate_names = data["company_name"][data["company_name"].duplicated(keep=False)].unique()

    def make_label(row: pd.Series) -> str:
        if row["company_name"] in duplicate_names:
            return f"{row['company_name']} ({str(row['tenant_id'])[:8]})"
        return row["company_name"]

    labels = data.apply(make_label, axis=1)
    label_to_position = dict(zip(labels, data.index))

    selected_label = st.selectbox("Select Enterprise Tenant Account:", sorted(labels))

    if selected_label:
        position = label_to_position[selected_label]
        selected_row = data.loc[position]
        risk = selected_row["churn_probability"]

        st.write(f"### {selected_row['company_name']}")
        st.caption(f"tenant_id: {selected_row['tenant_id']} | subscription tier: {selected_row['subscription_tier']}")

        st.progress(float(risk), text=f"Predicted churn risk: {risk:.0%}")

        st.subheader("What's driving this score")
        plt.figure()
        shap.plots.waterfall(explanation[position], show=False)
        fig = plt.gcf()
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        st.caption(
            "Each row shows one behavioral feature and how much it pushed this "
            "company's risk score up (red) or down (blue) from the average "
            "baseline risk, ending at this company's actual predicted score."
        )
