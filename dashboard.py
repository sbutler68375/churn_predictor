"""Customer Churn Prediction System -- Streamlit dashboard.

Week 5, Day 1: page layout and sidebar navigation skeleton.
Week 5, Day 2: Executive Overview KPI row.
Week 5, Day 3: Model Diagnostics page (SHAP chart, ROC curve, confusion matrix).
Week 5, Day 4: Account Deep-Dive company selector.
Week 5, Day 5: live risk score + local SHAP waterfall explanation.
Redesign: light theme, styled KPI cards, risk charts and an at-risk account
table on the Executive Overview.
"""

from datetime import datetime
from html import escape

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import shap
import streamlit as st
from sklearn.metrics import auc, confusion_matrix, precision_score, recall_score, roc_curve

from src.data_loader import load_churn_features
from src.evaluate import compute_shap_explanation
from src.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from src.paths import MODEL_PATH, TEST_PREDICTIONS_PATH

st.set_page_config(
    page_title="Churn Intelligence",
    page_icon="\U0001F4C9",
    layout="wide",
)

PAGES = ["Home", "Executive Overview", "Model Diagnostics", "Account Deep-Dive"]
HIGH_RISK_THRESHOLD = 0.70
MEDIUM_RISK_THRESHOLD = 0.30

# Design tokens -- text inks, surfaces, and the reserved status colors used
# for risk levels (always paired with a text label, never color alone).
COMPANY = "Northpeak"  # fictional company this internal tool belongs to
INK = "#18181b"
INK_2 = "#52525b"
INK_3 = "#71717a"
GRID = "#f0f0f1"
BORDER = "#e4e4e7"
ACCENT = "#b91c1c"
ACCENT_TINT = "#fdecec"
CHARCOAL = "#1f1f23"
OK_GREEN = "#16a34a"
RISK_COLORS = {"High": "#b91c1c", "Medium": "#d97706", "Low": "#71717a"}
RISK_TINTS = {"High": "#fdecec", "Medium": "#fef3e2", "Low": "#f1f1f3"}
NEUTRAL_BAR = "#d4d4d8"

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"], .stMarkdown, .stDataFrame, button, input {{
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }}
    #MainMenu, footer, [data-testid="stDecoration"] {{ visibility: hidden; height: 0; }}
    [data-testid="stHeader"] {{ background: transparent; height: 0; }}
    [data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"],
    [data-testid="stExpandSidebarButton"] {{ display: none; }}
    .block-container {{ padding-top: 1.2rem; padding-bottom: 3rem; max-width: 1320px; }}

    /* App header */
    .st-key-app_header {{
        background: {CHARCOAL}; border-radius: 12px; padding: 10px 20px; margin-bottom: 26px;
        border-bottom: 3px solid {ACCENT};
    }}
    .brand {{ display: flex; align-items: center; gap: 11px; }}
    .brand-mark {{
        width: 32px; height: 32px; border-radius: 7px; background: {ACCENT};
        color: #fff; display: grid; place-items: center; font-weight: 700; font-size: 15px; flex: none;
    }}
    .brand-name {{ font-weight: 700; font-size: 15px; color: #ffffff; line-height: 1.15; }}
    .brand-sub {{ font-size: 11.5px; color: #a1a1aa; }}
    .st-key-app_header [role="radiogroup"] {{ gap: 4px; flex-wrap: wrap; }}
    .st-key-app_header [role="radiogroup"] label {{
        padding: 7px 14px; border-radius: 7px; margin: 0;
    }}
    .st-key-app_header [role="radiogroup"] label p {{ color: #d4d4d8; font-size: 13.5px; font-weight: 500; }}
    .st-key-app_header [role="radiogroup"] label:hover {{ background: rgba(255,255,255,.07); }}
    .st-key-app_header [role="radiogroup"] label:has(input:checked) {{ background: rgba(255,255,255,.12); }}
    .st-key-app_header [role="radiogroup"] label:has(input:checked) p {{ color: #ffffff; font-weight: 600; }}
    .st-key-app_header [role="radiogroup"] label div:has(> [data-testid="stMarkdownContainer"]) > div:first-child {{
        display: none;
    }}
    .header-meta {{
        display: flex; align-items: center; justify-content: flex-end; gap: 14px;
        font-size: 12.5px; color: #d4d4d8; flex-wrap: wrap;
    }}
    .env-badge {{
        font-size: 10.5px; font-weight: 700; letter-spacing: .08em; text-transform: uppercase;
        color: #fff; background: {ACCENT}; border-radius: 4px; padding: 2px 7px;
    }}
    .fresh {{ display: inline-flex; align-items: center; gap: 6px; }}
    .avatar {{
        width: 28px; height: 28px; border-radius: 50%; background: #3f3f46; color: #fff;
        display: grid; place-items: center; font-size: 11px; font-weight: 600;
    }}

    /* Page header */
    .page-title {{ font-size: 26px; font-weight: 700; color: {INK}; margin: 0 0 4px; letter-spacing: -.02em; }}
    .page-sub {{ font-size: 14.5px; color: {INK_2}; margin-bottom: 22px; }}

    /* KPI cards */
    .kpi-grid {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 14px; margin-bottom: 18px; }}
    @media (max-width: 900px) {{ .kpi-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }} }}
    .kpi {{
        background: #fff; border: 1px solid {BORDER}; border-radius: 10px; padding: 16px 18px;
        box-shadow: 0 1px 2px rgba(24,24,27,.04);
    }}
    .kpi-label {{
        font-size: 11.5px; color: {INK_3}; font-weight: 600; letter-spacing: .06em; text-transform: uppercase;
    }}
    .kpi-value {{
        font-size: 26px; font-weight: 700; color: {INK}; margin: 8px 0 4px;
        letter-spacing: -.02em; font-variant-numeric: tabular-nums;
    }}
    .kpi-note {{ font-size: 12.5px; color: {INK_3}; }}
    .kpi-flag {{ display: inline-flex; align-items: center; gap: 6px; }}
    .dot {{ width: 8px; height: 8px; border-radius: 50%; display: inline-block; flex: none; }}

    /* Section cards (bordered st.container) */
    [data-testid="stVerticalBlockBorderWrapper"] {{
        background: #fff; border-radius: 10px; box-shadow: 0 1px 2px rgba(24,24,27,.04);
    }}
    .section-title {{ font-size: 15px; font-weight: 600; color: {INK}; margin: 0; }}
    .section-sub {{ font-size: 13px; color: {INK_2}; margin: 2px 0 10px; }}
    .section-label {{
        font-size: 12px; font-weight: 600; letter-spacing: .08em; text-transform: uppercase;
        color: {INK_3}; margin: 8px 0 10px;
    }}
    .fine-print {{ font-size: 12px; color: {INK_3}; margin-top: 18px; }}

    /* Buttons */
    .stButton button {{ border-radius: 7px; font-weight: 500; font-size: 13.5px; }}

    /* Risk pills */
    .pill {{
        display: inline-flex; align-items: center; gap: 6px; font-size: 12px; font-weight: 600;
        border-radius: 999px; padding: 3px 10px; white-space: nowrap; font-variant-numeric: tabular-nums;
    }}

    /* Home: key findings */
    .finding-eyebrow {{
        font-size: 11px; font-weight: 700; letter-spacing: .08em; text-transform: uppercase; color: {ACCENT};
    }}
    .finding-title {{ font-size: 16px; font-weight: 600; color: {INK}; margin: 3px 0 10px; }}
    .finding {{ padding: 10px 0; border-top: 1px solid {GRID}; }}
    .finding-value {{
        font-size: 20px; font-weight: 700; color: {INK}; letter-spacing: -.01em; font-variant-numeric: tabular-nums;
    }}
    .finding-text {{ font-size: 13px; color: {INK_2}; line-height: 1.45; margin-top: 1px; }}

    /* Home: priority queue */
    .queue-row {{
        display: grid; grid-template-columns: 26px minmax(0, 1fr) auto auto; align-items: center; gap: 14px;
        padding: 12px 2px; border-top: 1px solid {GRID};
    }}
    .queue-row:first-child {{ border-top: none; }}
    .queue-rank {{
        width: 24px; height: 24px; border-radius: 6px; background: #f4f4f5; color: {INK_2};
        display: grid; place-items: center; font-size: 12px; font-weight: 600;
    }}
    .queue-name {{ font-size: 14px; font-weight: 600; color: {INK}; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
    .queue-meta {{ font-size: 12.5px; color: {INK_3}; margin-top: 1px; }}
    .queue-arr {{ font-size: 14px; font-weight: 600; color: {INK}; font-variant-numeric: tabular-nums; text-align: right; }}

    /* Home: risk mix + status */
    .mix-bar {{ display: flex; height: 12px; border-radius: 6px; overflow: hidden; gap: 2px; margin: 4px 0 12px; }}
    .mix-legend {{ display: flex; flex-direction: column; gap: 8px; font-size: 13px; color: {INK_2}; }}
    .mix-legend div {{ display: flex; justify-content: space-between; gap: 12px; }}
    .mix-legend span {{ display: inline-flex; align-items: center; gap: 8px; }}
    .mix-legend b {{ color: {INK}; font-weight: 600; font-variant-numeric: tabular-nums; }}
    .status-row {{ display: flex; justify-content: space-between; gap: 12px; padding: 9px 0; border-top: 1px solid {GRID}; font-size: 13px; }}
    .status-row:first-of-type {{ border-top: none; }}
    .status-key {{ color: {INK_2}; display: inline-flex; align-items: center; gap: 8px; }}
    .status-val {{ color: {INK}; font-weight: 500; text-align: right; }}
    </style>
    """,
    unsafe_allow_html=True,
)


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
    data["risk_level"] = pd.cut(
        data["churn_probability"],
        bins=[-0.01, MEDIUM_RISK_THRESHOLD, HIGH_RISK_THRESHOLD, 1.0],
        labels=["Low", "Medium", "High"],
    ).astype(str)
    data.attrs["scored_at"] = datetime.now()
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


def money(value: float) -> str:
    if abs(value) >= 1_000_000:
        return f"${value / 1_000_000:.2f}M"
    if abs(value) >= 1_000:
        return f"${value / 1_000:.0f}K"
    return f"${value:,.0f}"


def risk_pill(probability: float) -> str:
    level = (
        "High" if probability > HIGH_RISK_THRESHOLD
        else "Medium" if probability > MEDIUM_RISK_THRESHOLD
        else "Low"
    )
    return (
        f'<span class="pill" style="background:{RISK_TINTS[level]};color:{RISK_COLORS[level]}">'
        f"{probability:.0%} &middot; {level}</span>"
    )


def clock(moment: datetime) -> str:
    return moment.strftime("%I:%M %p").lstrip("0")


def go_to(target: str) -> None:
    st.session_state["nav"] = target


def page_header(title: str, subtitle: str) -> None:
    st.markdown(
        f'<div class="page-title">{title}</div><div class="page-sub">{subtitle}</div>',
        unsafe_allow_html=True,
    )


def section_heading(title: str, subtitle: str = "") -> None:
    sub = f'<div class="section-sub">{subtitle}</div>' if subtitle else ""
    st.markdown(f'<div class="section-title">{title}</div>{sub}', unsafe_allow_html=True)


def style_figure(fig: go.Figure, height: int) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=8, r=16, t=8, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, system-ui, sans-serif", size=13, color=INK_2),
        hoverlabel=dict(bgcolor="#ffffff", bordercolor=BORDER, font=dict(color=INK, size=13)),
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False, linecolor=BORDER, tickfont=dict(color=INK_3))
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor=BORDER, tickfont=dict(color=INK_2))
    return fig


# ---------------------------------------------------------------- header
scored_at = get_live_predictions().attrs.get("scored_at", datetime.now())
with st.container(key="app_header"):
    brand_col, nav_col, meta_col = st.columns([1.25, 2.9, 1.45], vertical_alignment="center")
    with brand_col:
        st.markdown(
            f'<div class="brand"><div class="brand-mark">{COMPANY[0]}</div>'
            f'<div><div class="brand-name">{COMPANY}</div>'
            '<div class="brand-sub">Churn Intelligence</div></div></div>',
            unsafe_allow_html=True,
        )
    with nav_col:
        page = st.radio("Navigate", PAGES, key="nav", horizontal=True, label_visibility="collapsed")
    with meta_col:
        st.markdown(
            '<div class="header-meta"><span class="env-badge">Internal</span>'
            f'<span class="fresh"><span class="dot" style="background:{OK_GREEN}"></span>'
            f"Data as of {clock(scored_at)}</span>"
            '<span class="avatar" title="Customer Success Ops">CS</span></div>',
            unsafe_allow_html=True,
        )


def finding_card(eyebrow: str, title: str, items: list[tuple[str, str]], target: str) -> None:
    """One 'key findings' card on Home: a few headline numbers summarizing
    another page, plus a button that opens that page."""
    with st.container(border=True, height="stretch"):
        body = "".join(
            f'<div class="finding"><div class="finding-value">{value}</div>'
            f'<div class="finding-text">{text}</div></div>'
            for value, text in items
        )
        st.markdown(
            f'<div class="finding-eyebrow">{eyebrow}</div><div class="finding-title">{title}</div>{body}',
            unsafe_allow_html=True,
        )
        st.button(f"Open {target}", on_click=go_to, args=(target,), key=f"home_{target}", width="stretch")


# ---------------------------------------------------------------- pages
if page == "Home":
    predictions = get_live_predictions()
    total = len(predictions)
    total_arr = predictions["contract_value"].sum()
    is_high = predictions["churn_probability"] > HIGH_RISK_THRESHOLD
    high_risk = predictions[is_high]
    arr_at_risk = high_risk["contract_value"].sum()
    expected_loss = (predictions["churn_probability"] * predictions["contract_value"]).sum()
    portfolio_health = 1 - predictions["churn_probability"].mean()
    level_counts = predictions["risk_level"].value_counts().reindex(["High", "Medium", "Low"], fill_value=0)

    # Executive Overview: which plan tier carries the most risk
    by_tier = predictions.assign(is_high=is_high).groupby("subscription_tier")["is_high"].agg(["sum", "size"])
    by_tier["share"] = by_tier["sum"] / by_tier["size"]
    worst_tier = by_tier["share"].idxmax()
    worst = by_tier.loc[worst_tier]

    # Account Deep-Dive: what SHAP says pushes high-risk accounts up
    shap_data, explanation = get_deep_dive_data()
    shap_frame = pd.DataFrame(explanation.values, columns=explanation.feature_names)
    shap_high = shap_frame[shap_data["churn_probability"] > HIGH_RISK_THRESHOLD]
    top_drivers = shap_high.mean().sort_values(ascending=False).head(2).index.tolist()
    top_global = shap_frame.abs().mean().sort_values(ascending=False).head(3).index.tolist()
    rest = predictions[~is_high]
    tickets_high = high_risk["support_tickets_last_30d"].mean()
    tickets_rest = rest["support_tickets_last_30d"].mean()
    events_high = high_risk["events_last_7d"].mean()
    events_rest = rest["events_last_7d"].mean()

    # Model Diagnostics: held-out test performance
    test = pd.read_csv(TEST_PREDICTIONS_PATH)
    test_fpr, test_tpr, _ = roc_curve(test["y_true"], test["y_proba"])
    test_auc = auc(test_fpr, test_tpr)
    test_precision = precision_score(test["y_true"], test["y_pred"], zero_division=0)
    test_recall = recall_score(test["y_true"], test["y_pred"], zero_division=0)

    now = datetime.now()
    greeting = "Good morning" if now.hour < 12 else "Good afternoon" if now.hour < 17 else "Good evening"
    page_header(
        greeting,
        f"Retention briefing for {now:%A, %B} {now.day}. "
        "A summary of today's findings across the portfolio, the model and the accounts that need outreach.",
    )

    st.markdown(
        f"""
        <div class="kpi-grid">
          <div class="kpi">
            <div class="kpi-label">Accounts monitored</div>
            <div class="kpi-value">{total}</div>
            <div class="kpi-note">Scored live from the feature store</div>
          </div>
          <div class="kpi">
            <div class="kpi-label">Total ARR</div>
            <div class="kpi-value">{money(total_arr)}</div>
            <div class="kpi-note">Across all active contracts</div>
          </div>
          <div class="kpi">
            <div class="kpi-label">ARR at risk</div>
            <div class="kpi-value">{money(arr_at_risk)}</div>
            <div class="kpi-note kpi-flag"><span class="dot" style="background:{RISK_COLORS['High']}"></span>
              {arr_at_risk / total_arr:.1%} of total ARR</div>
          </div>
          <div class="kpi">
            <div class="kpi-label">High-risk accounts</div>
            <div class="kpi-value">{len(high_risk)}</div>
            <div class="kpi-note">Predicted churn above {HIGH_RISK_THRESHOLD:.0%}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-label">Key findings</div>', unsafe_allow_html=True)
    f1, f2, f3 = st.columns(3, gap="medium")
    with f1:
        finding_card(
            "Executive Overview",
            "Where the revenue risk sits",
            [
                (f"{money(arr_at_risk)}", f"of ARR ({arr_at_risk / total_arr:.0%}) is tied to "
                 f"{len(high_risk)} high-risk accounts"),
                (f"{worst['share']:.0%}", f"of {worst_tier} accounts are high risk "
                 f"({int(worst['sum'])} of {int(worst['size'])}), the highest of any plan"),
                (f"{money(expected_loss)}", f"probability-weighted ARR loss; portfolio health is "
                 f"{portfolio_health:.0%}"),
            ],
            "Executive Overview",
        )
    with f2:
        finding_card(
            "Account Deep-Dive",
            "What's driving churn",
            [
                (f"{top_drivers[0].capitalize()}", f"is the biggest factor pushing high-risk accounts up, "
                 f"followed by {top_drivers[1]}"),
                (f"{tickets_high:.0f} vs {tickets_rest:.0f}", "average support tickets (30d): "
                 "high-risk accounts vs. everyone else"),
                (f"{events_high:,.0f} vs {events_rest:,.0f}", "average product events (7d): "
                 "high-risk accounts vs. everyone else"),
            ],
            "Account Deep-Dive",
        )
    with f3:
        finding_card(
            "Model Diagnostics",
            "How far to trust the scores",
            [
                (f"{test_auc:.2f}", f"ROC AUC on {len(test)} held-out accounts "
                 "(0.99 mean across 5-fold cross-validation)"),
                (f"{test_precision:.0%} / {test_recall:.0%}", "precision / recall: share of flagged "
                 "accounts that churned, and share of churners caught"),
                (f"{top_global[0].capitalize()}", f"is the most important feature overall, ahead of "
                 f"{top_global[1]} and {top_global[2]}"),
            ],
            "Model Diagnostics",
        )

    st.markdown('<div class="section-label" style="margin-top:22px">Action items</div>', unsafe_allow_html=True)
    left, right = st.columns([1.65, 1], gap="medium")

    with left:
        with st.container(border=True):
            section_heading(
                "Priority outreach",
                "High-risk accounts ranked by contract value: the renewals to protect first",
            )
            queue = high_risk.sort_values("contract_value", ascending=False).head(6)
            if queue.empty:
                st.markdown(
                    '<div class="queue-meta">No accounts are above the high-risk threshold right now.</div>',
                    unsafe_allow_html=True,
                )
            else:
                # Built without indentation/newlines: indented lines inside st.markdown
                # are parsed as a Markdown code block instead of HTML.
                rows = "".join(
                    '<div class="queue-row">'
                    f'<div class="queue-rank">{rank}</div>'
                    '<div style="min-width:0">'
                    f'<div class="queue-name">{escape(str(row.company_name))}</div>'
                    f'<div class="queue-meta">{row.subscription_tier} plan &middot; '
                    f"{int(row.support_tickets_last_30d)} tickets &middot; "
                    f"{int(row.events_last_30d):,} events (30d)</div></div>"
                    f'<div class="queue-arr">{money(row.contract_value)}</div>'
                    f"{risk_pill(row.churn_probability)}</div>"
                    for rank, row in enumerate(queue.itertuples(), start=1)
                )
                st.markdown(f"<div>{rows}</div>", unsafe_allow_html=True)
            st.button(
                f"View all {len(high_risk)} high-risk accounts",
                on_click=go_to, args=("Executive Overview",), key="home_watchlist",
            )

    with right:
        with st.container(border=True):
            section_heading("Risk mix", "Accounts by predicted churn level")
            segments = "".join(
                f'<div style="flex:{max(int(n), 0)};background:{RISK_COLORS[lvl]}"></div>'
                for lvl, n in level_counts.items() if n > 0
            )
            legend = "".join(
                f'<div><span><span class="dot" style="background:{RISK_COLORS[lvl]}"></span>{lvl} risk</span>'
                f"<b>{int(n)} &middot; {n / total:.0%}</b></div>"
                for lvl, n in level_counts.items()
            )
            st.markdown(
                f'<div class="mix-bar">{segments}</div><div class="mix-legend">{legend}</div>',
                unsafe_allow_html=True,
            )

        with st.container(border=True):
            section_heading("System status")
            ok = f'<span class="dot" style="background:{OK_GREEN}"></span>'
            st.markdown(
                f'<div class="status-row"><span class="status-key">{ok}Feature store (PostgreSQL)</span>'
                f'<span class="status-val">Connected &middot; {total} accounts</span></div>'
                f'<div class="status-row"><span class="status-key">{ok}Churn model</span>'
                '<span class="status-val">Random Forest v1</span></div>'
                f'<div class="status-row"><span class="status-key">{ok}Scoring job</span>'
                f'<span class="status-val">Last run {clock(scored_at)} &middot; every 5 min</span></div>',
                unsafe_allow_html=True,
            )

    st.markdown(
        f'<div class="fine-print">Internal use only. {COMPANY} is a fictional company; all account data is '
        "synthetic. Cross-validated ROC AUC is the Random Forest mean from 5-fold stratified "
        "cross-validation; see MODEL_METRICS.md for limitations.</div>",
        unsafe_allow_html=True,
    )

elif page == "Executive Overview":
    predictions = get_live_predictions()
    total = len(predictions)
    total_arr = predictions["contract_value"].sum()
    high_risk = predictions[predictions["churn_probability"] > HIGH_RISK_THRESHOLD]
    arr_at_risk = high_risk["contract_value"].sum()
    portfolio_health = 1 - predictions["churn_probability"].mean()
    expected_loss = (predictions["churn_probability"] * predictions["contract_value"]).sum()

    page_header(
        "Executive Overview",
        f"Live churn risk across {total} customer accounts &middot; "
        f"updated {datetime.now():%b %d, %Y at %I:%M %p}",
    )

    st.markdown(
        f"""
        <div class="kpi-grid">
          <div class="kpi">
            <div class="kpi-label">Total ARR</div>
            <div class="kpi-value">{money(total_arr)}</div>
            <div class="kpi-note">{total} active accounts</div>
          </div>
          <div class="kpi">
            <div class="kpi-label">ARR at risk</div>
            <div class="kpi-value">{money(arr_at_risk)}</div>
            <div class="kpi-note kpi-flag"><span class="dot" style="background:{RISK_COLORS['High']}"></span>
              {arr_at_risk / total_arr:.1%} of total ARR in high-risk accounts</div>
          </div>
          <div class="kpi">
            <div class="kpi-label">High-risk accounts</div>
            <div class="kpi-value">{len(high_risk)} <span style="font-size:16px;color:{INK_3};font-weight:500">/ {total}</span></div>
            <div class="kpi-note">Predicted churn above {HIGH_RISK_THRESHOLD:.0%}</div>
          </div>
          <div class="kpi">
            <div class="kpi-label">Portfolio health</div>
            <div class="kpi-value">{portfolio_health:.0%}</div>
            <div class="kpi-note">Probability-weighted ARR loss: {money(expected_loss)}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.15, 1], gap="medium")

    with left:
        with st.container(border=True):
            section_heading("ARR by risk level", "Contract value grouped by predicted churn probability")
            levels = ["High", "Medium", "Low"]
            by_level = (
                predictions.groupby("risk_level")
                .agg(accounts=("tenant_id", "size"), arr=("contract_value", "sum"))
                .reindex(levels, fill_value=0)
            )
            band_text = {
                "High": f"> {HIGH_RISK_THRESHOLD:.0%}",
                "Medium": f"{MEDIUM_RISK_THRESHOLD:.0%}–{HIGH_RISK_THRESHOLD:.0%}",
                "Low": f"< {MEDIUM_RISK_THRESHOLD:.0%}",
            }
            fig = go.Figure(
                go.Bar(
                    x=by_level["arr"],
                    y=[f"{lvl} risk  " for lvl in levels],
                    orientation="h",
                    marker=dict(color=[RISK_COLORS[l] for l in levels], cornerradius=4),
                    text=[f"{money(a)} · {n} account{'' if n == 1 else 's'}" for a, n in zip(by_level["arr"], by_level["accounts"])],
                    textposition="outside",
                    textfont=dict(color=INK, size=13),
                    cliponaxis=False,
                    customdata=[[band_text[l], n] for l, n in zip(levels, by_level["accounts"])],
                    hovertemplate="<b>%{y}</b> (%{customdata[0]})<br>ARR: %{x:$,.0f}"
                    "<br>Accounts: %{customdata[1]}<extra></extra>",
                    width=0.55,
                )
            )
            fig.update_xaxes(tickprefix="$", tickformat="~s", range=[0, by_level["arr"].max() * 1.6])
            fig.update_yaxes(autorange="reversed", showgrid=False)
            st.plotly_chart(style_figure(fig, 260), use_container_width=True, config={"displayModeBar": False})

    with right:
        with st.container(border=True):
            section_heading("High-risk accounts by plan", "Share of each subscription tier above the risk threshold")
            tiers = ["Enterprise", "Pro", "Starter"]
            by_tier = (
                predictions.assign(is_high=predictions["churn_probability"] > HIGH_RISK_THRESHOLD)
                .groupby("subscription_tier")
                .agg(accounts=("tenant_id", "size"), high=("is_high", "sum"))
                .reindex(tiers, fill_value=0)
            )
            by_tier["other"] = by_tier["accounts"] - by_tier["high"]
            fig = go.Figure()
            fig.add_trace(
                go.Bar(
                    name="High risk",
                    y=tiers,
                    x=by_tier["high"],
                    orientation="h",
                    marker=dict(color=RISK_COLORS["High"], line=dict(color="#ffffff", width=2)),
                    hovertemplate="<b>%{y}</b><br>High risk: %{x} accounts<extra></extra>",
                    width=0.55,
                )
            )
            fig.add_trace(
                go.Bar(
                    name="Not high risk",
                    y=tiers,
                    x=by_tier["other"],
                    orientation="h",
                    marker=dict(color=NEUTRAL_BAR, line=dict(color="#ffffff", width=2)),
                    text=[f"{h} of {n} ({h / n:.0%})" for h, n in zip(by_tier["high"], by_tier["accounts"])],
                    textposition="outside",
                    textfont=dict(color=INK, size=13),
                    cliponaxis=False,
                    hovertemplate="<b>%{y}</b><br>Not high risk: %{x} accounts<extra></extra>",
                    width=0.55,
                )
            )
            fig.update_layout(
                barmode="stack",
                legend=dict(orientation="h", y=-0.18, x=0, font=dict(color=INK_2), traceorder="normal"),
            )
            fig.update_xaxes(title=None, range=[0, by_tier["accounts"].max() * 1.4])
            fig.update_yaxes(autorange="reversed", showgrid=False, ticksuffix="  ")
            st.plotly_chart(style_figure(fig, 260), use_container_width=True, config={"displayModeBar": False})

    with st.container(border=True):
        section_heading(
            "Accounts needing attention",
            "High-risk accounts ranked by contract value: the biggest renewals to protect first",
        )
        watchlist = (
            high_risk.sort_values("contract_value", ascending=False)
            .loc[
                :,
                [
                    "company_name",
                    "subscription_tier",
                    "contract_value",
                    "churn_probability",
                    "events_last_30d",
                    "support_tickets_last_30d",
                ],
            ]
            .rename(
                columns={
                    "company_name": "Account",
                    "subscription_tier": "Plan",
                    "contract_value": "ARR",
                    "churn_probability": "Churn risk",
                    "events_last_30d": "Events (30d)",
                    "support_tickets_last_30d": "Support tickets (30d)",
                }
            )
        )
        watchlist["Churn risk"] = watchlist["Churn risk"] * 100
        st.dataframe(
            watchlist,
            hide_index=True,
            use_container_width=True,
            height=min(38 + 35 * len(watchlist), 420),
            column_config={
                "Account": st.column_config.TextColumn(width="medium"),
                "ARR": st.column_config.NumberColumn(format="$%,.0f"),
                "Churn risk": st.column_config.ProgressColumn(
                    format="%.0f%%", min_value=0, max_value=100, width="medium"
                ),
                "Events (30d)": st.column_config.NumberColumn(format="%d"),
                "Support tickets (30d)": st.column_config.NumberColumn(format="%d"),
            },
        )
        st.caption("Open **Account Deep-Dive** in the sidebar to see what's driving any account's score.")

    st.markdown(
        f'<div class="fine-print">Risk levels: High &gt; {HIGH_RISK_THRESHOLD:.0%}, '
        f"Medium {MEDIUM_RISK_THRESHOLD:.0%}–{HIGH_RISK_THRESHOLD:.0%}, Low &lt; {MEDIUM_RISK_THRESHOLD:.0%} "
        "predicted churn probability. Portfolio health = 1 − average predicted churn probability.</div>",
        unsafe_allow_html=True,
    )

elif page == "Model Diagnostics":
    page_header(
        "Model Diagnostics",
        "Evaluation of the Random Forest model on the held-out test set",
    )

    with st.container(border=True):
        section_heading(
            "Global feature importance (SHAP)",
            "Mean absolute SHAP value per feature across all live accounts: how much each one moves the score",
        )
        _, explanation = get_deep_dive_data()
        importance = (
            pd.DataFrame(explanation.values, columns=explanation.feature_names)
            .abs().mean().sort_values(ascending=False)
        )
        fig = go.Figure(
            go.Bar(
                x=importance.values,
                y=[f"{name.capitalize()}  " for name in importance.index],
                orientation="h",
                marker=dict(
                    color=[ACCENT if i < 2 else NEUTRAL_BAR for i in range(len(importance))],
                    cornerradius=4,
                ),
                text=[f"{v:.3f}" for v in importance.values],
                textposition="outside",
                textfont=dict(color=INK, size=12),
                cliponaxis=False,
                hovertemplate="<b>%{y}</b><br>Mean |SHAP|: %{x:.3f}<extra></extra>",
                width=0.6,
            )
        )
        fig.update_xaxes(range=[0, importance.max() * 1.15], title=None)
        fig.update_yaxes(autorange="reversed", showgrid=False)
        st.plotly_chart(
            style_figure(fig, 40 * len(importance) + 40),
            use_container_width=True, config={"displayModeBar": False},
        )

    test_predictions = pd.read_csv(TEST_PREDICTIONS_PATH)
    y_true = test_predictions["y_true"]
    y_pred = test_predictions["y_pred"]
    y_proba = test_predictions["y_proba"]

    roc_col, cm_col = st.columns(2, gap="medium")

    with roc_col:
        with st.container(border=True):
            fpr, tpr, _ = roc_curve(y_true, y_proba)
            roc_auc = auc(fpr, tpr)
            section_heading("ROC curve", f"AUC = {roc_auc:.2f}")

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(x=fpr, y=tpr, mode="lines", name="Model", line=dict(color=ACCENT, width=2))
            )
            fig.add_trace(
                go.Scatter(
                    x=[0, 1], y=[0, 1], mode="lines", name="Random guess",
                    line=dict(dash="dash", color=INK_3, width=1.5),
                )
            )
            fig.update_layout(
                xaxis_title="False positive rate",
                yaxis_title="True positive rate",
                legend=dict(orientation="h", y=-0.25, x=0),
            )
            st.plotly_chart(style_figure(fig, 360), use_container_width=True, config={"displayModeBar": False})

    with cm_col:
        with st.container(border=True):
            section_heading("Confusion matrix", "Predicted vs. actual churn outcome")
            cm = confusion_matrix(y_true, y_pred)
            labels = ["Healthy", "At-risk"]

            fig = go.Figure(
                data=go.Heatmap(
                    z=cm,
                    x=[f"Predicted {l}" for l in labels],
                    y=[f"Actual {l}" for l in labels],
                    text=cm,
                    texttemplate="%{text}",
                    textfont=dict(size=18),
                    colorscale=[[0, "#f4f4f5"], [1, "#991b1b"]],
                    showscale=False,
                    xgap=2,
                    ygap=2,
                    hovertemplate="%{y} / %{x}: %{z}<extra></extra>",
                )
            )
            fig.update_yaxes(autorange="reversed", showgrid=False)
            fig.update_xaxes(showgrid=False)
            st.plotly_chart(style_figure(fig, 360), use_container_width=True, config={"displayModeBar": False})

    st.markdown(
        f'<div class="fine-print">Computed on the {len(test_predictions)}-row held-out test set '
        "(Random Forest model).</div>",
        unsafe_allow_html=True,
    )

elif page == "Account Deep-Dive":
    page_header(
        "Account Deep-Dive",
        "Pick an account to see its predicted churn risk and the behaviors driving it",
    )
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

    selected_label = st.selectbox("Account", sorted(labels))

    if selected_label:
        position = label_to_position[selected_label]
        selected_row = data.loc[position]
        risk = selected_row["churn_probability"]
        level = "High" if risk > HIGH_RISK_THRESHOLD else "Medium" if risk > MEDIUM_RISK_THRESHOLD else "Low"

        st.markdown(
            f"""
            <div class="kpi-grid">
              <div class="kpi">
                <div class="kpi-label">Predicted churn risk</div>
                <div class="kpi-value">{risk:.0%}</div>
                <div class="kpi-note kpi-flag"><span class="dot" style="background:{RISK_COLORS[level]}"></span>{level} risk</div>
              </div>
              <div class="kpi">
                <div class="kpi-label">Annual contract value</div>
                <div class="kpi-value">{money(selected_row['contract_value'])}</div>
                <div class="kpi-note">{selected_row['subscription_tier']} plan</div>
              </div>
              <div class="kpi">
                <div class="kpi-label">Product events (30d)</div>
                <div class="kpi-value">{int(selected_row['events_last_30d']):,}</div>
                <div class="kpi-note">{int(selected_row['events_last_7d']):,} in the last 7 days</div>
              </div>
              <div class="kpi">
                <div class="kpi-label">Support tickets (30d)</div>
                <div class="kpi-value">{int(selected_row['support_tickets_last_30d'])}</div>
                <div class="kpi-note">tenant {str(selected_row['tenant_id'])[:8]}</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.container(border=True):
            section_heading(
                "What's driving this score",
                "Each row shows how much one behavior pushed this account's risk up (red) or "
                "down (blue) from the average baseline, ending at its predicted score.",
            )
            plt.figure()
            shap.plots.waterfall(explanation[position], show=False)
            fig = plt.gcf()
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
