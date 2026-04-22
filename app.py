"""
app.py — AML Pattern Detection | Streamlit Dashboard
GlobalBank Compliance Desk
"""

import os
import sys
import warnings
import io

warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from PIL import Image
from sklearn.decomposition import PCA

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AML Pattern Detection — GlobalBank",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Inject CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ---- Global background ---- */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0d0f1a 0%, #111426 50%, #0a0c18 100%);
    color: #e4e6f1;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f1120 0%, #161930 100%);
    border-right: 1px solid #2a2d45;
}
[data-testid="stSidebar"] * { color: #c8cce8 !important; }

/* ---- Metric cards ---- */
[data-testid="metric-container"] {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px;
    padding: 14px 18px;
    backdrop-filter: blur(8px);
}
[data-testid="stMetricValue"] { color: #00d4ff !important; font-size: 2rem !important; }
[data-testid="stMetricLabel"] { color: #888db5 !important; font-size:.85rem !important; }
[data-testid="stMetricDelta"] { font-size:.85rem !important; }

/* ---- Section headers ---- */
.section-header {
    font-size: 1.35rem;
    font-weight: 700;
    color: #ffffff;
    padding: 10px 0 4px;
    border-bottom: 2px solid #00d4ff33;
    margin-bottom: 16px;
    letter-spacing:.5px;
}
.sub-header {
    font-size: 1.05rem;
    font-weight: 600;
    color: #a0a8d8;
    margin: 12px 0 6px;
}

/* ---- Alert / SAR boxes ---- */
.sar-box {
    background: rgba(255,60,60,0.07);
    border-left: 4px solid #ff4d4d;
    border-radius: 0 10px 10px 0;
    padding: 14px 20px;
    margin-bottom: 18px;
}
.sar-box h4 { color: #ff4d4d; margin:0 0 8px; font-size:1rem; }
.sar-box p  { color: #e0e0e0; margin:2px 0; font-size:.9rem; }

.medium-box {
    background: rgba(255,165,0,0.07);
    border-left: 4px solid #f7971e;
    border-radius: 0 10px 10px 0;
    padding: 14px 20px;
    margin-bottom: 18px;
}
.low-box {
    background: rgba(67,233,123,0.06);
    border-left: 4px solid #43e97b;
    border-radius: 0 10px 10px 0;
    padding: 14px 20px;
    margin-bottom: 18px;
}

/* ---- Tabs ---- */
[data-baseweb="tab"] { color: #888db5 !important; font-weight: 600; }
[aria-selected="true"] { color: #00d4ff !important; }
[data-baseweb="tab-border"] { background: #00d4ff !important; }

/* ---- Tables ---- */
[data-testid="stDataFrame"] thead th {
    background: #1a1d2e !important;
    color: #00d4ff !important;
}

/* ---- Sidebar radio labels ---- */
.stRadio { padding-top: 6px; }

/* ---- Image border ---- */
img { border-radius: 12px; }
</style>
""", unsafe_allow_html=True)

# ── Load Pipeline Modules ─────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from src.data_loader  import load_and_preprocess
from src.models       import apply_unsupervised_models
from src.typologies   import apply_typology_rules
from src.risk_scoring import calculate_risk_scores

FEATURE_COLS = [
    "Avg_Transaction_Amount_INR",
    "Transaction_Frequency_Monthly",
    "Unique_Counterparties_Monthly",
    "Cash_Deposit_Ratio",
    "International_Transfer_Ratio",
]
GRAPHS_DIR   = os.path.join(BASE_DIR, "graphs")
DATASET_PATH = os.path.join(BASE_DIR, "P9_AML_Accounts.csv")


# ── Pipeline (cached) ─────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def run_pipeline():
    df, features, X_scaled, scaler = load_and_preprocess(DATASET_PATH)
    df, kmeans, iso = apply_unsupervised_models(
        df, X_scaled, feature_cols=FEATURE_COLS, n_clusters=4, contamination=0.05
    )
    df = apply_typology_rules(df)
    df = calculate_risk_scores(df)
    return df, X_scaled


@st.cache_data(show_spinner=False)
def load_graph(filename):
    path = os.path.join(GRAPHS_DIR, filename)
    if os.path.exists(path):
        return Image.open(path)
    return None


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏦 GlobalBank")
    st.markdown("### AML Compliance Dashboard")
    st.markdown("---")

    page = st.radio(
        "Navigate",
        [
            "🏠 Overview",
            "📊 EDA Analysis",
            "🔵 K-Means Clustering",
            "🌲 Isolation Forest",
            "⚠️ Risk Scoring",
            "🔗 Cross-Validation",
            "📋 SAR Reports",
            "❓ Presentation Notes",
        ],
        index=0,
    )

    st.markdown("---")
    st.markdown("**Dataset:** `P9_AML_Accounts.csv`")
    st.markdown("**Accounts:** 200")
    st.markdown("**Period:** 30 days")
    st.markdown("**Model:** KMeans + Isolation Forest")
    st.markdown("---")
    st.markdown(
        "<small style='color:#555d8a'>© 2026 GlobalBank Compliance Desk</small>",
        unsafe_allow_html=True,
    )


# ── Load Data ─────────────────────────────────────────────────────────────────
with st.spinner("Running AML pipeline…"):
    df, X_scaled = run_pipeline()

top10 = df.nlargest(10, "Risk_Score")
top5  = df.nlargest(5,  "Risk_Score")

high_count   = int((df["Risk_Score"] >= 75).sum())
medium_count = int(((df["Risk_Score"] >= 50) & (df["Risk_Score"] < 75)).sum())
low_count    = int((df["Risk_Score"] < 50).sum())
anomaly_count = int(df["Is_Anomaly"].sum())
smurf_count  = int(df["Smurfing_Flag"].sum())
round_count  = int(df["RoundTripping_Flag"].sum())
layer_count  = int(df["Layering_Flag"].sum())


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.markdown("""
    <div style="text-align:center; padding: 30px 0 20px;">
        <div style="font-size:3.5rem; margin-bottom:10px;">🏦</div>
        <h1 style="font-size:2.4rem; font-weight:800; color:#ffffff; margin:0;">
            Anti-Money Laundering<br>
            <span style="color:#00d4ff;">Pattern Detection System</span>
        </h1>
        <p style="color:#888db5; font-size:1.1rem; margin-top:12px;">
            GlobalBank Compliance Desk · Unsupervised ML Pipeline · 22 April 2026
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ── KPI Row ──────────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
    c1.metric("Total Accounts",  "200")
    c2.metric("🔴 HIGH Risk",   str(high_count),   f"{high_count/200:.1%} of dataset",   delta_color="inverse")
    c3.metric("🟡 MEDIUM Risk", str(medium_count), f"{medium_count/200:.1%}")
    c4.metric("🟢 LOW Risk",    str(low_count),    f"{low_count/200:.1%}")
    c5.metric("IF Anomalies",   str(anomaly_count))
    c6.metric("Smurfing Flags", str(smurf_count))
    c7.metric("Layering Flags", str(layer_count))

    st.markdown("---")

    # ── Two-Column: Top 10 table + Risk Gauge ────────────────────────────────
    col_table, col_gauge = st.columns([3, 2])

    with col_table:
        st.markdown('<div class="section-header">🏆 Top 10 Highest-Risk Accounts</div>', unsafe_allow_html=True)
        display_df = top10[
            ["Account_ID", "Risk_Score", "Risk_Level",
             "Smurfing_Flag", "RoundTripping_Flag", "Layering_Flag", "Is_Anomaly"]
        ].copy()
        display_df.columns = ["Account", "Score", "Level", "Smurfing", "RndTrip", "Layering", "IF Anomaly"]
        display_df["Score"] = display_df["Score"].round(1)

        def colour_score(val):
            if val >= 75:
                return "background-color:#3a0a0a; color:#ff6b6b; font-weight:bold"
            elif val >= 50:
                return "background-color:#3a2a00; color:#f7971e; font-weight:bold"
            return "color:#43e97b"

        st.dataframe(
            display_df.style.map(colour_score, subset=["Score"]),
            use_container_width=True,
            height=360,
        )

    with col_gauge:
        st.markdown('<div class="section-header">📈 Risk Score Distribution</div>', unsafe_allow_html=True)
        img = load_graph("13_risk_score_distribution.png")
        if img:
            st.image(img, use_container_width=True)

    st.markdown("---")

    # ── Risk score bar chart live ────────────────────────────────────────────
    st.markdown('<div class="section-header">📉 Top 10 Account Risk Scores</div>', unsafe_allow_html=True)
    img16 = load_graph("16_top10_accounts_bar.png")
    if img16:
        st.image(img16, use_container_width=True)

    # ── Typology badges ──────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="section-header">🚩 AML Typology Summary</div>', unsafe_allow_html=True)
    b1, b2, b3 = st.columns(3)
    with b1:
        st.markdown(f"""
        <div style="background:rgba(255,107,107,0.12); border:1px solid #ff6b6b44;
                    border-radius:14px; padding:20px; text-align:center;">
            <div style="font-size:2.5rem;">💵</div>
            <div style="font-size:2rem; font-weight:800; color:#ff6b6b;">{smurf_count}</div>
            <div style="color:#e0e0e0; font-weight:600;">SMURFING Flags</div>
            <div style="color:#888; font-size:.82rem; margin-top:6px;">
                High freq + small amounts + high cash ratio
            </div>
        </div>""", unsafe_allow_html=True)
    with b2:
        st.markdown(f"""
        <div style="background:rgba(247,151,30,0.10); border:1px solid #f7971e44;
                    border-radius:14px; padding:20px; text-align:center;">
            <div style="font-size:2.5rem;">🔄</div>
            <div style="font-size:2rem; font-weight:800; color:#f7971e;">{round_count}</div>
            <div style="color:#e0e0e0; font-weight:600;">ROUND-TRIPPING Flags</div>
            <div style="color:#888; font-size:.82rem; margin-top:6px;">
                High freq + very few counterparties
            </div>
        </div>""", unsafe_allow_html=True)
    with b3:
        st.markdown(f"""
        <div style="background:rgba(0,201,255,0.08); border:1px solid #00c9ff33;
                    border-radius:14px; padding:20px; text-align:center;">
            <div style="font-size:2.5rem;">🌐</div>
            <div style="font-size:2rem; font-weight:800; color:#00c9ff;">{layer_count}</div>
            <div style="color:#e0e0e0; font-weight:600;">LAYERING Flags</div>
            <div style="color:#888; font-size:.82rem; margin-top:6px;">
                Many counterparties + high intl transfers
            </div>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: EDA
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 EDA Analysis":
    st.markdown('<div class="section-header">📊 Exploratory Data Analysis (EDA)</div>', unsafe_allow_html=True)
    st.caption("Understanding the data distribution before applying ML models.")

    # Raw data preview
    with st.expander("📂 Raw Dataset Preview", expanded=False):
        raw = pd.read_csv(DATASET_PATH)
        st.dataframe(raw, use_container_width=True, height=300)
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Rows", raw.shape[0])
        col_b.metric("Columns", raw.shape[1])
        col_c.metric("Missing Values", raw.isnull().sum().sum())

    st.markdown("---")

    # Summary stats
    st.markdown('<div class="sub-header">Descriptive Statistics</div>', unsafe_allow_html=True)
    st.dataframe(
        df[FEATURE_COLS].describe().round(3).style.background_gradient(cmap="Blues"),
        use_container_width=True,
    )

    st.markdown("---")

    # Chart 01
    st.markdown('<div class="sub-header">01 · Feature Distributions (Histogram + KDE)</div>', unsafe_allow_html=True)
    img = load_graph("01_feature_distributions.png")
    if img: st.image(img, use_container_width=True)
    st.caption("Each feature plotted with density curve. Red dashed line = mean. Note the bimodal shape in Transaction Frequency — indicating two distinct account behaviour groups.")

    st.markdown("---")

    # Chart 02
    st.markdown('<div class="sub-header">02 · Feature Correlation Heatmap</div>', unsafe_allow_html=True)
    col1, col2 = st.columns([2, 1])
    with col1:
        img = load_graph("02_correlation_heatmap.png")
        if img: st.image(img, use_container_width=True)
    with col2:
        st.markdown("#### Key Insights")
        st.markdown("""
        - **Transaction Frequency ↔ Unique Counterparties**: Moderate positive correlation — accounts making more transactions tend to use more counterparties (layering pattern)
        - **Cash Deposit Ratio ↔ International Transfer Ratio**: Slight positive correlation — smurfing accounts also tend to have higher international exposure
        - **Avg Amount ↔ Frequency**: Near-zero correlation — confirms two distinct behavioural archetypes (high-value-low-volume vs. low-value-high-volume)
        """)

    st.markdown("---")

    # Charts 03 & 04
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown('<div class="sub-header">03 · Boxplots: Normal vs Anomaly Accounts</div>', unsafe_allow_html=True)
        img = load_graph("03_boxplots_features.png")
        if img: st.image(img, use_container_width=True)
        st.caption("Anomaly accounts (right box each pair) show extreme Transaction Frequency and Cash Deposit Ratio values.")
    with col_b:
        st.markdown('<div class="sub-header">04 · Pairplot — All Feature Combinations</div>', unsafe_allow_html=True)
        img = load_graph("04_pairplot.png")
        if img: st.image(img, use_container_width=True)
        st.caption("Blue = normal accounts · Red = Isolation Forest anomalies. Clear separation in Frequency vs Cash Ratio and Frequency vs Intl Transfer axes.")


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: K-MEANS CLUSTERING
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔵 K-Means Clustering":
    st.markdown('<div class="section-header">🔵 K-Means Clustering — Behavioural Segmentation</div>', unsafe_allow_html=True)
    st.caption("K-Means groups accounts into 4 behavioural clusters. High-risk clusters have elevated Cash Deposit Ratio and International Transfer Ratio.")

    # Cluster stats table
    st.markdown('<div class="sub-header">Cluster Summary Statistics</div>', unsafe_allow_html=True)
    cluster_profile = df.groupby("Cluster")[FEATURE_COLS + ["Cluster_Risk", "Risk_Score"]].mean().round(3)
    cluster_profile.index = [f"Cluster {i}" for i in cluster_profile.index]
    st.dataframe(
        cluster_profile.style.background_gradient(cmap="YlOrRd", subset=["Cluster_Risk", "Risk_Score"]),
        use_container_width=True,
    )

    st.markdown("---")

    # Elbow + Silhouette
    col_e, col_s = st.columns(2)
    with col_e:
        st.markdown('<div class="sub-header">05 · Elbow Curve (Optimal k)</div>', unsafe_allow_html=True)
        img = load_graph("05_elbow_curve.png")
        if img: st.image(img, use_container_width=True)
        st.caption("The 'elbow' at k=4 indicates diminishing returns beyond 4 clusters. Red dashed line marks the chosen k.")
    with col_s:
        st.markdown('<div class="sub-header">06 · Silhouette Scores per k</div>', unsafe_allow_html=True)
        img = load_graph("06_silhouette_scores.png")
        if img: st.image(img, use_container_width=True)
        st.caption("Higher silhouette = tighter, better-separated clusters. Confirms k=4 as a strong choice.")

    st.markdown("---")

    # PCA scatter
    st.markdown('<div class="sub-header">07 · Cluster Map (PCA 2D Projection)</div>', unsafe_allow_html=True)
    img = load_graph("07_cluster_scatter_pca.png")
    if img: st.image(img, use_container_width=True)
    st.caption("Red circles = Isolation Forest anomalies. Note how anomalies concentrate in the high-frequency cluster.")

    st.markdown("---")

    col_r, col_h = st.columns(2)
    with col_r:
        st.markdown('<div class="sub-header">08 · Cluster Feature Profiles (Radar Chart)</div>', unsafe_allow_html=True)
        img = load_graph("08_cluster_profiles_radar.png")
        if img: st.image(img, use_container_width=True)
        st.caption("Each polygon represents one cluster's normalised feature means. The 'spiky' high-frequency cluster is clearly the highest-risk group.")
    with col_h:
        st.markdown('<div class="sub-header">09 · Cluster Centroids Heatmap</div>', unsafe_allow_html=True)
        img = load_graph("09_cluster_heatmap.png")
        if img: st.image(img, use_container_width=True)
        st.caption("Normalised centroid values. Darker red = higher relative value. Identifies which features drive each cluster's behaviour.")


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: ISOLATION FOREST
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🌲 Isolation Forest":
    st.markdown('<div class="section-header">🌲 Isolation Forest — Statistical Anomaly Detection</div>', unsafe_allow_html=True)
    st.caption("Isolation Forest isolates anomalies by randomly partitioning the feature space. Accounts that are isolated in fewer splits are anomalous.")

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Total Accounts", 200)
    col_b.metric("Anomalies Detected", anomaly_count, f"{anomaly_count/200:.1%} contamination")
    col_c.metric("Normal Accounts", 200 - anomaly_count)

    st.markdown("---")

    # Score distribution
    st.markdown('<div class="sub-header">10 · Anomaly Score Distribution</div>', unsafe_allow_html=True)
    img = load_graph("10_anomaly_score_distribution.png")
    if img: st.image(img, use_container_width=True)
    st.caption("Blue KDE = normal accounts (low scores). Red KDE = anomalies (scores > 0.8). Yellow dashed line marks the high-alert threshold.")

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="sub-header">11 · Anomaly Score Map (PCA 2D)</div>', unsafe_allow_html=True)
        img = load_graph("11_anomaly_scatter_pca.png")
        if img: st.image(img, use_container_width=True)
        st.caption("Continuous anomaly scores projected onto PCA. Green = normal, Red = highly anomalous. Top 10 accounts labelled.")
    with col2:
        st.markdown('<div class="sub-header">12 · Decision Boundary: Freq vs Cash Ratio</div>', unsafe_allow_html=True)
        img = load_graph("12_isolation_forest_decision.png")
        if img: st.image(img, use_container_width=True)
        st.caption("X marks = anomalies. Clear cluster of suspicious accounts in the high-frequency + high-cash zone (top-right). Top 5 labelled.")

    st.markdown("---")

    # Anomaly accounts table
    st.markdown('<div class="sub-header">Flagged Anomaly Accounts (Isolation Forest)</div>', unsafe_allow_html=True)
    anoms = df[df["Is_Anomaly"] == 1].sort_values("Anomaly_Score", ascending=False)
    st.dataframe(
        anoms[["Account_ID", "Anomaly_Score", "Risk_Score", "Risk_Level",
               "Transaction_Frequency_Monthly", "Cash_Deposit_Ratio",
               "International_Transfer_Ratio"]].round(3).reset_index(drop=True),
        use_container_width=True,
        height=350,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: RISK SCORING
# ══════════════════════════════════════════════════════════════════════════════
elif page == "⚠️ Risk Scoring":
    st.markdown('<div class="section-header">⚠️ Risk Scoring Model</div>', unsafe_allow_html=True)

    # Formula card
    st.markdown("""
    <div style="background:rgba(0,212,255,0.07); border:1px solid #00d4ff33;
                border-radius:14px; padding:22px 28px; margin-bottom:20px;">
        <h3 style="color:#00d4ff; margin:0 0 12px;">📐 Risk Scoring Formula</h3>
        <code style="color:#e4e6f1; font-size:1rem; line-height:1.9;">
        Risk_Score = (<br>
        &nbsp;&nbsp;&nbsp;<b style="color:#00c9ff">0.30</b> × Anomaly_Score_norm &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(Isolation Forest)<br>
        &nbsp;+ <b style="color:#f7971e">0.25</b> × Cash_Deposit_Ratio_norm &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(raw feature)<br>
        &nbsp;+ <b style="color:#a8edea">0.20</b> × Intl_Transfer_Ratio_norm &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(raw feature)<br>
        &nbsp;+ <b style="color:#43e97b">0.15</b> × Transaction_Frequency_norm &nbsp;&nbsp;&nbsp;(raw feature)<br>
        &nbsp;+ <b style="color:#ff6b6b">0.10</b> × Unique_Counterparties_norm &nbsp;&nbsp;(raw feature)<br>
        ) × 100 &nbsp;+ <b style="color:#f9ca24">10</b> if KMeans_Suspicious == 1 &nbsp;← clipped to [0, 100]
        </code>
    </div>
    """, unsafe_allow_html=True)

    col_l, col_m, col_h = st.columns(3)
    col_l.markdown("""
    <div class="low-box">
        <h4 style="color:#43e97b; margin:0 0 6px;">🟢 LOW Risk  (< 50)</h4>
        <p>Routine monitoring only · No escalation required</p>
    </div>""", unsafe_allow_html=True)
    col_m.markdown("""
    <div class="medium-box">
        <h4 style="color:#f7971e; margin:0 0 6px;">🟡 MEDIUM Risk  (50 – 75)</h4>
        <p>Enhanced Due Diligence · Analyst review within 7 days</p>
    </div>""", unsafe_allow_html=True)
    col_h.markdown("""
    <div class="sar-box">
        <h4>🔴 HIGH Risk  (≥ 75)</h4>
        <p>Mandatory SAR filing · Escalate immediately · Freeze intl transfers</p>
    </div>""", unsafe_allow_html=True)

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="sub-header">13 · Risk Score Distribution</div>', unsafe_allow_html=True)
        img = load_graph("13_risk_score_distribution.png")
        if img: st.image(img, use_container_width=True)
    with col2:
        st.markdown('<div class="sub-header">15 · Risk Score per Cluster</div>', unsafe_allow_html=True)
        img = load_graph("15_cluster_vs_risk_boxplot.png")
        if img: st.image(img, use_container_width=True)

    st.markdown("---")

    st.markdown('<div class="sub-header">14 · Typology Flag Counts</div>', unsafe_allow_html=True)
    img = load_graph("14_typology_flag_counts.png")
    if img: st.image(img, use_container_width=True)

    st.markdown("---")

    # Interactive score slider
    st.markdown('<div class="sub-header">🔍 Risk Score Lookup — Filter Accounts</div>', unsafe_allow_html=True)
    min_score = st.slider("Minimum Risk Score", 0, 100, 50, step=5)
    filtered = df[df["Risk_Score"] >= min_score].sort_values("Risk_Score", ascending=False)
    st.write(f"**{len(filtered)} accounts** with Risk Score ≥ {min_score}")
    st.dataframe(
        filtered[["Account_ID", "Risk_Score", "Risk_Level", "Cluster",
                  "Is_Anomaly", "Smurfing_Flag", "RoundTripping_Flag", "Layering_Flag",
                  "Reason"]].reset_index(drop=True),
        use_container_width=True, height=350,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: CROSS-VALIDATION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔗 Cross-Validation":
    st.markdown('<div class="section-header">🔗 Cross-Validation — Method Agreement</div>', unsafe_allow_html=True)
    st.caption("Accounts flagged by multiple independent methods are most reliably suspicious. The Venn diagram shows overlap.")

    # Compute sets
    kmeans_ids = set(df.loc[df["Cluster_Risk"] >= 0.7, "Account_ID"])
    iso_ids    = set(df.loc[df["Is_Anomaly"] == 1, "Account_ID"])
    typol_ids  = set(df.loc[
        (df["Smurfing_Flag"] | df["RoundTripping_Flag"] | df["Layering_Flag"]) == 1,
        "Account_ID"
    ])
    all_three  = kmeans_ids & iso_ids & typol_ids
    any_two    = (kmeans_ids & iso_ids) | (kmeans_ids & typol_ids) | (iso_ids & typol_ids)
    all_two    = any_two - all_three

    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("KMeans High-Risk", len(kmeans_ids))
    col_b.metric("Iso Forest Anomalies", len(iso_ids))
    col_c.metric("Typology Flagged", len(typol_ids))
    col_d.metric("🎯 ALL 3 Methods", len(all_three))

    st.markdown("---")

    col_venn, col_table = st.columns([2, 3])
    with col_venn:
        st.markdown('<div class="sub-header">18 · Cross-Validation Venn Diagram</div>', unsafe_allow_html=True)
        img = load_graph("18_cross_validation_venn.png")
        if img: st.image(img, use_container_width=True)
        st.caption("Yellow centre = accounts flagged by ALL three independent methods — highest confidence alerts.")

    with col_table:
        st.markdown('<div class="sub-header">Accounts Flagged by All 3 Methods</div>', unsafe_allow_html=True)
        df_all3 = df[df["Account_ID"].isin(all_three)].sort_values("Risk_Score", ascending=False)
        if not df_all3.empty:
            st.dataframe(
                df_all3[["Account_ID", "Risk_Score", "Risk_Level",
                          "Smurfing_Flag", "RoundTripping_Flag", "Layering_Flag",
                          "Is_Anomaly", "Cluster"]].reset_index(drop=True),
                use_container_width=True, height=400,
            )
        else:
            st.info("No accounts flagged by all three methods simultaneously.")

        st.markdown("---")
        st.markdown('<div class="sub-header">Flagged by Any 2 Methods</div>', unsafe_allow_html=True)
        df_any2 = df[df["Account_ID"].isin(all_two)].sort_values("Risk_Score", ascending=False)
        st.write(f"**{len(df_any2)} accounts** flagged by at least 2 methods")
        st.dataframe(
            df_any2[["Account_ID", "Risk_Score", "Risk_Level"]].reset_index(drop=True),
            use_container_width=True, height=200,
        )


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: SAR REPORTS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📋 SAR Reports":
    st.markdown('<div class="section-header">📋 Suspicious Activity Reports (SAR)</div>', unsafe_allow_html=True)
    st.caption("SARs are legally required filings for accounts exceeding the HIGH risk threshold. Failure to file attracts regulatory penalties.")

    st.markdown("---")

    # Feature heatmap
    st.markdown('<div class="sub-header">17 · Feature Heatmap — Top 20 Risk Accounts</div>', unsafe_allow_html=True)
    img = load_graph("17_risk_heatmap_features.png")
    if img: st.image(img, use_container_width=True)
    st.caption("Normalised feature values for the 20 highest-risk accounts. Darker red = more extreme value. Patterns across rows reveal typology clusters.")

    st.markdown("---")
    st.markdown('<div class="sub-header">Top 10 Bar Chart</div>', unsafe_allow_html=True)
    img16 = load_graph("16_top10_accounts_bar.png")
    if img16: st.image(img16, use_container_width=True)

    st.markdown("---")
    st.markdown("### 📄 Individual SAR Reports — Top 5 Accounts")

    for rank, (_, row) in enumerate(top5.iterrows(), 1):
        icon = "🔴"
        box_class = "sar-box"

        reason_clean = str(row["Reason"]).replace("→", "->")
        with st.expander(f"{icon} SAR #{rank}  |  {row['Account_ID']}  |  Score: {row['Risk_Score']:.1f}/100  |  {row['Risk_Level']}", expanded=(rank == 1)):
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"""
                <div class="{box_class}">
                    <h4>Account: {row['Account_ID']}</h4>
                    <p><b>Risk Score:</b> {row['Risk_Score']:.1f} / 100</p>
                    <p><b>Risk Level:</b> {row['Risk_Level']}</p>
                    <p><b>KMeans Cluster:</b> {int(row['Cluster'])}</p>
                    <p><b>IF Anomaly:</b> {"YES ⚠️" if row['Is_Anomaly'] else "No"}</p>
                    <hr style="border-color:#ff4d4d33; margin:10px 0;">
                    <p><b>Smurfing:</b> {"🚨 DETECTED" if row['Smurfing_Flag'] else "✅ Clear"}</p>
                    <p><b>Round-Tripping:</b> {"🚨 DETECTED" if row['RoundTripping_Flag'] else "✅ Clear"}</p>
                    <p><b>Layering:</b> {"🚨 DETECTED" if row['Layering_Flag'] else "✅ Clear"}</p>
                </div>""", unsafe_allow_html=True)
            with c2:
                metrics = {
                    "Avg Txn Amount (INR)":      f"₹{row['Avg_Transaction_Amount_INR']:,.0f}",
                    "Txn Frequency / month":     str(int(row["Transaction_Frequency_Monthly"])),
                    "Unique Counterparties":     str(int(row["Unique_Counterparties_Monthly"])),
                    "Cash Deposit Ratio":        f"{row['Cash_Deposit_Ratio']:.2f}",
                    "Intl Transfer Ratio":       f"{row['International_Transfer_Ratio']:.2f}",
                    "Anomaly Score":             f"{row['Anomaly_Score']:.3f}",
                    "Cluster Risk Weight":       f"{row['Cluster_Risk']:.3f}",
                }
                for label, val in metrics.items():
                    st.markdown(
                        f"<div style='display:flex; justify-content:space-between; "
                        f"padding:5px 0; border-bottom:1px solid #2a2d45;'>"
                        f"<span style='color:#888db5;'>{label}</span>"
                        f"<span style='color:#e4e6f1; font-weight:600;'>{val}</span>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
            st.markdown("**Analyst Narrative:**")
            st.info(reason_clean)
            st.markdown("**Recommended Actions:**")
            st.markdown("""
            1. 🔍 Escalate for Enhanced Due Diligence (EDD) immediately
            2. 🔒 Freeze international transfers pending investigation
            3. 📁 Conduct full 90-day transaction-level audit
            4. 📨 File SAR with FIU-IND/FinCEN within 30-day legal deadline
            """)

    st.markdown("---")
    # Download results CSV
    csv_buffer = io.StringIO()
    df.sort_values("Risk_Score", ascending=False).to_csv(csv_buffer, index=False)
    st.download_button(
        "⬇️  Download Full Results CSV",
        data=csv_buffer.getvalue(),
        file_name="AML_Results_AllAccounts.csv",
        mime="text/csv",
    )


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: PRESENTATION NOTES
# ══════════════════════════════════════════════════════════════════════════════
elif page == "❓ Presentation Notes":
    st.markdown('<div class="section-header">❓ Presentation Notes — Key Questions</div>', unsafe_allow_html=True)

    st.markdown("""
    <div style="background:rgba(0,212,255,0.06); border:1px solid #00d4ff22;
                border-radius:14px; padding:22px; margin-bottom:20px;">
        <h3 style="color:#00d4ff;">Q1: Why is unsupervised learning suitable for AML?</h3>
        <p style="color:#c8cce8; line-height:1.8;">
        Labeled fraud data in AML is <b>extremely scarce, expensive to obtain, and often biased</b>
        toward previously known schemes. New money-laundering patterns evolve faster than labeling
        can keep up. Unsupervised methods such as <b>K-Means</b> and <b>Isolation Forest</b> detect
        anomalies purely from the statistical structure of the data — no prior examples needed.
        This makes them ideal for discovering <i>novel</i> typologies that supervised classifiers
        would miss. Additionally, the class imbalance problem is severe (< 1% of accounts may be
        truly suspicious), making unsupervised approaches more robust than oversampled classifiers.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="background:rgba(247,151,30,0.07); border:1px solid #f7971e33;
                border-radius:14px; padding:22px; margin-bottom:20px;">
        <h3 style="color:#f7971e;">Q2: What is the legal obligation when suspicious activity is detected?</h3>
        <p style="color:#c8cce8; line-height:1.8;">
        Under the <b>Prevention of Money Laundering Act (PMLA)</b> in India and the 
        <b>Bank Secrecy Act (BSA)</b> in the US, a bank <i>must</i>:
        </p>
        <ol style="color:#c8cce8; line-height:2.0;">
            <li>File a <b>Suspicious Activity Report (SAR)</b> with the national Financial Intelligence Unit
                (FIU-IND / FinCEN) within <b>7 days</b> of detection (30 days if investigation in progress).</li>
            <li><b>Not tip off</b> the suspected customer (tipping-off offence).</li>
            <li>Retain all transaction records for a minimum of <b>5 years</b>.</li>
            <li>Subject the account to <b>Enhanced Due Diligence (EDD)</b>.</li>
            <li>Consider <b>de-risking</b> (account closure) if risk cannot be mitigated.</li>
        </ol>
        <p style="color:#c8cce8;">Failure to file attracts heavy fines, license revocation, and personal criminal liability for compliance officers.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="background:rgba(67,233,123,0.06); border:1px solid #43e97b33;
                border-radius:14px; padding:22px; margin-bottom:20px;">
        <h3 style="color:#43e97b;">Q3: How would you reduce false positives in production?</h3>
        <p style="color:#c8cce8; line-height:1.8; margin-bottom:12px;">
        False positives (innocent accounts flagged as suspicious) impose huge operational costs 
        and harm customer relationships. Strategies to reduce them:
        </p>
        <table style="width:100%; border-collapse:collapse; color:#c8cce8;">
            <tr style="border-bottom:1px solid #2a2d45;">
                <td style="padding:8px; font-weight:700; color:#43e97b; width:30%;">Method</td>
                <td style="padding:8px;">Rationale</td>
            </tr>
            <tr style="border-bottom:1px solid #1a1d2e;">
                <td style="padding:8px; color:#43e97b;">Ensemble agreement</td>
                <td style="padding:8px;">Only escalate to HIGH if flagged by ALL 3 methods (KMeans + IF + Typology)</td>
            </tr>
            <tr style="border-bottom:1px solid #1a1d2e;">
                <td style="padding:8px; color:#43e97b;">Peer-group comparison</td>
                <td style="padding:8px;">Compare account to industry/segment peers, not the whole population</td>
            </tr>
            <tr style="border-bottom:1px solid #1a1d2e;">
                <td style="padding:8px; color:#43e97b;">Contamination tuning</td>
                <td style="padding:8px;">Adjust Isolation Forest contamination % using analyst feedback loop</td>
            </tr>
            <tr style="border-bottom:1px solid #1a1d2e;">
                <td style="padding:8px; color:#43e97b;">Temporal analysis</td>
                <td style="padding:8px;">Flag only accounts with sudden behavioural change vs. historical baseline</td>
            </tr>
            <tr style="border-bottom:1px solid #1a1d2e;">
                <td style="padding:8px; color:#43e97b;">Graph features</td>
                <td style="padding:8px;">Add transaction network centrality — real launderers have unusual network roles</td>
            </tr>
            <tr>
                <td style="padding:8px; color:#43e97b;">Active learning</td>
                <td style="padding:8px;">Analyst decisions feed back into model retraining each month</td>
            </tr>
        </table>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-header">📐 Complete Risk Formula Reference</div>', unsafe_allow_html=True)
    st.code("""
# All raw features are min-max normalised to [0,1] before weighting
Risk_Score = (
      0.30 * Anomaly_Score_norm          # Isolation Forest (normalised to [0,1])
    + 0.25 * Cash_Deposit_Ratio_norm     # min-max normalised
    + 0.20 * Intl_Transfer_Ratio_norm    # min-max normalised
    + 0.15 * Transaction_Frequency_norm  # min-max normalised
    + 0.10 * Unique_Counterparties_norm  # min-max normalised
) * 100
+ 10 * KMeans_Suspicious                # bonus: account in highest-risk cluster
[clipped to 0-100]

# KMeans_Suspicious = 1  if account belongs to highest-risk cluster (by Cluster_Risk)
# Typology Rules (data-driven percentile thresholds):
Smurfing_Flag      = (Freq >= Q75) AND (Amount <= Q25) AND (CashRatio >= Q75)
RoundTripping_Flag = (Freq >= Q75) AND (UniqueCounterparties <= max(2, Q25))
Layering_Flag      = (UniqueCounterparties >= Q75) AND (IntlTransfer >= Q75)
    """, language="python")
