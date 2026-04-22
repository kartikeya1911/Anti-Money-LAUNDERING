# 🏦 Anti-Money Laundering (AML) Pattern Detection System

> **Project 9 · Regulatory Compliance / Risk Management**  
> GlobalBank Compliance Desk · Unsupervised ML Pipeline · April 2026

---

## 📖 Problem Statement

GlobalBank's compliance team needs to detect suspicious transaction patterns that may indicate money laundering. Common patterns include:

| Pattern | Description |
|---------|-------------|
| 💵 **Smurfing** | Multiple small deposits split to evade reporting thresholds |
| 🔄 **Round-Tripping** | Money cycling between a tight ring of counterparties |
| 🌐 **Layering** | Rapid account-to-account + cross-border transfers to obscure origin |

This project builds a fully **unsupervised** system to flag anomalous accounts, generate Suspicious Activity Reports (SARs), and visualise findings in an interactive Streamlit dashboard.

---

## 🗂️ Project Structure

```
project/
├── app.py                    # Streamlit dashboard (8 pages)
├── main.py                   # CLI pipeline (10-step orchestration)
├── P9_AML_Accounts.csv       # Synthetic dataset (200 accounts, 30 days)
├── AML_Risk_Report.pdf       # Auto-generated 4-page PDF report
├── graphs/                   # 18 saved PNG charts
│   ├── 01_feature_distributions.png
│   ├── 02_correlation_heatmap.png
│   ├── 03_boxplots_features.png
│   ├── 04_pairplot.png
│   ├── 05_elbow_curve.png
│   ├── 06_silhouette_scores.png
│   ├── 07_cluster_scatter_pca.png
│   ├── 08_cluster_profiles_radar.png
│   ├── 09_cluster_heatmap.png
│   ├── 10_anomaly_score_distribution.png
│   ├── 11_anomaly_scatter_pca.png
│   ├── 12_isolation_forest_decision.png
│   ├── 13_risk_score_distribution.png
│   ├── 14_typology_flag_counts.png
│   ├── 15_cluster_vs_risk_boxplot.png
│   ├── 16_top10_accounts_bar.png
│   ├── 17_risk_heatmap_features.png
│   └── 18_cross_validation_venn.png
└── src/
    ├── __init__.py
    ├── data_loader.py        # Load CSV + StandardScaler
    ├── models.py             # K-Means + Isolation Forest
    ├── typologies.py         # Smurfing / Round-Tripping / Layering rules
    ├── risk_scoring.py       # Composite Risk Score (0–100)
    ├── reporting.py          # SAR text reports + cross-validation table
    ├── pdf_generator.py      # 4-page FPDF report
    └── visualizations.py     # 18 dark-theme matplotlib/seaborn charts
```

---

## 📊 Dataset

**File:** `P9_AML_Accounts.csv`  
**Size:** 200 accounts · 5 features · 30-day aggregation window

| Feature | Description |
|---------|-------------|
| `Avg_Transaction_Amount_INR` | Average transaction value in INR |
| `Transaction_Frequency_Monthly` | Number of transactions per month |
| `Unique_Counterparties_Monthly` | Distinct parties transacted with |
| `Cash_Deposit_Ratio` | Proportion of transactions that are cash deposits |
| `International_Transfer_Ratio` | Proportion of transfers going abroad |

---

## ⚙️ Pipeline (10 Steps)

```
Step 1   Load & validate dataset → StandardScaler
Step 2   EDA charts: feature distributions + correlation heatmap
Step 3   K-Means clustering (k=4, silhouette-validated)
Step 4   Isolation Forest anomaly detection (contamination=5 %)
         └─ EDA phase 2: boxplots + pairplot (post-model)
Step 5   AML typology rule flags (Smurfing / Round-Tripping / Layering)
Step 6   Composite Risk Score 0–100
Step 7   Cross-validation: KMeans ∩ Isolation Forest ∩ Typology Rules
Step 8   SAR generation for Top 5 accounts
Step 9   Save all 18 charts → graphs/
Step 10  Export 4-page PDF report → AML_Risk_Report.pdf
```

---

## 🤖 ML Models

### K-Means Clustering
- Groups the 200 accounts into **4 behavioural segments**
- Optimal `k` selected via **Elbow Curve** + **Silhouette Score**
- Each cluster assigned a `Cluster_Risk` weight based on average cash deposit and international transfer densities

### Isolation Forest
- Detects statistical outliers by recursively partitioning the feature space
- Accounts isolated in fewer splits receive higher **Anomaly Scores**
- `contamination=0.05` → flags ~10 accounts as anomalous
- Raw decision scores normalised to `[0, 1]`

---

## 📐 Risk Scoring Formula

```python
# All raw features are min-max normalised to [0, 1] before weighting
Risk_Score = (
      0.30 * Anomaly_Score_norm          # Isolation Forest (normalised output)
    + 0.25 * Cash_Deposit_Ratio_norm     # min-max scaled
    + 0.20 * Intl_Transfer_Ratio_norm    # min-max scaled
    + 0.15 * Transaction_Frequency_norm  # min-max scaled
    + 0.10 * Unique_Counterparties_norm  # min-max scaled
) * 100
+ 10 * KMeans_Suspicious                # +10 bonus if in highest-risk cluster
[clipped to 0–100]

# KMeans_Suspicious = 1  if account belongs to the single highest-risk cluster
```

| Score Range | Risk Level | Action |
|-------------|-----------|--------|
| ≥ 75 | 🔴 HIGH | Mandatory SAR filing · Freeze intl transfers |
| 50 – 75 | 🟡 MEDIUM | Enhanced Due Diligence within 7 days |
| < 50 | 🟢 LOW | Routine monitoring only |

---

## 🚩 AML Typology Rules

All thresholds are **data-driven** (derived from dataset percentiles — no hardcoded values):

```python
# Smurfing: break large sums into many small cash deposits
Smurfing_Flag = (Freq >= Q75) AND (Amount <= Q25) AND (CashRatio >= Q75)

# Round-Tripping: money cycles between a tiny set of accounts
RoundTripping_Flag = (Freq >= Q75) AND (UniqueCounterparties <= max(2, Q25))

# Layering: rapidly move money across many accounts + borders
Layering_Flag = (UniqueCounterparties >= Q75) AND (IntlTransferRatio >= Q75)
```

---

## 📈 Results Summary

| Metric | Value |
|--------|-------|
| Total Accounts | 200 |
| 🔴 HIGH Risk Accounts | 20 (10 %) |
| 🟢 LOW Risk Accounts | 180 (90 %) |
| Isolation Forest Anomalies | 10 |
| Smurfing Flags | 15 |
| Round-Tripping Flags | 2 |
| Layering Flags | 27 |

### Top 5 Flagged Accounts

| Rank | Account | Score | Patterns |
|------|---------|-------|----------|
| 1 | ACC2177 | 100.0 | Smurfing + Layering + IF Anomaly |
| 2 | ACC2068 | 100.0 | Smurfing + Layering + IF Anomaly |
| 3 | ACC2069 | 100.0 | Smurfing + Layering + IF Anomaly |
| 4 | ACC2030 | 100.0 | Smurfing + Layering + IF Anomaly |
| 5 | ACC2165 | 100.0 | Smurfing + Layering + IF Anomaly |

---

## 🖥️ Streamlit Dashboard

An interactive dashboard (`app.py`) with **8 navigation pages**:

| Page | Contents |
|------|---------|
| 🏠 Overview | Hero banner · 7 KPI metrics · Top-10 table · Typology badges |
| 📊 EDA Analysis | Raw data preview · Descriptive stats · 4 EDA charts with insights |
| 🔵 K-Means Clustering | Cluster stats · Elbow · Silhouette · PCA map · Radar · Heatmap |
| 🌲 Isolation Forest | Score distribution · PCA anomaly map · Decision boundary · Flagged table |
| ⚠️ Risk Scoring | Formula card · Risk level bands · Interactive score filter slider |
| 🔗 Cross-Validation | Venn diagram · All-3-methods table · Any-2-methods table |
| 📋 SAR Reports | Feature heatmap · Expandable SAR cards for Top 5 · CSV download |
| ❓ Presentation Notes | Q&A for 3 graded questions · Full formula reference |

---

## 🚀 How to Run

### Prerequisites

```bash
pip install pandas scikit-learn matplotlib seaborn fpdf streamlit pillow
```

### Option A — Streamlit Dashboard (Recommended)

```powershell
# Windows (set UTF-8 encoding to avoid console issues)
$env:PYTHONIOENCODING="utf-8"
streamlit run app.py
```

Then open **http://localhost:8501** in your browser.

### Option B — CLI Pipeline (saves graphs + PDF)

```powershell
$env:PYTHONIOENCODING="utf-8"
python main.py
```

Outputs:
- `graphs/` — 18 PNG charts (dark theme)
- `AML_Risk_Report.pdf` — 4-page professional PDF

---

## 🎓 Presentation Q&A

**Q1: Why is unsupervised learning suitable for AML?**  
Labeled fraud data is scarce, imbalanced, and biased toward _known_ schemes. Unsupervised methods (K-Means, Isolation Forest) find novel anomalies without requiring prior fraud examples — critical for detecting emerging typologies.

**Q2: What is the legal obligation when suspicious activity is detected?**  
Under India's **PMLA** and the US **Bank Secrecy Act**, banks must file a **Suspicious Activity Report (SAR)** with the national FIU (FIU-IND / FinCEN) within **30 days** of detection. Tipping off the customer is a criminal offence. Records must be retained for **5 years**.

**Q3: How would you reduce false positives in production?**  
1. Require flagging by **all three methods** before escalating to HIGH
2. Use **peer-group segmentation** (compare within industry/size tier)
3. Tune `contamination` via analyst feedback (**active learning**)
4. Add **temporal trend analysis** — detect sudden behavioural change vs. historical baseline
5. Incorporate **transaction graph features** (network centrality)

---

## 🛠️ Tech Stack

| Library | Role |
|---------|------|
| `pandas` | Data loading, manipulation |
| `scikit-learn` | StandardScaler · KMeans · IsolationForest · PCA · silhouette_score |
| `matplotlib` / `seaborn` | 18 dark-theme charts |
| `fpdf` | 4-page PDF report |
| `streamlit` | Interactive dashboard |
| `pillow` | Image loading in Streamlit |

---

## 👥 Authors

**GlobalBank Compliance Desk — Data Science Team**  
Project 9 · Regulatory Compliance / Risk Management