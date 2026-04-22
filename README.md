# 🏦 Anti-Money Laundering (AML) Anomaly Detection System

## 📖 Project Overview
GlobalBank's compliance team requires a robust system to detect suspicious transaction patterns indicative of money laundering. Because labeled data for financial crimes is often scarce and adversarial behaviors constantly evolve, this project utilizes **Unsupervised Machine Learning** combined with **Rule-based Typology Detection** to flag anomalous accounts and generate automatic Suspicious Activity Reports (SAR).

## 🎯 What We Had To Do (Objectives)
The goal was to build a complete pipeline taking account-level transaction features and outputting highly explainable, risk-scored alerts. The core requirements were:
1. **Unsupervised ML:** Use K-Means clustering (to group normal vs. abnormal behavioral profiles) and Isolation Forest (to detect statistical outliers).
2. **AML Pattern Detection:** Explicitly flag known financial crime typologies:
   - **Smurfing (Structuring):** Multiple small deposits to avoid reporting thresholds.
   - **Round-Tripping:** Funds flowing out and immediately back in, often between a tiny closed ring of counterparties.
   - **Layering:** Rapid movement of money across borders and multiple accounts to obscure origins.
3. **Composite Risk Scoring:** Combine the unsupervised anomaly scores with the explicit typology flags to output a normalized `0–100` Risk Score.
4. **Explainability & SAR Generation:** Translate the math into plain-English "Reasons" and auto-generate compliance-ready Suspicious Activity Reports for top-tier alerts.

## 📊 Dataset
The system operates on synthetic account-aggregated transaction data (`P9_AML_Accounts.csv`). 
**Key Features include:**
* `Avg_Transaction_Amount_INR`
* `Transaction_Frequency_Monthly`
* `Unique_Counterparties_Monthly`
* `Cash_Deposit_Ratio`
* `International_Transfer_Ratio`

## ⚙️ How It Works (The Pipeline)

### 1. Data Preprocessing & EDA
The raw data is first loaded and numeric features are extracted. A `StandardScaler` normalizes the data so that large currency values do not overwhelm ratio-based features during distance calculations. Exploratory Data Analysis (EDA) charts are generated to understand feature distributions.

### 2. Machine Learning Core
* **K-Means Clustering:** Segments the accounts into 4 distinct behavioral clusters. A baseline "Cluster Risk" is calculated by checking the average cash and international transfer densities of each cluster.
* **Isolation Forest:** Evaluates the standardized dataset to find statistical outliers. Accounts residing in sparse regions of the multi-dimensional feature space are granted high "Anomaly Scores".

### 3. Typology Detection (Heuristics)
To complement the Unsupervised ML, quantile-based thresholds evaluate explicitly for:
* **Smurfing Flag:** Triggered if an account has *High Frequency*, *Low Amounts*, and *High Cash Ratios*.
* **Round-Tripping Flag:** Triggered if an account has *High Frequency* but interacts with an abnormally *Low number of unique counterparties*.
* **Layering Flag:** Triggered if an account has a *High number of unique counterparties* paired with *High International Transfers*.

### 4. Explainability & Risk Scoring
* **Reason Generation:** A custom function maps triggered flags and ML scores into a human-readable sentence (e.g., *"Many counterparties + high international transfers → possible layering network"*).
* **Final Risk Score (0-100):** 
  - Base Risk = Weighted blend of Isolation Forest score + Cluster Risk.
  - Pattern Penalty = Flat additive penalties for triggering any explicit AML flags.
  - The final sum is clipped to a strict `0 - 100` scale. Accounts are then binned into `Low`, `Medium`, or `High` Risk Levels.

### 5. Output & SAR Drafting
The system filters the highest-risk accounts and automatically generates a text-based **Suspicious Activity Report (SAR)**. This report lists the triggered flags, account metrics, the analyst narrative (Reason), and standard recommended actions (Enhanced Due Diligence, reporting to FIUs). Visualizations of the final risk distributions are rendered via `seaborn`.

## 🚀 How to Run

1. Ensure you have Python installed.
2. Install the required dependencies:
   ```bash
   pip install pandas scikit-learn matplotlib seaborn
   ```
3. Open and run `main.ipynb` sequentially from top to bottom.
4. View the generated SAR report, the Top 10 High-Risk DataFrame, and the output visualizations at the end of the notebook.