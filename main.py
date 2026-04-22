"""
main.py  —  Anti-Money Laundering (AML) Pattern Detection Pipeline
================================================================
GlobalBank Compliance Team | Unsupervised ML System

Pipeline Steps:
  1. Load & preprocess dataset
  2. Exploratory Data Analysis (EDA) + charts saved to graphs/
  3. K-Means clustering → behavioural segments
  4. Isolation Forest   → statistical anomaly detection
  5. AML Typology Rules → Smurfing / Round-Tripping / Layering
  6. Risk Scoring       → composite 0-100 score
  7. Cross-validation   → accounts flagged by both methods
  8. SAR Report         → top 5 accounts
  9. PDF Export
 10. Save all graphs   → graphs/ folder (18 charts)
"""

import warnings
import os

warnings.filterwarnings('ignore')

from src.data_loader   import load_and_preprocess
from src.models        import apply_unsupervised_models
from src.typologies    import apply_typology_rules
from src.risk_scoring  import calculate_risk_scores
from src.reporting     import generate_sar_report
from src.pdf_generator import export_pdf_report
from src.visualizations import (
    plot_eda, plot_eda_post, plot_clustering, plot_anomaly, plot_aml_results
)

DATASET_PATH  = 'P9_AML_Accounts.csv'
PDF_OUTPUT    = 'AML_Risk_Report.pdf'
N_CLUSTERS    = 4
CONTAMINATION = 0.05   # % of dataset expected to be anomalous

FEATURE_COLS = [
    'Avg_Transaction_Amount_INR',
    'Transaction_Frequency_Monthly',
    'Unique_Counterparties_Monthly',
    'Cash_Deposit_Ratio',
    'International_Transfer_Ratio',
]


def separator(text):
    w = 65
    print(f"\n{'-' * w}")
    print(f"  STEP {text}")
    print(f"{'-' * w}")


def main():
    # -- 1. Load & Preprocess --
    separator("1 | Loading and Pre-processing Data")
    df, features, X_scaled, scaler = load_and_preprocess(DATASET_PATH)

    # -- 2. EDA Charts --
    separator("2 | Exploratory Data Analysis (EDA)")
    print("  Generating and saving EDA charts to graphs/ ...")
    plot_eda(df)   # saves charts 01-04

    # -- 3 & 4. K-Means + Isolation Forest --
    separator("3 & 4 | K-Means Clustering + Isolation Forest")
    df, kmeans_model, iso_model = apply_unsupervised_models(
        df, X_scaled, feature_cols=FEATURE_COLS,
        n_clusters=N_CLUSTERS, contamination=CONTAMINATION
    )
    plot_clustering(df, X_scaled)   # saves charts 05-09
    plot_anomaly(df, X_scaled)      # saves charts 10-12
    plot_eda_post(df)               # saves charts 03-04 (boxplots + pairplot)

    # -- 5. AML Typology Rules --
    separator("5 | AML Typology Rules (Smurfing / Round-Tripping / Layering)")
    df = apply_typology_rules(df)

    # -- 6. Risk Scoring --
    separator("6 | Risk Scoring Formula")
    df = calculate_risk_scores(df)

    # -- 7-8. Cross-Validation + SAR Report --
    separator("7 & 8 | Cross-Validation + SAR Report")
    top_alerts = generate_sar_report(df)

    # -- Save final output charts --
    separator("9 | Saving Risk & Output Charts")
    plot_aml_results(df)   # saves charts 13-18

    # -- 10. PDF Export --
    separator("10 | Exporting PDF Report")
    export_pdf_report(top_alerts, df=df, output_filename=PDF_OUTPUT)

    # -- Summary --
    separator("COMPLETE - Summary")
    print(f"  Dataset          : {DATASET_PATH}  ({len(df)} accounts)")
    print(f"  High Risk        : {(df['Risk_Score'] >= 75).sum()} accounts")
    print(f"  Medium Risk      : {((df['Risk_Score'] >= 50) & (df['Risk_Score'] < 75)).sum()} accounts")
    print(f"  Low Risk         : {(df['Risk_Score'] < 50).sum()} accounts")
    print(f"  Graphs saved to  : {os.path.abspath('graphs/')}")
    print(f"  PDF saved to     : {os.path.abspath(PDF_OUTPUT)}")
    print(f"\n  ✅ Pipeline complete.")


if __name__ == '__main__':
    main()