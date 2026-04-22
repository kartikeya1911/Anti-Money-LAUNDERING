"""
risk_scoring.py — Composite Risk Score Formula
================================================

Risk_Score = (  0.30 × Anomaly_Score_norm           (Isolation Forest)
              + 0.25 × Cash_Deposit_Ratio_norm
              + 0.20 × International_Transfer_Ratio_norm
              + 0.15 × Transaction_Frequency_norm
              + 0.10 × Unique_Counterparties_norm    ) × 100
              + 10 bonus points if KMeans_Suspicious == 1

All raw features are min-max normalised to [0, 1] before weighting.
Anomaly_Score is already normalised by models.py.
Final score is clipped to [0, 100].
"""

import numpy as np


def _minmax(series):
    """Min-max normalise a pandas Series to [0, 1]."""
    mn, mx = series.min(), series.max()
    return (series - mn) / (mx - mn + 1e-9)


def generate_reason(row):
    """
    Generates a human-readable explanation based on triggered AML typologies
    and statistical anomaly scores.
    """
    reasons = []
    if row['Smurfing_Flag']:
        reasons.append("High frequency + low amounts + high cash ratio -> possible SMURFING")
    if row['RoundTripping_Flag']:
        reasons.append("High frequency with very few counterparties -> possible ROUND-TRIPPING")
    if row['Layering_Flag']:
        reasons.append("Many counterparties + high international transfers -> possible LAYERING")
    if row['KMeans_Suspicious']:
        reasons.append("Account belongs to highest-risk KMeans cluster")
    if row['Anomaly_Score'] > 0.80:
        reasons.append("Statistically anomalous vs. peer group (Isolation Forest)")
    if not reasons:
        return "Behavior consistent with peer norms -- no specific typology triggered"
    return " | ".join(reasons)


def assign_risk_level(score):
    """Classifies Risk Level from numerical Risk Score (0-100)."""
    if score >= 75:
        return "HIGH"
    elif score >= 50:
        return "MEDIUM"
    return "LOW"


def calculate_risk_scores(df):
    """
    Implements the composite AML Risk Scoring formula:

        Risk_Score = (
              0.30 * Anomaly_Score_norm
            + 0.25 * Cash_Deposit_Ratio_norm
            + 0.20 * International_Transfer_Ratio_norm
            + 0.15 * Transaction_Frequency_norm
            + 0.10 * Unique_Counterparties_norm
        ) * 100
        + 10 * KMeans_Suspicious          <- bonus for highest-risk cluster
        clipped to [0, 100]

    Returns df with added columns: Reason, Risk_Score, Risk_Level
    """
    # -- Normalise raw features to [0, 1] -----------------------------------
    # Anomaly_Score is already normalised by models.py; others need min-max
    cash_norm   = _minmax(df['Cash_Deposit_Ratio'])
    intl_norm   = _minmax(df['International_Transfer_Ratio'])
    freq_norm   = _minmax(df['Transaction_Frequency_Monthly'])
    cp_norm     = _minmax(df['Unique_Counterparties_Monthly'])
    anom_norm   = df['Anomaly_Score']          # already [0, 1]

    # -- Weighted base score (0-100) ----------------------------------------
    base = (
        0.30 * anom_norm
      + 0.25 * cash_norm
      + 0.20 * intl_norm
      + 0.15 * freq_norm
      + 0.10 * cp_norm
    ) * 100

    # -- KMeans bonus: +10 pts if account is in highest-risk cluster --------
    km_bonus = 10 * df['KMeans_Suspicious']

    # -- Final score (clip to [0, 100]) -------------------------------------
    df['Risk_Score'] = (base + km_bonus).clip(0, 100).round(2)
    df['Risk_Level'] = df['Risk_Score'].apply(assign_risk_level)

    # -- Reason generation --------------------------------------------------
    df['Reason'] = df.apply(generate_reason, axis=1)

    high   = (df['Risk_Score'] >= 75).sum()
    medium = ((df['Risk_Score'] >= 50) & (df['Risk_Score'] < 75)).sum()
    low    = (df['Risk_Score'] < 50).sum()
    print(f"   High Risk: {high} | Medium Risk: {medium} | Low Risk: {low}")
    print(f"   Risk scoring complete.")

    return df