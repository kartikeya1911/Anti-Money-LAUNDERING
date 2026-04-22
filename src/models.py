import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score

FEATURE_COLS = [
    'Avg_Transaction_Amount_INR',
    'Transaction_Frequency_Monthly',
    'Unique_Counterparties_Monthly',
    'Cash_Deposit_Ratio',
    'International_Transfer_Ratio',
]

# Weights used for cluster risk scoring
DRIVER_WEIGHTS = {
    'Cash_Deposit_Ratio':             0.35,
    'International_Transfer_Ratio':   0.35,
    'Transaction_Frequency_Monthly':  0.20,
    'Unique_Counterparties_Monthly':  0.10,
}

def _optimal_k(X_scaled, k_range=range(2, 9)):
    """Finds optimal K using silhouette score (informational)."""
    best_k, best_score = 4, -1
    for k in k_range:
        labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        if score > best_score:
            best_score, best_k = score, k
    print(f"   Silhouette analysis → optimal k = {best_k} (score={best_score:.3f})")
    return best_k

def apply_unsupervised_models(df, X_scaled, feature_cols=None, n_clusters=4, contamination=0.05):
    """
    Applies KMeans and Isolation Forest to detect behavioral clusters and statistical anomalies.
    Adds columns:
        Cluster, Cluster_Risk, KMeans_Suspicious  (1 = account in highest-risk cluster)
        Anomaly_Label, Is_Anomaly, Anomaly_Score
    Returns updated dataframe, kmeans model, iso model.
    """
    if feature_cols is None:
        feature_cols = FEATURE_COLS

    # ── 1. KMeans Clustering ──────────────────────────────────────────────────
    print(f"   Running KMeans with n_clusters={n_clusters} ...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(X_scaled)
    df['Cluster'] = df['Cluster'].astype(int)

    # Compute cluster risk based on weighted average of risk-driving features
    cluster_profile = df.groupby('Cluster')[feature_cols].mean(numeric_only=True).copy()
    weighted_risk = sum(
        cluster_profile[col] * wt
        for col, wt in DRIVER_WEIGHTS.items()
        if col in cluster_profile.columns
    )
    min_r, max_r = weighted_risk.min(), weighted_risk.max()
    cluster_risk_norm = (weighted_risk - min_r) / (max_r - min_r + 1e-9)
    df['Cluster_Risk'] = df['Cluster'].map(cluster_risk_norm.to_dict())

    # KMeans_Suspicious = 1 if the account belongs to the single highest-risk cluster
    top_risk_cluster = int(cluster_risk_norm.idxmax())
    df['KMeans_Suspicious'] = (df['Cluster'] == top_risk_cluster).astype(int)
    n_km_susp = df['KMeans_Suspicious'].sum()
    print(f"   KMeans highest-risk cluster = {top_risk_cluster}  ({n_km_susp} accounts flagged).")

    # ── 2. Isolation Forest (Anomaly Detection) ───────────────────────────────
    print(f"   Running Isolation Forest (contamination={contamination}) ...")
    iso = IsolationForest(contamination=contamination, random_state=42, n_estimators=200)
    df['Anomaly_Label'] = iso.fit_predict(X_scaled)
    df['Is_Anomaly']    = (df['Anomaly_Label'] == -1).astype(int)

    # Normalize anomaly score to [0, 1]  (higher = more anomalous)
    raw_anomaly = -iso.decision_function(X_scaled)
    df['Anomaly_Score'] = (raw_anomaly - raw_anomaly.min()) / (raw_anomaly.max() - raw_anomaly.min() + 1e-9)

    n_flagged = df['Is_Anomaly'].sum()
    print(f"   Isolation Forest flagged {n_flagged} accounts as anomalous ({n_flagged/len(df)*100:.1f}%).")

    return df, kmeans, iso