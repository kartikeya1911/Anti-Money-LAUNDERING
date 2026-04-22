"""
visualizations.py
Saves every important AML analysis chart to the `graphs/` directory.
Charts produced:
  EDA:
    01_feature_distributions.png
    02_correlation_heatmap.png
    03_boxplots_features.png
    04_pairplot.png
  Clustering:
    05_elbow_curve.png
    06_silhouette_scores.png
    07_cluster_scatter_pca.png
    08_cluster_profiles_radar.png
    09_cluster_heatmap.png
  Anomaly:
    10_anomaly_score_distribution.png
    11_anomaly_scatter_pca.png
    12_isolation_forest_decision.png
  Risk:
    13_risk_score_distribution.png
    14_typology_flag_counts.png
    15_cluster_vs_risk_boxplot.png
    16_top10_accounts_bar.png
    17_risk_heatmap_features.png
  Cross-validation:
    18_cross_validation_venn.png
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')          # non-interactive backend → save to files
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from matplotlib.patches import FancyBboxPatch
import matplotlib.cm as cm

# ── Style ─────────────────────────────────────────────────────────────────────
sns.set_theme(style="darkgrid", palette="viridis")
plt.rcParams.update({
    'figure.facecolor': '#0f1117',
    'axes.facecolor':   '#1a1d27',
    'axes.edgecolor':   '#444',
    'axes.labelcolor':  '#e0e0e0',
    'xtick.color':      '#b0b0b0',
    'ytick.color':      '#b0b0b0',
    'text.color':       '#e0e0e0',
    'grid.color':       '#2a2d3a',
    'grid.linewidth':   0.6,
    'font.family':      'DejaVu Sans',
    'font.size':        11,
})

GRAPHS_DIR = 'graphs'
PALETTE    = ['#00c9ff', '#f7971e', '#ff6b6b', '#a8edea', '#d4fc79',
               '#667eea', '#f093fb', '#4facfe', '#43e97b', '#fa709a']


def _save(fig, filename):
    """Save figure to graphs/ and close it."""
    path = os.path.join(GRAPHS_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"   Saved → {path}")


def _title(ax, text, fontsize=13):
    ax.set_title(text, fontsize=fontsize, fontweight='bold',
                 color='#ffffff', pad=10)


def _suptitle(fig, text, fontsize=15):
    fig.suptitle(text, fontsize=fontsize, fontweight='bold',
                 color='#ffffff', y=1.01)


# ══════════════════════════════════════════════════════════════════════════════
#  EDA CHARTS
# ══════════════════════════════════════════════════════════════════════════════

def plot_feature_distributions(df):
    """01 — Histogram KDE for every numeric feature."""
    features = [
        'Avg_Transaction_Amount_INR', 'Transaction_Frequency_Monthly',
        'Unique_Counterparties_Monthly', 'Cash_Deposit_Ratio',
        'International_Transfer_Ratio',
    ]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    colors = PALETTE

    for i, feat in enumerate(features):
        ax = axes[i]
        data = df[feat].dropna()
        ax.hist(data, bins=25, color=colors[i % len(colors)],
                alpha=0.55, edgecolor='white', linewidth=0.4, density=True)
        data.plot.kde(ax=ax, color='white', linewidth=2)
        _title(ax, feat.replace('_', ' '))
        ax.set_xlabel('Value', color='#b0b0b0')
        ax.set_ylabel('Density', color='#b0b0b0')
        # Annotate mean
        ax.axvline(data.mean(), color='#ff6b6b', linestyle='--', linewidth=1.5,
                   label=f'Mean={data.mean():.1f}')
        ax.legend(fontsize=9)

    # Hide unused subplot
    axes[-1].set_visible(False)
    _suptitle(fig, '📊 AML Dataset — Feature Distributions (EDA)')
    plt.tight_layout()
    _save(fig, '01_feature_distributions.png')


def plot_correlation_heatmap(df):
    """02 — Correlation matrix heatmap."""
    feat_cols = [
        'Avg_Transaction_Amount_INR', 'Transaction_Frequency_Monthly',
        'Unique_Counterparties_Monthly', 'Cash_Deposit_Ratio',
        'International_Transfer_Ratio',
    ]
    corr = df[feat_cols].corr()
    short = ['Avg Amt', 'Freq', 'Uniq CP', 'Cash Dep', 'Intl Tfr']
    corr.index   = short
    corr.columns = short

    fig, ax = plt.subplots(figsize=(9, 7))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f',
                cmap='coolwarm', center=0, linewidths=0.5,
                linecolor='#0f1117', ax=ax, vmin=-1, vmax=1,
                annot_kws={'size': 12, 'weight': 'bold'})
    _title(ax, '🔥 Feature Correlation Heatmap', fontsize=14)
    _suptitle(fig, 'Understanding Relationships Between AML Features')
    plt.tight_layout()
    _save(fig, '02_correlation_heatmap.png')


def plot_boxplots(df):
    """03 — Boxplots for each feature (normal vs anomaly)."""
    features = [
        'Avg_Transaction_Amount_INR', 'Transaction_Frequency_Monthly',
        'Unique_Counterparties_Monthly', 'Cash_Deposit_Ratio',
        'International_Transfer_Ratio',
    ]
    fig, axes = plt.subplots(1, 5, figsize=(22, 6))

    for i, feat in enumerate(features):
        ax = axes[i]
        data_n = df.loc[df['Is_Anomaly'] == 0, feat]
        data_a = df.loc[df['Is_Anomaly'] == 1, feat]
        ax.boxplot([data_n, data_a],
                   patch_artist=True,
                   medianprops=dict(color='yellow', linewidth=2),
                   boxprops=dict(facecolor=PALETTE[i % len(PALETTE)], alpha=0.7),
                   whiskerprops=dict(color='#aaa'),
                   capprops=dict(color='#aaa'),
                   flierprops=dict(marker='o', color='#ff6b6b', markersize=4))
        ax.set_xticklabels(['Normal', 'Anomaly'], fontsize=10)
        _title(ax, feat.replace('_', ' '), fontsize=10)

    _suptitle(fig, '📦 Feature Boxplots: Normal vs Anomaly Accounts')
    plt.tight_layout()
    _save(fig, '03_boxplots_features.png')


def plot_pairplot(df):
    """04 — Pairplot coloured by anomaly."""
    features = [
        'Transaction_Frequency_Monthly', 'Cash_Deposit_Ratio',
        'International_Transfer_Ratio', 'Unique_Counterparties_Monthly',
    ]
    plot_df = df[features + ['Is_Anomaly']].copy()
    plot_df['Type'] = plot_df['Is_Anomaly'].map({0: 'Normal', 1: 'Anomaly'})

    with plt.style.context('dark_background'):
        g = sns.pairplot(
            plot_df.drop(columns='Is_Anomaly'),
            hue='Type',
            palette={'Normal': '#00c9ff', 'Anomaly': '#ff6b6b'},
            plot_kws=dict(alpha=0.7, s=30, edgecolor='none'),
            diag_kind='kde',
        )
        g.figure.suptitle('🔍 Pairplot — Normal vs Anomaly Accounts', y=1.02,
                           fontsize=14, fontweight='bold', color='white')
        g.figure.patch.set_facecolor('#0f1117')
        _save(g.figure, '04_pairplot.png')


# ══════════════════════════════════════════════════════════════════════════════
#  CLUSTERING CHARTS
# ══════════════════════════════════════════════════════════════════════════════

def plot_elbow_curve(X_scaled):
    """05 — Elbow curve for K-Means."""
    inertias = []
    k_range  = range(2, 11)
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertias.append(km.inertia_)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(list(k_range), inertias, 'o-', color='#00c9ff',
            linewidth=2.5, markersize=8, markerfacecolor='#ff6b6b')
    ax.fill_between(list(k_range), inertias,
                    alpha=0.15, color='#00c9ff')
    ax.axvline(4, color='#ff6b6b', linestyle='--', linewidth=1.8,
               label='Chosen k=4')
    ax.legend(fontsize=11)
    _title(ax, '📐 Elbow Curve — Optimal Number of Clusters (K-Means)', fontsize=13)
    ax.set_xlabel('Number of Clusters (k)')
    ax.set_ylabel('Inertia (Within-cluster SSE)')
    plt.tight_layout()
    _save(fig, '05_elbow_curve.png')


def plot_silhouette_scores(X_scaled):
    """06 — Silhouette score per k."""
    from sklearn.metrics import silhouette_score
    scores  = []
    k_range = range(2, 11)
    for k in k_range:
        labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X_scaled)
        scores.append(silhouette_score(X_scaled, labels))

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(list(k_range), scores,
                  color=[PALETTE[i % len(PALETTE)] for i in range(len(k_range))],
                  edgecolor='white', linewidth=0.5, alpha=0.85)
    best_k = list(k_range)[np.argmax(scores)]
    ax.axvline(best_k, color='#ff6b6b', linestyle='--',
               linewidth=2, label=f'Best k={best_k}')
    for bar, s in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f'{s:.3f}', ha='center', va='bottom', fontsize=9, color='white')
    ax.legend(fontsize=11)
    _title(ax, '📏 Silhouette Scores vs Number of Clusters', fontsize=13)
    ax.set_xlabel('Number of Clusters (k)')
    ax.set_ylabel('Silhouette Score')
    plt.tight_layout()
    _save(fig, '06_silhouette_scores.png')


def plot_cluster_scatter_pca(df, X_scaled):
    """07 — 2D PCA scatter coloured by cluster, anomalies marked."""
    pca   = PCA(n_components=2, random_state=42)
    comps = pca.fit_transform(X_scaled)
    ev    = pca.explained_variance_ratio_

    fig, ax = plt.subplots(figsize=(11, 8))
    clusters = sorted(df['Cluster'].unique())
    for cl in clusters:
        mask = df['Cluster'] == cl
        ax.scatter(comps[mask, 0], comps[mask, 1],
                   s=60, alpha=0.75, label=f'Cluster {cl}',
                   color=PALETTE[cl % len(PALETTE)], edgecolors='none')

    # Overlay anomalies
    anom_mask = df['Is_Anomaly'] == 1
    ax.scatter(comps[anom_mask, 0], comps[anom_mask, 1],
               s=120, facecolors='none', edgecolors='#ff6b6b',
               linewidths=2, zorder=5, label='Anomaly (IF)')

    ax.legend(fontsize=10, loc='upper right')
    _title(ax, f'🗺️ PCA 2D Cluster Map  (PC1={ev[0]:.1%}, PC2={ev[1]:.1%})', fontsize=13)
    ax.set_xlabel(f'PC1 ({ev[0]:.1%} variance)')
    ax.set_ylabel(f'PC2 ({ev[1]:.1%} variance)')
    plt.tight_layout()
    _save(fig, '07_cluster_scatter_pca.png')


def plot_cluster_profiles(df):
    """08 — Radar/spider chart of cluster mean profiles."""
    features = [
        'Avg_Transaction_Amount_INR', 'Transaction_Frequency_Monthly',
        'Unique_Counterparties_Monthly', 'Cash_Deposit_Ratio',
        'International_Transfer_Ratio',
    ]
    labels_short = ['Avg Amt', 'Freq', 'Uniq CP', 'Cash Dep', 'Intl Tfr']
    N = len(labels_short)

    # Normalize each feature 0-1 for radar
    profile = df.groupby('Cluster')[features].mean()
    for col in features:
        mn, mx = profile[col].min(), profile[col].max()
        profile[col] = (profile[col] - mn) / (mx - mn + 1e-9)

    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
    ax.set_facecolor('#1a1d27')
    ax.spines['polar'].set_color('#444')

    for cl_idx, (cl, row) in enumerate(profile.iterrows()):
        values = row.tolist() + row.tolist()[:1]
        color  = PALETTE[cl_idx % len(PALETTE)]
        ax.plot(angles, values, 'o-', linewidth=2.2, color=color)
        ax.fill(angles, values, alpha=0.18, color=color,
                label=f'Cluster {cl}')

    ax.set_thetagrids(np.degrees(angles[:-1]), labels_short, fontsize=11, color='white')
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['25%', '50%', '75%', '100%'], color='#aaa', fontsize=9)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)
    _title(ax, '🕷️ Cluster Feature Profiles (Radar Chart)', fontsize=13)
    plt.tight_layout()
    _save(fig, '08_cluster_profiles_radar.png')


def plot_cluster_heatmap(df):
    """09 — Heatmap of normalised cluster centroids."""
    features = [
        'Avg_Transaction_Amount_INR', 'Transaction_Frequency_Monthly',
        'Unique_Counterparties_Monthly', 'Cash_Deposit_Ratio',
        'International_Transfer_Ratio',
    ]
    profile = df.groupby('Cluster')[features].mean()
    # Normalise per column for visual comparison
    norm_profile = (profile - profile.min()) / (profile.max() - profile.min() + 1e-9)
    norm_profile.columns = ['Avg Amt', 'Freq', 'Uniq CP', 'Cash Dep', 'Intl Tfr']
    norm_profile.index   = [f'Cluster {i}' for i in norm_profile.index]

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(norm_profile, annot=True, fmt='.2f', cmap='YlOrRd',
                linewidths=0.5, linecolor='#0f1117', ax=ax,
                annot_kws={'size': 12, 'weight': 'bold'}, vmin=0, vmax=1)
    _title(ax, '🌡️ Cluster Centroids Heatmap (Normalised)', fontsize=13)
    plt.tight_layout()
    _save(fig, '09_cluster_heatmap.png')


# ══════════════════════════════════════════════════════════════════════════════
#  ANOMALY / ISOLATION FOREST CHARTS
# ══════════════════════════════════════════════════════════════════════════════

def plot_anomaly_score_distribution(df):
    """10 — Anomaly score KDE + rug plot."""
    fig, ax = plt.subplots(figsize=(11, 5))
    normal = df.loc[df['Is_Anomaly'] == 0, 'Anomaly_Score']
    anom   = df.loc[df['Is_Anomaly'] == 1, 'Anomaly_Score']

    normal.plot.kde(ax=ax, color='#00c9ff', linewidth=2.5, label='Normal')
    anom.plot.kde(ax=ax, color='#ff6b6b', linewidth=2.5, label='Anomaly')
    ax.fill_between(
        np.linspace(0, 1, 300),
        0,
        [ax.lines[0].get_ydata()[np.argmin(np.abs(
            ax.lines[0].get_xdata() - x))] for x in np.linspace(0, 1, 300)],
        alpha=0.12, color='#00c9ff'
    )
    ax.axvline(0.8, color='yellow', linestyle='--', linewidth=1.5,
               label='High Anomaly Threshold (0.80)')
    ax.set_xlabel('Anomaly Score (0=normal, 1=most anomalous)')
    ax.set_ylabel('Density')
    ax.legend(fontsize=11)
    _title(ax, '🚨 Isolation Forest Anomaly Score Distribution', fontsize=13)
    plt.tight_layout()
    _save(fig, '10_anomaly_score_distribution.png')


def plot_anomaly_scatter_pca(df, X_scaled):
    """11 — PCA scatter coloured by continuous anomaly score."""
    pca   = PCA(n_components=2, random_state=42)
    comps = pca.fit_transform(X_scaled)

    fig, ax = plt.subplots(figsize=(11, 8))
    sc = ax.scatter(comps[:, 0], comps[:, 1],
                    c=df['Anomaly_Score'], cmap='RdYlGn_r',
                    s=70, alpha=0.85, edgecolors='none')
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label('Anomaly Score', color='white')
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')

    # Label top anomalies
    top10 = df.nlargest(10, 'Anomaly_Score')
    for idx in top10.index:
        ax.annotate(df.loc[idx, 'Account_ID'],
                    (comps[idx, 0], comps[idx, 1]),
                    fontsize=7, color='white',
                    xytext=(5, 5), textcoords='offset points')

    _title(ax, '🎯 Anomaly Score Map (PCA 2D projection)', fontsize=13)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    plt.tight_layout()
    _save(fig, '11_anomaly_scatter_pca.png')


def plot_isolation_forest_decision(df):
    """12 — Scatter: Transaction Frequency vs Cash Ratio coloured by IF label."""
    fig, ax = plt.subplots(figsize=(10, 7))
    normal = df[df['Is_Anomaly'] == 0]
    anom   = df[df['Is_Anomaly'] == 1]

    ax.scatter(normal['Transaction_Frequency_Monthly'],
               normal['Cash_Deposit_Ratio'],
               s=60, alpha=0.65, color='#00c9ff', label='Normal', edgecolors='none')
    ax.scatter(anom['Transaction_Frequency_Monthly'],
               anom['Cash_Deposit_Ratio'],
               s=120, alpha=0.9, color='#ff6b6b',
               marker='X', label='Anomaly (Isolation Forest)', edgecolors='white', linewidth=0.5)

    # Annotate top anomaly accounts
    for _, row in df.nlargest(5, 'Anomaly_Score').iterrows():
        ax.annotate(row['Account_ID'],
                    (row['Transaction_Frequency_Monthly'], row['Cash_Deposit_Ratio']),
                    fontsize=8, color='yellow',
                    xytext=(7, 4), textcoords='offset points',
                    arrowprops=dict(arrowstyle='->', color='yellow', lw=1))

    ax.set_xlabel('Transaction Frequency (Monthly)')
    ax.set_ylabel('Cash Deposit Ratio')
    ax.legend(fontsize=11)
    _title(ax, '🌲 Isolation Forest Decision: Freq vs Cash Ratio', fontsize=13)
    plt.tight_layout()
    _save(fig, '12_isolation_forest_decision.png')


# ══════════════════════════════════════════════════════════════════════════════
#  RISK / OUTPUT CHARTS
# ══════════════════════════════════════════════════════════════════════════════

def plot_risk_score_distribution(df):
    """13 — Risk score histogram + KDE with thresholds annotated."""
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.hist(df['Risk_Score'], bins=30, color='#667eea',
            alpha=0.55, edgecolor='white', linewidth=0.4, density=True)
    df['Risk_Score'].plot.kde(ax=ax, color='white', linewidth=2)

    ax.axvspan(75, 100, alpha=0.15, color='#ff6b6b', label='High Risk (≥75)')
    ax.axvspan(50,  75, alpha=0.15, color='#f7971e', label='Medium Risk (50-75)')
    ax.axvspan(0,   50, alpha=0.10, color='#43e97b', label='Low Risk (<50)')
    ax.axvline(75, color='#ff6b6b', linestyle='--', linewidth=1.5)
    ax.axvline(50, color='#f7971e', linestyle='--', linewidth=1.5)

    ax.set_xlabel('Risk Score (0–100)')
    ax.set_ylabel('Density')
    ax.legend(fontsize=10)
    _title(ax, '⚠️ Risk Score Distribution with Threshold Bands', fontsize=13)
    plt.tight_layout()
    _save(fig, '13_risk_score_distribution.png')


def plot_typology_flag_counts(df):
    """14 — Bar chart of typology flag counts."""
    flags  = ['Smurfing_Flag', 'RoundTripping_Flag', 'Layering_Flag']
    counts = df[flags].sum()
    labels = ['Smurfing', 'Round-Tripping', 'Layering']
    colors = ['#ff6b6b', '#f7971e', '#00c9ff']

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(labels, counts.values, color=colors,
                  edgecolor='white', linewidth=0.7, alpha=0.85, width=0.5)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3,
                str(int(val)), ha='center', va='bottom',
                fontsize=13, fontweight='bold', color='white')

    ax.set_ylim(0, counts.max() * 1.25)
    ax.set_ylabel('Number of Accounts Flagged')
    _title(ax, '🚩 AML Typology Flag Counts across All Accounts', fontsize=13)
    plt.tight_layout()
    _save(fig, '14_typology_flag_counts.png')


def plot_cluster_vs_risk_boxplot(df):
    """15 — Boxplot of risk scores per KMeans cluster."""
    fig, ax = plt.subplots(figsize=(10, 6))
    clusters = sorted(df['Cluster'].unique())
    data     = [df.loc[df['Cluster'] == cl, 'Risk_Score'].values for cl in clusters]

    bp = ax.boxplot(data,
                    patch_artist=True,
                    medianprops=dict(color='yellow', linewidth=2.5),
                    whiskerprops=dict(color='#aaa', linewidth=1.3),
                    capprops=dict(color='#aaa'),
                    flierprops=dict(marker='o', color='#ff6b6b', markersize=5))
    for patch, color in zip(bp['boxes'], PALETTE):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    ax.set_xticklabels([f'Cluster {c}' for c in clusters])
    ax.set_ylabel('Risk Score (0–100)')
    _title(ax, '📦 Risk Score Distribution per KMeans Cluster', fontsize=13)
    plt.tight_layout()
    _save(fig, '15_cluster_vs_risk_boxplot.png')


def plot_top10_accounts_bar(df):
    """16 — Horizontal bar chart of top 10 accounts by risk score."""
    top10 = df.nlargest(10, 'Risk_Score')[
        ['Account_ID', 'Risk_Score', 'Is_Anomaly', 'Smurfing_Flag',
         'RoundTripping_Flag', 'Layering_Flag']
    ].copy()
    top10 = top10.sort_values('Risk_Score')

    colors = ['#ff6b6b' if r >= 75 else '#f7971e' for r in top10['Risk_Score']]

    fig, ax = plt.subplots(figsize=(11, 7))
    bars = ax.barh(top10['Account_ID'], top10['Risk_Score'],
                   color=colors, edgecolor='white', linewidth=0.5, alpha=0.88)
    for bar, val in zip(bars, top10['Risk_Score']):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f'{val:.1f}', va='center', fontsize=10, fontweight='bold', color='white')

    ax.axvline(75, color='#ff6b6b', linestyle='--', linewidth=1.5, label='High threshold')
    ax.axvline(50, color='#f7971e', linestyle='--', linewidth=1.5, label='Medium threshold')
    ax.set_xlim(0, 110)
    ax.set_xlabel('Risk Score (0–100)')
    ax.legend(fontsize=10)
    _title(ax, '🏆 Top 10 Highest-Risk Accounts', fontsize=13)
    plt.tight_layout()
    _save(fig, '16_top10_accounts_bar.png')


def plot_risk_heatmap_features(df):
    """17 — Heatmap showing feature values for top 20 riskiest accounts."""
    feats = [
        'Avg_Transaction_Amount_INR', 'Transaction_Frequency_Monthly',
        'Unique_Counterparties_Monthly', 'Cash_Deposit_Ratio',
        'International_Transfer_Ratio',
    ]
    top20 = df.nlargest(20, 'Risk_Score').set_index('Account_ID')[feats]
    # Normalise per column
    norm = (top20 - top20.min()) / (top20.max() - top20.min() + 1e-9)
    norm.columns = ['Avg Amt', 'Freq', 'Uniq CP', 'Cash Dep', 'Intl Tfr']

    fig, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(norm, annot=top20.round(2), fmt='g', cmap='Reds',
                linewidths=0.4, linecolor='#0f1117', ax=ax,
                annot_kws={'size': 8})
    _title(ax, '🌡️ Feature Values for Top 20 Risk Accounts (Normalised colour)', fontsize=13)
    plt.tight_layout()
    _save(fig, '17_risk_heatmap_features.png')


def plot_cross_validation_venn(df):
    """18 — Venn diagram showing method agreement (manual fallback)."""
    _plot_venn_manual(df)


def _plot_venn_lib(df):
    from matplotlib_venn import venn3
    kmeans_set   = set(df.loc[df['Cluster_Risk'] >= 0.7, 'Account_ID'])
    iso_set      = set(df.loc[df['Is_Anomaly'] == 1, 'Account_ID'])
    typol_set    = set(df.loc[
        (df['Smurfing_Flag'] | df['RoundTripping_Flag'] | df['Layering_Flag']) == 1,
        'Account_ID'
    ])

    fig, ax = plt.subplots(figsize=(9, 7))
    v = venn3([kmeans_set, iso_set, typol_set],
              set_labels=('KMeans\n(high risk cluster)',
                          'Isolation Forest', 'Typology Rules'),
              ax=ax,
              set_colors=('#667eea', '#ff6b6b', '#f7971e'))
    _suptitle(fig, '🔗 Cross-Validation Venn Diagram — Method Agreement')
    plt.tight_layout()
    _save(fig, '18_cross_validation_venn.png')


def _plot_venn_manual(df):
    """Manual Venn fallback using matplotlib patches."""
    kmeans_ids  = set(df.loc[df['Cluster_Risk'] >= 0.7, 'Account_ID'])
    iso_ids     = set(df.loc[df['Is_Anomaly'] == 1, 'Account_ID'])
    typol_ids   = set(df.loc[
        (df['Smurfing_Flag'] | df['RoundTripping_Flag'] | df['Layering_Flag']) == 1,
        'Account_ID'
    ])

    only_km      = kmeans_ids - iso_ids - typol_ids
    only_iso     = iso_ids - kmeans_ids - typol_ids
    only_typ     = typol_ids - kmeans_ids - iso_ids
    km_iso       = kmeans_ids & iso_ids - typol_ids
    km_typ       = kmeans_ids & typol_ids - iso_ids
    iso_typ      = iso_ids & typol_ids - kmeans_ids
    all_three    = kmeans_ids & iso_ids & typol_ids

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis('off')

    circle_kw = dict(fill=True, linewidth=2)
    ax.add_patch(plt.Circle((3.8, 5.5), 2.5, color='#667eea', alpha=0.35, **circle_kw))
    ax.add_patch(plt.Circle((6.2, 5.5), 2.5, color='#ff6b6b', alpha=0.35, **circle_kw))
    ax.add_patch(plt.Circle((5.0, 3.5), 2.5, color='#f7971e', alpha=0.35, **circle_kw))

    ax.text(2.3, 6.8, 'KMeans', fontsize=11, fontweight='bold', color='#667eea', ha='center')
    ax.text(7.7, 6.8, 'Iso Forest', fontsize=11, fontweight='bold', color='#ff6b6b', ha='center')
    ax.text(5.0, 1.4, 'Typology Rules', fontsize=11, fontweight='bold', color='#f7971e', ha='center')

    ax.text(2.0, 5.5, f'{len(only_km):d}', fontsize=14, ha='center', color='white', fontweight='bold')
    ax.text(8.0, 5.5, f'{len(only_iso):d}', fontsize=14, ha='center', color='white', fontweight='bold')
    ax.text(5.0, 2.2, f'{len(only_typ):d}', fontsize=14, ha='center', color='white', fontweight='bold')
    ax.text(5.0, 6.2, f'{len(km_iso):d}', fontsize=13, ha='center', color='white')
    ax.text(3.3, 4.2, f'{len(km_typ):d}', fontsize=13, ha='center', color='white')
    ax.text(6.7, 4.2, f'{len(iso_typ):d}', fontsize=13, ha='center', color='white')
    ax.text(5.0, 5.1, f'{len(all_three):d}', fontsize=16, ha='center',
            color='yellow', fontweight='bold')

    legend_elems = [
        mpatches.Patch(facecolor='#667eea', alpha=0.6, label='KMeans high-risk cluster'),
        mpatches.Patch(facecolor='#ff6b6b', alpha=0.6, label='Isolation Forest anomaly'),
        mpatches.Patch(facecolor='#f7971e', alpha=0.6, label='Typology rule triggered'),
    ]
    ax.legend(handles=legend_elems, loc='upper right', fontsize=10)
    ax.set_title('🔗 Cross-Validation — Method Agreement (Venn Diagram)\nYellow centre = flagged by ALL three methods',
                 fontsize=13, fontweight='bold', color='white', pad=12)
    fig.set_facecolor('#0f1117')
    _save(fig, '18_cross_validation_venn.png')


# ══════════════════════════════════════════════════════════════════════════════
#  MASTER ENTRY POINTS
# ══════════════════════════════════════════════════════════════════════════════

def plot_eda(df):
    """Generate EDA charts that do NOT require model columns (pre-model)."""
    os.makedirs(GRAPHS_DIR, exist_ok=True)
    print("\n[EDA Charts - Phase 1: Pre-model]")
    plot_feature_distributions(df)
    plot_correlation_heatmap(df)


def plot_eda_post(df):
    """Generate EDA charts that require Is_Anomaly column (post-model)."""
    os.makedirs(GRAPHS_DIR, exist_ok=True)
    print("\n[EDA Charts - Phase 2: Post-model]")
    plot_boxplots(df)
    plot_pairplot(df)


def plot_clustering(df, X_scaled):
    """Generate all clustering charts and save to graphs/."""
    os.makedirs(GRAPHS_DIR, exist_ok=True)
    print("\n[Clustering Charts]")
    plot_elbow_curve(X_scaled)
    plot_silhouette_scores(X_scaled)
    plot_cluster_scatter_pca(df, X_scaled)
    plot_cluster_profiles(df)
    plot_cluster_heatmap(df)


def plot_anomaly(df, X_scaled):
    """Generate all anomaly detection charts and save to graphs/."""
    os.makedirs(GRAPHS_DIR, exist_ok=True)
    print("\n[Anomaly Detection Charts]")
    plot_anomaly_score_distribution(df)
    plot_anomaly_scatter_pca(df, X_scaled)
    plot_isolation_forest_decision(df)


def plot_aml_results(df):
    """Generate all risk / output charts and save to graphs/."""
    os.makedirs(GRAPHS_DIR, exist_ok=True)
    print("\n[Risk & Output Charts]")
    plot_risk_score_distribution(df)
    plot_typology_flag_counts(df)
    plot_cluster_vs_risk_boxplot(df)
    plot_top10_accounts_bar(df)
    plot_risk_heatmap_features(df)
    plot_cross_validation_venn(df)