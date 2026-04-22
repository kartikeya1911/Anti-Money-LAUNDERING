import textwrap
from datetime import datetime

REPORT_DATE = datetime.today().strftime('%d %B %Y')


# ─────────────────────────────────────────────────────────────── #
#  Helpers (ASCII-only for Windows cp1252 compatibility)
# ─────────────────────────────────────────────────────────────── #

def _divider(char='=', width=65):
    return char * width


def _section(title):
    print("\n" + _divider())
    print("  " + title)
    print(_divider())


def _sar_block(account_row, rank=1):
    """Returns an ASCII-safe formatted SAR string for one account."""
    r = account_row
    # Pre-build strings that cannot go inside f-strings (Python < 3.12)
    narrative_lines = textwrap.wrap(str(r["Reason"]), width=57)
    narrative_str   = "\n".join("  |  " + line for line in narrative_lines)
    divider65       = "*" * 65
    smurfing_val    = "YES (!)" if r['Smurfing_Flag']      else "No"
    roundtrip_val   = "YES (!)" if r['RoundTripping_Flag'] else "No"
    layering_val    = "YES (!)" if r['Layering_Flag']      else "No"
    anomaly_val     = "YES (!)" if r['Is_Anomaly']         else "NO"
    level_clean     = str(r['Risk_Level']).replace('\U0001f534', '').replace('\U0001f7e1', '').replace('\U0001f7e2', '').strip()

    return f"""
{divider65}
            SUSPICIOUS ACTIVITY REPORT (SAR)  #{rank}
        Reporting Entity : GlobalBank Compliance Desk
        Report Date      : {REPORT_DATE}
{divider65}

  ACCOUNT ID    : {r['Account_ID']}
  RISK SCORE    : {r['Risk_Score']:.1f} / 100
  RISK LEVEL    : {level_clean}
  KMEANS CLUSTER: {int(r['Cluster'])}  |  ANOMALY FLAG: {anomaly_val}

  +-- TYPOLOGY TRIGGERS ------------------------------------------+
  |  Smurfing Detected       : {smurfing_val}
  |  Round-Tripping Detected : {roundtrip_val}
  |  Layering Detected       : {layering_val}
  +----------------------------------------------------------------+

  +-- ANALYST NARRATIVE ------------------------------------------+
{narrative_str}
  +----------------------------------------------------------------+

  +-- ACCOUNT METRICS --------------------------------------------+
  |  Avg Transaction Amount (INR)    : {r['Avg_Transaction_Amount_INR']:>12,.2f}
  |  Transaction Frequency (monthly) : {r['Transaction_Frequency_Monthly']:>12}
  |  Unique Counterparties (monthly) : {r['Unique_Counterparties_Monthly']:>12}
  |  Cash Deposit Ratio              : {r['Cash_Deposit_Ratio']:>12.2f}
  |  International Transfer Ratio    : {r['International_Transfer_Ratio']:>12.2f}
  +----------------------------------------------------------------+

  +-- RECOMMENDED ACTIONS ----------------------------------------+
  |  1) Escalate for Enhanced Due Diligence (EDD) immediately.
  |  2) Freeze international transfers pending investigation.
  |  3) Conduct full transaction-level audit for past 90 days.
  |  4) File SAR with FIU/FinCEN within 30-day legal deadline.
  +----------------------------------------------------------------+
{divider65}
"""


def print_cross_validation_table(df):
    """Prints which accounts are flagged by both models + rules."""
    _section("CROSS-VALIDATION: KMeans x Isolation Forest x Typology Rules")

    df_cv = df.copy()
    df_cv['HighRiskCluster'] = (df_cv['Cluster_Risk'] >= 0.7).astype(int)
    df_cv['AnyTypology']     = (
        (df_cv['Smurfing_Flag'] | df_cv['RoundTripping_Flag'] | df_cv['Layering_Flag'])
    ).astype(int)
    df_cv['FlagCount'] = (
        df_cv['HighRiskCluster'] + df_cv['Is_Anomaly'] + df_cv['AnyTypology']
    )

    all_three = df_cv[df_cv['FlagCount'] == 3].sort_values('Risk_Score', ascending=False)
    iso_typol = df_cv[(df_cv['Is_Anomaly'] == 1) & (df_cv['AnyTypology'] == 1)]
    km_iso    = df_cv[(df_cv['HighRiskCluster'] == 1) & (df_cv['Is_Anomaly'] == 1)]

    print(f"\n  >> Accounts flagged by ALL three methods : {len(all_three)}")
    print(f"  >> Iso Forest + Typology Rules          : {len(iso_typol)}")
    print(f"  >> KMeans + Isolation Forest            : {len(km_iso)}")

    if not all_three.empty:
        print(f"\n  {'Account ID':<12} {'Risk Score':>10} {'Cluster':>8} "
              f"{'IF':>5} {'Smurf':>6} {'RndTrp':>7} {'Layer':>6}")
        print("  " + "-" * 60)
        for _, row in all_three.head(10).iterrows():
            print(f"  {row['Account_ID']:<12} {row['Risk_Score']:>10.1f} "
                  f"{int(row['Cluster']):>8} "
                  f"{int(row['Is_Anomaly']):>5} {int(row['Smurfing_Flag']):>6} "
                  f"{int(row['RoundTripping_Flag']):>7} {int(row['Layering_Flag']):>6}")
    else:
        print("\n  (No accounts flagged simultaneously by all three methods.)")

    return all_three


def generate_sar_report(df):
    """
    Prints Top 10 summary, cross-validation table, and SAR blocks for Top 5 accounts.
    Returns the top 10 DataFrame.
    """
    top10 = df.sort_values('Risk_Score', ascending=False).head(10)
    top5  = top10.head(5)

    # -- Top 10 Summary --
    _section("TOP 10 HIGH-RISK ACCOUNTS SUMMARY")
    print(f"\n  {'Rank':<5} {'Account ID':<12} {'Score':>7} {'Level':<8} "
          f"{'Smurf':>6} {'RndTrp':>7} {'Layer':>6} {'IF':>4}")
    print("  " + "-" * 65)
    for rank, (_, row) in enumerate(top10.iterrows(), 1):
        level_plain = (str(row['Risk_Level'])
                       .replace('\U0001f534 ', '')
                       .replace('\U0001f7e1 ', '')
                       .replace('\U0001f7e2 ', '')
                       .strip())
        print(f"  {rank:<5} {row['Account_ID']:<12} {row['Risk_Score']:>7.1f} "
              f"{level_plain:<8} {int(row['Smurfing_Flag']):>6} "
              f"{int(row['RoundTripping_Flag']):>7} {int(row['Layering_Flag']):>6} "
              f"{int(row['Is_Anomaly']):>4}")

    # -- Cross-Validation --
    print_cross_validation_table(df)

    # -- SAR Reports for Top 5 --
    _section("SUSPICIOUS ACTIVITY REPORTS (SAR) - TOP 5 ACCOUNTS")
    for rank, (_, row) in enumerate(top5.iterrows(), 1):
        print(_sar_block(row, rank=rank))

    # -- Risk Scoring Formula --
    _section("RISK SCORING FORMULA")
    print("""
  Risk_Score = (
      0.30 x Anomaly_Score_norm          (Isolation Forest - normalised to [0,1])
    + 0.25 x Cash_Deposit_Ratio_norm     (min-max normalised)
    + 0.20 x Intl_Transfer_Ratio_norm    (min-max normalised)
    + 0.15 x Transaction_Frequency_norm  (min-max normalised)
    + 0.10 x Unique_Counterparties_norm  (min-max normalised)
  ) x 100
  + 10 x KMeans_Suspicious              (bonus: account in highest-risk cluster)
  <- Clipped to [0, 100]

  Risk Level thresholds:
    >= 75  ->  HIGH   (Mandatory SAR filing)
    >= 50  ->  MEDIUM (Enhanced Due Diligence)
    <  50  ->  LOW    (Routine monitoring)
""")

    return top10