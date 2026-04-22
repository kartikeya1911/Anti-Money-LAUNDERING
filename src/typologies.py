def apply_typology_rules(df):
    """
    Applies business-logic heuristics to flag specific financial crime typologies.
    Flags: Smurfing, Round-Tripping, Layering.
    Uses data-driven percentile thresholds so rules adapt to any dataset.
    """
    q75_freq    = df['Transaction_Frequency_Monthly'].quantile(0.75)
    q25_amt     = df['Avg_Transaction_Amount_INR'].quantile(0.25)
    q75_cash    = df['Cash_Deposit_Ratio'].quantile(0.75)
    q25_counter = df['Unique_Counterparties_Monthly'].quantile(0.25)
    q75_counter = df['Unique_Counterparties_Monthly'].quantile(0.75)
    q75_intl    = df['International_Transfer_Ratio'].quantile(0.75)

    # 1. SMURFING  — High frequency + small amounts + heavy cash deposits
    #    (breaking large sums into many small deposits to evade reporting)
    df['Smurfing_Flag'] = (
        (df['Transaction_Frequency_Monthly'] >= q75_freq) &
        (df['Avg_Transaction_Amount_INR']    <= q25_amt) &
        (df['Cash_Deposit_Ratio']            >= q75_cash)
    ).astype(int)

    # 2. ROUND-TRIPPING — High frequency but very few unique counterparties
    #    (money cycles between a small set of accounts)
    df['RoundTripping_Flag'] = (
        (df['Transaction_Frequency_Monthly']   >= q75_freq) &
        (df['Unique_Counterparties_Monthly']   <= max(2, q25_counter))
    ).astype(int)

    # 3. LAYERING — Many counterparties + heavy international transfers
    #    (rapidly moving money across accounts/borders to obscure origin)
    df['Layering_Flag'] = (
        (df['Unique_Counterparties_Monthly']  >= q75_counter) &
        (df['International_Transfer_Ratio']   >= q75_intl)
    ).astype(int)

    print(f"   Smurfing flags     : {df['Smurfing_Flag'].sum()}")
    print(f"   Round-Tripping flags: {df['RoundTripping_Flag'].sum()}")
    print(f"   Layering flags     : {df['Layering_Flag'].sum()}")

    return df