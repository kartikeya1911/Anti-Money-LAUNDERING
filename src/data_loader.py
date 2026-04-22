import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

FEATURE_COLS = [
    'Avg_Transaction_Amount_INR',
    'Transaction_Frequency_Monthly',
    'Unique_Counterparties_Monthly',
    'Cash_Deposit_Ratio',
    'International_Transfer_Ratio',
]

def load_and_preprocess(filepath):
    """
    Loads dataset, performs basic validation, scales numeric features for modeling.
    Returns: (df, features_df, X_scaled, scaler)
    """
    df = pd.read_csv(filepath)

    # Validate expected columns
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    print(f"   Loaded {len(df)} accounts with {len(df.columns)} columns.")
    print(f"   Missing values: {df[FEATURE_COLS].isnull().sum().sum()}")

    # Select only numeric feature columns
    features = df[FEATURE_COLS].copy()

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    return df, features, X_scaled, scaler
