"""Utility functions."""

import pandas as pd


def transform_numeric_features_binary_target(
    data, drop_cols=None, target_col=None, target_vals=None
):
    if drop_cols is None:
        drop_cols = []
    if target_col is None:
        target_col = data.columns[-1]
    if target_vals is None:
        target_vals = [1]
    y = data[target_col].rename('target').isin(target_vals).astype(int)
    X = data.drop(columns=[target_col] + list(drop_cols)).astype(float)
    columns_mapping = dict(zip(X.columns, range(X.shape[1])))
    X = X.rename(columns=columns_mapping)
    data = pd.concat([X, y], axis=1)
    data = data.dropna()
    return data
