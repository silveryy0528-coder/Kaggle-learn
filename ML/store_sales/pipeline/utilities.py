#%%
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.stattools import acf


def build_features(df_fam, meta=None):
    df_fam = df_fam.sort_values(['store_nbr', 'date'])
    df_fam['store_id'] = df_fam['store_nbr'].astype(str)

    # Calendar features
    if meta is None:
        unique_dates = df_fam['date'].sort_values().unique()
        date_to_idx = {date: i for i, date in enumerate(unique_dates)}
        max_time_idx = len(unique_dates) - 1
    else:
        date_to_idx = meta['date_to_idx']
        max_time_idx = meta['max_time_idx']

    df_fam['time_idx'] = df_fam['date'].map(date_to_idx)
    # Handle future date (for test)
    mask_future = df_fam["time_idx"].isna()
    if mask_future.any():
        df_fam.loc[mask_future, "time_idx"] = (
            max_time_idx
            + (df_fam.loc[mask_future, "date"]
            - pd.to_datetime(max(date_to_idx)))
            .dt.days
        )

    df_fam['month'] = df_fam['date'].dt.month
    df_fam['dayofweek'] = df_fam['date'].dt.dayofweek
    df_fam['year'] = df_fam['date'].dt.year

    # Feature lists
    lr_features = ['time_idx', 'dayofweek', 'month', 'store_id']
    xgb_features = ['store_id', 'dayofweek', 'month', 'year']

    X_lr = df_fam[lr_features]
    X_xgb = df_fam[xgb_features]

    # One-hot encoding
    X_lr = pd.get_dummies(
        X_lr,
        columns=['dayofweek', 'month', 'store_id'],
        drop_first=True
    )

    X_xgb = pd.get_dummies(
        X_xgb,
        columns=['store_id', 'dayofweek', 'month'],
        drop_first=False
    )

    # Store-specific trend (LR only)
    store_dummies = pd.get_dummies(
        df_fam['store_id'], prefix='store', drop_first=True
    )
    for c in store_dummies.columns:
        X_lr[f'{c}_time'] = store_dummies[c] * df_fam['time_idx']

    y = df_fam['sales'] if 'sales' in df_fam.columns else None

    # Column alignment
    if meta is None:
        meta = {
            "lr_cols": X_lr.columns,
            "xgb_cols": X_xgb.columns,
            "date_to_idx": date_to_idx,
            "max_time_idx": max_time_idx
        }
    else:
        X_lr = X_lr.reindex(columns=meta["lr_cols"], fill_value=0)
        X_xgb = X_xgb.reindex(columns=meta["xgb_cols"], fill_value=0)

    return df_fam, X_lr, X_xgb, y, meta


#%%
if __name__ == '__main__':
    train_file_path = r"C:\Users\guoya\Documents\Git_repo\Kaggle-learn\ML\store_sales\data\train.csv"
    train_df = pd.read_csv(train_file_path, parse_dates=['date'])

    test_file_path = r"C:\Users\guoya\Documents\Git_repo\Kaggle-learn\ML\store_sales\data\test.csv"
    test_df = pd.read_csv(test_file_path, parse_dates=['date'])
    #%%
    df_fam, X_lr_tr, X_xgb_tr, y_tr, meta  = build_features(
        train_df[train_df['family'] == 'GROCERY I'].copy()
    )

    df_fam, X_lr_te, X_xgb_te, y_te, meta  = build_features(
        test_df[test_df['family'] == 'GROCERY I'].copy(), meta=meta
    )
