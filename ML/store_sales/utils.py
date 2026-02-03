#%%
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error

def compute_family_metrics(
    df_fam,
    split_date,
    lr_model,
    X_lr,
    xgb_pred,
):
    # --- Test mask ---
    test_mask = df_fam["date"] >= split_date

    y_true = df_fam.loc[test_mask, "sales"].values
    lr_pred = df_fam.loc[test_mask, "lr_pred_corr"].values
    hybrid_pred = df_fam.loc[test_mask, "hybrid_pred"].values

    # --- 1. LR vs Hybrid MAE ---
    lr_mae = mean_absolute_error(y_true, lr_pred)
    hybrid_mae = mean_absolute_error(y_true, hybrid_pred)

    # --- 2. Residual ACF (store-averaged) ---
    mean_resid_acf = df_fam['lr_residual'].autocorr(lag=1)  # simple lag-1 ACF as proxy

    # --- 3. XGB contribution size ---
    xgb_contrib_ratio = (
        np.mean(np.abs(xgb_pred[test_mask])) /
        (np.mean(np.abs(lr_pred)) + 1e-8)
    )

    # --- 4. Store-level slope variance ---
    # Extract store_time coefficients from LR
    coef_series = pd.Series(lr_model.coef_, index=X_lr.columns)
    store_time_coefs = coef_series[coef_series.index.str.endswith('_time')]

    store_slopes = (
        store_time_coefs
        .rename(lambda x: x.replace('_time', ''))
        .sort_values()
    )
    slope_variance = np.var(store_slopes)

    return {
        "lr_mae": lr_mae,
        "hybrid_mae": hybrid_mae,
        "mae_improvement_pct": (lr_mae - hybrid_mae) / lr_mae * 100,
        "mean_residual_acf": mean_resid_acf,
        "xgb_contrib_ratio": xgb_contrib_ratio,
        "store_slope_variance": slope_variance
    }


def family_policy(metrics,
                  min_explained_frac=0.05,
                  min_mae_improvement_pct=5.0):
    if metrics['explained_frac'] < min_explained_frac:
        return 'lr'
    if metrics['mae_improvement_pct'] < min_mae_improvement_pct:
        return 'lr'
    return 'hybrid'


def build_features(
        df_fam, lags=(1, 7, 28), rolls=(7,)
):
    """
    Build LR and XGB features for a single family.
    """
    df_fam = df_fam.sort_values(['store_nbr', 'date'])
    df_fam['store_id'] = df_fam['store_nbr'].astype(str)

    # ------- time features -------
    unique_dates = df_fam['date'].sort_values().unique()
    date_to_idx = {date: i for i, date in enumerate(unique_dates)}

    df_fam['time_idx'] = df_fam['date'].map(date_to_idx)
    df_fam['month'] = df_fam['date'].dt.month
    df_fam['dayofweek'] = df_fam['date'].dt.dayofweek
    df_fam['year'] = df_fam['date'].dt.year

    # ------- lag / rolling -------
    for lag in lags:
        df_fam[f'lag_{lag}'] = (
            df_fam.groupby('store_nbr')['sales'].shift(lag)
        )
    for roll in rolls:
        df_fam[f'roll_{roll}'] = (
            df_fam.groupby('store_nbr')['sales']
            .shift(1)
            .rolling(window=roll)
            .mean()
        )

    # ------- activity gate -------
    df_fam["is_active"] = (
        (df_fam["lag_1"] > 0) | (df_fam["lag_7"] > 0)
    ).astype(int)

    # ------- feature selection -------
    lr_features = ['time_idx', 'dayofweek', 'month', 'store_id']
    xgb_features = [
        'lag_1', 'lag_7', 'roll_7',
        'store_id', 'dayofweek', 'month', 'year'
    ]

    X_lr = df_fam[lr_features]
    X_xgb = df_fam[xgb_features]
    y = df_fam["sales"]

    # ------- dummies -------
    dummy_features = ['store_id', 'dayofweek', 'month']
    X_lr = pd.get_dummies(
        X_lr, columns=dummy_features, drop_first=True
    )
    X_xgb = pd.get_dummies(
        X_xgb, columns=dummy_features, drop_first=False
    )

    # ------- store-specific slopes (LR only) -------
    store_dummies = pd.get_dummies(
        df_fam['store_id'], prefix='store', drop_first=True
    )
    for c in store_dummies.columns:
        X_lr[f'{c}_time'] = store_dummies[c] * df_fam['time_idx']

    # ------- drop invalid rows (NaNs in XGB) -------
    valid_idx = X_xgb.dropna().index
    X_lr = X_lr.loc[valid_idx]
    X_xgb = X_xgb.loc[valid_idx]
    if y is not None:
        y = y.loc[valid_idx]
    df_fam = df_fam.loc[valid_idx]

    return X_lr, X_xgb, y, df_fam


#%%
if __name__ == '__main__':
    train_file_path = r"C:\Users\guoya\Documents\Git_repo\Kaggle-learn\ML\store_sales\data\train.csv"
    train_df = pd.read_csv(train_file_path, parse_dates=['date'])

    test_file_path = r"C:\Users\guoya\Documents\Git_repo\Kaggle-learn\ML\store_sales\data\test.csv"
    test_df = pd.read_csv(test_file_path, parse_dates=['date'])

    X_lr_tr, X_xgb_tr, y_tr,  = build_features(
        train_df[train_df['family'] == 'GROCERY I'].copy()
    )
    X_lr_tr.head()

