import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.stattools import acf


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
