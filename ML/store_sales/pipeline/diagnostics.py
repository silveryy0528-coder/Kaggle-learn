#%%
import sys
path = r'C:\Users\guoya\Documents\Git_repo\Kaggle-learn\ML\store_sales\pipeline'
sys.path.insert(0, path)

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.linear_model import Ridge
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import mean_absolute_error

import utilities as utils


split_date = '2016-01-01'
lr_alpha = 4.0
param_grid = {
    'max_depth': [4, 5, 6],
    'learning_rate': [0.05, 0.1],
    'n_estimators': [400, 500, 600],
    'subsample': [0.7],
    'colsample_bytree': [0.8],
    'reg_alpha': [0.5],
    'reg_lambda': [1., 2.]
}


def compute_family_metrics(
    df_fam,
    split_date,
    lr_model,
    X_lr
):
    # --- Test mask ---
    test_mask = df_fam["date"] >= split_date

    y_true = df_fam.loc[test_mask, "sales"].values
    lr_pred = df_fam.loc[test_mask, "lr_pred_corr"].values
    lr_residual = df_fam.loc[test_mask, "lr_residual"].values
    xgb_pred = df_fam.loc[test_mask, "xgb_pred"].values
    hybrid_pred = df_fam.loc[test_mask, "hybrid_pred"].values

    # --- 1. LR vs Hybrid MAE ---
    lr_mae = mean_absolute_error(y_true, lr_pred)
    hybrid_mae = mean_absolute_error(y_true, hybrid_pred)

    # --- 2. Residual ACF (store-averaged) ---
    # Simple lag-1 ACF as proxy
    mean_resid_acf = df_fam['lr_residual'].autocorr(lag=1)

    # --- 3. XGB contribution size ---
    xgb_contrib_ratio = (
        np.mean(np.abs(xgb_pred)) /
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

    # --- 5. Explained fraction ---
    explained_frac = 1 - np.var(lr_residual - xgb_pred) / np.var(lr_residual)

    return {
        "lr_mae": lr_mae,
        "hybrid_mae": hybrid_mae,
        "mae_improvement_pct": (lr_mae - hybrid_mae) / lr_mae * 100,
        "mean_residual_acf": mean_resid_acf,
        "xgb_contrib_ratio": xgb_contrib_ratio,
        "store_slope_variance": slope_variance,
        "explained_frac": explained_frac
    }


def run_family_hybrid(df, family):

    df_fam = df[df['family'] == family].copy()

    df_fam, X_lr, X_xgb, y, meta = utils.build_features(df_fam, meta=None)

    train_mask = df_fam['date'] < split_date
    test_mask  = ~train_mask

    # Stage 1 - LR
    X_train_lr, y_train_lr = X_lr[train_mask], y[train_mask]

    lr_model = Ridge(alpha=lr_alpha)
    lr_model.fit(X_train_lr, y_train_lr)

    y_train_pred_lr = lr_model.predict(X_train_lr)
    y_pred_lr = lr_model.predict(X_lr)

    bias = (y_train_lr - y_train_pred_lr).mean()
    lr_pred_corr = y_pred_lr + bias

    df_fam['lr_pred_corr'] = lr_pred_corr
    df_fam['lr_residual'] = y - lr_pred_corr

    # Stage 2 - XGB on residuals
    X_train_xgb, X_test_xgb = X_xgb[train_mask], X_xgb[test_mask]
    y_train_xgb = df_fam.loc[train_mask, 'lr_residual']
    y_test_xgb = df_fam.loc[test_mask, 'lr_residual']

    best_mae = float('inf')
    best_params = None
    for params in ParameterGrid(param_grid):
        model = xgb.XGBRegressor(
            n_estimators=params['n_estimators'],
            max_depth=params['max_depth'],
            learning_rate=params['learning_rate'],
            reg_alpha=params['reg_alpha'],
            reg_lambda=params['reg_lambda'],
            subsample=params['subsample'],
            colsample_bytree=params['colsample_bytree'],
            random_state=42,
            early_stopping_rounds=50
        )
        model.fit(
            X_train_xgb, y_train_xgb,
            eval_set=[(X_test_xgb, y_test_xgb)],
            verbose=False
        )
        y_train_pred_xgb = model.predict(X_train_xgb)

        mae = mean_absolute_error(y_train_xgb, y_train_pred_xgb)
        if mae < best_mae:
            best_mae = mae
            xgb_model = model
            best_params = params

    xgb_pred = np.full(len(df_fam), np.nan)
    xgb_pred[train_mask] = xgb_model.predict(X_train_xgb)
    xgb_pred[test_mask] = xgb_model.predict(X_test_xgb)
    df_fam['xgb_pred'] = xgb_pred

    # Stage 3 - Hybrid
    hybrid_pred = lr_pred_corr + xgb_pred
    df_fam['hybrid_pred'] = hybrid_pred

    metrics = compute_family_metrics(df_fam, split_date, lr_model, X_lr)
    results = {
        "family": family,
        "metrics": metrics,
        "xgb_params": best_params
    }
    return results


def family_policy(
        metrics,
        min_explained_frac=0.05,
        min_mae_improvement_pct=5.0
):
    if metrics['explained_frac'] < min_explained_frac:
        return 'lr'
    if metrics['mae_improvement_pct'] < min_mae_improvement_pct:
        return 'lr'
    return 'hybrid'


def run_single_family_diagnostics(df, family):
    results = run_family_hybrid(df, family)

    metrics = results['metrics']
    policy = family_policy(metrics)

    return {
        "family": family,
        "policy": policy,
        "xgb_params": results['xgb_params'],
        **metrics
    }


def run_all_family_diagnostics(train_df, families):
    results = []

    for family in families:
        diag = run_single_family_diagnostics(train_df, family=family)
        results.append({k: v for k, v in diag.items()})

    results = pd.DataFrame(results)
    results = results.set_index('family')

    return results


#%%
if __name__ == '__main__':
    train_file_path = r"C:\Users\guoya\Documents\Git_repo\Kaggle-learn\ML\store_sales\data\train.csv"
    train_df = pd.read_csv(train_file_path, parse_dates=['date'])
    families = [
        # 'GROCERY I',
        'PRODUCE',
        'MEATS',
        # 'BOOKS',
    ]

    results = run_all_family_diagnostics(train_df, families)