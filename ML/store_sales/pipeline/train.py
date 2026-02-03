#%%
import os
import sys
import joblib
folder_path = r'C:\Users\guoya\Documents\Git_repo\Kaggle-learn\ML\store_sales\pipeline'
sys.path.insert(0, folder_path)

import xgboost as xgb
import pandas as pd
from sklearn.linear_model import Ridge

import utilities as utils
import diagnostics


lr_alpha = 4.0

def train_single_family(df, family, policy, xgb_params):

    df_fam = df[df['family'] == family].copy()

    df_fam, X_lr, X_xgb, y, meta = utils.build_features(df_fam, meta=None)

    # Stage 1 - LR
    lr_model = Ridge(alpha=lr_alpha)
    lr_model.fit(X_lr, y)

    lr_pred = lr_model.predict(X_lr)
    bias = (y - lr_pred).mean()

    if policy == 'lr':
        return {
            'family': family,
            'policy': 'lr',
            'lr_model': lr_model,
            'lr_alpha': lr_alpha,
            'xgb_model': None,
            'xgb_params': None,
            'bias': bias,
            'meta': meta
        }

    # Stage 2 - XGB on residuals
    lr_residual = y - (lr_pred + bias)

    xgb_model = xgb.XGBRegressor(
        random_state=42,
        **xgb_params
    )
    xgb_model.fit(X_xgb, lr_residual)

    return {
        'family': family,
        'policy': 'hybrid',
        'lr_model': lr_model,
        'lr_alpha': lr_alpha,
        'xgb_model': xgb_model,
        'xgb_params': xgb_params,
        'bias': bias,
        'meta': meta
    }


def train_all_families(train_df, diagnostic_results):
    models_by_family = {}

    for family, row in diagnostic_results.iterrows():
        policy = row['policy']
        xgb_params = row['xgb_params']

        print(f'Family {family}, training policy: {policy}')

        model_bundle = train_single_family(
            train_df,
            family=family,
            policy=policy,
            xgb_params=xgb_params
        )

        models_by_family[family] = model_bundle

    return models_by_family


#%%
train_file_path = r"C:\Users\guoya\Documents\Git_repo\Kaggle-learn\ML\store_sales\data\train.csv"
train_df = pd.read_csv(train_file_path, parse_dates=['date'])
families = train_df['family'].unique()

diagnostics_results = diagnostics.run_all_family_diagnostics(train_df, families)
diagnostics_results

#%%
models_by_family = train_all_families(
    train_df, diagnostic_results=diagnostics_results)
joblib.dump(models_by_family, os.path.join(folder_path, 'family_models.joblib'))