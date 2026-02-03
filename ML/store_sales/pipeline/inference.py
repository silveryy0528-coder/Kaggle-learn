#%%
import os
import sys
import joblib
folder_path = r'C:\Users\guoya\Documents\Git_repo\Kaggle-learn\ML\store_sales\pipeline'
sys.path.insert(0, folder_path)

import pandas as pd
import numpy as np

import utilities as utils


def predict_single_family(test_df, model_bundle):
    family = model_bundle['family']
    policy = model_bundle['policy']
    lr_model = model_bundle['lr_model']
    xgb_model = model_bundle['xgb_model']
    bias = model_bundle['bias']
    meta = model_bundle['meta']

    df_fam = test_df[test_df['family'] == family].copy()

    df_fam, X_lr, X_xgb, _, _ = utils.build_features(df_fam, meta=meta)

    # Stage 1 - LR
    lr_pred = lr_model.predict(X_lr) + bias

    if policy == 'lr':
        final_pred = lr_pred
    else:
        xgb_pred = xgb_model.predict(X_xgb)
        final_pred = lr_pred + xgb_pred

    out = df_fam[['id']].copy()
    out['sales'] = np.clip(final_pred, 0, None)

    return out


def predict_all_families(test_df, models_by_family):
    preds = []

    for family, model_bundle in models_by_family.items():
        print(f'Predicting family: {family}')
        fam_pred = predict_single_family(test_df, model_bundle)
        preds.append(fam_pred)

    submission = pd.concat(preds, axis=0)
    submission = submission.sort_values(by='id').reset_index(drop=True)

    return submission


#%%
test_file_path = r"C:\Users\guoya\Documents\Git_repo\Kaggle-learn\ML\store_sales\data\test.csv"
test_df = pd.read_csv(test_file_path, parse_dates=['date'])

models_by_family = joblib.load(
    os.path.join(folder_path, 'family_models.joblib')
)

submission = predict_all_families(test_df, models_by_family)

submission.to_csv(
    os.path.join(folder_path, 'submission.csv'),
    index=False
)
print("Submission saved.")

assert submission.shape[0] == test_df.shape[0]
assert submission['sales'].isna().sum() == 0
assert (submission['sales'] >= 0).all()