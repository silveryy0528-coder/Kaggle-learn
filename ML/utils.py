import copy
import pandas as pd


def select_high_corr_features(df, target='SalePrice', lower_bound=0.25, print_corr=False):
    corr = df.corr(numeric_only=True)[target].sort_values(ascending=False)
    corr = corr[abs(corr) > lower_bound].iloc[1:]
    if print_corr:
        print(f'#### {len(corr)} Numerical features with high correlation ####\n{corr}')
    return corr


def find_feature_importance(model, corr):
    data = {
        'corr': corr.values,
        'importance': model.feature_importances_
    }
    importances = pd.DataFrame(data=data, index=corr.index)
    print(f'#### Importance on selected features ####\n{\
        importances.sort_values(by='importance', ascending=False)}')


def combine_numeric_features(df_in, source_features, target_name):
    df_out = copy.deepcopy(df_in)
    df_out[target_name] = 0
    for f in source_features:
        df_out[target_name] += df_in[f]
    df_out.drop(columns=source_features, inplace=True)
    return df_out