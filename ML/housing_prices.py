#%%
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import plot_tree
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

random_state = 1

def select_high_corr_features(df, target='SalePrice', lower_bound=0.25, print_corr=False):
    corr = df.corr(numeric_only=True)[target].sort_values(ascending=False)
    corr = corr[abs(corr) > lower_bound].iloc[1:]
    if print_corr:
        print(f'#### Numerical features with high correlation ####\n{corr}')
    return corr


def find_feature_importance(model, corr):
    data = {
        'corr': corr.values,
        'importance': model.feature_importances_
    }
    importances = pd.DataFrame(data=data, index=corr.index)
    print(f'#### Importance on selected features ####\n{\
        importances.sort_values(by='importance', ascending=False)}')


#%%
train_data_path = '../ML/housing_prices_train.csv'
train_df = pd.read_csv(train_data_path)

corr = select_high_corr_features(train_df, lower_bound=0.25)
features = [
    'OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea', 'TotalBsmtSF',
    '1stFlrSF', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt', 'YearRemodAdd',
    'GarageYrBlt', 'MasVnrArea', 'Fireplaces', 'BsmtFinSF1', 'LotFrontage',
    'WoodDeckSF', '2ndFlrSF', 'OpenPorchSF', 'LotArea'
]

print(f'#### Numeric features to use ####\n{train_df[features].columns}')
X = train_df[features]
y = train_df.SalePrice

#%%
X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=random_state)

preprocessor = SimpleImputer(strategy='median')
rf_model = RandomForestRegressor(random_state=random_state)

pipeline = Pipeline([
    ('imputer', preprocessor),
    ('model', rf_model)
])

#%%
pipeline.fit(X_train, y_train)

rf_val_predictions = pipeline.predict(X_val)
rf_val_mae = mean_absolute_error(rf_val_predictions, y_val)

print("Validation MAE for Random Forest Model: {:,.0f}".format(rf_val_mae))
# find_feature_importance(pipeline['model'], corr[features])


# #%%
# rf_model_on_full_data = RandomForestRegressor(random_state=random_state, max_depth=3)
# rf_model_on_full_data.fit(X, y)

# test_data_path = '../ML/housing_prices_test.csv'
# test_df = pd.read_csv(test_data_path)

# #%%
# X_test = test_df[features]
# y_test = test_df.SalePrice

# test_preds = rf_model_on_full_data.predict(X_test)
# print(test_preds[:5])
# rf_test_mae = mean_absolute_error(test_preds, y_test)

# print("Test MAE for Random Forest Model: {:,.0f}".format(rf_test_mae))


