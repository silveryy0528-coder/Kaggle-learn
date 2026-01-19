#%%
import copy
import sys
sys.path.insert(0, '../ML')

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import plot_tree
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

import utils
random_state = 1


#%%
train_data_path = '../ML/housing_prices_train.csv'
df = pd.read_csv(train_data_path)
train_df = copy.deepcopy(df)

corr = utils.select_high_corr_features(train_df, lower_bound=0.25, print_corr=False)
numerical_features = corr.index

# print(f'#### Numeric features to use ####\n{train_df[features].columns}')
X = train_df[numerical_features]
y = train_df.SalePrice

#%%
X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=random_state)

preprocessor = SimpleImputer(strategy='median')
rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=None,
    min_samples_split=2,
    random_state=random_state,
    n_jobs=-1)

pipeline = Pipeline([
    ('imputer', preprocessor),
    ('model', rf_model)
])

#%%
pipeline.fit(X_train, y_train)

rf_train_preds = pipeline.predict(X_train)
rf_val_preds = pipeline.predict(X_val)

print("Train MAE:", mean_absolute_error(y_train, rf_train_preds))
print("Valid MAE:", mean_absolute_error(y_val, rf_val_preds))

scores = cross_val_score(pipeline, X, y, cv=5, scoring='neg_mean_absolute_error')
print(f'CV score: {-1 * np.mean(scores)}')

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


