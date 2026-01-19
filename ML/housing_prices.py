#%%
import copy
import sys
import os
sys.path.insert(0, '../ML')

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
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
categorical_features = [
    'Neighborhood',
    # 'MSZoning',
    # 'BldgType',
    # 'HouseStyle',
    'KitchenQual',
    'ExterQual',
    # 'Foundation',
    # 'HeatingQC'
]

features = copy.deepcopy(list(numerical_features))
features.extend(categorical_features)

# print(f'#### Numeric features to use ####\n{train_df[features].columns}')
X = train_df[features]
y = train_df.SalePrice

#%%
X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=random_state)

numeric_transformer = SimpleImputer(strategy='median')
categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numerical_features),
    ('cat', categorical_transformer, categorical_features)
])

rf_model = RandomForestRegressor(
    n_estimators=600,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=5,
    max_features=0.7,
    random_state=random_state,
    n_jobs=-1)

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', rf_model)
])

#%%
pipeline.fit(X_train, y_train)

rf_train_preds = pipeline.predict(X_train)
rf_val_preds = pipeline.predict(X_val)

print("Train MAE:", mean_absolute_error(y_train, rf_train_preds))
print("Valid MAE:", mean_absolute_error(y_val, rf_val_preds))

scores = cross_val_score(pipeline, X, y, cv=5, scoring='neg_mean_absolute_error')
print(f"CV score: {-scores.mean()}")

#%%
pipeline.fit(X, y)

test_data_path = '../ML/housing_prices_test.csv'
test_df = pd.read_csv(test_data_path)

X_test = test_df[features]

test_preds = pipeline.predict(X_test)

data = {'Id': test_df.Id, 'SalePrice': test_preds}
df_to_save = pd.DataFrame(data)
outfile = '../ML/housing_prices_prediction.csv'
df_to_save.to_csv(outfile, sep=',', index=False)

