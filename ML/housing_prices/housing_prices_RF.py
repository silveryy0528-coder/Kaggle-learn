#%%
import copy
import sys
import os
print(os.getcwd())
sys.path.insert(0, '../ML')

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.tree import plot_tree
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

import utils
random_state = 1

#%%
train_data_path = '../housing_prices/data/housing_prices_train.csv'
df = pd.read_csv(train_data_path)

# Remove columns with too many missing values
missing_val_count_by_column = df.isnull().sum()
cols_to_keep = missing_val_count_by_column[missing_val_count_by_column < 100]
train_df = df[cols_to_keep.index].copy()

#%%
corr = utils.select_high_corr_features(train_df, lower_bound=0.0, print_corr=False)
numerical_features = corr.index

object_cols = train_df.select_dtypes(include=['object']).columns
categorical_features = [col for col in object_cols if train_df[col].nunique() < 10]
ordinal_features = ['KitchenQual', 'ExterQual', 'ExterCond', 'HeatingQC']
nominal_features = list(set(categorical_features) - set(ordinal_features))
grading_categories = ['Po', 'Fa', 'TA', 'Gd', 'Ex']
categories = [
    grading_categories,
    grading_categories,
    grading_categories,
    grading_categories
]
features = list(numerical_features) + categorical_features

#%%
X = train_df[features]
y = train_df.SalePrice

numeric_transformer = SimpleImputer(strategy='median')
nominal_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])
ordinal_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('ordinal', OrdinalEncoder(categories=categories))
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numerical_features),
    ('nom', nominal_transformer, nominal_features),
    ('ord', ordinal_transformer, ordinal_features)
])

param_grid = {
    'model__n_estimators': [400, 600, 800],
    'model__min_samples_leaf': [3, 5, 7],
    'model__max_features': [0.5, 0.7, 0.9]
}

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(random_state=random_state))
])

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=5,
    scoring='neg_mean_absolute_error',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X, y)
print("Best CV MAE:", -grid_search.best_score_)
print("Best parameters:")
print(grid_search.best_params_)

#%%
pipeline = grid_search.best_estimator_
pipeline.fit(X, y)

test_data_path = '../housing_prices/data/housing_prices_test.csv'
test_df = pd.read_csv(test_data_path)

X_test = test_df[features]

test_preds = pipeline.predict(X_test)

data = {'Id': test_df.Id, 'SalePrice': test_preds}
df_to_save = pd.DataFrame(data)
outfile = '../housing_prices/data/housing_prices_prediction_RF.csv'
print(f'Saving predictions to {outfile}')
df_to_save.to_csv(outfile, sep=',', index=False)

