#%%
import sys
import os
sys.path.insert(0, '../ML')

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.tree import plot_tree
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from sklearn.base import BaseEstimator, TransformerMixin
from category_encoders import MEstimateEncoder

import utils

random_state = 42
TARGET_ENCODING = True

class TargetEncoderWrapper(BaseEstimator, TransformerMixin):
    def __init__(self, m=5.0):
        self.m = m
        self.encoder = MEstimateEncoder(m=m)

    def fit(self, X, y):
        self.encoder.fit(X, y)
        return self

    def transform(self, X):
        return self.encoder.transform(X)


def create_area_features(df):
    X_out = pd.DataFrame()
    X_out['LivLotRatio'] = df['GrLivArea'] / df['LotArea']
    X_out['Spaciousness'] = (df['1stFlrSF'] + df['2ndFlrSF']) / df['TotRmsAbvGrd']
    X_out['TotalOutsideSF'] = df['OpenPorchSF'] + df['EnclosedPorch'] + \
        df['3SsnPorch'] + df['ScreenPorch'] + df['WoodDeckSF']
    return X_out


def create_num_cat_interaction_feature(df, cat_col, num_col):
    X_out = pd.get_dummies(df[cat_col], prefix=cat_col)
    X_out = X_out.mul(df[num_col], axis=0)
    return X_out


def create_count_features(df, cat_cols, new_col_name):
    X_out = pd.DataFrame()
    X_out[new_col_name] = df[cat_cols].gt(0).sum(axis=1)
    return X_out


def create_cat_num_grouped_feature(df, cat_col, num_col, transform, new_col_name):
    X_out = pd.DataFrame()
    X_out[new_col_name] = df.groupby(cat_col)[num_col].transform(transform)
    return X_out


#%%
train_data_path = r"../housing_prices/data/housing_prices_train_cleaned.csv"
df = pd.read_csv(train_data_path)

# Remove columns with too many missing values
missing_val_count_by_column = df.isnull().sum()
cols_to_keep = missing_val_count_by_column[missing_val_count_by_column < 100]
train_df = df[cols_to_keep.index].copy()

#%%
corr = utils.select_high_corr_features(train_df, lower_bound=0.0, print_corr=False)
numerical_features = list(corr.index)

object_cols = train_df.select_dtypes(include=['object']).columns
low_cardinality_features = [col for col in object_cols if train_df[col].nunique() < 10]
ordinal_features = [
    'KitchenQual', 'ExterQual', 'ExterCond', 'HeatingQC',
    'BsmtQual', 'BsmtCond', 'FireplaceQu', 'GarageQual',
    'GarageCond', 'PoolQC']
nominal_features = list(set(low_cardinality_features) - set(ordinal_features))
target_encoded_features = ['Neighborhood']
grading_categories = ['Missing', 'Po', 'Fa', 'TA', 'Gd', 'Ex']
categories = [grading_categories for _ in ordinal_features]
features = numerical_features + nominal_features + ordinal_features + target_encoded_features

#%%
X = train_df[features]
y = train_df.SalePrice

numeric_transformer = SimpleImputer(strategy='constant', fill_value=0)
nominal_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])
ordinal_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
    ('ordinal', OrdinalEncoder(categories=categories))
])
target_transformer = TargetEncoderWrapper(m=5)

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numerical_features),
    ('nom', nominal_transformer, nominal_features),
    ('ord', ordinal_transformer, ordinal_features),
    ('tgt', target_transformer, target_encoded_features),

])

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', XGBRegressor(
        objective='reg:squarederror',
        random_state=42,
        learning_rate=0.01,
        subsample=0.7,
        colsample_bytree=0.7,
        n_jobs=-1
    ))
])

grid_search = GridSearchCV(
    pipeline,
    param_grid={
        'model__n_estimators': [3000, 4000],
        'model__max_depth': [4, 5, 6],
        'model__min_child_weight': [1, 3, 5],
        'model__gamma': [0, 0.1, 0.3],
    },
    scoring='neg_mean_absolute_error',
    cv=5,
    n_jobs=-1
)
grid_search.fit(X, y)
print(f'Best parameters: {grid_search.best_params_}')
print(f'Best CV score: {-grid_search.best_score_}')

pipeline = grid_search.best_estimator_
pipeline.fit(X, y)

test_data_path = '../housing_prices/data/housing_prices_test_cleaned.csv'
test_df = pd.read_csv(test_data_path)

X_test = test_df[features]
test_preds = pipeline.predict(X_test)

data = {'Id': test_df.Id, 'SalePrice': test_preds}
df_to_save = pd.DataFrame(data)
outfile = '../housing_prices/data/housing_prices_prediction.csv'
print(f'Saving predictions to {outfile}')
df_to_save.to_csv(outfile, sep=',', index=False)

# %%
