#%%
import sys
import os
sys.path.insert(0, '../ML')

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.tree import plot_tree
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

import utils
random_state = 1


#%%
train_data_path = '../ML/housing_prices_train.csv'
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

params_grid = {
    'model__n_estimators': [500, 700, 900],
    'model__learning_rate': [0.01, 0.05, 0.1],
    'model__max_depth': [3, 4, 5],
    'model__subsample': [0.6, 0.7, 0.8],
    'model__colsample_bytree': [0.6, 0.7, 0.8]
}

xgb = XGBRegressor(
    random_state=random_state,
    objective='reg:squarederror'
)
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', xgb)
])

grid_search = GridSearchCV(
    pipeline,
    param_grid=params_grid,
    scoring='neg_mean_absolute_error',
    cv=5,
    n_jobs=-1,
    verbose=False
)
grid_search.fit(X, y)
print(f'Best parameters: {grid_search.best_params_}')
print(f'Best CV score: {-grid_search.best_score_}')

pipeline = grid_search.best_estimator_

pipeline.fit(X, y)
preds = pipeline.predict(X)
print(f'Training MAE: {mean_absolute_error(y, preds)}')

# test_data_path = '../ML/housing_prices_test.csv'
# test_df = pd.read_csv(test_data_path)

# X_test = test_df[features]

# test_preds = pipeline.predict(X_test)

# data = {'Id': test_df.Id, 'SalePrice': test_preds}
# df_to_save = pd.DataFrame(data)
# outfile = '../ML/housing_prices_prediction.csv'
# print(f'Saving predictions to {outfile}')
# df_to_save.to_csv(outfile, sep=',', index=False)

