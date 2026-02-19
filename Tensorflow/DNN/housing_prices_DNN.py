#%%
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras

from keras import layers, callbacks
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import mutual_info_regression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import ParameterSampler


random_seed = 42

def build_model(layer_1st, layer_2nd, dropout):
    optimizer = keras.optimizers.Adam(
        learning_rate=3e-4
    )
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Dense(layer_1st, activation='relu'),
        layers.Dropout(dropout),
        layers.Dense(layer_2nd, activation='relu'),
        layers.Dropout(dropout),
        layers.Dense(1)
    ])
    model.compile(
        optimizer=optimizer,
        loss='mae'
    )
    return model


def make_mi_scores(train_df):
    df = train_df.dropna(axis=0).copy()

    X = df.drop('SalePrice', axis=1)
    cat_cols = X.select_dtypes("object").columns
    for col in cat_cols:
        X[col], _ = X[col].factorize()
    y = df.SalePrice
    discrete_features = X.columns.isin(cat_cols)

    mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores


def plot_mi_scores(scores):
    plt.figure(dpi=100, figsize=(5, len(scores) * 0.3))
    scores = scores.sort_values(ascending=True)
    width = np.arange(len(scores))
    ticks = list(scores.index)
    plt.barh(width, scores)
    plt.yticks(width, ticks)
    plt.title("Mutual Information Scores")
    plt.tight_layout()
    plt.show()


#%% Replaced nan values by 'not available'
train_data_path = r"C:\Users\guoya\Documents\Git_repo\Kaggle-learn\DL\DNN\data\housing_prices_train.csv"
train_df = pd.read_csv(train_data_path)

test_data_path = r"C:\Users\guoya\Documents\Git_repo\Kaggle-learn\DL\DNN\data\housing_prices_test.csv"
test_df = pd.read_csv(test_data_path)

columns_to_drop = ['MiscFeature', 'Fence', 'MasVnrType', 'LotFrontage']
train_df = train_df.drop(columns=columns_to_drop)
test_df = test_df.drop(columns=columns_to_drop)

imputer = SimpleImputer(
    missing_values=np.nan,
    strategy='constant',
    fill_value='Not Available'
)
columns_to_fill = [
    'PoolQC', 'Alley', 'FireplaceQu',
    'GarageFinish', 'GarageType', 'GarageQual', 'GarageCond',
    'BsmtFinType1', 'BsmtFinType2', 'BsmtExposure', 'BsmtCond', 'BsmtQual'
]
train_df[columns_to_fill] = imputer.fit_transform(train_df[columns_to_fill])
test_df[columns_to_fill] = imputer.transform(test_df[columns_to_fill])

imputer = SimpleImputer(
    missing_values=np.nan,
    strategy='constant',
    fill_value=0
)
columns_to_fill = ['GarageYrBlt']
train_df[columns_to_fill] = imputer.fit_transform(train_df[columns_to_fill])
test_df[columns_to_fill] = imputer.transform(test_df[columns_to_fill])

train_df = train_df.dropna(axis=0, subset=['SalePrice'])

#%% Choose the top ten features
mi_scores = make_mi_scores(train_df=train_df)
feature_columns = mi_scores.index[:20]
feature_columns
#%%
X = train_df[feature_columns]
y = train_df.SalePrice

X_train, X_val, y_train, y_val = train_test_split(
    X, y, train_size=0.7, random_state=random_seed
)

# log transform
y_train_log = np.log1p(y_train)
y_val_log = np.log1p(y_val)

y_mean = y_train_log.mean()
y_std = y_train_log.std()

y_train_scaled = (y_train_log - y_mean) / y_std
y_val_scaled = (y_val_log - y_mean) / y_std

numerical_cols = [cname for cname in X.columns if X[cname].dtype in ['int64', 'float64']]
ordinal_cols = ['KitchenQual', 'BsmtQual', 'ExterQual']
nominal_cols = ['Neighborhood']

#%% Preprocessing pipeline
numerical_transformer = Pipeline(
    steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
        ('std', StandardScaler())
    ]
)
nominal_transformer = Pipeline(
    steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ]
)
grades = ['Not Available', 'Po', 'Fa', 'TA', 'Gd', 'Ex']
ordinal_transformer = Pipeline(
    steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='Not Available')),
        ('ordinal', OrdinalEncoder(categories=[grades] * len(ordinal_cols))),
        ('std', StandardScaler())
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('nom', nominal_transformer, nominal_cols),
        ('ord', ordinal_transformer, ordinal_cols)
    ]
)

X_train = preprocessor.fit_transform(X_train)
X_val = preprocessor.transform(X_val)

input_shape = [X_train.shape[1]]
print("Input shape: {}".format(input_shape))

#%%
early_stopping = callbacks.EarlyStopping(
    min_delta=0.001,
    patience=15,
    restore_best_weights=True
)
reduce_lr = callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5
)
params_dist = {
    'layer_1st': [32, 64, 128],
    'layer_2nd': [32, 64, 128],
    'dropout': [0.0, 0.05, 0.1]
}

best_model = None
best_params = None
best_mae = float('inf')
for params in ParameterSampler(params_dist, n_iter=20, random_state=random_seed):
    model = build_model(**params)

    history = model.fit(
        X_train, y_train_scaled,
        validation_data=(X_val, y_val_scaled),
        batch_size=32,
        epochs=500,
        callbacks=[early_stopping, reduce_lr],
        verbose=False
    )
    history_df = pd.DataFrame(history.history)
    history_df.loc[:, ['loss', 'val_loss']].plot()
    print("Minimum Validation Loss: {:0.4f}".format(history_df['val_loss'].min()))


    y_val_pred_scaled = model.predict(X_val)
    y_val_pred = np.expm1(y_val_pred_scaled * y_std + y_mean)
    mae = mean_absolute_error(y_val, y_val_pred)
    if mae < best_mae:
        best_mae = mae
        best_params = params

print(f'Best params: {best_params}')
print(f'Best MAE: {best_mae}')

#%% Retrain using full data
model = build_model(**best_params)
X = preprocessor.fit_transform(X)
y_log = np.log1p(y)
mean, std = y_log.mean(), y_log.std()
y_log_scaled = (y_log - mean) / std

history = model.fit(
    X, y_log_scaled,
    validation_split=0.1,
    batch_size=32,
    epochs=500,
    callbacks=[early_stopping, reduce_lr],
    verbose=False
)
history_df = pd.DataFrame(history.history)
history_df.loc[:, ['loss', 'val_loss']].plot()
print("Minimum Validation Loss: {:0.4f}".format(history_df['val_loss'].min()))

X_test = test_df[feature_columns]
X_test = preprocessor.transform(X_test)
test_preds = model.predict(X_test)
test_preds = np.expm1(test_preds * std + mean)
data = {'Id': test_df.Id, 'SalePrice': test_preds[:, 0]}
df_to_save = pd.DataFrame(data)
outfile = './data/housing_prices_prediction_NN.csv'
print(f'Saving predictions to {outfile}')
df_to_save.to_csv(outfile, sep=',', index=False)
