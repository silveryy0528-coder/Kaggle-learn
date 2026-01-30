#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from statsmodels.graphics.tsaplots import plot_acf


#%%
train_file_path = r"C:\Users\guoya\Documents\Git_repo\Kaggle-learn\ML\store_sales\data\train.csv"
train_df = pd.read_csv(train_file_path, parse_dates=['date'])

family = 'GROCERY I'

df_fam = train_df[train_df['family'] == family].copy()
df_fam = df_fam.sort_values(['store_nbr', 'date'])
df_fam['store_id'] = df_fam['store_nbr'].astype(str)

#%% Feature engineering
unique_dates = df_fam['date'].sort_values().unique()
date_to_idx = {date: i for i, date in enumerate(unique_dates)}
df_fam['time_idx'] = df_fam['date'].map(date_to_idx)
df_fam['month'] = df_fam['date'].dt.month
df_fam['dayofweek'] = df_fam['date'].dt.dayofweek
df_fam['year'] = df_fam['date'].dt.year

lags = [1, 7]
for lag in lags:
    df_fam[f'lag_{lag}'] = (
        df_fam
        .groupby('store_nbr')['sales']
        .shift(lag))

rolls = [7, 14]
for roll in rolls:
    df_fam[f'roll_{roll}'] = (
        df_fam
        .groupby('store_nbr')['sales']
        .shift(1)
        .rolling(window=roll)
        .mean())

#%%
lr_features = ['time_idx', 'dayofweek', 'month', 'store_id']
xgb_features = ['lag_1', 'lag_7', 'store_id', 'dayofweek', 'month', 'year']

X_lr = df_fam[lr_features]
X_xgb = df_fam[xgb_features]

X_lr = pd.get_dummies(
    X_lr, columns=['dayofweek', 'month', 'store_id'], drop_first=True)
X_xgb = pd.get_dummies(
    X_xgb, columns=['store_id', 'dayofweek', 'month'])
y = df_fam['sales']

store_dummies = pd.get_dummies(df_fam['store_id'], prefix='store', drop_first=True)
for c in store_dummies.columns:
    X_lr[f'{c}_time'] = store_dummies[c] * df_fam['time_idx']

valid_idx = X_xgb.dropna().index
X_lr = X_lr.loc[valid_idx]
X_xgb = X_xgb.loc[valid_idx]
y = y.loc[valid_idx]
df_fam = df_fam.loc[valid_idx]

#%% Train/Test split
split_date = '2016-01-01'
train_mask = df_fam['date'] < split_date
test_mask  = df_fam['date'] >= split_date

# LR target and train
X_train_lr, X_test_lr = X_lr[train_mask], X_lr[test_mask]
y_train_lr, y_test_lr = y[train_mask], y[test_mask]

lr = Ridge(alpha=4.0)
lr.fit(X_train_lr, y_train_lr)

y_train_pred_lr = lr.predict(X_train_lr)
y_test_pred_lr = lr.predict(X_test_lr)

df_fam['lr_pred'] = lr.predict(X_lr)
bias = (y_train_lr - y_train_pred_lr).mean()

df_fam['lr_pred_corr'] = df_fam['lr_pred'] + bias
df_fam['lr_residual'] = df_fam['sales'] - df_fam['lr_pred_corr']

#%% Extra features for XGB
# store_dummies = pd.get_dummies(df_fam['store_id'], prefix='store', drop_first=False)
# time_idx_store_cols = {f"time_idx_{col}": df_fam['time_idx'].values * store_dummies[col].values
#                        for col in store_dummies.columns}
# time_idx_store_df = pd.DataFrame(time_idx_store_cols, index=df_fam.index)

# lag_cols = ['lag_1', 'lag_7']
# lag_store_cols = {}
# for lag in lag_cols:
#     for col in store_dummies.columns:
#         lag_store_cols[f"{lag}_{col}"] = df_fam[lag].values * store_dummies[col].values

# lag_store_df = pd.DataFrame(lag_store_cols, index=df_fam.index)

# residual_store_cols = {f"resid_{col}": df_fam['lr_residual'].values * store_dummies[col].values
#                        for col in store_dummies.columns}

# resid_store_df = pd.DataFrame(residual_store_cols, index=df_fam.index)
# X_xgb = pd.concat([X_xgb, time_idx_store_df], axis=1)

#%%
X_train_xgb, X_test_xgb = X_xgb[train_mask], X_xgb[test_mask]
y_train_xgb = df_fam.loc[train_mask, 'lr_residual']

model = xgb.XGBRegressor(
    n_estimators=500,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.7,
    colsample_bytree=0.7,
    reg_alpha=0.,
    reg_lambda=1.,
    random_state=42
)
model.fit(X_train_xgb, y_train_xgb)

y_train_pred_xgb = model.predict(X_train_xgb)
y_test_pred_xgb  = model.predict(X_test_xgb)

#%% Hybrid forecast
lr_train_pred = df_fam.loc[train_mask, 'lr_pred_corr']
lr_test_pred = df_fam.loc[test_mask, 'lr_pred_corr']

hybrid_train = lr_train_pred.values + y_train_pred_xgb
hybrid_test  = lr_test_pred.values  + y_test_pred_xgb

hybrid_pred_full = np.full(len(df_fam), np.nan)
hybrid_pred_full[train_mask] = hybrid_train
hybrid_pred_full[test_mask]  = hybrid_test

#%%
n_rows, n_cols = 3, 3
num_stores = n_rows * n_cols
stores_to_check = sorted([1, 8, 14, 16, 23, 27, 7, 40, 42])

fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 12), sharex=True)

for i, store in enumerate(stores_to_check):
    ax = axes[i // n_cols, i % n_cols]

    # Mask for this store
    store_mask = df_fam['store_nbr'] == store
    df_store = df_fam[store_mask]

    # Mask for train/test inside this store
    hybrid_store = hybrid_pred_full[store_mask.values]

    ax.plot(df_store['date'], df_store['sales'], label='Actual', alpha=0.5)
    ax.plot(df_store['date'], df_store['lr_pred_corr'], label='LR Forecast', alpha=0.6)
    ax.plot(df_store['date'], hybrid_store, label='Hybrid Forecast', alpha=0.8)

    ax.set_title(f'Store {store}')
    ax.axvline(pd.to_datetime(split_date), color='black', linestyle='--', lw=1)
    if i % n_cols == 0:
        ax.set_ylabel('Sales')
    if i // n_cols == n_rows - 1:
        ax.set_xlabel('Date')

handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right')
plt.suptitle(f'Actual vs Hybrid Forecast: {family} (All Stores)', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.draw()

#%%
stats = []
for store in stores_to_check:
    store_mask = (df_fam['store_nbr'] == store) & (df_fam['date'] >= split_date)
    y_true = df_fam.loc[store_mask, 'sales'].values
    y_pred = hybrid_pred_full[store_mask.values]

    mae = mean_absolute_error(y_true, y_pred)
    smape_val = 100 * np.mean(
        2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + 1e-8)
    )
    stats.append((store, mae, smape_val))

stats_df = pd.DataFrame(stats, columns=['store_nbr', 'MAE', 'sMAPE'])
print(stats_df)
print("Overall MAE:", stats_df['MAE'].mean())
print("Overall sMAPE:", stats_df['sMAPE'].mean())
plt.show()
