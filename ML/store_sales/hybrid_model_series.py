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
train_df = pd.read_csv(train_file_path, parse_dates=['date']) # explict about which column has date

store_nbr = 1
family = 'DELI'

df_s = train_df[(train_df['store_nbr'] == store_nbr) & (train_df['family'] == family)].copy()
df_s = df_s.sort_values('date')

#%% Feature engineering
df_s['year'] = df_s['date'].dt.year
df_s['month'] = df_s['date'].dt.month
df_s['weekofyear'] = df_s['date'].dt.isocalendar().week.astype(int)
df_s['dayofweek'] = df_s['date'].dt.dayofweek
df_s['time_idx'] = np.arange(len(df_s))

lags = [1, 7, 14]
for lag in lags:
    df_s[f'lag_{lag}'] = df_s['sales'].shift(lag)

rolls = [7, 14]
for roll in rolls:
    df_s[f'roll_{roll}'] = df_s['sales'].shift(1).rolling(window=roll).mean()

#%% Create LR and XGB features
lr_features = ['time_idx', 'dayofweek', 'month']
xgb_features = ['lag_1', 'lag_7', 'roll_7']

X_lr = df_s[lr_features]
X_xgb = df_s[xgb_features]
X_lr = pd.get_dummies(X_lr, columns=['dayofweek', 'month'], drop_first=True)
# X_xgb = pd.get_dummies(X_xgb, columns=['dayofweek', 'month'], drop_first=False)
y = df_s['sales']

valid_idx = X_xgb.dropna().index
X_lr = X_lr.loc[valid_idx]
X_xgb = X_xgb.loc[valid_idx]
y = y.loc[valid_idx]
df_s = df_s.loc[valid_idx]

#%% Train/Test split
split_date = '2016-01-01'
train_mask = df_s['date'] < split_date
test_mask  = df_s['date'] >= split_date

# LR target and train
X_train_lr, X_test_lr = X_lr[train_mask], X_lr[test_mask]
y_train_lr, y_test_lr = y[train_mask], y[test_mask]

lr = Ridge(alpha=1.0)
lr.fit(X_train_lr, y_train_lr)

y_train_pred_lr = lr.predict(X_train_lr)
y_test_pred_lr = lr.predict(X_test_lr)

df_s['lr_pred'] = lr.predict(X_lr)

bias = (y_train_lr - y_train_pred_lr).mean()
df_s['lr_pred_corr'] = df_s['lr_pred'] + bias
df_s['residual'] = df_s['sales'] - df_s['lr_pred_corr']

y_test_sales = df_s.loc[test_mask, "sales"]
smape_lr = np.mean(
    2 * np.abs(y_test_sales.values - y_test_pred_lr) /
    (np.abs(y_test_sales.values) + np.abs(y_test_pred_lr) + 1e-8)
) * 100

#%% Plot LR results
plt.figure(figsize=(14, 5))
plt.plot(df_s.loc[X_lr.index, "date"], y, label="Actual", alpha=0.5)
plt.plot(df_s.loc[X_train_lr.index, "date"], y_train_pred_lr, label="LR Train Fit")
plt.plot(df_s.loc[X_test_lr.index, "date"], y_test_pred_lr, label="LR Forecast")

plt.axvline(pd.to_datetime(split_date), color="black", linestyle="--", label="Train/Test Split")
plt.legend()
plt.title("Linear Regression: Trend + Seasonality")
plt.draw()

plot_acf(df_s["residual"].dropna(), lags=14)
plt.draw()

# coef = pd.Series(lr.coef_, index=X_lr.columns)
# coef.sort_values().head(10), coef.sort_values().tail(10)
# print("Train MAE:", mean_absolute_error(y_train_lr, y_train_pred_lr))
# print("Test  MAE:", mean_absolute_error(y_test_lr, y_test_pred_lr))

#%% Residual analysis
# plt.figure(figsize=(14, 4))
# plt.plot(df_s.loc[X_lr.index, "date"], df_s.loc[X_lr.index, "residual"])
# plt.axhline(0, color="black", linewidth=1)
# plt.title(f"LR Residuals over Time (Store {store_nbr}, {family})")
# plt.show()

# df_s["residual"].hist(bins=50)
# plt.title("Residual Distribution")
# plt.show()

# df_s.groupby('month')["residual"].mean().plot(kind="bar")
# plt.title("Mean Residual by Month")
# plt.show()

#%%
# XGB features and target for aligned rows
X_train_xgb, X_test_xgb = X_xgb[train_mask], X_xgb[test_mask]
y_train_xgb = df_s.loc[train_mask, 'residual']

xgb_model = xgb.XGBRegressor(
    n_estimators=200,
    max_depth=3,
    learning_rate=0.1,
    min_child_weight=20,
    subsample=0.6,
    colsample_bytree=0.8,
    random_state=42
)

xgb_model.fit(X_train_xgb, y_train_xgb)

y_train_pred_xgb = xgb_model.predict(X_train_xgb)
y_test_pred_xgb = xgb_model.predict(X_test_xgb)

#%%
# Hybrid forecast
lr_train_pred = df_s.loc[train_mask, 'lr_pred_corr']
lr_test_pred = df_s.loc[test_mask, 'lr_pred_corr']
hybrid_train = lr_train_pred.values + y_train_pred_xgb
hybrid_test  = lr_test_pred.values  + y_test_pred_xgb

plt.figure(figsize=(14, 5))
plt.plot(df_s['date'], df_s['sales'], label="Actual", alpha=0.5)
plt.plot(df_s.loc[train_mask, "date"], hybrid_train, label="Hybrid Train")
plt.plot(df_s.loc[test_mask, "date"], hybrid_test, label="Hybrid Test")
plt.axvline(pd.to_datetime(split_date), color="black", linestyle="--", label="Train/Test Split")
plt.legend()
plt.title("Hybrid Forecast: LR + XGB Residuals")
plt.show()

y_test_sales = df_s.loc[test_mask, "sales"]
smape_hybrid = np.mean(
    2 * np.abs(y_test_sales.values - hybrid_test) /
    (np.abs(y_test_sales.values) + np.abs(hybrid_test) + 1e-8)
) * 100

print("LR MAE:", mean_absolute_error(y_test_sales, lr_test_pred))
print("Hybrid MAE:", mean_absolute_error(y_test_sales, hybrid_test))

print("LR sMAPE:", smape_lr)
print("Hybrid sMAPE:", smape_hybrid)