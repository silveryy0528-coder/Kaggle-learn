#%%
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# %%
train_file_path = r"C:\Users\guoya\Documents\Git_repo\Kaggle-learn\ML\store_sales\data\train.csv"
train_df = pd.read_csv(train_file_path, parse_dates=['date']) # explict about which column has date
print(f'Products families: {train_df.family.unique()}')

#%%
store_nbr = 1
family = 'GROCERY I'

df_s = train_df[
    (train_df['store_nbr'] == store_nbr)
    & (train_df['family'] == family)
].copy()
df_s = df_s.sort_values('date')

df_s['year'] = df_s['date'].dt.year
df_s['month'] = df_s['date'].dt.month
df_s['weekofyear'] = df_s['date'].dt.isocalendar().week.astype(int)
df_s['dayofweek'] = df_s['date'].dt.dayofweek
df_s['time_idx'] = np.arange(len(df_s))

#%%
X_lr = df_s[['time_idx', 'dayofweek', 'month']]
X_lr = pd.get_dummies(X_lr, columns=['dayofweek', 'month'], drop_first=True)
y = df_s['sales']

#%%
lags = [1, 7, 14]
for lag in lags:
    df_s[f'lag_{lag}'] = df_s['sales'].shift(lag)

rolls = [7, 14]
for roll in rolls:
    df_s[f'roll_{roll}'] = df_s['sales'].shift(1).rolling(window=roll).mean()

X_xgb = df_s[['lag_1', 'lag_7', 'lag_14', 'roll_7', 'roll_14', 'dayofweek', 'month']]
X_xgb = pd.get_dummies(X_xgb, columns=['dayofweek', 'month'], drop_first=False)

#%%
valid_idx = X_xgb.dropna().index
X_lr = X_lr.loc[valid_idx]
X_xgb = X_xgb.loc[valid_idx]
y = y.loc[valid_idx]

#%%
split_date = '2016-01-01'

train_idx = df_s.loc[X_lr.index, 'date'] < split_date
test_idx = df_s.loc[X_lr.index, 'date'] >= split_date

X_train, X_test = X_lr[train_idx], X_lr[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

lr = LinearRegression()
lr.fit(X_train, y_train)

y_train_pred = lr.predict(X_train)
y_test_pred = lr.predict(X_test)

df_s["residual"] = df_s["sales"] - df_s["lr_pred"]


#%%
plt.figure(figsize=(14, 5))
plt.plot(df_s.loc[X_lr.index, "date"], y, label="Actual", alpha=0.5)
plt.plot(df_s.loc[X_train.index, "date"], y_train_pred, label="LR Train Fit")
plt.plot(df_s.loc[X_test.index, "date"], y_test_pred, label="LR Forecast")

plt.axvline(pd.to_datetime(split_date), color="black", linestyle="--", label="Train/Test Split")
plt.legend()
plt.title("Linear Regression: Trend + Seasonality")
plt.show()

#%%
coef = pd.Series(lr.coef_, index=X_lr.columns)
coef.sort_values().head(10), coef.sort_values().tail(10)
print("Train MAE:", mean_absolute_error(y_train, y_train_pred))
print("Test  MAE:", mean_absolute_error(y_test, y_test_pred))

#%%
# ----- residual analysis -----


plt.figure(figsize=(14, 4))
plt.plot(df_s.loc[X_lr.index, "date"], df_s.loc[X_lr.index, "residual"])
plt.axhline(0, color="black", linewidth=1)
plt.title(f"LR Residuals over Time (Store {store_nbr}, {family})")
plt.show()

df_s["residual"].hist(bins=50)
plt.title("Residual Distribution")
plt.show()

from statsmodels.graphics.tsaplots import plot_acf

plot_acf(df_s["residual"].dropna(), lags=30)
plt.show()

df_s.groupby('month')["residual"].mean().plot(kind="bar")
plt.title("Mean Residual by Day of Week")
plt.show()

mask = (df_s["date"] > "2016-04-01") & (df_s["date"] < "2016-05-01")

plt.figure(figsize=(12, 4))
plt.plot(df_s.loc[mask, "date"], df_s.loc[mask, "residual"])
plt.axhline(0, color="black")
plt.title("Residuals around April 2016")
plt.show()

#%%
valid_idx = X_xgb.index
X_train_xgb = X_xgb.loc[valid_idx]
y_train_xgb = df_s.loc[valid_idx, 'residual']

split_date = '2016-01-01'

train_mask = (df_s["date"] < split_date) & df_s.index.isin(valid_idx)
test_mask  = (df_s["date"] >= split_date) & df_s.index.isin(valid_idx)

# LR predictions for the aligned rows
lr_train_pred = df_s.loc[train_mask, "lr_pred"]
lr_test_pred  = df_s.loc[test_mask, "lr_pred"]

# XGB features and target for aligned rows
X_train = X_xgb.loc[train_mask]
X_test  = X_xgb.loc[test_mask]
y_train = df_s.loc[train_mask, "residual"]
y_test  = df_s.loc[test_mask, "residual"]

import xgboost as xgb

xgb_model = xgb.XGBRegressor(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

xgb_model.fit(X_train, y_train)
y_train_pred_xgb = xgb_model.predict(X_train)
y_test_pred_xgb = xgb_model.predict(X_test)

#%%
# Hybrid forecast
hybrid_train = lr_train_pred + y_train_pred_xgb
hybrid_test  = lr_test_pred + y_test_pred_xgb

plt.figure(figsize=(14,5))
plt.plot(df_s.loc[valid_idx, "date"], df_s.loc[valid_idx, "sales"], label="Actual", alpha=0.5)
plt.plot(df_s.loc[train_mask, "date"], hybrid_train, label="Hybrid Train")
plt.plot(df_s.loc[test_mask, "date"], hybrid_test, label="Hybrid Test")
plt.axvline(pd.to_datetime(split_date), color="black", linestyle="--", label="Train/Test Split")
plt.legend()
plt.title("Hybrid Forecast: LR + XGB Residuals")
plt.show()
