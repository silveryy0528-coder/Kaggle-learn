#%%
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.graphics.tsaplots import plot_pacf


def family_sanity_check(train_df):
    results = []

    for fam, df_fam in train_df.groupby("family"):
        df_fam = df_fam.copy()
        sales = df_fam['sales']

        # total rows = store-days
        total_days = len(df_fam)

        # active store-days (sales > 0)
        active_days = (sales > 0).sum()

        # sales scale & variability
        mean_sales = df_fam.loc[sales > 0, "sales"].mean()
        min_sales = df_fam.loc[sales > 0, "sales"].min()
        max_sales = df_fam.loc[sales > 0, "sales"].max()
        std_sales  = df_fam.loc[sales > 0, "sales"].std()
        cv_sales   = std_sales / mean_sales if mean_sales > 0 else 0.0

        results.append({
            "family": fam,
            # "total_days": total_days,
            # "active_days": active_days,
            "activity_ratio": active_days / total_days,
            "mean_sales_active": mean_sales,
            "std_sales_active": std_sales,
            "cv_active": cv_sales,
            # "mean_sales_all": sales.mean(),
            # "min_sales_active": min_sales,
            # "max_sales_active": max_sales,
        })

    results = pd.DataFrame(results)
    mean_sales_active = results['mean_sales_active']
    flag_high_sales = mean_sales_active > (mean_sales_active.mean() + mean_sales_active.std())

    results['flag_sparsity'] = results['activity_ratio'] < 0.1
    results['flag_high_sales'] = flag_high_sales
    results['flag_high_volatility'] = results['cv_active'] > 1.0

    return results


# %%
train_file_path = r"C:\Users\guoya\Documents\Git_repo\Kaggle-learn\ML\store_sales\data\train.csv"
train_df = pd.read_csv(train_file_path, parse_dates=['date']) # explict about which column has date

sanity_df = family_sanity_check(train_df=train_df)
sanity_df

#%%
sanity_df.sort_values(by='mean_sales_active', ascending=False)

#%%
store_nbr = 40
families = ['GROCERY I', 'CLEANING']
family = families[0]

series = train_df[
    (train_df['store_nbr'] == store_nbr)
    & (train_df['family'] == family)
].copy()
series = series[['date', 'sales']]
series = series.sort_values(by='date')
series = series.set_index('date')
print(series.head())

plot_pacf(series, lags=12)
plt.show()

#%%
moving_average = series.rolling(
    window=14,
    center=True,
    min_periods=8
).mean()

plt.figure()
plt.plot(series.index, series['sales'], label='sales')
plt.plot(moving_average.index, moving_average['sales'], label='MA')
plt.title(f'Store {store_nbr}, {family} Sales')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.xticks(rotation=45)
plt.show()

fig = series.loc['2016-03':'2016-05'].plot(title='Zoomed view', marker='.')
plt.xticks(rotation=45)
plt.show()

#%%
df = series.copy()
df['date'] = df.index
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['weekofyear'] = df['date'].dt.isocalendar().week.astype(int)
df['dayofweek'] = df['date'].dt.dayofweek
df['time_idx'] = np.arange(len(df))

#%%
df.groupby(['dayofweek'])['sales'].mean().plot(
    figsize=(8, 5),
    kind='bar', title='Average sales by day of week')
plt.show()
df.groupby(['month'])['sales'].mean().plot(
    figsize=(8, 5),
    kind='bar', title='Average sales by month')
plt.show()

# #%%
# store_data = train_df[train_df['store_nbr'] == store_nbr]
# compare_df = store_data[store_data['family'].isin(families)]

# compare_df = compare_df[['date', 'family', 'sales']]
# compare_df = compare_df.sort_values(['family', 'date'])
# compare_df.head()

# #%%
# plt.figure(figsize=(12,5))
# for fam in families:
#     fam_data = compare_df[compare_df["family"] == fam]
#     plt.plot(fam_data["date"], fam_data["sales"], label=fam)

# plt.title(f"Store {store_nbr} — Sales comparison: {families[0]} vs {families[1]}")
# plt.xlabel("Date")
# plt.ylabel("Sales")
# plt.legend()
# plt.show()

# compare_df.groupby(["family", compare_df["date"].dt.dayofweek])["sales"].mean().unstack(0).plot(kind="bar", figsize=(10,4))
# plt.title("Average sales by day of week")
# plt.show()

# # Monthly comparison
# compare_df.groupby(["family", compare_df["date"].dt.month])["sales"].mean().unstack(0).plot(kind="bar", figsize=(10,4))
# plt.title("Average sales by month")
# plt.show()

# #%%
# df = make_lags(df, 'sales', lags=[1, 7, 14])
# df = make_rollings(df, 'sales', rolls=[7, 14])

# df = df.dropna()
# df.head()

# #%%
# train = pd.read_csv(train_file_path, parse_dates=['date'])
# train = train.drop(columns=['id', 'onpromotion'])

# #%%
# # Add time features
# train['time_idx'] = (train['date'] - train['date'].min()).dt.days
# train['dayofweek'] = train['date'].dt.dayofweek
# train['month'] = train['date'].dt.month
# train['weekofyear'] = train['date'].dt.isocalendar().week.astype(int)

# # Add lag and rolling features
# train = train.sort_values(['store_nbr', 'family', 'date']).reset_index(drop=True)

# lags = [1, 7, 14, 28]
# for lag in lags:
#     train[f'lag_{lag}'] = train.groupby(['store_nbr', 'family'])['sales'].shift(lag)

# windows = [7, 14]
# for window in windows:
#     train[f'rolling_mean_{window}'] = (
#         train
#         .groupby(['store_nbr', 'family'])['sales']
#         .shift(1)
#         .rolling(window)
#         .mean()
#     )
# train = train.dropna().reset_index(drop=True)
# train.head()

# #%%
# train = pd.get_dummies(train, columns=['family', 'store_nbr'])

# #%%
# train.columns