#%%
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess


plot_params = dict(
    color="0.75",
    style=".-",
    markeredgecolor="0.25",
    markerfacecolor="0.25",
    legend=False,
)

#%%
data_folder = r'C:\Users\guoya\Documents\Git_repo\Kaggle-learn\ML\store_sales\data'

store_sales = pd.read_csv(
    os.path.join(data_folder, 'train.csv'),
    parse_dates=['date'],
    dtype={
        'store_nbr': 'category',
        'family': 'category',
        'sales': 'float32',
    }
)
store_sales['date'] = store_sales.date.dt.to_period('D')
store_sales = store_sales.drop(['onpromotion'], axis=1)
store_sales = store_sales.set_index(['store_nbr', 'family', 'date']).sort_index()

#%%
y = store_sales['sales'].unstack(['store_nbr', 'family']).loc['2017']

# Create training data
fourier = CalendarFourier(freq='ME', order=4)
dp = DeterministicProcess(
    index=y.index,
    constant=True,
    order=1,
    seasonal=True,
    additional_terms=[fourier],
    drop=True,
)
X = dp.in_sample()
X['NewYear'] = (X.index.dayofyear == 1)

model = LinearRegression(fit_intercept=False)
model.fit(X, y)

y_pred = pd.DataFrame(model.predict(X), index=X.index, columns=y.columns)
STORE_NBR = '1'  # 1 - 54
FAMILY = 'PRODUCE'

ax = y.loc(axis=1)[STORE_NBR, FAMILY].plot(**plot_params)
ax = y_pred.loc(axis=1)[STORE_NBR, FAMILY].plot(ax=ax, c='m', alpha=0.7)
ax.set_title(f'{FAMILY} Sales at Store {STORE_NBR}')
ax.legend()

#%%
df_test = pd.read_csv(
    os.path.join(data_folder, 'test.csv'),
    dtype={
        'store_nbr': 'category',
        'family': 'category',
        'onpromotion': 'uint32',
    },
    parse_dates=['date']
)
df_test['date'] = df_test.date.dt.to_period('D')
df_test = df_test.set_index(['store_nbr', 'family', 'date']).sort_index()

X_test = dp.out_of_sample(steps=16)
X_test.index.name = 'date'
X_test['NewYear'] = (X_test.index.dayofyear == 1)

y_submit = (
    pd.DataFrame(model.predict(X_test), index=X_test.index, columns=y.columns)
    .stack(['store_nbr', 'family'])
    .rename('sales')
    .reset_index()
)

y_submit = y_submit.merge(
    df_test.reset_index()[['id', 'store_nbr', 'family', 'date']],
    on=['store_nbr', 'family', 'date'],
    how='left'
)

y_submit[['id', 'sales']].to_csv('submission.csv', index=False)