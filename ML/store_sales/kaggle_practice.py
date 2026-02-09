#%%
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import root_mean_squared_log_error
from sklearn.linear_model import Ridge
from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess
from sklearn.preprocessing import StandardScaler


add_promo = False

def time_series_cv_ridge(y, alphas, promo=None, horizon=16, logscale=False):

    add_promo = False if promo is None else True
    y_orig = y.copy()
    results = []

    dates = y_orig.index
    train_idx = dates[:-horizon]
    val_idx = dates[-horizon:]

    # scale promo features
    if add_promo:
        promo_train = promo.loc[train_idx]
        promo_val = promo.loc[val_idx]

        scaler = StandardScaler()
        promo_train_scaled = pd.DataFrame(
            scaler.fit_transform(promo_train),
            index=promo_train.index,
            columns=promo_train.columns
        )
        promo_val_scaled = pd.DataFrame(
            scaler.transform(promo_val),
            index=promo_val.index,
            columns=promo_val.columns
        )

    fourier = CalendarFourier(freq='ME', order=4)
    dp_fold = DeterministicProcess(
        index=train_idx,
        constant=True,
        order=1,
        seasonal=True,
        additional_terms=[fourier],
        drop=True,
    )

    X_train = dp_fold.in_sample().assign(NewYear=lambda x: x.index.dayofyear == 1)
    X_val = dp_fold.out_of_sample(steps=horizon)
    X_val.index = val_idx
    X_val = X_val.assign(NewYear=lambda x: x.index.dayofyear == 1)

    # join promo
    if add_promo:
        X_train = pd.concat([X_train, promo_train_scaled], axis=1)
        X_val = pd.concat([X_val, promo_val_scaled], axis=1)

    # make all column names strings
    X_train.columns = X_train.columns.map(lambda x: '_'.join(x) if isinstance(x, tuple) else str(x))
    X_val.columns = X_val.columns.map(lambda x: '_'.join(x) if isinstance(x, tuple) else str(x))

    if logscale:
        y_train = np.log1p(y_orig.loc[train_idx])
    else:
        y_train = y_orig.loc[train_idx]

    for alpha in alphas:
        model = Ridge(alpha=alpha, fit_intercept=False)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)

        if logscale:
            y_pred = np.expm1(y_pred)

        y_pred = np.clip(y_pred, 0, None)

        results.append({
            'alpha': alpha,
            'rmsle': root_mean_squared_log_error(
                y_orig.loc[val_idx], y_pred)
        })

    results = pd.DataFrame(results)
    results.plot(x='alpha', y='rmsle', logx=True)
    plt.show()

    results = results.sort_values('rmsle')
    best_alpha = results.iloc[0]['alpha']
    print(f"Best alpha: {best_alpha}, rmsle: {results['rmsle'].min():.4f}")

    return best_alpha


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
store_sales = store_sales.set_index(['store_nbr', 'family', 'date']).sort_index()
y = (
    store_sales['sales']
    .unstack(['store_nbr', 'family'])
    .loc['2017']
)
promo = (
    store_sales['onpromotion']
    .unstack(['store_nbr', 'family'])
    .loc[y.index]
).fillna(0.)

#%%
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

best_alpha = time_series_cv_ridge(
    y,
    alphas=np.arange(5),
    promo=None,
    horizon=16,
)

X = dp.in_sample().assign(
    NewYear=lambda x: x.index.dayofyear == 1)

if add_promo:
    scaler = StandardScaler()
    promo_scaled = pd.DataFrame(
        scaler.fit_transform(promo),
        index=promo.index,
        columns=promo.columns
    )
    X = pd.concat([X, promo_scaled], axis=1)
X.columns = X.columns.map(lambda x: '_'.join(x) if isinstance(x, tuple) else str(x))

#%%
best_model = Ridge(alpha=best_alpha, fit_intercept=False)
best_model.fit(X, y)

y_pred = best_model.predict(X)
y_pred = np.clip(y_pred, 0, None)
y_pred = pd.DataFrame(y_pred, index=X.index, columns=y.columns)

STORE_NBR = '5'
FAMILY = 'GROCERY I'

plt.figure()
ax = y.loc(axis=1)[STORE_NBR, FAMILY].plot(**plot_params)
ax = y_pred.loc(axis=1)[STORE_NBR, FAMILY].plot(ax=ax, c='m', alpha=0.7)
ax.set_title(f'{FAMILY} Sales at Store {STORE_NBR}')
ax.legend()
plt.show()

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
promo_test = (
    df_test['onpromotion']
    .unstack(['store_nbr','family'])
).fillna(0.)

#%%
X_test = dp.out_of_sample(steps=16)
X_test.index.name = 'date'
X_test = X_test.assign(
    NewYear=lambda x: x.index.dayofyear == 1)
if add_promo:
    X_test = pd.concat([X_test, promo_test], axis=1)
X_test.columns = X_test.columns.map(
    lambda x: '_'.join(x) if isinstance(x, tuple) else str(x))

y_pred_test = best_model.predict(X_test)
y_pred_test = np.clip(y_pred_test, 0, None)

y_pred_test = pd.DataFrame(
    y_pred_test,
    index=X_test.index,
    columns=y.columns
)

y_submit = (
    y_pred_test
    .stack(['store_nbr', 'family'], future_stack=True)
    .rename('sales')
    .reset_index()
)

y_submit = y_submit.merge(
    df_test.reset_index()[['id', 'store_nbr', 'family', 'date']],
    on=['store_nbr', 'family', 'date'],
    how='left'
)
assert len(y_submit) == len(df_test)
y_submit[['id', 'sales']].to_csv('submission.csv', index=False)