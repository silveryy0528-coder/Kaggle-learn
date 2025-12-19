#%%
import pandas as pd
import kagglehub
import seaborn as sns
import matplotlib.pyplot as plt
import calendar
import numpy as np
import sys
from matplotlib import cm

sys.path.insert(0, r'C:\Users\YanGuo\Documents\Kaggle-learn\EDA')
from viz_utils import *
sns.set_style("white")


def create_time_features(df):
    df = df.copy()
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['dayname'] = df.index.day_name()
    df['quarter'] = df.index.quarter
    df['year'] = df.index.year
    df['month'] = df.index.month
    df['dayofyear'] = df.index.dayofyear
    df['dayofmonth'] = df.index.day
    df['weekofyear'] = df.index.isocalendar().week
    df['timeofday'] = pd.cut(df['hour'], bins=[-1, 5, 11, 17, 23],
                             labels=['Night', 'Morning', 'Afternoon', 'Evening'])

    return df

#%%
raw_path = kagglehub.dataset_download("olistbr/brazilian-ecommerce")
olist_customer = pd.read_csv(path_join(raw_path, 'olist_customers_dataset.csv'))
olist_geolocation = pd.read_csv(path_join(raw_path, 'olist_geolocation_dataset.csv'))
olist_orders = pd.read_csv(path_join(raw_path, 'olist_orders_dataset.csv'))
olist_order_items = pd.read_csv(path_join(raw_path, 'olist_order_items_dataset.csv'))
olist_order_payments = pd.read_csv(path_join(raw_path, 'olist_order_payments_dataset.csv'))
olist_order_reviews = pd.read_csv(path_join(raw_path, 'olist_order_reviews_dataset.csv'))
olist_products = pd.read_csv(path_join(raw_path, 'olist_products_dataset.csv'))
olist_sellers = pd.read_csv(path_join(raw_path, 'olist_sellers_dataset.csv'))

datasets = {
    'olist_customer': olist_customer,
    'olist_geolocation': olist_geolocation,
    'olist_orders': olist_orders,
    'olist_order_items': olist_order_items,
    'olist_order_payments': olist_order_payments,
    'olist_order_reviews': olist_order_reviews,
    'olist_products': olist_products,
    'olist_sellers': olist_sellers
}
summary = summarize_datasets(datasets)

olist_orders = olist_orders.set_index('order_purchase_timestamp')
olist_orders.index = pd.to_datetime(olist_orders.index)
olist_orders = olist_orders.sort_index()
olist_orders = create_time_features(olist_orders)

#%% Total orders on E-Commerce
olist_orders = olist_orders
counts = olist_orders['order_status'].value_counts()
fig = plt.figure(figsize=(8, 4))
sns.barplot(counts)
plt.xticks(rotation=45)
plt.tight_layout()
plt.draw()

orders_over_time = (
    olist_orders
    .groupby(["year", "month"])
    .size()
    .reset_index(name='num_orders')
)
orders_over_time['year_month'] = pd.to_datetime(
    orders_over_time[['year', 'month']].assign(day=1)
)
orders_per_day = (
    olist_orders
    .groupby('dayname')
    .size()
    .reset_index(name='num_orders')
)
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
time_order = ['Night', 'Morning', 'Afternoon', 'Evening']
orders_over_period = (
    olist_orders
    .groupby('timeofday')
    .size()
    .reset_index(name='num_orders')
)
fig, ax = plt.subplot_mosaic(
    [
        ["main", "main"],
        ["left", "right"]
    ],
    figsize=(12, 8)
)
sns.lineplot(x=orders_over_time['year_month'], y=orders_over_time['num_orders'],
             ax=ax['main'])
ax['main'].set_xlabel('Year - Month')
ax['main'].set_ylabel('Number of orders')
ax['main'].set_title('Evolution of total orders in Brazilian e-Commerce')

sns.barplot(data=orders_per_day, x='dayname', y='num_orders',
            ax=ax['left'], hue='dayname', order=day_order)
ax['left'].set_xlabel('')
ax['left'].set_ylabel('Number of orders')
ax['left'].set_title('Total orders by day of week')
sns.barplot(data=orders_over_period, x='timeofday', y='num_orders',
            ax=ax['right'], hue='timeofday', order=time_order)
ax['right'].set_xlabel('')
ax['right'].set_ylabel('Number of orders')
ax['right'].set_title('Total orders by time of day')
plt.tight_layout()
plt.show()

orders_per_month = (
    olist_orders
    .groupby(['year', 'month'])
    .size()
    .reset_index(name='num_orders')
)
orders_per_month['month_name'] = orders_per_month['month'] \
    .apply(lambda x: calendar.month_name[x])
fig = plt.figure(figsize=(10, 5))
sns.barplot(
    data=orders_per_month.query('(month <= 8) and (year in [2017, 2018])'),
    x='month_name',
    y='num_orders',
    hue='year',  # Different colors for each year
    palette='Set2'
)
plt.xlabel('')
plt.ylabel('Number of orders')
plt.title('Total orders comparison between 2017 and 2018 (till August)')

# %% E-Commerce around Brazil
olist_order_location = pd.merge(olist_orders, olist_customer, on='customer_id', how='inner')
orders_per_state = (
    olist_order_location
    .groupby('customer_state')
    .size()
    .reset_index(name='num_orders')
    .sort_values(by='num_orders', ascending=False)
)
orders_per_city = (
    olist_order_location
    .groupby('customer_city')
    .size()
    .reset_index(name='num_orders')
)
region_map = {
    "AC": "North", "AP": "North", "AM": "North", "PA": "North", "RO": "North", "RR": "North", "TO": "North",
    "AL": "Northeast", "BA": "Northeast", "CE": "Northeast", "MA": "Northeast", "PB": "Northeast",
    "PE": "Northeast", "PI": "Northeast", "RN": "Northeast", "SE": "Northeast",
    "DF": "Central-West", "GO": "Central-West", "MT": "Central-West", "MS": "Central-West",
    "ES": "Southeast", "MG": "Southeast", "RJ": "Southeast", "SP": "Southeast",
    "PR": "South", "RS": "South", "SC": "South"
}
olist_order_location['region'] = olist_order_location['customer_state'].map(region_map)
olist_order_location['year_month'] = pd.to_datetime(
    olist_order_location[['year', 'month']].assign(day=1)
)
orders_per_region = (
    olist_order_location
    .groupby(['year_month', 'region'])
    .size()
    .reset_index(name='num_orders')
)
orders_per_region = orders_per_region.query(
    "year_month >= '2017-01' and year_month <= '2018-08'")
fig, ax = plt.subplot_mosaic(
    [
        ["top", "main"],
        ["bottom", "main"]
    ],
    figsize=(12, 8)
)
sns.barplot(data=orders_per_state, y='customer_state', x='num_orders',
            ax=ax['main'], hue='customer_state')
ax['main'].set_xlabel('Number of orders')
ax['main'].set_ylabel('State name')
ax['main'].set_title('Total of customer orders by state')

sns.lineplot(data=orders_per_region, x='year_month', y='num_orders', hue='region', ax=ax['top'])
ax['top'].set_xlabel('')
ax['top'].set_ylabel('Number of orders')
ax['top'].set_title('Evaluation of ordres per region')
ax['top'].tick_params(axis='x', rotation=45)

sns.barplot(data=orders_per_city.sort_values(by='num_orders', ascending=False).head(10),
            x='num_orders', y='customer_city', ax=ax['bottom'], hue='customer_city')
ax['bottom'].set_xlabel('Number of orders')
ax['bottom'].set_ylabel('City name')
ax['bottom'].set_title('Total of customer orders by city (top 10)')

plt.tight_layout()
plt.show()

#%% E-commerce impact on economy
olist_order_price = pd.merge(
    olist_orders, olist_order_items, on='order_id', how='inner')
olist_order_price['year_month'] = pd.to_datetime(
    olist_order_price[['year', 'month']].assign(day=1)
)
olist_order_price = pd.merge(
    olist_order_price, olist_customer, on='customer_id', how='inner')

olist_order_price = olist_order_price.query(
    "year_month >= '2017-01' and year_month <= '2018-08'").reset_index(drop=True)
price_per_month = (
    olist_order_price
    .groupby('year_month')['price']
    .sum()
    .reset_index(name='price')
)
order_per_month = (
    olist_order_price
    .groupby('year_month')
    .size()
    .reset_index(name='num_orders')
)
avg_freight_value_per_month = (
    olist_order_price
    .groupby('year_month')['freight_value']
    .mean()
    .reset_index(name='avg_value')
)

fig, ax = plt.subplots(2, 1, figsize=(12, 10))
ax2 = ax[0].twinx()
colors = cm.Blues(np.linspace(0.4, 0.9, len(order_per_month)))
sns.lineplot(data=price_per_month, x='year_month', y='price',
             color='k', ax=ax[0], marker='o', lw=1)
ax2.bar(order_per_month['year_month'], order_per_month['num_orders'],
        width=20, color=colors)
ax[0].set_xlabel('Year - Month')
ax[0].set_ylabel('Price')
ax[0].tick_params(axis='x', rotation=45)
ax[0].ticklabel_format(axis='y', style='plain')
ax[0].set_title('Evolution of E-commerce: total orders and total amount sold')
ax2.set_ylim([0, 2 * order_per_month['num_orders'].max()])
ax2.set_ylabel('Number of orders')

sns.lineplot(data=avg_freight_value_per_month, x='year_month', y='avg_value',
             color='k', ax=ax[1], marker='o', lw=1)
ax[1].set_xlabel('Year - Month')
ax[1].set_ylabel('Average freight value')
ax[1].tick_params(axis='x', rotation=45)
ax[1].ticklabel_format(axis='y', style='plain')
ax[1].set_title('Evolution of average freight value paid by customers')
# for x, y in zip(avg_freight_value_per_month['year_month'],
#                 avg_freight_value_per_month['avg_value']):
#     ax[1].annotate(
#         f"{y:,.2f}",          # format number
#         (x, y),               # point location
#         textcoords="offset points",
#         xytext=(0, -16),         # move text slightly above marker
#         ha="center",
#         fontsize=9,
#         color="k"
#     )
plt.tight_layout()
plt.show()

avg_price_per_state = (
    olist_order_price
    .groupby('customer_state')['price']
    .mean()
    .reset_index(name='avg_value')
    .sort_values(by='avg_value', ascending=False)
)
sum_price_per_state = (
    olist_order_price
    .groupby('customer_state')['price']
    .sum()
    .reset_index(name='sum_value')
)
price_per_state = pd.merge(avg_price_per_state, sum_price_per_state, on='customer_state')
fig, ax = plt.subplots(1, 2, figsize=(12, 10))
sns.barplot(price_per_state, x='avg_value', y='customer_state',
            ax=ax[0], hue='avg_value', width=0.8)
ax[0].set_ylabel('State')
ax[0].set_xlabel('Price')
ax[0].set_title('Average price by state')
sns.barplot(price_per_state, x='sum_value', y='customer_state',
            ax=ax[1], hue='sum_value', width=0.8)
ax[1].set_ylabel('State')
ax[1].set_xlabel('Price')
ax[1].set_title('Sum of price by state')
plt.tight_layout()
plt.show()

#%%
colors = ['red', 'blue', 'green', 'magenta', 'yellow']

payment_value = (
    olist_order_payments
    .groupby(by='payment_type')['payment_value']
    .mean()
    .reset_index(name='avg_value')
    .sort_values(by='avg_value', ascending=False)
)
ref_type = payment_value['payment_type']
payment_type = (
    olist_order_payments['payment_type']
    .value_counts()
    .reset_index(name='count')
    .set_index('payment_type').reindex(ref_type).reset_index()
)
olist_order_clean = olist_orders[['order_id', 'year', 'month']].copy()
olist_order_clean = pd.merge(
    olist_order_clean, olist_order_payments, on='order_id', how='left')
olist_order_clean['year_month'] = pd.to_datetime(
    olist_order_clean[['year', 'month']].assign(day=1))

orders_per_payment = (
    olist_order_clean
    .groupby(['year_month', 'payment_type'])
    .size()
    .reset_index(name='sum_value')
)
orders_per_payment = orders_per_payment.query(
    "year_month >= '2017-01' and year_month <= '2018-08'"
).reset_index(drop=True)
orders_per_payment['payment_type'] = pd.Categorical(
    orders_per_payment['payment_type'],
    categories=ref_type,
    ordered=True
)

fig, ax = plt.subplot_mosaic(
    [
        ["left", "right"],
        ["main", "main"]
    ],
    figsize=(10, 10))
ax['left'].pie(payment_type['count'], labels=payment_type['payment_type'],
        wedgeprops={'width':0.3}, autopct='%1.1f%%', colors=colors)
ax['left'].set_title('Count of transactions by payment type')
sns.barplot(data=payment_value, x='payment_type', y='avg_value',
            legend='brief', ax=ax['right'], palette=colors, hue='payment_type')
ax['right'].set_xlabel('Payment type')
ax['right'].set_ylabel('Average payment')
ax['right'].set_title('Average payment by payment type')
sns.lineplot(
    data=orders_per_payment,
    x='year_month',
    y='sum_value',
    marker='o',
    ax=ax['main'],
    palette=colors,
    hue='payment_type'
)
plt.tight_layout()
plt.show()

#%%
