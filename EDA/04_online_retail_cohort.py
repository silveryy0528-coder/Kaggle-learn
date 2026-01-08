#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
import kagglehub
from viz_utils import *
sns.set_style("whitegrid")
# Cohort analysis for online retail dataset, time-based customer behavior.
# Focusing on customer retention and purchase behavior over time after joining.

folder_path = kagglehub.dataset_download("mashlyn/online-retail-ii-uci")
print("Path to dataset files:", folder_path)
file_path = path_join(folder_path, "online_retail_II.csv")
retail = pd.read_csv(file_path)

# %%
retail['InvoiceDate_DT'] = pd.to_datetime(retail['InvoiceDate'])
retail = retail.dropna(subset=['Customer ID'], axis=0)
retail['Customer ID'] = retail['Customer ID'].astype(int)

retail = retail.sort_values(by='InvoiceDate_DT', ascending=True).reset_index(drop=True)
retail['TotalPrice'] = retail['Quantity'] * retail['Price']

#%%
most_expensive = retail.loc[retail['Price'] == retail['Price'].max()]
least_expensive = retail.loc[retail['Price'] == retail['Price'].min()]
least_expensive = (
    least_expensive
    .groupby(['Description', 'Price'], as_index=False)['Quantity']
    .agg('sum')
)

#%%
retail_non_zero_price = retail.loc[retail['Price'] > 0]
least_expensive_not_null = (
    retail_non_zero_price
    .loc[retail_non_zero_price['Price'] == retail_non_zero_price['Price'].min()])
least_expensive_not_null = (
    least_expensive_not_null
    .groupby(['Description', 'Price'], as_index=False)['Quantity']
    .agg('sum')
)

#%%
retail_customers = (
    retail.groupby(['Customer ID', 'Country'], as_index=False)['TotalPrice']
    .agg('sum')
)
retail_customers = (
    retail_customers
    .sort_values(by='TotalPrice', ascending=True)
    .reset_index(drop=True)
)
retail_customers['Customer ID'] = retail_customers['Customer ID'].astype(str)

sorted_customers = pd.concat([
    retail_customers.head(5), 
    retail_customers.tail(5)
])
fig = plt.figure(figsize=(10, 6))
sns.barplot(
    data=sorted_customers,
    x='Customer ID',
    y='TotalPrice',
    hue='Country',
)
plt.title("Customers With Min and Max Total Purchase Amount")

#%%
retail_country = retail.groupby('Country', as_index=False)['TotalPrice'].agg('sum')
retail_country.sort_values(by='TotalPrice', ascending=True, inplace=True)

top = 10
fig, ax = plt.subplots(2, 1, figsize=(10, 6))
sns.barplot(
    data=retail_country.head(top),
    x='Country',
    y='TotalPrice',
    hue='Country',
    ax=ax[0],
)
ax[0].set_title("Country With Min Total Purchase Amount")
ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=45)
sns.barplot(
    data=retail_country.tail(top),
    x='Country',
    y='TotalPrice',
    hue='Country',
    ax=ax[1],
)
ax[1].set_title("Country With Max Total Purchase Amount")
plt.tight_layout()
ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation=45)
plt.show()

#%%
retail_countries = (
    retail.groupby(['Country'], as_index=False)['Customer ID']
    .agg('nunique')
)
retail_countries = retail_countries.rename(columns={'Customer ID': 'NumCustomers'})
retail_countries = (
    retail_countries
    .sort_values(by='NumCustomers', ascending=True)
    .reset_index(drop=True)
)

top = 10
fig, ax = plt.subplots(2, 1, figsize=(10, 6))
sns.barplot(
    data=retail_countries.head(top),
    x='Country',
    y='NumCustomers',
    hue='Country',
    ax=ax[0],
)
ax[0].set_title("Country With Min Unique Customer")
ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=45)
sns.barplot(
    data=retail_countries.tail(top),
    x='Country',
    y='NumCustomers',
    hue='Country',
    ax=ax[1],
)
ax[1].set_title("Country With Max Unique Customer")
plt.tight_layout()
ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation=45)
plt.show()

#%%
retail_products = retail.groupby(['Description'], as_index=False)['Quantity'].agg('sum')
retail_products = retail_products.loc[~retail_products['Description'].isin(['Discount', 'CRUK Commission'])]
retail_products = (
    retail_products
    .sort_values(by='Quantity', ascending=True)
    .reset_index(drop=True)
)
print("The most sold product:", retail_products.iloc[-1])
print("The least sold product:", retail_products.iloc[0])

#%%
retail = retail[retail['InvoiceDate_DT'].dt.year > 2009]
retail['InvoiceQuarter'] = (
    'Q'
    + retail['InvoiceDate_DT'].dt.quarter.astype(str)
    + '/'
    + retail['InvoiceDate_DT'].dt.year.astype(str)
)
quarters_map = dict(zip(
    retail['InvoiceQuarter'].unique(),
    range(len(retail['InvoiceQuarter'].unique()))
))
# When did the purchase happen in quarter IDs
retail['InvoiceQuarterID'] = retail['InvoiceQuarter'].map(quarters_map)
# When did the customer first purchase in quarter IDs
retail['CohortQuarterID'] = retail.groupby('Customer ID')['InvoiceQuarterID'].transform('min')
retail['CohortQuarter'] = retail['CohortQuarterID'].map({v: k for k, v in quarters_map.items()})
# How many quarters since the first purchase
retail['CohortIndex'] = retail['InvoiceQuarterID'] - retail['CohortQuarterID']

# How many unique customers are active in each cohort quarter and cohort index
cohort_retention = (
    retail.groupby(['CohortQuarterID', 'CohortIndex'], as_index=False)['Customer ID']
    .nunique()
    .reset_index()
)
cohort_retention.rename(columns={'Customer ID': 'NumCustomers'}, inplace=True)
# Pivot the data to create a cohort retention table
cohort_retention_count = cohort_retention.pivot_table(
    index='CohortQuarterID', columns='CohortIndex', values='NumCustomers'
)
cohort_retention_count['CohortQuarter'] = cohort_retention_count.index.map({v: k for k, v in quarters_map.items()})
cohort_retention_count.set_index('CohortQuarter', inplace=True)
# Number of customers in each cohort (per quarter)
cohort_sizes = cohort_retention_count.iloc[:, 0]
retention = cohort_retention_count.divide(cohort_sizes, axis=0)
retention = retention.round(2) * 100
# retention = retention.iloc[::-1]

fig = plt.figure(figsize=(10, 6))
sns.heatmap(
    retention,
    annot=True,
    fmt='.0f',
    cmap='Blues',
    cbar_kws={'label': 'Retention Rate (%)'},
)
plt.title("Cohort Analysis - Customer Retention Rate (%)")
plt.ylabel("Cohort Quarter")
plt.xlabel("Cohort Index (Quarters Since First Purchase)")
plt.show()

#%%
cohort_quantity = (
    retail
    .groupby(['CohortQuarterID', 'CohortIndex'], as_index=False)['Quantity']
    .mean()
    .reset_index()
)
cohort_quantity.rename(columns={'Quantity': 'Average Quantity'}, inplace=True)
average_quantity = cohort_quantity.pivot_table(
    index='CohortQuarterID', columns='CohortIndex', values='Average Quantity'
)
average_quantity['CohortQuarter'] = average_quantity.index.map({v: k for k, v in quarters_map.items()})
average_quantity.set_index('CohortQuarter', inplace=True)
fig = plt.figure(figsize=(10, 6))
sns.heatmap(
    average_quantity,
    annot=True,
    fmt='.0f',
    cmap='Blues',
    cbar_kws={'label': 'Average Quantity'},
)
plt.title("Cohort Analysis - Average Quantity")
plt.ylabel("Cohort Quarter")
plt.xlabel("Cohort Index (Quarters Since First Purchase)")
plt.show()

#%%
retail_quarters_by_sales = (
    retail
    .groupby(['InvoiceQuarterID'], as_index=False)['TotalPrice']
    .sum()
)
retail_quarters_by_sales['InvoiceQuarter'] = (
    retail_quarters_by_sales['InvoiceQuarterID']
    .map({v: k for k, v in quarters_map.items()}))

fig = plt.figure(figsize=(10, 6))
sns.barplot(
    data=retail_quarters_by_sales,
    x='InvoiceQuarter',
    y='TotalPrice',
)
plt.title("Total Sales by Invoice Quarter")
plt.xlabel("Invoice Quarter")
plt.ylabel("Total Sales")
plt.xticks(rotation=45)
plt.show()

#%%
retail_monthly = retail.copy()
retail_monthly['InvoiceMonth'] = (
    retail_monthly['InvoiceDate_DT']
    .dt.to_period('M')
    .dt.to_timestamp()
)
retail_monthly_by_sales = (
    retail_monthly.groupby(['InvoiceMonth'], as_index=False)['TotalPrice']
    .sum()
)
fig = plt.figure(figsize=(10, 6))
sns.lineplot(
    data=retail_monthly_by_sales,
    x='InvoiceMonth',
    y='TotalPrice',
    marker='o',
)
plt.title("Total Sales by Invoice Month")
plt.xlabel("Invoice Month")
plt.ylabel("Total Sales")
plt.xticks(rotation=45)
plt.show()
