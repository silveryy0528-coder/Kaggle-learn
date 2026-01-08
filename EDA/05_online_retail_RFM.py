#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
import kagglehub
from viz_utils import *
from warnings import simplefilter

# RFM analysis for online retail dataset: customer value and engagement.
# Focusing on segmenting customers based on their purchasing behavior.

simplefilter(action='ignore', category=FutureWarning)
sns.set_style("whitegrid")


folder_path = kagglehub.dataset_download("mashlyn/online-retail-ii-uci")
print("Path to dataset files:", folder_path)
file_path = path_join(folder_path, "online_retail_II.csv")
retail = pd.read_csv(file_path)
retail['InvoiceDate'] = pd.to_datetime(retail['InvoiceDate'])

#%%
retail.drop_duplicates(inplace=True)
retail.dropna(subset=['Customer ID'], axis=0, inplace=True)
retail['Customer ID'] = retail['Customer ID'].astype(int)
retail['Total'] = retail['Quantity'] * retail['Price']
retail.head()

#%%
latest_date = retail['InvoiceDate'].max() + dt.timedelta(days=1)

rfm = retail.groupby('Customer ID').agg({
    'InvoiceDate': lambda x: (latest_date - x.max()).days,
    'Invoice': 'count',
    'Total': 'sum'
}).reset_index()

rfm.rename(columns={
    'InvoiceDate': 'Recency',
    'Invoice': 'Frequency',
    'Total': 'MonetaryValue'
}, inplace=True)

#%%
# The unit of recency is days, and it is mostly unique. Hence, qcut is suitable here.
rfm["Recency_score"] = pd.qcut(rfm['Recency'], 5, labels = [5, 4, 3, 2, 1])

# Frequency might have many duplicated values, hence, using rank method before qcut
rfm["Frequency_rank"] = rfm['Frequency'].rank(method='first')
rfm['Frequency_score'] = pd.qcut(rfm['Frequency_rank'], 5, labels=[1, 2, 3, 4, 5])
rfm.drop('Frequency_rank', axis=1, inplace=True)

rfm['rfm_segment'] = rfm['Recency_score'].astype(str) + rfm['Frequency_score'].astype(str)

#%%
fig = plt.figure(figsize=(10, 6))
sns.histplot(rfm['Recency'], bins=30, kde=True)
plt.title('Distribution of Recency')
plt.xlabel('Recency (days)')
plt.ylabel('Number of Customers')
plt.show()

fig = plt.figure(figsize=(10, 6))
sns.histplot(rfm['Frequency'], bins=30, kde=False)
plt.title('Distribution of Frequency')
plt.xlabel('Frequency (number of purchases)')
plt.ylabel('Number of Customers')
plt.show()

fig = plt.figure(figsize=(10, 6))
sns.histplot(rfm['MonetaryValue'], bins=30, kde=False)
plt.title('Distribution of Monetary Value')
plt.xlabel('Monetary Value (total spend)')
plt.ylabel('Number of Customers')
plt.show()

#%%
segment_map = {
    r'[1-2][1-2]' : 'Hibernating', # Not bought for a long time and low frequency
    r'[1-2][3-4]' : 'At-Risk', # Not bought for a long time but used to buy frequently
    r'[1-2]5' : "Can\'t Lose", # Not bought for a long time but used to spend a lot
    r'3[1-2]' : 'About to Slip',
    r'33' : 'Need Attention', # Bought moderately recently and moderately frequently
    r'[3-4][4-5]' : 'Loyal Customers', # Bought frequently and spent a decent amount
    r'41' : 'Promising', # Bought recently
    r'51' : 'New Customers', # Bought recently but low frequency
    r'[4-5][2-3]' : 'Potential Loyalists', # Bought recently and spent a decent amount
    r'5[4-5]' : 'Champions', # Bought recently, frequently and spent the most
}
rfm['rfm_segment'] = rfm['rfm_segment'].replace(segment_map, regex=True)

#%%
new_rfm = rfm[['Recency', 'Frequency', 'MonetaryValue', 'rfm_segment']]
fig = plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=new_rfm,
    x='Recency',
    y='Frequency',
    hue='rfm_segment',
    palette='viridis',
    alpha=0.7,)
plt.title('RFM Segments')
plt.xlabel('Recency (days)')
plt.ylabel('Frequency (number of purchases)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

segments = new_rfm['rfm_segment'].value_counts()
fig = plt.figure(figsize=(10, 6))
sns.barplot(
    x=segments.index,
    y=segments.values,
    palette='viridis',
)
plt.title('Number of Customers in Each RFM Segment')
plt.xlabel('RFM Segment')
plt.ylabel('Number of Customers')
plt.xticks(rotation=45)
plt.show()

fig = plt.figure(figsize=(10, 6))
explode = [0.1 if i == segments.idxmax() else 0 for i in segments.index]
plt.pie(
    segments.values,
    labels=segments.index,
    autopct='%1.1f%%',
    explode=explode,
)
plt.title('Proportion of Customers in Each RFM Segment')
plt.axis('equal')
plt.show()

#%%
(
    new_rfm[['rfm_segment', 'Recency', 'Frequency', 'MonetaryValue']]
    .groupby('rfm_segment')
    .agg(['mean', 'count', 'sum'])
)