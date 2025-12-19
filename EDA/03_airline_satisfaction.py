#%%
import kagglehub
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from viz_utils import *

# Download latest version
raw_path = kagglehub.dataset_download("teejmahal20/airline-passenger-satisfaction")

df = pd.read_csv(path_join(raw_path, 'train.csv'))
df = df.dropna()
df = df.drop('Unnamed: 0', axis=1)

#%%
satisfaction = df['satisfaction'].value_counts() / len(df['satisfaction'])
fig = plt.figure()
plt.pie(
    satisfaction.values,
    labels=satisfaction.index,
    explode=(0.0, 0.02),
    autopct='%1.1f%%',
    startangle=0,
)
plt.title('Overall customer satisfaction')

df_delay = df[['Departure Delay in Minutes', 'Arrival Delay in Minutes']].copy()
df_delay = df_delay.rename(columns={'Departure Delay in Minutes': 'Departure Delay',
                                'Arrival Delay in Minutes': 'Arrival Delay'})

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
sns.heatmap(df_delay.corr(), annot=True, ax=ax[0])
ax[0].set_title('Correlation between departure and arrival delays')
sns.scatterplot(
    data=df,
    x='Departure Delay in Minutes',
    y='Arrival Delay in Minutes',
    marker='o',
    color='.3',
    hue='satisfaction'
)
plt.tight_layout()
plt.show()

df_score = df[['Ease of Online booking', 'Checkin service', 'Inflight service',
               'Cleanliness', 'Food and drink', 'Seat comfort']]
fig = plt.figure()
sns.heatmap(df_score.corr(), annot=True)
plt.show()

#%% Explore gender vs. class vs. travel type
df_sum = df.groupby(['Gender', 'Class', 'Type of Travel']).size().reset_index(name='Number of Customers')
pivot = (
    df_sum.pivot_table(
        index=['Class', 'Type of Travel'],
        columns='Gender',
        values='Number of Customers',
        aggfunc='sum'
    )
    .fillna(0)
)
classes = pivot.index.get_level_values(0).unique()
travel_types = pivot.index.get_level_values(1).unique()

x = np.arange(len(classes))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))

for i, travel in enumerate(travel_types):
    data = pivot.xs(travel, level='Type of Travel')
    ax.bar(
        x + i * width,
        data['Male'],
        width,
        label=f'Male - {travel}'
    )
    ax.bar(
        x + i * width,
        data['Female'],
        width,
        bottom=data['Male'],
        label=f'Female - {travel}'
    )
ax.set_xticks(x + width / 2)
ax.set_xticklabels(classes)
ax.set_ylabel('Number of Passengers')
ax.set_xlabel('Flight Class')
ax.set_title('Passengers by Class, Travel Type, and Gender')
ax.legend(ncol=2)
plt.show()

#%%