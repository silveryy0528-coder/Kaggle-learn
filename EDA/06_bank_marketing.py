#%%
import kagglehub
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from viz_utils import *


# Download latest version
dir_path = kagglehub.dataset_download("janiobachmann/bank-marketing-dataset")
file_path = path_join(dir_path, "bank.csv")
df = pd.read_csv(file_path)
# df['balance'] = df['balance'].clip(upper=df['balance'].quantile(0.99))

#%%
deposit_by_education = df.groupby(['education', 'deposit']).size()
fig, ax = plt.subplots(1, 2, figsize=(15, 6))
ax[0].pie(
    df['deposit'].value_counts(),
    labels=df['deposit'].value_counts().index,
    autopct='%1.1f%%'
)
ax[0].set_title('Distribution of Deposit Subscription')
sns.countplot(
    data=df,
    x='education',
    hue='deposit',
    stat='percent',
    ax=ax[1]
)
ax[1].set_title('Deposit Subscription by Education Level')
plt.tight_layout()
plt.show()

numeric_features = df.select_dtypes(include='number')
fig = plt.figure(figsize=(15, 12))
for i, column in enumerate(numeric_features.columns):
    ax = fig.add_subplot(3, 3, i+1)
    sns.histplot(numeric_features[column], bins=30, kde=False, ax=ax)
    ax.set_title(f'Distribution of {column}')
plt.tight_layout()
plt.show()

#%%
fig, ax = plt.subplot_mosaic(
    [
        ["left", "right", "right"],
        ["main", "main", "main"]
    ],
    figsize=(15, 8)
)
sns.boxplot(data=df, x='default', y='balance', hue='deposit', ax=ax['left'])
ax['left'].set_title('Balance by Default Status and Deposit Subscription')
sns.boxplot(data=df, x='education', y='balance', hue='deposit', ax=ax['right'])
ax['right'].set_title('Balance by Education Level and Deposit Subscription')
sns.boxplot(data=df, x='job', y='balance', hue='deposit', ax=ax['main'])
ax['main'].set_title('Balance by Job and Deposit Subscription')
ax['main'].tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.show()

fig = plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='job', y='age', hue='job')
plt.title('Age Distribution by Job')
plt.tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.show()

#%%
df.columns
fig = plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='marital', hue='deposit', stat='percent')
plt.title('Deposit Subscription by Marital Status')
plt.tight_layout()
plt.show()

fig = plt.figure()
sns.pairplot(data=numeric_features, diag_kind='kde')
plt.suptitle('Pairplot of Numeric Features', y=1.02)
plt.show()