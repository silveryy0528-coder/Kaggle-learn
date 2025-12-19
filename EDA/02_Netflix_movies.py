#%%
import kagglehub
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import calendar
import seaborn as sns
from viz_utils import *

sns.set_style('darkgrid')

colors = ['#221f1f', '#b20710', '#e50914','#f5f5f1']

# Download latest version
raw_path = kagglehub.dataset_download("shivamb/netflix-shows")
df = pd.read_csv(path_join(raw_path, 'netflix_titles.csv'))
df['country'] = df['country'].fillna(df['country'].mode()[0])
df['director'] = df['director'].replace(np.nan, 'No data')
df['cast'] = df['cast'].replace(np.nan, 'No data')
df = df.dropna()
df = df.drop_duplicates()

df['date_added'] = pd.to_datetime(df['date_added'].str.strip(), format='%B %d, %Y')
df['month'] = df['date_added'].dt.month
df['month_name'] = df['date_added'].dt.month_name()
df['year'] = df['date_added'].dt.year
ref_type = df['type'].unique()

#%% Explore content over time
df['year_month'] = pd.to_datetime(df[['year', 'month']].assign(day=1))
content_per_year = (
    df
    .groupby(['type', 'year'])
    .size()
    .reset_index(name='num_content')
)
content_per_year['type'] = pd.Categorical(
    content_per_year['type'],
    categories=ref_type,
    ordered=True
)

type_ratio = df.groupby(by='type').size() / len(df['type'])
type_ratio.index = pd.Categorical(
    type_ratio.index,
    categories=ref_type,
    ordered=True
)

fig, ax = plt.subplot_mosaic(
    [['left', 'main', 'main']], figsize=(12, 4))

ax['left'].pie(
    type_ratio.values,
    labels=type_ratio.index,
    explode=(0.0, 0.05),
    autopct='%1.1f%%',
    startangle=90,
)
ax['left'].legend()
ax['left'].set_title('Netflix content distribution')
ax['left'].axis('equal')

sns.lineplot(data=content_per_year, x='year', y='num_content',
             hue='type', marker='o', ax=ax['main'])
ax['main'].set_ylabel('Number of contents')
ax['main'].set_title('Content growth over years')
ax['main'].set_xlabel('')
plt.tight_layout()
plt.show()

release_years = (
    df
    .groupby(['type', 'release_year'])
    .size()
    .reset_index(name='num_content')
)
release_years['type'] = pd.Categorical(
    release_years['type'],
    categories=ref_type,
    ordered=True
)
fig, ax = plt.subplots(figsize=(8, 4))
sns.barplot(data=release_years, x='release_year', y='num_content',
            hue='type', ax=ax, width=1)
for i, label in enumerate(ax.get_xticklabels()):
    if i % 10 != 0: label.set_visible(False)
ax.set_xlabel('')
ax.set_ylabel('Number of contents')
ax.set_title('Content released per year')
plt.show()

content_per_month = (
    df
    .groupby(['month_name', 'type'])
    .size()
    .reset_index(name='num_content')
)
content_per_month['type'] = pd.Categorical(
    content_per_month['type'],
    ordered=True,
    categories=ref_type
)
fig = plt.figure(figsize=(10, 4))
sns.barplot(data=content_per_month, x='month_name', y='num_content',
            order=[calendar.month_name[i] for i in range(1, 13)],
            hue='type')
plt.xlabel('')
plt.ylabel('Number of content')
plt.xticks(rotation=45)
plt.title('Content added per month')
plt.show()

movie_sorted = (
    df
    .query("type == 'Movie'")
    .sort_values(by='release_year', ascending=True)
    [['title', 'release_year']]
)
print('\nFive oldest movies', movie_sorted.head())
shows_sorted = (
    df
    .query("type == 'TV Show'")
    .sort_values(by='release_year', ascending=True)
    [['title', 'release_year']]
)
print('\nFive oldest TV shows', shows_sorted.head())

#%%