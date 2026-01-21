#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_selection import mutual_info_regression


def make_mi_scores(X, y, discrete_features):
    mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores


def plot_mi_scores(scores):
    plt.figure(dpi=100, figsize=(5, len(scores) * 0.3))
    scores = scores.sort_values(ascending=True)
    width = np.arange(len(scores))
    ticks = list(scores.index)
    plt.barh(width, scores)
    plt.yticks(width, ticks)
    plt.title("Mutual Information Scores")
    plt.tight_layout()
    plt.show()


#%%
train_data_path = '../ML/housing_prices_train.csv'
df = pd.read_csv(train_data_path)

missing_val_count_by_column = df.isnull().sum()
cols_to_keep = missing_val_count_by_column[missing_val_count_by_column == 0]
train_df = df[cols_to_keep.index].copy()

y = train_df.SalePrice
X = train_df.drop(columns=['SalePrice'], axis=1)

#%%
cat_cols = X.select_dtypes("object").columns
for col in cat_cols:
    X[col], _ = X[col].factorize()

discrete_features = X.columns.isin(cat_cols)

mi_scores = make_mi_scores(X, y, discrete_features)
plot_mi_scores(mi_scores[:10])

#%%
X.groupby('Neighborhood')['GrLivArea'].median().sort_values().plot.barh()