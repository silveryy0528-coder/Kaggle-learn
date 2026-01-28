import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from category_encoders import MEstimateEncoder


def target_encoding(df, cat_col, num_col, random_state, m=5.0):
    X_encode = df.sample(frac=0.20, random_state=random_state)
    y_encode = X_encode.pop(num_col)
    X_pretrain = df.drop(X_encode.index)
    y_train = X_pretrain.pop(num_col)
    encoder = MEstimateEncoder(cols=[cat_col], m=m)
    encoder.fit(X_encode, y_encode)
    X_train = encoder.transform(X_pretrain, y_train)
    return X_train, y_train, encoder


def select_high_corr_features(df, target='SalePrice', lower_bound=0.25, print_corr=False):
    corr = df.corr(numeric_only=True)[target].sort_values(ascending=False)
    corr = corr[abs(corr) > lower_bound].iloc[1:]
    if print_corr:
        print(f'#### {len(corr)} Numerical features with high correlation ####\n{corr}')
    return corr


def find_feature_importance(model, corr):
    data = {
        'corr': corr.values,
        'importance': model.feature_importances_
    }
    importances = pd.DataFrame(data=data, index=corr.index)
    print(f'#### Importance on selected features ####\n{\
        importances.sort_values(by='importance', ascending=False)}')


def combine_numeric_features(df_in, source_features, target_name):
    df_out = copy.deepcopy(df_in)
    df_out[target_name] = 0
    for f in source_features:
        df_out[target_name] += df_in[f]
    df_out.drop(columns=source_features, inplace=True)
    return df_out


def create_kmeans_cluster_feature(df, feature_cols, random_state, Ks=[5, 15], new_col_name='Cluster'):
    X_out = pd.DataFrame()
    X_scaled = df.loc[:, feature_cols]
    X_scaled = (X_scaled - X_scaled.mean()) / X_scaled.std()

    scores = []
    for k in Ks:
        kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        scores.append(silhouette_score(X_scaled, labels))

    plt.figure()
    plt.plot(range(2, 11), scores, marker='o')
    plt.title('Silhouette Scores for KMeans Clustering')
    plt.xlabel('Number of clusters (K)')
    plt.ylabel('Silhouette Score')
    plt.show()
    kmeans = KMeans(n_clusters=Ks[np.argmax(scores)], random_state=random_state)
    X_out[new_col_name] = kmeans.fit(X_scaled).labels_
    return X_out


def apply_pca(X, standardize=True):
    # Standardize
    if standardize:
        X = (X - X.mean(axis=0)) / X.std(axis=0)
    # Create principal components
    pca = PCA()
    X_pca = pca.fit_transform(X)
    print(f'Explained variance ratios by PCA components: {pca.explained_variance_ratio_}')
    # Convert to dataframe
    component_names = [f"PC{i+1}" for i in range(X_pca.shape[1])]
    X_pca = pd.DataFrame(X_pca, columns=component_names)
    # Create loadings
    loadings = pd.DataFrame(
        pca.components_.T,  # transpose the matrix of loadings
        columns=component_names,  # so the columns are the principal components
        index=X.columns,  # and the rows are the original features
    )
    return pca, X_pca, loadings