import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score


def compute_core_metrics(X, labels):
    """
    Compute the core metrics used for hyperparameter selection.
    """
    X_sc = StandardScaler().fit_transform(X)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    explained_2 = pca.explained_variance_ratio_[:2].sum()

    unique_labels = np.unique(labels)
    if len(unique_labels) < 2 or len(labels) <= len(unique_labels):
        sil_full = np.nan
        sil_pca2 = np.nan
        db = np.nan
        mean_centroid_distance = np.nan
    else:
        sil_full = silhouette_score(X_sc, labels)
        sil_pca2 = silhouette_score(X_pca, labels)
        db = davies_bouldin_score(X_sc, labels)

        centroids = {}
        for lab in unique_labels:
            pts = X_pca[np.array(labels) == lab]
            centroids[lab] = pts.mean(axis=0)

        dists = []
        labs = list(centroids.keys())
        for i in range(len(labs)):
            for j in range(i + 1, len(labs)):
                dists.append(np.linalg.norm(centroids[labs[i]] - centroids[labs[j]]))
        mean_centroid_distance = np.mean(dists) if dists else np.nan

    return {
        'explained_2': explained_2,
        'silhouette_full': sil_full,
        'silhouette_pca2': sil_pca2,
        'davies_bouldin': db,
        'mean_centroid_distance': mean_centroid_distance
    }


def compute_baseline_metrics(df_valid, tuning_slices):
    """
    Compute baseline metrics once using features_orig.
    """
    baseline_rows = []
    baseline_dict = {}

    for slice_name, meta in tuning_slices.items():
        sub = df_valid.loc[meta['index']].copy()
        X = np.stack(sub['features_orig'].values)
        labels = sub[meta['label_col']].values

        metrics = compute_core_metrics(X, labels)
        baseline_dict[slice_name] = metrics

        baseline_rows.append({
            'slice': slice_name,
            'method': 'Baseline',
            **metrics
        })

    baseline_df = pd.DataFrame(baseline_rows)
    return baseline_df, baseline_dict
