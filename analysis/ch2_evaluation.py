import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.model_selection import LeaveOneOut
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


def get_label_order(labels):
    """Get a consistent label order.
    INPUT: labels (array-like) - class labels
    RETURN: list - ordered unique labels
    """
    expected_order = ["Haydn", "Mozart", "Beethoven"]
    present = list(pd.unique(labels))
    ordered = [label for label in expected_order if label in present]
    ordered += [label for label in sorted(present) if label not in ordered]
    return ordered


def compute_class_centroids(X, y):
    """Compute class centroids in feature space.
    INPUT: X (np.ndarray shape (n_samples, n_features)) - feature matrix
           y (np.ndarray shape (n_samples,)) - class labels
    RETURN: dict - centroid vectors keyed by label
    """
    centroids = {}
    for label in get_label_order(y):
        centroids[label] = X[y == label].mean(axis=0)
    return centroids


def compute_within_class_dispersion(X, y):
    """Compute within-class dispersion statistics in the full feature space.
    INPUT: X (np.ndarray shape (n_samples, n_features)) - feature matrix
           y (np.ndarray shape (n_samples,)) - class labels
    RETURN: pd.DataFrame - per-class dispersion table
    """
    rows = []
    centroids = compute_class_centroids(X, y)

    for label in get_label_order(y):
        Xc = X[y == label]
        centroid = centroids[label]
        dists = np.linalg.norm(Xc - centroid, axis=1)
        sq_dists = np.sum((Xc - centroid) ** 2, axis=1)

        if len(Xc) > 1:
            rms_distance = float(np.sqrt(np.sum(sq_dists) / (len(Xc) - 1)))
        else:
            rms_distance = 0.0
        rows.append({
            "label": label,
            "n_samples": int(len(Xc)),
            "mean_distance": float(np.mean(dists)),
            "median_distance": float(np.median(dists)),
            "max_distance": float(np.max(dists)),
            "rms_distance": rms_distance,
        })

    return pd.DataFrame(rows)


def compute_centroid_distance_matrix(X, y):
    """Compute pairwise distances between class centroids.
    INPUT: X (np.ndarray shape (n_samples, n_features)) - feature matrix
           y (np.ndarray shape (n_samples,)) - class labels
    RETURN: pd.DataFrame - centroid distance matrix
    """
    centroids = compute_class_centroids(X, y)
    labels = get_label_order(y)

    D = np.zeros((len(labels), len(labels)), dtype=float)

    for i, li in enumerate(labels):
        for j, lj in enumerate(labels):
            D[i, j] = np.linalg.norm(centroids[li] - centroids[lj])

    return pd.DataFrame(D, index=labels, columns=labels)


def compute_pca_dispersion(X_scaled, y):
    """Compute per-class dispersion in the 2D PCA space.
    INPUT: X_scaled (np.ndarray shape (n_samples, n_features)) - standardized feature matrix
           y (np.ndarray shape (n_samples,)) - class labels
    RETURN: X_pca (np.ndarray shape (n_samples, 2)) - PCA coordinates
            pca (PCA) - fitted PCA object
            pd.DataFrame - per-class dispersion table in PCA space
    """
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    rows = []
    for label in get_label_order(y):
        points = X_pca[y == label]
        center = points.mean(axis=0)
        sq_dists = np.sum((points - center) ** 2, axis=1)

        if len(points) > 1:
            dispersion = float(np.sqrt(np.sum(sq_dists) / (len(points) - 1)))
        else:
            dispersion = 0.0

        rows.append({
            "label": label,
            "n_samples": int(len(points)),
            "pca_dispersion": dispersion,
        })

    return X_pca, pca, pd.DataFrame(rows)


def loo_1nn_accuracy(X, y):
    """Compute leave-one-out 1-NN accuracy.
    INPUT: X (np.ndarray shape (n_samples, n_features)) - feature matrix
           y (np.ndarray shape (n_samples,)) - class labels
    RETURN: float - LOOCV 1-NN accuracy
    """
    loo = LeaveOneOut()
    y_true = []
    y_pred = []

    for train_idx, test_idx in loo.split(X):
        clf = KNeighborsClassifier(n_neighbors=1)
        clf.fit(X[train_idx], y[train_idx])
        pred = clf.predict(X[test_idx])[0]

        y_true.append(y[test_idx][0])
        y_pred.append(pred)

    return float(np.mean(np.array(y_true) == np.array(y_pred)))


def loo_nearest_centroid_accuracy(X, y):
    """Compute leave-one-out nearest-centroid accuracy.
    INPUT: X (np.ndarray shape (n_samples, n_features)) - feature matrix
           y (np.ndarray shape (n_samples,)) - class labels
    RETURN: float - LOOCV nearest-centroid accuracy
    """
    loo = LeaveOneOut()
    y_true = []
    y_pred = []

    for train_idx, test_idx in loo.split(X):
        X_train = X[train_idx]
        y_train = y[train_idx]
        x_test = X[test_idx][0]

        centroids = compute_class_centroids(X_train, y_train)

        best_label = None
        best_dist = None

        for label, centroid in centroids.items():
            dist = np.linalg.norm(x_test - centroid)
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_label = label

        y_true.append(y[test_idx][0])
        y_pred.append(best_label)

    return float(np.mean(np.array(y_true) == np.array(y_pred)))


def evaluate_branch(df_features, feature_cols, label_col="composer", dataset_name="dataset", model_name="model"):
    """Evaluate one model by clustering and classification metrics.
    INPUT: df_features (pd.DataFrame) - feature table
           feature_cols (list[str]) - feature columns used in evaluation
           label_col (str) - target label column
           dataset_name (str) - dataset name
           model_name (str) - model name
    RETURN: summary_df (pd.DataFrame) - one-row summary table
            dispersion_df (pd.DataFrame) - within-class dispersion table
            centroid_df (pd.DataFrame) - centroid distance matrix
            pca_disp_df (pd.DataFrame) - PCA-space dispersion table
            X_scaled (np.ndarray) - standardized feature matrix
            X_pca (np.ndarray) - 2D PCA coordinates
            y (np.ndarray) - labels
            pca (PCA) - fitted PCA object
    """
    X = df_features[feature_cols].fillna(0.0).values
    y = df_features[label_col].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    dispersion_df = compute_within_class_dispersion(X_scaled, y)
    centroid_df = compute_centroid_distance_matrix(X_scaled, y)
    X_pca, pca, pca_disp_df = compute_pca_dispersion(X_scaled, y)

    centroid_vals = centroid_df.values.copy()
    centroid_vals = centroid_vals[np.triu_indices_from(centroid_vals, k=1)]

    n_labels = len(np.unique(y))
    silhouette_full = np.nan
    silhouette_pca2 = np.nan

    if 2 <= n_labels <= len(y) - 1:
        silhouette_full = float(silhouette_score(X_scaled, y))
        silhouette_pca2 = float(silhouette_score(X_pca, y))

    summary = pd.DataFrame([{
        "dataset": dataset_name,
        "model": model_name,
        "label_col": label_col,
        "n_samples": int(len(df_features)),
        "n_features": int(len(feature_cols)),
        "pc1_variance": float(pca.explained_variance_ratio_[0]),
        "pc2_variance": float(pca.explained_variance_ratio_[1]),
        "pc12_variance": float(pca.explained_variance_ratio_[:2].sum()),
        "silhouette_full": silhouette_full,
        "silhouette_pca2": silhouette_pca2,
        "davies_bouldin": float(davies_bouldin_score(X_scaled, y)),
        "mean_within_class_rms": float(dispersion_df["rms_distance"].mean()),
        "max_within_class_rms": float(dispersion_df["rms_distance"].max()),
        "min_centroid_distance": float(np.min(centroid_vals)),
        "mean_centroid_distance": float(np.mean(centroid_vals)),
        "max_centroid_distance": float(np.max(centroid_vals)),
        "loo_1nn_accuracy": float(loo_1nn_accuracy(X_scaled, y)),
        "loo_nearest_centroid_accuracy": float(loo_nearest_centroid_accuracy(X_scaled, y)),
    }])

    return summary, dispersion_df, centroid_df, pca_disp_df, X_scaled, X_pca, y, pca
