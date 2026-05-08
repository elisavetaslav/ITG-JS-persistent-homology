import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score


def compare_methods(X_orig, X_breg, composers):
    results = {}

    for name, X in [("Baseline", X_orig), ("Weighted_JS", X_breg)]:
        print(f"=== {name} ===")

        # 1. Explained variance
        X_sc = StandardScaler().fit_transform(X)
        pca = PCA()
        pca.fit(X)
        expl = pca.explained_variance_ratio_
        print(f"Explained variance (PC1+PC2): {expl[:2].sum():.3f}")
        print(f"PC1: {expl[0]:.3f}, PC2: {expl[1]:.3f}")

        # 2. Silhouette in full space
        sil_full = silhouette_score(X_sc, composers)
        print(f"Silhouette (full space): {sil_full:.4f}")

        # 3. Silhouette в PC2
        X_pca2 = PCA(n_components=2).fit_transform(X)
        sil_pca = silhouette_score(X_pca2, composers)
        print(f"Silhouette (PC1+PC2): {sil_pca:.4f}")

        # 4. Davies-Bouldin index
        db = davies_bouldin_score(X_sc, composers)
        print(f"Davies-Bouldin: {db:.4f}")

        # 5. Intraclass variance (RMS from the centroid)
        print("Intraclass variance (RMS):")
        unique_comp = np.unique(composers)
        for comp in unique_comp:
            mask = np.array(composers) == comp
            if mask.sum() < 2: continue
            points = X_pca2[mask]
            center = np.mean(points, axis=0)
            sq_dists = np.sum((points - center) ** 2, axis=1)
            disp = np.sqrt(np.sum(sq_dists) / (len(points) - 1))
            print(f"  {comp:10} = {disp:.4f}")

        # 6. Inter-class distance (average distance between centroids)
        centroids = {}
        for comp in unique_comp:
            mask = np.array(composers) == comp
            if mask.sum() < 1: continue
            centroids[comp] = np.mean(X_pca2[mask], axis=0)

        inter_dist = []
        for i, c1 in enumerate(unique_comp):
            for c2 in unique_comp[i+1:]:
                if c1 in centroids and c2 in centroids:
                    d = np.linalg.norm(centroids[c1] - centroids[c2])
                    inter_dist.append(d)
        if inter_dist:
            print(f"The average distance between the centroids (PCA space): {np.mean(inter_dist):.4f}\n")

        results[name] = {
            'explained_2': expl[:2].sum(),
            'sil_full': sil_full,
            'sil_pca': sil_pca,
            'db': db,
        }

    print(f"{'Metric':<20} {'Baseline':<12} {'Weighted_JS':<12} {'Diff':<10}")
    print("-"*54)
    print(f"Explained PC1+PC2   {results['Baseline']['explained_2']:.4f}   {results['Weighted_JS']['explained_2']:.4f}   {results['Weighted_JS']['explained_2'] - results['Baseline']['explained_2']:+.4f}")
    print(f"Silhouette (full)   {results['Baseline']['sil_full']:.4f}   {results['Weighted_JS']['sil_full']:.4f}   {results['Weighted_JS']['sil_full'] - results['Baseline']['sil_full']:+.4f}")
    print(f"Silhouette (PCA2)   {results['Baseline']['sil_pca']:.4f}   {results['Weighted_JS']['sil_pca']:.4f}   {results['Weighted_JS']['sil_pca'] - results['Baseline']['sil_pca']:+.4f}")
    print(f"Davies-Bouldin      {results['Baseline']['db']:.4f}   {results['Weighted_JS']['db']:.4f}   {results['Weighted_JS']['db'] - results['Baseline']['db']:+.4f}")
