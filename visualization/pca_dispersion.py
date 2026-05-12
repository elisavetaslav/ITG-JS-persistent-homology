from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt


def plot_pca_and_dispersion(X, title, ax_scatter, ax_disp, composers, scale=False, composer_order=['Haydn', 'Mozart', 'Beethoven']):
    if scale:
        X = StandardScaler().fit_transform(X)
    
    # Closer to Mijangos's GitHub scripts: PCA is applied directly to X
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    if composer_order is None:
        unique_comp = np.unique(composers)
    else:
        unique_comp = [c for c in composer_order if c in set(composers)]

    n_colors = len(unique_comp)
    cmap = plt.cm.tab20 if n_colors > 10 else plt.cm.tab10
    colors = cmap(np.linspace(0, 1, n_colors))
    color_map = dict(zip(unique_comp, colors))

    for comp in unique_comp:
        mask = composers == comp
        if len(unique_comp) <= 3:
            ax_scatter.scatter(
                X_pca[mask, 0], X_pca[mask, 1],
                color=color_map[comp], s=80, label=comp, alpha=0.9
            )
        else:
            ax_scatter.scatter(
                X_pca[mask, 0], X_pca[mask, 1],
                color=color_map[comp], s=40, label=comp, alpha=0.7
            )

    ax_scatter.set_title(f"{title}\nExplained: {pca.explained_variance_ratio_[:2].sum():.1%}")
    ax_scatter.set_xlabel("PC1")
    ax_scatter.set_ylabel("PC2")
    ax_scatter.grid(True, alpha=0.3)
    if len(unique_comp) > 3:
        ax_scatter.legend(
            title="Composer", fontsize='small', title_fontsize='small', loc='center left',
            bbox_to_anchor=(1, 0.5), markerscale=0.8, frameon=True
        )
        plt.subplots_adjust(hspace=0.35, wspace=0.25)
    else:
        ax_scatter.legend(title="Composer")

    # Dispersion in PCA space, with denominator (n-1), as in their scripts
    disp_values = []
    for comp in unique_comp:
        mask = composers == comp
        if mask.sum() < 2:
            continue
        points = X_pca[mask]
        center = np.mean(points, axis=0)
        sq_dists = np.sum((points - center) ** 2, axis=1)
        disp = np.sqrt(np.sum(sq_dists) / (len(points) - 1))
        disp_values.append((comp, disp))

    ax_disp.bar([c[0] for c in disp_values], [c[1] for c in disp_values], color='skyblue')
    ax_disp.set_title(f"Dispersion ({title}, PCA space)")
    ax_disp.set_ylabel("Dispersion")
    ax_disp.grid(True, axis='y', alpha=0.3)

    if len(disp_values) > 3:
        comp_names = [c[0] for c in disp_values]
        ax_disp.set_xticks(range(len(comp_names)))
        ax_disp.set_xticklabels(comp_names, rotation=90, fontsize=8)
