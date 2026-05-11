import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def _add_pca_columns(df, features_col, prefix, use_scaling=True):
    X = np.stack(df[features_col].values)
    if use_scaling:
      X = StandardScaler().fit_transform(X)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    df[f'{prefix}_PC1'] = X_pca[:, 0]
    df[f'{prefix}_PC2'] = X_pca[:, 1]
    explained = pca.explained_variance_ratio_[:2].sum()
    return df, explained


def experiment_4(df_valid, mod_col="features_js", method_name="Weighted_JS"):
    df_plot = df_valid.copy()
    df_plot['year'] = pd.to_numeric(df_plot['date'].astype(str).str.split('-').str[0], errors='coerce')
    df_plot = df_plot.dropna(subset=['year']).copy()

    df_plot, orig_explained = _add_pca_columns(df_plot, 'features_orig', 'orig', use_scaling=False)
    df_plot, js_explained = _add_pca_columns(df_plot, mod_col, 'js', use_scaling=False)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Experiment 4: Evolution of style by dates", fontsize=16, y=1.02)

    colors = {'Haydn': '#d62728', 'Mozart': '#1f77b4', 'Beethoven': '#2ca02c'}

    for i, method in enumerate(['orig', 'js']):
        prefix = method
        label = "Baseline method" if method == 'orig' else method_name
        ax = axes[0, i]
        for comp in ['Haydn', 'Mozart', 'Beethoven']:
            sub = df_plot[df_plot['composer'] == comp].sort_values('year')
            ax.scatter(sub['year'], sub[f'{prefix}_PC1'],
                      color=colors[comp], s=70, alpha=0.85, label=comp)
            # Trend line (optional)
            # z = np.polyfit(sub['year'], sub[f'{prefix}_PC1'], 1)
            # p = np.poly1d(z)
            # ax.plot(sub['year'], p(sub['year']), color=colors[comp], linestyle='--', alpha=0.5)

        ax.set_xlabel("Year")
        ax.set_ylabel("PC1 score")
        explained_val = orig_explained if prefix == 'orig' else js_explained
        ax.set_title(f"PC1 evolution — {label}\nExplained: {explained_val:.1%}")
        ax.grid(True, alpha=0.3)
        ax.legend(title="Composer")

        # Plot 2: RMS Evolution (sliding window 10 years)
        ax = axes[1, i]
        window_years  = 15
        for comp in ['Haydn', 'Mozart', 'Beethoven']:
            sub = df_plot[df_plot['composer'] == comp].sort_values('year')
            if len(sub) < 3:
                continue

            years = sub['year'].values
            pc_points = sub[[f'{prefix}_PC1', f'{prefix}_PC2']].values
            rms_values = []
            rms_years = []
            year_grid = np.arange(int(years.min()), int(years.max()) + 1)

            for center_year in year_grid:
                mask = (years >= center_year - window_years / 2) & (years <= center_year + window_years / 2)
                window_points = pc_points[mask]
                if len(window_points) < 2:
                    continue
                center = np.mean(window_points, axis=0)
                sq_dists = np.sum((window_points - center) ** 2, axis=1)
                rms = np.sqrt(np.sum(sq_dists) / (len(window_points) - 1))
                rms_values.append(rms)
                rms_years.append(center_year)

            if rms_values:
                ax.plot(rms_years, rms_values, color=colors[comp], marker='o', linewidth=2, label=comp)

        ax.set_xlabel(f"Year (moving window {window_years} years)")
        ax.set_ylabel("Dispersion in PCA space")
        ax.set_title(f"Evolution of intraclass variance — {label}")
        ax.grid(True, alpha=0.3)
        ax.legend(title="Composer")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
