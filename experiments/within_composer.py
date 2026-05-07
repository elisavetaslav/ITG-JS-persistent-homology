import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def _plot_single_genre_dispersion(X, title, ax_scatter, ax_disp):
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_sc)

    ax_scatter.scatter(X_pca[:, 0], X_pca[:, 1], c='skyblue', s=80, alpha=0.7)
    ax_scatter.set_title(f"{title}\nExplained: {pca.explained_variance_ratio_[:2].sum():.1%}")
    ax_scatter.set_xlabel("PC1")
    ax_scatter.set_ylabel("PC2")
    ax_scatter.grid(True, alpha=0.3)

    center = np.mean(X_pca, axis=0)
    dists = np.linalg.norm(X_pca - center, axis=1)
    rms = np.sqrt(np.mean(dists**2))

    ax_disp.bar([title], [rms], color='lightcoral')
    ax_disp.set_title(f"Dispersion (RMS from centroid)")
    ax_disp.set_ylabel("RMS")
    ax_disp.grid(True, axis='y', alpha=0.3)


def _plot_pca_and_dispersion_single_composer(X, genre_labels, title, ax_scatter, ax_disp):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    unique_genres = sorted(pd.unique(genre_labels))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_genres)))
    color_map = dict(zip(unique_genres, colors))

    for genre in unique_genres:
        mask = (genre_labels == genre)
        ax_scatter.scatter(
            X_pca[mask, 0], X_pca[mask, 1],
            color=color_map[genre], s=80, alpha=0.8, label=genre
        )

    ax_scatter.set_title(f"{title}\nExplained: {pca.explained_variance_ratio_[:2].sum():.1%}")
    ax_scatter.set_xlabel("PC1")
    ax_scatter.set_ylabel("PC2")
    ax_scatter.grid(True, alpha=0.3)
    ax_scatter.legend(title="Genre")

    disp_values = []
    for genre in unique_genres:
        mask = (genre_labels == genre)
        pts = X_pca[mask]
        if len(pts) < 2:
            continue
        center = np.mean(pts, axis=0)
        sq_dists = np.sum((pts - center) ** 2, axis=1)
        disp = np.sqrt(np.sum(sq_dists) / (len(pts) - 1))
        disp_values.append((genre, disp))

    ax_disp.bar(
        [g for g, _ in disp_values],
        [d for _, d in disp_values],
        color=[color_map[g] for g, _ in disp_values]
    )
    ax_disp.set_title("Dispersion by genre (PCA space)")
    ax_disp.set_ylabel("Dispersion")
    ax_disp.grid(True, axis='y', alpha=0.3)
    ax_disp.tick_params(axis='x', rotation=30)


def experiment_3a(df_valid):
  results = {}
  for comp in ['Haydn', 'Mozart', 'Beethoven']:
      print(f"\n=== {comp} — within each composer by genre ===")
      sub_comp = df_valid[df_valid['composer'] == comp].copy()

      genre_counts = sub_comp['genre'].value_counts()
      valid_genres = genre_counts[genre_counts >= 3].index.tolist()

      for gname, cnt in genre_counts.items():
          if cnt < 3:
              print(f"  {gname}: {cnt} — Insufficient data")
          else:
              print(f"  {gname}: {cnt}")

      if len(valid_genres) < 2:
          print(f"  {comp}: fewer than 2 valid genres, comparison is not meaningful")
          continue

      sub_comp = sub_comp[sub_comp['genre'].isin(valid_genres)].copy()

      X_orig = np.stack(sub_comp['features_orig'].values)
      X_breg = np.stack(sub_comp['features_breg'].values)
      genre_labels = sub_comp['genre'].values

      results[comp] = {
            'X_orig': X_orig,
            'X_breg': X_breg,
            'genre_labels': genre_labels
        }

      fig = plt.figure(figsize=(14, 8))
      fig.suptitle(f"{comp} — comparison of genres", fontsize=14, y=1.02)

      ax1 = fig.add_subplot(2, 2, 1)
      ax2 = fig.add_subplot(2, 2, 2)
      ax3 = fig.add_subplot(2, 2, 3)
      ax4 = fig.add_subplot(2, 2, 4)

      _plot_pca_and_dispersion_single_composer(X_orig, genre_labels, "Original method", ax1, ax3)
      _plot_pca_and_dispersion_single_composer(X_breg, genre_labels, "Bregman-enhanced", ax2, ax4)

      plt.tight_layout(rect=[0, 0, 1, 0.96])
      plt.show()

  return results


def experiment_3b(df_valid):
  results = {}
  for comp in ['Haydn', 'Mozart', 'Beethoven']:
      print(f"\n=== {comp} — within each composer by genre ===")
      sub_comp = df_valid[df_valid['composer'] == comp]

      genre_groups = sub_comp.groupby('genre')
      valid_genres = []

      for gname, group in genre_groups:
          if len(group) < 3:
              print(f"  {gname}: {len(group)} — Insufficient data")
              continue
          valid_genres.append((gname, group))
          print(f"  {gname}: {len(group)}")

      if not valid_genres:
          print(f"  {comp}: no valid genres with more than 2 compositions")
          continue

      for i, (gname, group) in enumerate(valid_genres, 1):
          print(f"    = Genre: {gname}")
          X_orig = np.stack(group['features_orig'])
          X_breg = np.stack(group['features_breg'])

          results[f"{comp}_{gname}"] = {
              'X_orig': X_orig,
              'X_breg': X_breg
          }
          fig = plt.figure(figsize=(14, 6))
          fig.suptitle(f"{comp} — {gname} ({len(group)} comp.)", fontsize=14, y=1.02)

          ax1 = fig.add_subplot(1, 4, 1)
          ax2 = fig.add_subplot(1, 4, 2)
          ax3 = fig.add_subplot(1, 4, 3)
          ax4 = fig.add_subplot(1, 4, 4)

          _plot_single_genre_dispersion(X_orig, "Original method", ax1, ax3)
          _plot_single_genre_dispersion(X_breg, "Bregman-enhanced", ax2, ax4)

          plt.tight_layout(rect=[0, 0, 1, 0.95])
          plt.show()

  return results
