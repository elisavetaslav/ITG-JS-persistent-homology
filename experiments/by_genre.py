import numpy as np
import matplotlib.pyplot as plt
from analysis.comparisons import compare_methods
from visualization.pca_dispersion import plot_pca_and_dispersion


def run_genre_experiment(genre_name, df_valid):
    sub = df_valid[df_valid['genre'].str.lower() == genre_name.lower()].copy()

    X_orig = np.stack(sub['features_orig'])
    X_breg = np.stack(sub['features_breg'])
    composers = sub['composer'].values

    print(f"\n=== GENRE: {genre_name} ({len(sub)}) ===")
    compare_methods(X_orig, X_breg, composers)
    return X_orig, X_breg, composers


def experiment_1(
    df_valid,
    genres=("Minuet", "Allegro", "Adagio")
):
    """
    Runs Experiment 1: comparison of composers within the same genre.
    INPUT:
        df_valid (pd.DataFrame): DataFrame with columns
            'features_orig', 'features_breg', 'composer', 'genre'.
        genres (tuple/list): genres to analyze.
    OUTPUT:
        results (dict): dictionary of the form
            results[genre] = {
                'X_orig': ...,
                'X_breg': ...,
                'composers': ...
            }
        Genres with no valid data are skipped.
    """
    results = {}

    for genre_name in genres:
        X_orig, X_breg, composers = run_genre_experiment(genre_name, df_valid)

        if X_orig is None:
            continue

        results[genre_name] = {
            'X_orig': X_orig,
            'X_breg': X_breg,
            'composers': composers
        }

        fig = plt.figure(figsize=(14, 10))

        ax1 = fig.add_subplot(2, 2, 1)
        ax2 = fig.add_subplot(2, 2, 2)
        ax3 = fig.add_subplot(2, 2, 3)
        ax4 = fig.add_subplot(2, 2, 4)

        plot_pca_and_dispersion(
            X_orig,
            f"{genre_name.upper()}: Original method",
            ax1, ax3,
            composers
        )
        plot_pca_and_dispersion(
            X_breg,
            f"{genre_name.upper()}: Bregman-enhanced",
            ax2, ax4,
            composers
        )

        plt.tight_layout()
        plt.show()

    return results
