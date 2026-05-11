import numpy as np
import matplotlib.pyplot as plt
from analysis.comparisons import compare_methods
from visualization.pca_dispersion import plot_pca_and_dispersion


def run_genre_experiment(genre_name, df_valid, mod_col="features_js", method_name="Weighted_JS"):
    sub = df_valid[df_valid['genre'].str.lower() == genre_name.lower()].copy()

    if len(sub) == 0:
        print(f"\n=== GENRE: {genre_name} (0) ===")
        print("No data for this genre.")
        return None, None, None

    X_orig = np.stack(sub['features_orig'])
    X_mod = np.stack(sub[mod_col].values)
    composers = sub['composer'].values

    print(f"\n=== GENRE: {genre_name} ({len(sub)}) ===")
    compare_methods(X_orig, X_mod, composers, method_name=method_name)
    return X_orig, X_mod, composers


def experiment_1(
    df_valid,
    genres=("Minuet", "Allegro", "Adagio"),
    mod_col="features_js",
    method_name="Weighted_JS"
):
    """
    Runs Experiment 1: comparison of composers within the same genre.
    INPUT:
        df_valid (pd.DataFrame): DataFrame with columns
            'features_orig', 'features_js', 'composer', 'genre'.
        genres (tuple/list): genres to analyze.
    OUTPUT:
        results (dict): dictionary of the form
            results[genre] = {
                'X_orig': ...,
                'X_mod': ...,
                'composers': ...
            }
        Genres with no valid data are skipped.
    """
    results = {}

    for genre_name in genres:
        X_orig, X_mod, composers = run_genre_experiment(genre_name, df_valid, mod_col, method_name)

        if X_orig is None:
            continue

        results[genre_name] = {
            'X_orig': X_orig,
            'X_mod': X_mod,
            'composers': composers
        }

        fig = plt.figure(figsize=(14, 10))

        ax1 = fig.add_subplot(2, 2, 1)
        ax2 = fig.add_subplot(2, 2, 2)
        ax3 = fig.add_subplot(2, 2, 3)
        ax4 = fig.add_subplot(2, 2, 4)

        plot_pca_and_dispersion(
            X_orig,
            f"{genre_name.upper()}: Baseline method",
            ax1, ax3,
            composers
        )
        plot_pca_and_dispersion(
            X_mod,
            f"{genre_name.upper()}: {method_name}",
            ax2, ax4,
            composers
        )

        plt.tight_layout()
        plt.show()

    return results
