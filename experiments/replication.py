import numpy as np
import matplotlib.pyplot as plt
from parts.part_extraction import add_parts_column
from parts.duration_normalization import compute_global_unit, normalize_durations
from features.feature_building import compute_features_for_row, compute_features_for_row_js
from visualization.pca_dispersion import plot_pca_and_dispersion
from analysis.comparisons import compare_methods


def run_small_dataset_pipeline(
    df_small,
    data_root=None,
    refresh_parts=False,
    apply_sqrt=False,
    use_pure_js=True,
    js_scale=1.0,
    edge_eps=1e-6,
    alpha=1.0,
    beta=1.0,
    lam=1.0,
    min_num_parts=4,
    plot_title_suffix="",
    scale=False,
    composer_order=['Haydn', 'Mozart', 'Beethoven'],
    figsize=(14, 10)
):
    """
    Full pipeline for the small dataset.

    INPUT: df_small : pd.DataFrame - Small metadata dataframe.
    data_root : str or None
        Path to midi root. Needed only if parts must be added/refreshed.
    refresh_parts : bool - If True, force reloading parts from disk.
    apply_sqrt : bool - Passed into normalize_durations.
    use_pure_js : bool
        If True, uses pure JS version inside compute_features_for_row.
        Otherwise uses mixed Bregman version with alpha, beta, lam.
    js_scale : float - Scale parameter for pure JS version.
    edge_eps : float - Numerical epsilon for pure JS version.
    alpha, beta, lam : float - Hyperparameters for mixed Bregman version.
    plot_title_suffix : str - Optional text to append to titles.
    figsize : tuple - Figure size for the final 2x2 plot.
    OUTPUT: dict with df_valid, X_orig, X_breg, composers
    """
    df_work = df_small.copy()

    # 1. Add / refresh parts if needed
    need_parts = ('parts' not in df_work.columns) or refresh_parts
    if need_parts:
        if data_root is None:
            raise ValueError("data_root must be provided if parts need to be loaded.")
        add_parts_column(df_work, data_root, flat_structure=True, inplace=True)

    # 2. Normalize durations using the current global-unit logic
    global_mult = compute_global_unit(df_work)
    normalize_durations(df_work, global_mult, apply_sqrt=apply_sqrt, inplace=True)

    # 3. Recompute features
    df_work['features_orig'] = None
    df_work['features_breg'] = None

    for idx, row in df_work.iterrows():
        if use_pure_js:
            vec_o, vec_b = compute_features_for_row_js(row, js_scale=js_scale, edge_eps=edge_eps, min_num_parts=min_num_parts)
        else:
            vec_o, vec_b = compute_features_for_row(row, use_pure_js=False, alpha=alpha, beta=beta, lam=lam, min_num_parts=min_num_parts)

        if vec_o is not None:
            df_work.at[idx, 'features_orig'] = vec_o
            df_work.at[idx, 'features_breg'] = vec_b

    df_valid = df_work.dropna(subset=['features_orig', 'features_breg']).copy()
    print(f"Successfully processed: {len(df_valid)} / {len(df_work)}")

    # 4. Prepare matrices
    X_orig = np.stack(df_valid['features_orig'].values)
    X_breg = np.stack(df_valid['features_breg'].values)
    composers = df_valid['composer'].values

    # 5. Print numeric comparison
    print("\n=== SMALL DATASET ANALYSIS ===")
    print(df_valid['composer'].value_counts())
    compare_methods(X_orig, X_breg, composers)

    # 6. Plot PCA + dispersion
    if use_pure_js:
        breg_label = f"JS-enhanced{plot_title_suffix}"
    else:
        breg_label = f"Bregman-enhanced (α={alpha}, β={beta}, λ={lam}){plot_title_suffix}"

    fig = plt.figure(figsize=figsize)

    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 3)
    ax4 = fig.add_subplot(2, 2, 4)

    plot_pca_and_dispersion(X_orig, f"Original method{plot_title_suffix}", ax1, ax3, composers, scale=scale, composer_order=composer_order)
    plot_pca_and_dispersion(X_breg, breg_label, ax2, ax4, composers, scale=scale, composer_order=composer_order)

    plt.tight_layout()
    plt.show()

    return {
        'df_valid': df_valid,
        'X_orig': X_orig,
        'X_breg': X_breg,
        'composers': composers
    }
