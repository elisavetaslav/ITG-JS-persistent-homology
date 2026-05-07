import numpy as np
import pandas as pd
from itertools import product
from analysis.metrics import compute_baseline_metrics, compute_core_metrics
from experiments.recompute import recompute_bregman_features


def build_tuning_slices(df_valid):
    """
    Build slices for hyperparameter tuning: Exp1 (within genre), Exp2 (all genres), Exp3A (within composer, labels=genre).
    INPUT: df_valid - pandas DataFrame with columns 'genre', 'composer'.
    RETURN: dict {slice_name: {'index': index, 'label_col': 'composer' or 'genre'}}
    """
    slices = {}

    # Experiment 1
    for genre_name in ["Minuet", "Allegro", "Adagio"]:
        sub = df_valid[df_valid['genre'].str.lower() == genre_name.lower()].copy()
        if len(sub) >= 3 and sub['composer'].nunique() >= 2:
            slices[f"Exp1:{genre_name}"] = {
                'index': sub.index,
                'label_col': 'composer'
            }

    # Experiment 2
    if len(df_valid) >= 3 and df_valid['composer'].nunique() >= 2:
        slices["Exp2:AllGenres"] = {
            'index': df_valid.index,
            'label_col': 'composer'
        }

    # Experiment 3A
    for comp in ['Haydn', 'Mozart', 'Beethoven']:
        sub_comp = df_valid[df_valid['composer'] == comp].copy()
        genre_counts = sub_comp['genre'].value_counts()
        valid_genres = genre_counts[genre_counts >= 3].index.tolist()

        if len(valid_genres) < 2:
            continue

        sub_comp = sub_comp[sub_comp['genre'].isin(valid_genres)].copy()
        if len(sub_comp) >= 3 and sub_comp['genre'].nunique() >= 2:
            slices[f"Exp3A:{comp}"] = {
                'index': sub_comp.index,
                'label_col': 'genre'
            }

    return slices


def evaluate_hyperparameter_grid(df, alpha_grid, beta_grid, lam_grid):
    """
    Grid search over JS parameters evaluating on tuning slices.
    INPUT: df - pandas DataFrame with existing 'features_orig' column.
           alpha_grid, beta_grid, lam_grid - iterables of numbers.
    RETURN: (baseline_df, detail_df, summary_df)
            baseline_df: metrics per slice for original features.
            detail_df: per (params, slice) metrics and deltas.
            summary_df: aggregated per-params ranking, sorted by quality.
    """
    if 'features_orig' not in df.columns:
        raise ValueError("Column 'features_orig' not found. Compute baseline features first.")

    df_valid_base = df.dropna(subset=['features_orig']).copy()
    tuning_slices = build_tuning_slices(df_valid_base)

    baseline_df, baseline_dict = compute_baseline_metrics(df_valid_base, tuning_slices)

    detail_rows = []

    for alpha, beta, lam in product(alpha_grid, beta_grid, lam_grid):
        print(f"Running alpha={alpha}, beta={beta}, lam={lam} ...")

        df_valid_param = recompute_bregman_features(df, alpha=alpha, beta=beta, lam=lam)

        for slice_name, meta in tuning_slices.items():
            sub = df_valid_param.loc[df_valid_param.index.intersection(meta['index'])].copy()

            if len(sub) < 3 or sub[meta['label_col']].nunique() < 2:
                continue

            X_breg = np.stack(sub['features_breg'].values)
            labels = sub[meta['label_col']].values

            breg_metrics = compute_core_metrics(X_breg, labels)
            base_metrics = baseline_dict[slice_name]

            db_improvement = base_metrics['davies_bouldin'] - breg_metrics['davies_bouldin']

            detail_rows.append({
                'alpha': alpha,
                'beta': beta,
                'lam': lam,
                'slice': slice_name,

                'orig_explained_2': base_metrics['explained_2'],
                'breg_explained_2': breg_metrics['explained_2'],
                'delta_explained_2': breg_metrics['explained_2'] - base_metrics['explained_2'],

                'orig_silhouette_full': base_metrics['silhouette_full'],
                'breg_silhouette_full': breg_metrics['silhouette_full'],
                'delta_silhouette_full': breg_metrics['silhouette_full'] - base_metrics['silhouette_full'],

                'orig_silhouette_pca2': base_metrics['silhouette_pca2'],
                'breg_silhouette_pca2': breg_metrics['silhouette_pca2'],
                'delta_silhouette_pca2': breg_metrics['silhouette_pca2'] - base_metrics['silhouette_pca2'],

                'orig_davies_bouldin': base_metrics['davies_bouldin'],
                'breg_davies_bouldin': breg_metrics['davies_bouldin'],
                'db_improvement': db_improvement,  # positive = better

                'orig_centroid_distance': base_metrics['mean_centroid_distance'],
                'breg_centroid_distance': breg_metrics['mean_centroid_distance'],
                'delta_centroid_distance': breg_metrics['mean_centroid_distance'] - base_metrics['mean_centroid_distance'],
            })

    detail_df = pd.DataFrame(detail_rows)

    summary_df = (
        detail_df
        .groupby(['alpha', 'beta', 'lam'], as_index=False)
        .agg(
            n_slices=('slice', 'count'),

            mean_delta_silhouette_full=('delta_silhouette_full', 'mean'),
            mean_delta_silhouette_pca2=('delta_silhouette_pca2', 'mean'),
            mean_db_improvement=('db_improvement', 'mean'),
            mean_delta_centroid_distance=('delta_centroid_distance', 'mean'),
            mean_delta_explained_2=('delta_explained_2', 'mean'),

            min_delta_silhouette_full=('delta_silhouette_full', 'min'),
            min_db_improvement=('db_improvement', 'min'),

            n_better_silhouette_full=('delta_silhouette_full', lambda x: int(np.sum(x > 0))),
            n_better_db=('db_improvement', lambda x: int(np.sum(x > 0))),
            n_better_centroid=('delta_centroid_distance', lambda x: int(np.sum(x > 0))),
        )
    )

    # Majority flags
    summary_df['majority_silhouette'] = summary_df['n_better_silhouette_full'] >= np.ceil(summary_df['n_slices'] / 2)
    summary_df['majority_db'] = summary_df['n_better_db'] >= np.ceil(summary_df['n_slices'] / 2)

    # Rank-based robust score: smaller is better
    rank_specs = {
        'mean_delta_silhouette_full': 3.0,
        'mean_db_improvement': 2.5,
        'mean_delta_centroid_distance': 1.0,
        'min_delta_silhouette_full': 2.0,
        'n_better_silhouette_full': 1.5,
        'n_better_db': 1.5,
        'mean_delta_silhouette_pca2': 0.75,
        'mean_delta_explained_2': 0.25,
    }

    for col, weight in rank_specs.items():
        summary_df[f'rank_{col}'] = summary_df[col].rank(ascending=False, method='min')
        summary_df[f'weighted_rank_{col}'] = weight * summary_df[f'rank_{col}']

    weighted_rank_cols = [f'weighted_rank_{col}' for col in rank_specs]
    summary_df['global_rank_score'] = summary_df[weighted_rank_cols].sum(axis=1)

    summary_df = summary_df.sort_values(
        by=[
            'majority_silhouette',
            'majority_db',
            'global_rank_score',
            'mean_delta_silhouette_full',
            'mean_db_improvement'
        ],
        ascending=[False, False, True, False, False]
    ).reset_index(drop=True)

    return baseline_df, detail_df, summary_df
