from features.feature_building import (
    compute_features_for_row,
    compute_features_for_row_js
)

def recompute_features_pure_js(df, js_scale=1.0, edge_eps=1e-6, min_num_parts=4):
    """
    Recompute features using pure Jensen-Shannon distance.
    INPUT: df - pandas DataFrame with 'parts' column.
           js_scale - float, scaling factor for JS distances.
           edge_eps - float, small constant to avoid zero edges.
           min_num_parts - int, minimum number of parts required.
    RETURN: DataFrame with added columns 'features_orig' (original method) and
            'features_breg' (JS-based), rows with failed processing dropped.
            Prints number of successfully processed pieces.
    """
    df_work = df.copy()
    df_work['features_orig'] = None
    df_work['features_breg'] = None

    for idx, row in df_work.iterrows():
        vec_o, vec_b = compute_features_for_row_js(
            row,
            js_scale=js_scale,
            edge_eps=edge_eps,
            min_num_parts=min_num_parts
        )
        if vec_o is not None:
            df_work.at[idx, 'features_orig'] = vec_o
            df_work.at[idx, 'features_breg'] = vec_b

    df_valid = df_work.dropna(subset=['features_orig', 'features_breg']).copy()
    print(f"Successfully processed: {len(df_valid)} / {len(df_work)}")
    return df_valid


def recompute_bregman_features(df, alpha, beta, lam):
    """
    Recompute only Bregman (divergence-based) features for fixed parameters.
    INPUT: df - pandas DataFrame with existing 'features_orig' column.
           alpha, beta, lam - parameters for Bregman edge lengths.
    RETURN: DataFrame with updated 'features_breg' column, rows where processing
            failed or 'features_orig' missing are dropped.
    """
    df_tmp = df.copy()
    df_tmp['features_breg'] = None

    for idx, row in df_tmp.iterrows():
        if row.get('features_orig') is None:
            continue

        _, vec_b = compute_features_for_row(
            row,
            alpha=alpha,
            beta=beta,
            lam=lam
        )
        if vec_b is not None:
            df_tmp.at[idx, 'features_breg'] = vec_b

    df_valid_param = df_tmp.dropna(subset=['features_orig', 'features_breg']).copy()
    return df_valid_param
