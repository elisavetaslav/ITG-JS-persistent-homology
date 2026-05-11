from experiments.by_genre import experiment_1
from experiments.general import experiment_2
from experiments.within_composer import experiment_3a, experiment_3b
from experiments.temporal import experiment_4
from experiments.recompute import recompute_features_pure_js
from parts.part_extraction import add_parts_column
from parts.duration_normalization import compute_global_unit, normalize_durations
from features.feature_building import compute_features_for_row


def run_all_experiments_pipeline(df, alpha=1.0, beta=1.0, lam=1.0,
    genres=("Minuet", "Allegro", "Adagio"), verbose=True, method_name=None,
    run_exp1=True, run_exp2=True, run_exp3a=True, run_exp3b=True, run_exp4=True,
):
    """
    Recompute features for the given hyperparameters and run all experiments.

    INPUT:
        df (pd.DataFrame): dataframe that already contains parsed parts and
            transition matrices inside the 'parts' column.
        alpha, beta, lam (float): hyperparameters of the weighted-JS metric.
        genres (tuple/list): genres for Experiment 1.
        run_exp1 ... run_exp4 (bool): whether to run each experiment.
        verbose (bool): whether to print progress messages.

    OUTPUT:
        results (dict) with keys:
            'params'   : chosen hyperparameters
            'df_valid' : dataframe with valid original and weighted-JS features
            'exp1'     : results of Experiment 1 or None
            'exp2'     : results of Experiment 2 or None
            'exp3a'    : results of Experiment 3A or None
            'exp3b'    : results of Experiment 3B or None
            'exp4'     : results of Experiment 4 or None
    """
    df_run = df.copy()

    if verbose:
        print(f"Recomputing features with alpha={alpha}, beta={beta}, lam={lam} ...")

    df_run['features_orig'] = None
    df_run['features_js'] = None

    for idx, row in df_run.iterrows():
        vec_o, vec_b = compute_features_for_row(row, alpha=alpha, beta=beta, lam=lam)
        if vec_o is not None:
            df_run.at[idx, 'features_orig'] = vec_o
            df_run.at[idx, 'features_js'] = vec_b

    df_valid = df_run.dropna(subset=['features_orig', 'features_js']).copy()
    if method_name is None:
        method_name = "Weighted_JS"

    if verbose:
        print(f"Successfully processed: {len(df_valid)} / {len(df_run)}")

    results = {
        'params': {'alpha': alpha, 'beta': beta, 'lam': lam},
        'df_valid': df_valid,
        'exp1': None,
        'exp2': None,
        'exp3a': None,
        'exp3b': None,
        'exp4': None
    }

    if run_exp1:
        if verbose:
            print("\n===== RUNNING EXPERIMENT 1 =====")
        results['exp1'] = experiment_1(df_valid, genres=genres, mod_col="features_js", method_name=method_name)

    if run_exp2:
        if verbose:
            print("\n===== RUNNING EXPERIMENT 2 =====")
        results['exp2'] = experiment_2(df_valid, mod_col="features_js", method_name=method_name)

    if run_exp3a:
        if verbose:
            print("\n===== RUNNING EXPERIMENT 3A =====")
        results['exp3a'] = experiment_3a(df_valid, mod_col="features_js", method_name=method_name)

    if run_exp3b:
        if verbose:
            print("\n===== RUNNING EXPERIMENT 3B =====")
        results['exp3b'] = experiment_3b(df_valid, mod_col="features_js", method_name=method_name)

    if run_exp4:
        if verbose:
            print("\n===== RUNNING EXPERIMENT 4 =====")
        results['exp4'] = experiment_4(df_valid, mod_col="features_js", method_name=method_name)

    return results


def run_all_experiments_pipeline_pure_js(
    df,
    data_root='/content/data_big',
    flat_structure=False,
    apply_sqrt=False,
    js_scale=1.0,
    edge_eps=1e-6, 
    min_num_parts=4,
    method_name="Pure_JS"
):
    """
    Run all experiments using pure Jensen-Shannon metric (no frequency term).
    INPUT: df - DataFrame (may lack 'parts' column).
           data_root - root directory containing MIDI files.
           flat_structure - if True, files directly in data_root; else nested by composer/genre.
           apply_sqrt - apply sqrt transform to durations.
           js_scale - scaling factor for JS distances.
           edge_eps - small constant to avoid zero edges.
           min_num_parts - minimum number of parts required.
    RETURN: dict with keys 'df_valid', 'exp1', 'exp2', 'exp3a', 'exp3b', 'exp4'.
    """
    df_work = df.copy()

    if 'parts' not in df_work.columns or df_work['parts'].isna().all():
        add_parts_column(df_work, data_root, flat_structure=flat_structure, inplace=True)

    GLOBAL_DF_MULT = compute_global_unit(df_work)
    normalize_durations(df_work, GLOBAL_DF_MULT, apply_sqrt=apply_sqrt, inplace=True)

    print(f"Recomputing features with pure JS: js_scale={js_scale}, edge_eps={edge_eps} ...")
    df_valid = recompute_features_pure_js(df_work, js_scale=js_scale, edge_eps=edge_eps, min_num_parts=min_num_parts)

    print("\n===== RUNNING EXPERIMENT 1 =====")
    exp1 = experiment_1(df_valid, mod_col="features_js", method_name=method_name)

    print("\n===== RUNNING EXPERIMENT 2 =====")
    exp2 = experiment_2(df_valid, mod_col="features_js", method_name=method_name)

    print("\n===== RUNNING EXPERIMENT 3A =====")
    exp3a = experiment_3a(df_valid, mod_col="features_js", method_name=method_name)

    print("\n===== RUNNING EXPERIMENT 3B =====")
    exp3b = experiment_3b(df_valid, mod_col="features_js", method_name=method_name)

    print("\n===== RUNNING EXPERIMENT 4 =====")
    exp4 = experiment_4(df_valid, mod_col="features_js", method_name=method_name)

    return {
        'df_valid': df_valid,
        'exp1': exp1,
        'exp2': exp2,
        'exp3a': exp3a,
        'exp3b': exp3b,
        'exp4': exp4
    }
