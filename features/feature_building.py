import numpy as np
from features.distributions import assign_distributions
from features.graph_construction import (
    build_expanded_graph,
    compute_bregman_edge_lengths,
    compute_original_distance_matrix,
    compute_pure_semantic_edge_lengths
)
from features.persistence import compute_persistence, extract_barcode_stats


def compute_features_for_row(row, alpha=0.75, beta=1.0, lam=1.0, min_num_parts=4):
    """Computes 24 topological features for a single piece: 4 voices × (H0 and H1) × (mean, std, entropy).
    INPUT: row (pandas.Series): a row from a DataFrame. Must contain a key 'parts' that holds a list
          of part dictionaries.
        alpha (float, optional): parameter for the Bregman divergence. Defaults to 0.75.
    RETURN: tuple of two numpy.ndarrays or (None, None):
            vec_orig (numpy.ndarray): 24-dimensional vector for the original distance method.
            vec_breg (numpy.ndarray): 24-dimensional vector for the Bregman distance method.
            If the piece has fewer than 4 parts, both return values are None.
    """
    parts = row.get('parts')
    if not parts or len(parts) < min_num_parts:
        return None, None

    vec_orig = []
    vec_breg = []
    for part in parts[:4]:
        P = part['transition_matrix']
        if P is None:
            vec_orig.extend([0.]*6)
            vec_breg.extend([0.]*6)
            continue

        vertices, vertex_map, edges = build_expanded_graph(P, min_weight=0.0, return_edges=True)

        # Original
        D_orig = compute_original_distance_matrix(vertices, edges)
        
        finite_vals = D_orig[np.isfinite(D_orig)]
        D_orig[~np.isfinite(D_orig)] = 1.1 * (np.max(finite_vals) if len(finite_vals) > 0 else 1.0)
        dgms_orig = compute_persistence(D_orig)

        # Bregman
        dist_dict, mass_dict = assign_distributions(vertices, vertex_map, P, eps=1e-8)
        edges_br = compute_bregman_edge_lengths(vertices, vertex_map, P, dist_dict, mass_dict, alpha=alpha,
                                                beta=beta, lam=lam, eps=1e-10, min_weight=1e-8, loop_multiplier=1.0)
        D_br = compute_original_distance_matrix(vertices, edges_br)

        finite_vals = D_br[np.isfinite(D_br)]
        D_br[~np.isfinite(D_br)] = 1.1 * (np.max(finite_vals) if len(finite_vals) > 0 else 1.0)
        dgms_br = compute_persistence(D_br)

        for dim in [0, 1]:
            vec_orig.extend(extract_barcode_stats(dgms_orig[dim]))
            vec_breg.extend(extract_barcode_stats(dgms_br[dim]))

    return np.array(vec_orig, dtype=float), np.array(vec_breg, dtype=float)


def compute_features_for_row_js(row, js_scale=1.0, edge_eps=1e-6, min_num_parts=4):
    """Computes 24 topological features for a single piece: 4 voices × (H0 and H1) × (mean, std, entropy).
    INPUT: row (pandas.Series): a row from a DataFrame. Must contain a key 'parts' that holds a list
          of part dictionaries.
        alpha (float, optional): parameter for the Bregman divergence. Defaults to 0.75.
    RETURN: tuple of two numpy.ndarrays or (None, None):
            vec_orig (numpy.ndarray): 24-dimensional vector for the original distance method.
            vec_breg (numpy.ndarray): 24-dimensional vector for the Bregman distance method.
            If the piece has fewer than 4 parts, both return values are None.
    """
    parts = row.get('parts')
    if not parts or len(parts) < min_num_parts:
        return None, None

    vec_orig = []
    vec_breg = []
    for part in parts[:4]:
        P = part['transition_matrix']
        if P is None:
            vec_orig.extend([0.]*6)
            vec_breg.extend([0.]*6)
            continue

        vertices, vertex_map, edges = build_expanded_graph(P, min_weight=0.0, return_edges=True)

        # Original
        D_orig = compute_original_distance_matrix(vertices, edges)

        finite_vals = D_orig[np.isfinite(D_orig)]
        D_orig[~np.isfinite(D_orig)] = 1.1 * (np.max(finite_vals) if len(finite_vals) > 0 else 1.0)
        dgms_orig = compute_persistence(D_orig)

        # Bregman
        dist_dict, mass_dict = assign_distributions(vertices, vertex_map, P, eps=1e-8)
        edges_br = compute_pure_semantic_edge_lengths(
            vertices, vertex_map, P, dist_dict, mass_dict,
            js_scale=js_scale, eps=edge_eps, min_weight=1e-8
        )
        D_br = compute_original_distance_matrix(vertices, edges_br)
        finite_vals = D_br[np.isfinite(D_br)]
        D_br[~np.isfinite(D_br)] = 1.1 * (np.max(finite_vals) if len(finite_vals) > 0 else 1.0)
        dgms_br = compute_persistence(D_br)

        for dim in [0, 1]:
            vec_orig.extend(extract_barcode_stats(dgms_orig[dim]))
            vec_breg.extend(extract_barcode_stats(dgms_br[dim]))

    return np.array(vec_orig, dtype=float), np.array(vec_breg, dtype=float)
