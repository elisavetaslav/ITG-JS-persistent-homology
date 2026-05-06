import numpy as np
from scipy.special import rel_entr


def weighted_js_distance(p, q, alpha_p, alpha_q, eps=1e-12):
    """Weighted Jensen-Shannon distance.
    INPUT: p, q: probability distributions (numpy arrays)
           alpha_p, alpha_q: nonnegative vertex masses
           eps: float – small constant for numerical stability (default 1e-12)
    OUTPUT: scalar distance
    """
    if alpha_p + alpha_q == 0:
        return 0.0
    w_p = alpha_p / (alpha_p + alpha_q)
    w_q = alpha_q / (alpha_p + alpha_q)
    m = w_p * p + w_q * q

    kl_pm = np.sum(rel_entr(p + eps, m + eps))
    kl_qm = np.sum(rel_entr(q + eps, m + eps))
    js = w_p * kl_pm + w_q * kl_qm
    return np.sqrt(np.clip(js, 0.0, None))


def compute_distance_matrix(vertices, dist_dict, mass_dict):
    """Compute pairwise weighted Jensen–Shannon distances between all vertices."""
    n = len(vertices)
    D = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            v_i = vertices[i]
            v_j = vertices[j]

            d = weighted_js_distance(dist_dict[v_i], dist_dict[v_j],
                                     mass_dict[v_i], mass_dict[v_j])
            D[i, j], D[j, i] = d, d
    return D
