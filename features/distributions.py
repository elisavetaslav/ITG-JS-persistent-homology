import numpy as np


def assign_distributions(P, eps=1e-8):
    """
    Assign smoothed conditional transition distributions and vertex masses to the original pitch-class vertices.
    INPUT:
        P: 12x12 numpy array (globally normalized transition matrix)
        eps: Dirichlet smoothing parameter
    OUTPUT:
        dist_dict: dict {vertex_id: probability distribution (length 12)}
        mass_dict: dict {vertex_id: scalar mass}
    Only the original pitch-class vertices 0,...,11 receive probability distributions and masses.
    Auxiliary vertices of the expanded graph do not receive independent distributions.
    """
    n = 12
    dist_dict = {}
    mass_dict = {}
    row_sums = P.sum(axis=1)

    # Original vertices
    cond_P = np.zeros_like(P)
    for i in range(n):
        denom = row_sums[i] + n * eps
        cond_P[i] = (P[i] + eps) / denom
        dist_dict[i] = cond_P[i]
        mass_dict[i] = row_sums[i]

    return dist_dict, mass_dict
