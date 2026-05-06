import numpy as np


def assign_distributions(vertices, vertex_map, P, eps=1e-8):
    """ Assign probability distributions and vertex masses for the expanded graph.
    INPUT: vertices: list[int] – all vertex IDs (original + added)
           vertex_map: dict {(i,j): new_id}
           P: 12x12 numpy array (globally normalized transition matrix)
           eps: Dirichlet smoothing parameter
    OUTPUT: dist_dict: dict {vertex_id: probability distribution (length 12)}
            mass_dict: dict {vertex_id: scalar mass alpha}
    Probability distributions and masses are assigned only to the original
    vertices 0,...,11. Added vertices of the expanded graph are treated as
    transition labels and do not receive independent probability distributions.
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
