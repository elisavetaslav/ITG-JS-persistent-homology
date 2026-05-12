import numpy as np
from ripser import ripser


def compute_persistence(D, max_dim=1):
    """
    INPUT: distance matrix D (n x n)
    OUTPUT: list of persistence diagrams [H0, H1] as returned by ripser
    """
    result = ripser(D, maxdim=max_dim, distance_matrix=True)
    return result['dgms']


def extract_barcode_stats(dgm):
    """
    INPUT: persistence diagram (array of [birth, death])
    OUTPUT: (mean, std, entropy) of bar lengths, computed in a way close to
            the authors' statistics.m script:
            - mean and std are computed only from finite bar lengths
            - each infinite bar is assigned surrogate length m+1, where
              m is the maximum finite bar length in the same diagram
            - entropy is computed using both finite bars and these surrogate
              infinite bars
    """
    if len(dgm) == 0:
        return (0.0, 0.0, 0.0)

    dgm = np.asarray(dgm, dtype=float)
    births = dgm[:, 0]
    deaths = dgm[:, 1]

    finite_mask = np.isfinite(deaths)
    finite_lengths = deaths[finite_mask] - births[finite_mask]
    finite_lengths = finite_lengths[finite_lengths >= 0]

    n_inf = np.sum(~finite_mask)

    # mean and std: only finite bars, as in statistics.m
    if len(finite_lengths) > 0:
        mean_len = float(np.mean(finite_lengths))
        std_len = float(np.std(finite_lengths, ddof=1)) if len(finite_lengths) > 1 else 0.0
        m = float(np.max(finite_lengths))
    else:
        mean_len = 0.0
        std_len = 0.0
        m = 0.0

    # entropy: finite bars + surrogate infinite bars of length m+1
    surrogate = m + 1.0
    total_len = float(np.sum(finite_lengths) + n_inf * surrogate)

    if total_len > 0:
        probs = []
        if len(finite_lengths) > 0:
            probs.append(finite_lengths / total_len)
        if n_inf > 0:
            probs.append(np.full(n_inf, surrogate / total_len))
        probs = np.concatenate(probs) if probs else np.array([])
        entropy = float(-np.sum(probs * np.log(probs)))
    else:
        entropy = 0.0

    return mean_len, std_len, entropy


def extract_filtered_barcode_stats(dgm, min_lifetime=0.0, top_k=None):
    """Extract filtered barcode statistics from a persistence diagram.
    INPUT: dgm (np.ndarray shape (n_bars, 2)) - persistence diagram
           min_lifetime (float) - minimum finite lifetime to retain
           top_k (int or None) - if not None, keep only the top-k longest finite bars
    RETURN: tuple - mean, std, entropy of the filtered bar lengths
    """
    if len(dgm) == 0:
        return (0.0, 0.0, 0.0)

    dgm = np.asarray(dgm, dtype=float)
    births = dgm[:, 0]
    deaths = dgm[:, 1]

    finite_mask = np.isfinite(deaths)
    finite_lengths = deaths[finite_mask] - births[finite_mask]
    finite_lengths = finite_lengths[finite_lengths >= min_lifetime]

    if top_k is not None and len(finite_lengths) > top_k:
        finite_lengths = np.sort(finite_lengths)[-top_k:]

    n_inf = np.sum(~finite_mask)

    if len(finite_lengths) > 0:
        mean_len = float(np.mean(finite_lengths))
        std_len = float(np.std(finite_lengths, ddof=1)) if len(finite_lengths) > 1 else 0.0
        m = float(np.max(finite_lengths))
    else:
        mean_len = 0.0
        std_len = 0.0
        m = 0.0

    surrogate = m + 1.0
    total_len = float(np.sum(finite_lengths) + n_inf * surrogate)

    if total_len > 0:
        probs = []
        if len(finite_lengths) > 0:
            probs.append(finite_lengths / total_len)
        if n_inf > 0:
            probs.append(np.full(n_inf, surrogate / total_len))
        probs = np.concatenate(probs) if probs else np.array([])
        entropy = float(-np.sum(probs * np.log(probs)))
    else:
        entropy = 0.0

    return mean_len, std_len, entropy
