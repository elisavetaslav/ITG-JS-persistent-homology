import numpy as np


def build_transition_matrix(pitches, durations, onsets):
    """Builds a 12x12 weighted transition probability matrix.
    The matrix is globally normalized so that sum_{i,j} P[i,j] = 1

    Thus P[i,j] represents the relative frequency of transitions
    between pitch classes i -> j across the piece.
    (This is not a row-stochastic Markov matrix but a joint
    distribution over transition pairs, following the construction
    used in Mijangos et al. (2022).)

    INPUT: pitches (list[int])      - pitch classes (0-11)
           durations (list[float])  - durations (quarterLength)
           onsets (list[float])     - onset positions (quarterLength)
    RETURN: P (np.ndarray shape (12,12)) - normalized transition matrix
            None if fewer than 2 notes
    """
    if len(pitches) < 2:
        return None
    P = np.zeros((12, 12))
    for k in range(len(pitches) - 1):
        i = pitches[k]
        j = pitches[k + 1]
        # A transition is counted only if there is no temporal gap.
        # Simultaneous onsets may occur due to encoding artifacts,
        # but since the data are extracted per voice, these events
        # still represent sequential notes within the same part.
        if onsets[k] + durations[k] < onsets[k + 1]:
            continue  #     A transition between two notes is counted only if there is no temporal gap between them.
        w = durations[k] * durations[k+1]
        P[i, j] += w
    total = np.sum(P)
    if total > 0:
        P /= total
    return P
