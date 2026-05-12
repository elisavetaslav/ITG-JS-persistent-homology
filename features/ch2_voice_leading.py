import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from tqdm.auto import tqdm
from features.distances import weighted_js_distance
from features.persistence import compute_persistence, extract_barcode_stats


def transition_cost(a, b, rest_cost=12.0):
    """Compute the matching cost between two pitch states.
    INPUT: a (int or None) - source MIDI pitch or None for rest
           b (int or None) - target MIDI pitch or None for rest
           rest_cost (float) - penalty used for note-to-rest or rest-to-note matching
    RETURN: float - pairwise matching cost
    """
    if a is None and b is None:
        return 0.0
    if a is None or b is None:
        return float(rest_cost)
    return float(abs(a - b))


def match_pitch_sets(source_set, target_set, rest_cost=12.0):
    """Match two pitch sets by minimum-cost bipartite assignment.
    INPUT: source_set (tuple[int]) - source pitches in one part at time t
           target_set (tuple[int]) - target pitches in the same part at time t+1
           rest_cost (float) - penalty used for note-to-rest or rest-to-note matching
    RETURN: list[tuple[int or None, int or None]] - matched source/target pairs
    """
    source = list(source_set)
    target = list(target_set)

    if len(source) == 0:
        source = [None]
    if len(target) == 0:
        target = [None]

    n = max(len(source), len(target))
    source = source + [None] * (n - len(source))
    target = target + [None] * (n - len(target))

    C = np.zeros((n, n), dtype=float)
    for i, a in enumerate(source):
        for j, b in enumerate(target):
            C[i, j] = transition_cost(a, b, rest_cost=rest_cost)

    row_ind, col_ind = linear_sum_assignment(C)
    pairs = [(source[i], target[j]) for i, j in zip(row_ind, col_ind)]

    return pairs


def match_pitch_items(source_items, target_items, rest_cost=12.0):
    """Match two lists of atomic pitch items by minimum-cost bipartite assignment.
    INPUT: source_items (list[dict]) - source atomic items with keys 'pitch', 'full_dur'
           target_items (list[dict]) - target atomic items with keys 'pitch', 'full_dur'
           rest_cost (float) - penalty used for note-to-rest or rest-to-note matching
    RETURN: list[tuple[dict or None, dict or None]] - matched source/target item pairs
    """
    source = [dict(x) for x in source_items]
    target = [dict(x) for x in target_items]

    if len(source) == 0:
        source = [None]
    if len(target) == 0:
        target = [None]

    n = max(len(source), len(target))
    source = source + [None] * (n - len(source))
    target = target + [None] * (n - len(target))

    C = np.zeros((n, n), dtype=float)
    for i, a in enumerate(source):
        for j, b in enumerate(target):
            a_pitch = None if a is None else a["pitch"]
            b_pitch = None if b is None else b["pitch"]
            C[i, j] = transition_cost(a_pitch, b_pitch, rest_cost=rest_cost)

    row_ind, col_ind = linear_sum_assignment(C)
    pairs = [(source[i], target[j]) for i, j in zip(row_ind, col_ind)]

    return pairs


def get_interval_state_count(modulus=24):
    """Return the number of voice-leading states in Z_modulus plus rest-related states.
    INPUT: modulus (int) - cyclic modulus for directed interval classes
    RETURN: int - total number of states, equal to modulus + 2
    """
    return int(modulus) + 2


def interval_to_index_mod(delta, modulus=24):
    """Encode a directed interval as a class in Z_modulus.
    INPUT: delta (int) - signed semitone difference b - a
           modulus (int) - cyclic modulus for directed interval classes
    RETURN: int - residue class in {0, ..., modulus - 1}
    """
    return int(delta) % int(modulus)


def encode_matched_pair_as_interval_mod(source_item, target_item, modulus=24):
    """Encode one matched atomic pair as a directed interval class or a rest-related state.
    INPUT: source_item (dict or None) - source atomic item with keys 'pitch', 'full_dur', or None for rest
           target_item (dict or None) - target atomic item with keys 'pitch', 'full_dur', or None for rest
           modulus (int) - cyclic modulus for directed interval classes
    RETURN: int or None - encoded state index, or None if the pair is static and must be dropped
    """
    if source_item is None and target_item is None:
        return None

    if source_item is not None and target_item is not None:
        a = int(source_item["pitch"])
        b = int(target_item["pitch"])

        if a == b:
            return None

        delta = b - a
        return interval_to_index_mod(delta, modulus=modulus)

    if source_item is None and target_item is not None:
        return int(modulus)

    if source_item is not None and target_item is None:
        return int(modulus) + 1

    return None


def compute_motion_weight(source_item, target_item):
    """Compute the local duration-aware weight of one matched atomic pair.
    INPUT: source_item (dict or None) - source atomic item with keys 'pitch', 'full_dur', or None for rest
           target_item (dict or None) - target atomic item with keys 'pitch', 'full_dur', or None for rest
    RETURN: float - positive local weight, or 0.0 for static / empty pairs
    """
    if source_item is None and target_item is None:
        return 0.0

    if source_item is not None and target_item is not None:
        a = int(source_item["pitch"])
        b = int(target_item["pitch"])

        if a == b:
            return 0.0

        return float(source_item["full_dur"]) * float(target_item["full_dur"])

    if source_item is None and target_item is not None:
        return float(target_item["full_dur"])

    if source_item is not None and target_item is None:
        return float(source_item["full_dur"])

    return 0.0


def build_interval_distribution(event_a, event_b, modulus=24, rest_cost=12.0):
    """Build a normalized motion-only duration-aware interval-transition distribution between two consecutive vertical events.
    INPUT: event_a (dict) - first vertical event
           event_b (dict) - second vertical event
           modulus (int) - cyclic modulus for directed interval classes
           rest_cost (float) - penalty used in minimum-cost matching
    RETURN: np.ndarray shape (m,) - normalized interval-transition distribution,
            where m = modulus + 2, or None if no actual motion is present
    """
    state_count = get_interval_state_count(modulus)
    v = np.zeros(state_count, dtype=float)

    n_parts = min(len(event_a["part_pitch_items"]), len(event_b["part_pitch_items"]))
    total_weight = 0.0

    for r in range(n_parts):
        source_items = event_a["part_pitch_items"][r]
        target_items = event_b["part_pitch_items"][r]

        matched_pairs = match_pitch_items(source_items, target_items, rest_cost=rest_cost)

        for source_item, target_item in matched_pairs:
            idx = encode_matched_pair_as_interval_mod(
                source_item,
                target_item,
                modulus=modulus
            )

            if idx is None:
                continue

            w = compute_motion_weight(source_item, target_item)

            if w <= 0:
                continue

            v[idx] += w
            total_weight += w

    if total_weight <= 0:
        return None

    v /= float(total_weight)
    return v


def build_voice_leading_cloud(events, modulus=24, rest_cost=12.0, unique_only=True, round_decimals=6):
    """Build the cloud of motion-only interval-based voice-leading transition objects for one piece.
    INPUT: events (list[dict]) - sequence of vertical events
           modulus (int) - cyclic modulus for directed interval classes
           rest_cost (float) - penalty used in minimum-cost matching
           unique_only (bool) - if True, merge repeated transition objects after rounding
           round_decimals (int) - rounding precision before uniqueness filtering
    RETURN: cloud (np.ndarray shape (n_points, m)) - interval-transition distributions
            masses (np.ndarray shape (n_points,)) - aggregated transition masses
            n_transitions (int) - number of effective motion transitions before uniqueness filtering
    """
    vectors = []
    masses = []

    if len(events) < 2:
        state_count = get_interval_state_count(modulus)
        return np.empty((0, state_count), dtype=float), np.empty((0,), dtype=float), 0

    for k in range(len(events) - 1):
        event_a = events[k]
        event_b = events[k + 1]

        vec = build_interval_distribution(
            event_a,
            event_b,
            modulus=modulus,
            rest_cost=rest_cost
        )

        if vec is None:
            continue


        vectors.append(vec)
        masses.append(1.0)

    if len(vectors) == 0:
        state_count = get_interval_state_count(modulus)
        return np.empty((0, state_count), dtype=float), np.empty((0,), dtype=float), 0

    vectors = np.asarray(vectors, dtype=float)
    masses = np.asarray(masses, dtype=float)

    if not unique_only:
        return vectors, masses, len(vectors)

    merged = {}
    for vec, mass in zip(vectors, masses):
        key = tuple(np.round(vec, round_decimals))
        if key not in merged:
            merged[key] = 0.0
        merged[key] += mass

    cloud = np.asarray([np.array(key, dtype=float) for key in merged.keys()], dtype=float)
    merged_masses = np.asarray(list(merged.values()), dtype=float)

    return cloud, merged_masses, len(vectors)


def compute_voice_distance_matrix(cloud, masses):
    """Compute the pairwise weighted Jensen-Shannon distance matrix for transition objects.
    INPUT: cloud (np.ndarray shape (n_points, m)) - transition distributions
           masses (np.ndarray shape (n_points,)) - transition masses
    RETURN: np.ndarray shape (n_points, n_points) - symmetric distance matrix
    """
    n = len(cloud)
    D = np.zeros((n, n), dtype=float)

    for i in range(n):
        for j in range(i + 1, n):
            d = weighted_js_distance(
                cloud[i],
                cloud[j],
                masses[i],
                masses[j]
            )
            D[i, j] = d
            D[j, i] = d

    return D


def compute_voice_persistence(events, modulus=24, rest_cost=12.0, unique_only=True, round_decimals=6, maxdim=1):
    """Compute persistent homology for the motion-only duration-aware voice-leading cloud of one piece.
    INPUT: events (list[dict]) - sequence of vertical events
           modulus (int) - cyclic modulus for directed interval classes
           rest_cost (float) - penalty used in minimum-cost matching
           unique_only (bool) - if True, merge repeated transition objects
           round_decimals (int) - rounding precision before uniqueness filtering
           maxdim (int) - maximum homology dimension
    RETURN: cloud (np.ndarray) - voice-leading cloud
            masses (np.ndarray) - masses of the transition objects
            diagrams (list[np.ndarray]) - persistence diagrams
            n_transitions (int) - number of effective motion transitions before uniqueness filtering
    """
    cloud, masses, n_transitions = build_voice_leading_cloud(
        events,
        modulus=modulus,
        rest_cost=rest_cost,
        unique_only=unique_only,
        round_decimals=round_decimals
    )

    if len(cloud) < 2:
        diagrams = [np.empty((0, 2), dtype=float) for _ in range(maxdim + 1)]
        return cloud, masses, diagrams, n_transitions

    D = compute_voice_distance_matrix(cloud, masses)
    diagrams = compute_persistence(D, max_dim=maxdim)

    return cloud, masses, diagrams, n_transitions


def voice_feature_vector_from_events(events, modulus=24, rest_cost=12.0, unique_only=True, round_decimals=6, maxdim=1, prefix="VOICE_"):
    """Compute piece-level voice-leading topological features from vertical events.
    INPUT: events (list[dict]) - sequence of vertical events
           modulus (int) - cyclic modulus for directed interval classes
           rest_cost (float) - penalty used in minimum-cost matching
           unique_only (bool) - if True, merge repeated transition objects
           round_decimals (int) - rounding precision before uniqueness filtering
           maxdim (int) - maximum homology dimension
           prefix (str) - feature name prefix
    RETURN: dict - piece-level voice-leading feature dictionary
    """
    cloud, masses, diagrams, n_transitions = compute_voice_persistence(
        events,
        modulus=modulus,
        rest_cost=rest_cost,
        unique_only=unique_only,
        round_decimals=round_decimals,
        maxdim=maxdim
    )

    h0_mean, h0_std, h0_entropy = extract_barcode_stats(diagrams[0])
    h1_mean, h1_std, h1_entropy = extract_barcode_stats(diagrams[1])

    feat = {
        f"{prefix}H0_mean": float(h0_mean),
        f"{prefix}H0_std": float(h0_std),
        f"{prefix}H0_entropy": float(h0_entropy),
        f"{prefix}H1_mean": float(h1_mean),
        f"{prefix}H1_std": float(h1_std),
        f"{prefix}H1_entropy": float(h1_entropy),
        f"{prefix}n_transitions": float(n_transitions),
        f"{prefix}n_points": float(len(cloud)),
    }

    return feat


def build_voice_feature_table(df_meta, event_cache, modulus=24, rest_cost=12.0, unique_only=True, round_decimals=6, maxdim=1, prefix="VOICE_"):
    """Build a voice-leading feature table for a dataset.
    INPUT: df_meta (pd.DataFrame) - metadata table with column 'filename'
           event_cache (dict) - mapping filename to vertical-event sequences
           modulus (int) - cyclic modulus for directed interval classes
           rest_cost (float) - penalty used in minimum-cost matching
           unique_only (bool) - if True, merge repeated transition objects
           round_decimals (int) - rounding precision before uniqueness filtering
           maxdim (int) - maximum homology dimension
           prefix (str) - feature name prefix
    RETURN: pd.DataFrame - one row per piece with voice-leading topological features
    """
    rows = []

    for _, row in tqdm(df_meta.iterrows(), total=len(df_meta)):
        filename = row["filename"]
        events = event_cache.get(filename, None)

        if events is None:
            continue

        feat = voice_feature_vector_from_events(
            events,
            modulus=modulus,
            rest_cost=rest_cost,
            unique_only=unique_only,
            round_decimals=round_decimals,
            maxdim=maxdim,
            prefix=prefix
        )

        out = row.to_dict()
        out.update(feat)
        rows.append(out)

    return pd.DataFrame(rows)
