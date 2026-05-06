import numpy as np
from features.distances import weighted_js_distance
from scipy.sparse.csgraph import shortest_path


def build_expanded_graph(P, min_weight=0.0, return_edges=True, n_orig = 12):
    """
    Builds the expanded undirected graph G' from a directed weighted graph P (12x12)
    as per Definition 18. Can return edges with lengths (for original method) and/or
    a mapping from (i,j) to new vertex IDs (for Bregman divergence method).
    INPUT: P - 12x12 numpy array, transition probabilities
           min_weight: float, ignore edges with weight <= min_weight
           return_edges: bool, if True, compute edge lengths (1/weight) for original method
    OUTPUT: vertices: list of int, vertex IDs (0..11 original, then new IDs starting from 12)
            vertex_map: dict {(i,j): new_id} for each directed edge with weight > min_weight
            edges: (optional) list of tuples (u, v, length) if return_edges=True, else None
    """
    next_id = n_orig
    vertex_map = {}
    edges = [] if return_edges else None

    for i in range(n_orig):
        for j in range(n_orig):
            w = P[i, j]
            if w <= min_weight:
                continue
            new_id = next_id
            next_id += 1
            vertex_map[(i, j)] = new_id
            if return_edges:
                if i == j:
                    # loop: single edge from i to new vertex, length = 1/w
                    edges.append((i, new_id, 1.0 / w))
                else:
                    # two edges: i - new and new - j, each length = 1/2w (unlike in original method 2/w)
                    length = 1.0 / (2.0 * w)
                    edges.append((i, new_id, length))
                    edges.append((new_id, j, length))

    vertices = list(range(n_orig)) + list(range(n_orig, next_id))
    return vertices, vertex_map, edges


def compute_original_distance_matrix(vertices, edges):
    """Computes pairwise shortest-path distances in the undirected weighted graph.
    INPUT: vertices: list of int, all vertex IDs
           edges: list of tuples (u, v, length) – undirected edges with positive lengths
    OUTPUT: D - numpy array of shape (n, n) with distances (inf if disconnected)
    """
    n = len(vertices)
    D = np.full((n, n), np.inf)
    np.fill_diagonal(D, 0)
    idx_map = {v: i for i, v in enumerate(vertices)}
    for u, v, w in edges:
        i, j = idx_map[u], idx_map[v]
        D[i, j] = w
        D[j, i] = w
    D_short = shortest_path(D, directed=False, method='D')
    return D_short


def compute_bregman_edge_lengths(
    vertices, vertex_map, P, dist_dict, mass_dict,
    alpha=0.7, beta=1.0, lam=0.25,
    eps=1e-10, min_weight=1e-8, loop_multiplier=1.0
):
    """Recomputes edge lengths in the expanded graph using weighted Jensen-Shannon distance
    combined with transition probability rarity.
    INPUT: vertices: list[int] – all vertex IDs (original 0–11 + added)
           vertex_map: dict {(i,j): new_id} – mapping from directed edge to added vertex ID
           P: np.ndarray (12,12) – transition probability matrix
           dist_dict: dict {vertex_id: np.array(12)} – probability distributions per original vertex
           mass_dict: dict {vertex_id: float} – vertex masses for original vertices
           alpha: float – exponent for rarity penalty (w ** alpha), typical range 0.3–1.0
           beta: float – strength of the Jensen-Shannon correction
           lam: float – additive baseline term used only in 'js_additive' mode
           eps: float – small value for numerical stability (default 1e-10)
           min_weight: float – ignore transitions with P[i,j] <= min_weight
           loop_multiplier: float – scaling factor for self-loop edge lengths (default 1.0)
    OUTPUT: edges: list of tuples (u, v, new_length) – undirected edges with updated Bregman-based lengths
    """
    edges = []

    for (i, j), v_new in vertex_map.items():
        w = P[i, j]
        if w <= min_weight:
            continue
        if i == j:  # self-loop
            length = loop_multiplier / (w ** alpha + eps)
            edges.append((i, v_new, length))
        else:
            js = weighted_js_distance(
                dist_dict[i],
                dist_dict[j],
                mass_dict[i],
                mass_dict[j],
                eps=1e-12
            )
            length = (lam + beta * js) / (2 * (w ** alpha + eps))
            edges.append((i, v_new, length))
            edges.append((v_new, j, length))
    return edges


def compute_pure_semantic_edge_lengths(
    vertices, vertex_map, P, dist_dict, mass_dict,
    js_scale=1.0, eps=1e-6, min_weight=1e-8
):
    """
    Recomputes edge lengths in the expanded graph using only weighted Jensen-Shannon distance.
    INPUT:
        vertices: list[int] - All vertex IDs (original + added).
        vertex_map: dict[(i, j) -> new_id]
            Mapping from directed edge (i, j) to the inserted vertex ID.
        P: np.ndarray - Transition probability matrix.
        dist_dict: dict[vertex_id -> np.ndarray]
            Probability distributions assigned to original vertices.
        mass_dict: dict[vertex_id -> float] - Vertex masses for weighted JS.
        js_scale: float - Global multiplicative scale for JS lengths.
        eps: float - Small positive floor to avoid zero-length edges.
        min_weight: float - Ignore transitions with P[i, j] <= min_weight.

    OUTPUT: edges: list[tuple]
            Undirected weighted edges (u, v, length).
    """
    edges = []

    for (i, j), v_new in vertex_map.items():
        w = P[i, j]
        if w <= min_weight:
            continue

        if i == j:
            edges.append((i, v_new, eps))
        else:
            js = weighted_js_distance(
                dist_dict[i],
                dist_dict[j],
                mass_dict[i],
                mass_dict[j],
                eps=1e-12
            )
            length = js_scale * max(js, eps)
            edges.append((i, v_new, length))
            edges.append((v_new, j, length))

    return edges
