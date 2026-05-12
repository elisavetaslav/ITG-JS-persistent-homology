from .transitions import build_transition_matrix
from .distances import weighted_js_distance, compute_distance_matrix
from .distributions import assign_distributions
from .graph_construction import (
    build_expanded_graph,
    compute_original_distance_matrix,
    compute_weighted_js_edge_lengths,
    compute_pure_js_edge_lengths
)
from .persistence import (
    compute_persistence,
    extract_barcode_stats,
    extract_filtered_barcode_stats,
)
from .feature_building import compute_features_for_row, compute_features_for_row_js

from .ch2_preprocessing import (
    build_lookup,
    qround,
    build_part_note_events,
    extract_vertical_events,
    build_event_cache,
)

from .ch2_harmonic import (
    extract_measure_intervals,
    collect_bar_pitch_class_vector,
    pc_vector_to_harmonic_descriptor,
    build_harmonic_cloud,
    compute_harmonic_persistence,
    harmonic_feature_vector_from_events,
)

from .ch2_voice_leading import (
    transition_cost,
    match_pitch_sets,
    match_pitch_items,
    get_interval_state_count,
    interval_to_index_mod,
    encode_matched_pair_as_interval_mod,
    compute_motion_weight,
    build_interval_distribution,
    build_voice_leading_cloud,
    compute_voice_distance_matrix,
    compute_voice_persistence,
    voice_feature_vector_from_events,
)

from .ch2_feature_tables import (
    build_harmonic_feature_table,
    build_voice_feature_table,
    build_fusion_feature_table,
)


__all__ = [
    "build_transition_matrix",
    "weighted_js_distance",
    "compute_distance_matrix",
    "assign_distributions",
    "build_expanded_graph",
    "compute_original_distance_matrix",
    "compute_weighted_js_edge_lengths",
    "compute_pure_js_edge_lengths",
    "compute_persistence",
    "extract_barcode_stats",
    "extract_filtered_barcode_stats",
    "compute_features_for_row",
    "compute_features_for_row_js",

    "build_lookup",
    "qround",
    "build_part_note_events",
    "extract_vertical_events",
    "build_event_cache",

    "extract_measure_intervals",
    "collect_bar_pitch_class_vector",
    "pc_vector_to_harmonic_descriptor",
    "build_harmonic_cloud",
    "compute_harmonic_persistence",
    "harmonic_feature_vector_from_events",

    "transition_cost",
    "match_pitch_sets",
    "match_pitch_items",
    "get_interval_state_count",
    "interval_to_index_mod",
    "encode_matched_pair_as_interval_mod",
    "compute_motion_weight",
    "build_interval_distribution",
    "build_voice_leading_cloud",
    "compute_voice_distance_matrix",
    "compute_voice_persistence",
    "voice_feature_vector_from_events",

    "build_harmonic_feature_table",
    "build_voice_feature_table",
    "build_fusion_feature_table",
]