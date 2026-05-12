import numpy as np
import pandas as pd
from music21 import converter
from ripser import ripser
from features.ch2_preprocessing import qround, EPS
from features.persistence import extract_barcode_stats


def extract_measure_intervals(file_path, quant_step=0.125):
    """Extract measure intervals from the first part of a score.
    INPUT: file_path (str) - symbolic score file path readable by music21
           quant_step (float) - quantization step
    RETURN: list[tuple[float, float]] - measure intervals (start, end)
    """
    score = converter.parse(file_path)
    parts = list(score.parts)

    if len(parts) == 0:
        raise ValueError("No parts found in the score.")

    measures = list(parts[0].recurse().getElementsByClass("Measure"))
    intervals = []

    for measure in measures:
        start = qround(measure.offset, step=quant_step)

        if measure.barDuration is not None:
            dur = qround(measure.barDuration.quarterLength, step=quant_step)
        else:
            dur = qround(measure.duration.quarterLength, step=quant_step)

        if dur <= 0:
            continue

        end = qround(start + dur, step=None)
        intervals.append((start, end))

    if len(intervals) == 0:
        total_dur = qround(score.highestTime, step=None)
        if total_dur <= 0:
            raise ValueError("Could not infer a valid measure structure.")
        intervals = [(0.0, total_dur)]

    return intervals


def collect_bar_pitch_class_vector(events, bar_start, bar_end, n_pc=12):
    """Build a duration-weighted pitch-class vector for one measure interval.
    INPUT: events (list[dict]) - vertical events
           bar_start (float) - measure start
           bar_end (float) - measure end
           n_pc (int) - number of pitch classes
    RETURN: np.ndarray shape (n_pc,) - duration-weighted pitch-class vector
    """
    x = np.zeros(n_pc, dtype=float)

    for event in events:
        overlap_start = max(bar_start, event["start"])
        overlap_end = min(bar_end, event["end"])
        overlap = overlap_end - overlap_start

        if overlap <= EPS:
            continue

        pcs = sorted({
            int(pitch) % n_pc
            for part_pitch_set in event["part_pitch_sets"]
            for pitch in part_pitch_set
        })

        if len(pcs) == 0:
            continue

        weight = float(overlap) / float(len(pcs))

        for pc in pcs:
            x[pc] += weight

    return x


def pc_vector_to_harmonic_descriptor(x):
    """Convert one pitch-class vector into a 6-dimensional DFT descriptor.
    INPUT: x (np.ndarray shape (12,)) - pitch-class vector
    RETURN: np.ndarray shape (6,) - magnitudes of DFT coefficients 1..6,
            or None if the vector is empty
    """
    total = float(x.sum())

    if total <= 0:
        return None

    x = x / total
    fft_vals = np.fft.fft(x)
    desc = np.abs(fft_vals[1:7]).astype(float)

    return desc


def build_harmonic_cloud(file_path, events, quant_step=0.125, unique_only=True, round_decimals=8):
    """Build a bar-level harmonic point cloud from a sequence of vertical events.
    INPUT: file_path (str) - symbolic score file path readable by music21
           events (list[dict]) - sequence of vertical events
           quant_step (float) - quantization step for measure boundaries
           unique_only (bool) - if True, remove repeated descriptors after rounding
           round_decimals (int) - rounding precision before uniqueness filtering
    RETURN: np.ndarray shape (n_points, 6) - bar-level harmonic point cloud
            int - number of measure intervals
    """
    measure_intervals = extract_measure_intervals(file_path, quant_step=quant_step)
    points = []

    for bar_start, bar_end in measure_intervals:
        x = collect_bar_pitch_class_vector(
            events,
            bar_start=bar_start,
            bar_end=bar_end,
            n_pc=12
        )
        
        desc = pc_vector_to_harmonic_descriptor(x)
        if desc is not None:
            points.append(desc)

    if len(points) == 0:
        return np.empty((0, 6), dtype=float), len(measure_intervals)

    cloud = np.vstack(points).astype(float)

    if unique_only:
        cloud = np.unique(np.round(cloud, round_decimals), axis=0)

    return cloud, len(measure_intervals)


def compute_harmonic_persistence(file_path, events, quant_step=0.125, unique_only=True, round_decimals=8, maxdim=1):
    """Compute persistent homology for the bar-level harmonic point cloud of one piece.
    INPUT: file_path (str) - symbolic score file path readable by music21
           events (list[dict]) - sequence of vertical events
           quant_step (float) - quantization step for measure boundaries
           unique_only (bool) - if True, remove repeated descriptors
           round_decimals (int) - rounding precision before uniqueness filtering
           maxdim (int) - maximum homology dimension for ripser
    RETURN: cloud (np.ndarray) - harmonic point cloud
            diagrams (list[np.ndarray]) - persistence diagrams
            int - number of measure intervals
    """
    cloud, n_bars = build_harmonic_cloud(
        file_path,
        events,
        quant_step=quant_step,
        unique_only=unique_only,
        round_decimals=round_decimals
    )

    if len(cloud) < 2:
        diagrams = [np.empty((0, 2), dtype=float) for _ in range(maxdim + 1)]
        return cloud, diagrams, n_bars

    result = ripser(cloud, maxdim=maxdim)
    diagrams = result["dgms"]

    return cloud, diagrams, n_bars


def harmonic_feature_vector_from_events(file_path, events, quant_step=0.125, unique_only=True, round_decimals=8, maxdim=1, prefix="HARM_"):
    """Compute piece-level bar-level harmonic topological features.
    INPUT: file_path (str) - symbolic score file path readable by music21
           events (list[dict]) - sequence of vertical events
           quant_step (float) - quantization step for measure boundaries
           unique_only (bool) - if True, remove repeated descriptors
           round_decimals (int) - rounding precision before uniqueness filtering
           maxdim (int) - maximum homology dimension for ripser
           prefix (str) - feature name prefix
    RETURN: dict - piece-level harmonic feature dictionary
    """
    cloud, diagrams, n_bars = compute_harmonic_persistence(
        file_path,
        events,
        quant_step=quant_step,
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
        f"{prefix}n_bars": float(n_bars),
        f"{prefix}n_points": float(len(cloud)),
    }

    return feat
