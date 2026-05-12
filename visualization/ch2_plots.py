import matplotlib.pyplot as plt
from features.ch2_harmonic import compute_harmonic_persistence
from features.ch2_voice_leading import compute_voice_persistence
from visualization.barcodes import plot_barcodes


def plot_harmonic_barcodes_for_piece(file_path, filename, event_cache, title, quant_step=0.125, unique_only=True, round_decimals=8, maxdim=1):
    """Plot H0 and H1 barcodes for one piece in the bar-level harmonic branch.
    INPUT: file_path (str) - MIDI file path
           filename (str) - piece filename
           event_cache (dict) - mapping filename to vertical events
           title (str) - plot title
           unique_only (bool) - if True, merge repeated harmonic descriptors
           round_decimals (int) - rounding precision before uniqueness filtering
           maxdim (int) - maximum homology dimension
    RETURN: None
    """
    events = event_cache[filename]

    _, dgms, _ = compute_harmonic_persistence(
        file_path,
        events,
        quant_step=quant_step,
        unique_only=unique_only,
        round_decimals=round_decimals,
        maxdim=maxdim
    )

    plot_barcodes(dgms, title=title)


def plot_voice_barcodes_for_piece(filename, event_cache, modulus, title, rest_cost=12.0, unique_only=True, round_decimals=6, maxdim=1):
    """Plot H0 and H1 barcodes for one piece in the voice-leading branch.
    INPUT: filename (str) - piece filename
           event_cache (dict) - mapping filename to vertical events
           modulus (int) - cyclic modulus for directed interval classes
           title (str) - plot title
           rest_cost (float) - penalty used in minimum-cost matching
           unique_only (bool) - if True, merge repeated transition objects
           round_decimals (int) - rounding precision before uniqueness filtering
           maxdim (int) - maximum homology dimension
    RETURN: None
    """
    events = event_cache[filename]
    _, _, dgms, _ = compute_voice_persistence(
        events,
        modulus=modulus,
        rest_cost=rest_cost,
        unique_only=unique_only,
        round_decimals=round_decimals,
        maxdim=maxdim
    )
    plot_barcodes(dgms, title=title)


def plot_fusion_barcodes_for_piece(
    filename,
    file_path,
    event_cache,
    voice_modulus,
    title,
    quant_step=0.125,
    voice_rest_cost=12.0,
    voice_unique_only=True,
    voice_round_decimals=6,
    harm_unique_only=True,
    harm_round_decimals=8,
    maxdim=1
):
    """Plot harmonic and voice-leading barcodes for one piece side by side.
    INPUT: filename (str) - piece filename
           file_path (str) - MIDI file path
           event_cache (dict) - mapping filename to vertical events
           voice_modulus (int) - cyclic modulus for the voice-leading interval coding
           title (str) - plot title prefix
           voice_rest_cost (float) - penalty used in minimum-cost matching
           voice_unique_only (bool) - if True, merge repeated voice-leading transition objects
           voice_round_decimals (int) - rounding precision for the voice-leading branch
           harm_unique_only (bool) - if True, merge repeated harmonic descriptors
           harm_round_decimals (int) - rounding precision for the harmonic branch
           maxdim (int) - maximum homology dimension
    RETURN: None
    """
    events = event_cache[filename]

    _, harm_dgms, _ = compute_harmonic_persistence(
        file_path,
        events,
        quant_step=quant_step,
        unique_only=harm_unique_only,
        round_decimals=harm_round_decimals,
        maxdim=maxdim
    )

    _, _, voice_dgms, _ = compute_voice_persistence(
        events,
        modulus=voice_modulus,
        rest_cost=voice_rest_cost,
        unique_only=voice_unique_only,
        round_decimals=voice_round_decimals,
        maxdim=maxdim
    )

    plot_barcodes(harm_dgms, title=f"{title} - Harmonic")
    plot_barcodes(voice_dgms, title=f"{title} - Voice-leading")


def plot_model_metric_comparison(df_comparison, dataset_name, metrics):
    """Plot selected evaluation metrics for all models in one dataset.
    INPUT: df_comparison (pd.DataFrame) - comparison table with one row per model
           dataset_name (str) - dataset name used in the plot title
           metrics (list[str]) - metric columns to visualize
    RETURN: None
    """
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 4))

    if n_metrics == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        ax.bar(df_comparison["model"], df_comparison[metric])
        ax.set_title(f"{dataset_name}: {metric}")
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.show()
