import pandas as pd
from tqdm.auto import tqdm
from features.ch2_harmonic import harmonic_feature_vector_from_events
from features.ch2_voice_leading import voice_feature_vector_from_events


def build_harmonic_feature_table(df_meta, event_cache, quant_step=0.125, unique_only=True, round_decimals=8, maxdim=1, prefix="HARM_"):
    """Build a bar-level harmonic feature table for a dataset.
    INPUT: df_meta (pd.DataFrame) - metadata table with columns 'filename' and 'filepath'
           event_cache (dict) - mapping filename -> list of vertical events
           quant_step (float) - quantization step for measure boundaries
           unique_only (bool) - if True, remove repeated descriptors
           round_decimals (int) - rounding precision before uniqueness filtering
           maxdim (int) - maximum homology dimension for ripser
           prefix (str) - feature prefix
    RETURN: pd.DataFrame - one row per piece with harmonic topological features
    """
    rows = []

    for _, row in tqdm(df_meta.iterrows(), total=len(df_meta)):
        filename = row["filename"]
        file_path = row["filepath"]
        events = event_cache.get(filename, None)

        if events is None:
            continue

        feat = harmonic_feature_vector_from_events(
            file_path,
            events,
            quant_step=quant_step,
            unique_only=unique_only,
            round_decimals=round_decimals,
            maxdim=maxdim,
            prefix=prefix
        )

        out = row.to_dict()
        out.update(feat)
        rows.append(out)

    return pd.DataFrame(rows)


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


def build_fusion_feature_table(df_harm, df_voice):
    """Build a late-fusion feature table by merging harmonic and voice-leading features.
    INPUT: df_harm (pd.DataFrame) - harmonic feature table
           df_voice (pd.DataFrame) - voice-leading feature table
    RETURN: pd.DataFrame - merged table with shared metadata and both feature sets
    """
    voice_feature_cols = [c for c in df_voice.columns if c.startswith("VOICE_")]

    df_fusion = df_harm.merge(
        df_voice[["filename"] + voice_feature_cols],
        on="filename",
        how="inner"
    )

    return df_fusion
