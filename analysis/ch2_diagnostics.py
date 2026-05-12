import os
import numpy as np
import pandas as pd
from music21 import chord, converter, note
from tqdm.auto import tqdm
from features.ch2_preprocessing import extract_vertical_events


# ----------------------------------------------------------------------------
# Diagnostic utilities for Chapter 2 symbolic-score preprocessing.

# These functions are used to inspect score quality, part structure, chord burden,
# and eligibility for the harmonic and voice-leading experiments.
# ----------------------------------------------------------------------------


def inspect_file_quality(file_path, quant_step=0.125, min_event_dur=0.125):
    result = {
        "filepath": file_path,
        "filename": os.path.basename(file_path),
        "parse_ok": False,
        "error": None,

        "n_parts": np.nan,
        "part_note_counts": None,
        "part_rest_counts": None,
        "part_chord_counts": None,
        "part_max_chord_size": None,

        "all_parts_have_notes": False,
        "all_parts_monophonic": False,

        "n_vertical_events": np.nan,
        "min_part_notes": np.nan,
        "max_part_chords": np.nan,

        "problems": [],
        "warnings": [],
    }

    try:
        score = converter.parse(file_path)
        parts = list(score.parts)

        result["parse_ok"] = True
        result["n_parts"] = len(parts)

        note_counts = []
        rest_counts = []
        chord_counts = []
        max_chord_sizes = []
        pitched_counts = []

        for p in parts:
            elems = list(p.flatten().notesAndRests)

            n_notes = sum(isinstance(el, note.Note) for el in elems)
            n_rests = sum(isinstance(el, note.Rest) for el in elems)
            n_chords = sum(isinstance(el, chord.Chord) for el in elems)

            chord_sizes = [len(el.pitches) for el in elems if isinstance(el, chord.Chord)]
            max_chord_size = max(chord_sizes) if chord_sizes else 0

            note_counts.append(n_notes)
            rest_counts.append(n_rests)
            chord_counts.append(n_chords)
            max_chord_sizes.append(max_chord_size)
            pitched_counts.append(n_notes + n_chords)

        result["part_note_counts"] = note_counts
        result["part_rest_counts"] = rest_counts
        result["part_chord_counts"] = chord_counts
        result["part_max_chord_size"] = max_chord_sizes

        result["min_part_notes"] = min(pitched_counts) if len(pitched_counts) > 0 else 0
        result["max_part_chords"] = max(chord_counts) if len(chord_counts) > 0 else 0

        result["all_parts_have_notes"] = (len(pitched_counts) > 0 and min(pitched_counts) > 0)
        result["all_parts_monophonic"] = (len(chord_counts) > 0 and max(chord_counts) == 0)

        events = extract_vertical_events(
            file_path,
            quant_step=quant_step,
            min_event_dur=min_event_dur,
            drop_all_rest=True
        )
        result["n_vertical_events"] = len(events)

        if len(parts) != 4:
            result["problems"].append(f"n_parts={len(parts)} (expected 4)")

        if not result["all_parts_have_notes"]:
            result["problems"].append("some parts have zero notes")

        if len(events) < 5:
            result["problems"].append(f"too few vertical events: {len(events)}")

        if not result["all_parts_monophonic"]:
            result["warnings"].append("some parts contain chord objects")

        if len(events) < 20:
            result["warnings"].append(f"few vertical events: {len(events)}")

    except Exception as e:
        result["parse_ok"] = False
        result["error"] = str(e)
        result["problems"].append(f"parse_error: {str(e)}")

    return result


def audit_dataset(df_meta, quant_step=0.125, min_event_dur=0.125):
    rows = []

    for _, row in tqdm(df_meta.iterrows(), total=len(df_meta)):
        fp = row.get("filepath", None)
        base = row.to_dict()

        if pd.isna(fp) or fp is None or not os.path.exists(fp):
            out = dict(base)
            out.update({
                "parse_ok": False,
                "error": "file_not_found",
                "n_parts": np.nan,
                "part_note_counts": None,
                "part_rest_counts": None,
                "part_chord_counts": None,
                "part_max_chord_size": None,
                "all_parts_have_notes": False,
                "all_parts_monophonic": False,
                "n_vertical_events": np.nan,
                "min_part_notes": np.nan,
                "max_part_chords": np.nan,
                "problems": ["file_not_found"],
                "warnings": [],
            })
            rows.append(out)
            continue

        q = inspect_file_quality(fp, quant_step=quant_step, min_event_dur=min_event_dur)
        out = dict(base)
        out.update(q)
        rows.append(out)

    df_audit = pd.DataFrame(rows)

    df_audit["eligible_core"] = (
        df_audit["parse_ok"].fillna(False)
        & (df_audit["n_parts"] == 4)
        & df_audit["all_parts_have_notes"].fillna(False)
        & (df_audit["n_vertical_events"].fillna(0) >= 5)
    )

    df_audit["eligible_strict_vl"] = (
        df_audit["eligible_core"]
        & df_audit["all_parts_monophonic"].fillna(False)
    )

    df_audit["n_problems"] = df_audit["problems"].apply(lambda x: len(x) if isinstance(x, list) else 0)
    df_audit["n_warnings"] = df_audit["warnings"].apply(lambda x: len(x) if isinstance(x, list) else 0)

    return df_audit


def compute_chord_burden(df_audit):
    rows = []

    for _, row in df_audit.iterrows():
        part_note_counts = row["part_note_counts"] if isinstance(row["part_note_counts"], list) else []
        part_chord_counts = row["part_chord_counts"] if isinstance(row["part_chord_counts"], list) else []
        part_max_chord_size = row["part_max_chord_size"] if isinstance(row["part_max_chord_size"], list) else []

        total_notes = sum(part_note_counts)
        total_chords = sum(part_chord_counts)
        total_note_like = total_notes + total_chords

        chord_share = total_chords / total_note_like if total_note_like > 0 else np.nan
        max_chord_size = max(part_max_chord_size) if len(part_max_chord_size) > 0 else 0

        rows.append({
            "filename": row.get("filename", None),
            "composer": row.get("composer", None),
            "genre": row.get("genre", None),
            "eligible_core": row["eligible_core"],
            "eligible_strict_vl": row["eligible_strict_vl"],
            "total_notes": total_notes,
            "total_chords": total_chords,
            "total_note_like": total_note_like,
            "chord_share": chord_share,
            "max_chord_size": max_chord_size,
            "n_vertical_events": row["n_vertical_events"],
            "warnings": row["warnings"],
            "problems": row["problems"],
        })

    return pd.DataFrame(rows)
