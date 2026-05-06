import os
from music21 import converter
from parts.part_extraction import extract_part_sequence


def process_midi_file_piano(midi_path):
    """
    Process a MIDI file and extract per-part sequences.
    INPUT: midi_path - str, path to MIDI file.
    RETURN: list of dicts with keys 'part_index', 'num_notes', 'pitches', 'durations', 'onsets';
            or None if parsing fails or no parts.
    """
    try:
        score = converter.parse(midi_path)
    except Exception as e:
        print(f"Error parsing {midi_path}: {e}")
        return None

    parts = score.parts
    if len(parts) == 0:
        return None

    parts_data = []
    for idx, part in enumerate(parts):
        pitches, durations, onsets = extract_part_sequence(part)
        parts_data.append({
            'part_index': idx,
            'num_notes': len(pitches),
            'pitches': pitches,
            'durations': durations,
            'onsets': onsets,
        })
    return parts_data


def add_parts_column_from_midi_path(df, inplace=False, split_n=None):
    """
    Add 'parts' column using 'midi_path' column.
    INPUT: df - pandas DataFrame.
           inplace - bool, modify df in place or return new copy.
           split_n - None or 'train'/'validation'/'test', filter by split column.
    RETURN: DataFrame with 'parts' column if inplace=False, else None.
            Rows with errors have parts=None. Prints success count.
    """
    if not inplace or split_n is not None:
        df = df.copy()
    if split_n is not None and split_n in ['train', 'validation', 'test']:
        df = df[df['split'] == split_n]
    df['parts'] = df['midi_path'].apply(process_midi_file_piano)
    success = df['parts'].notna().sum()
    print(f"Processed {success} / {len(df)} files successfully.")
    if not inplace or split_n is not None:
        return df
