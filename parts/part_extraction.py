import os
from music21 import converter


def extract_part_sequence(part):
    """Extract pitch classes, durations, and onset positions from a single part.
    INPUT: part (music21.stream.Part)
    RETURN: pitches (list[int])      - pitch classes (0-11) in order of appearance
            durations (list[float])  - durations in quarterLength units
            onsets (list[float])     - onset positions (quarterLength offsets)
    """
    pitches = []
    durations = []
    onsets = []

    for element in part.recurse().notes:
        if element.isNote:
            pitches.append(element.pitch.midi % 12)
            durations.append(element.quarterLength)
            onsets.append(float(element.offset))
        elif element.isChord:  # for chords, we don't reduce it to the highest note, instead we use arpeggios
            for note in element.notes:
                pitches.append(note.pitch.midi % 12)
                durations.append(note.quarterLength)
                onsets.append(float(element.offset))
    return pitches, durations, onsets


def process_midi_file(midi_path):
    """Processes a single MIDI file (assumed to be a string quartet).
    INPUT: midi_path (str) - full path to the .mid file.
    RETURN: list of dicts for each part (first 4 parts), each dict contains:
            'part_index', 'num_notes', 'pitches', 'durations', 'transition_matrix'.
            Returns None if parsing fails or fewer than 4 parts.
    """
    try:
        score = converter.parse(midi_path)
    except Exception as e:
        print(f"Error parsing {midi_path}: {e}")
        return None

    parts = score.parts
    if len(parts) < 4:
        print(f"Warning: {midi_path} has only {len(parts)} parts, skipping.")
        return None

    parts_data = []
    for idx, part in enumerate(parts[:4]):
        pitches, durations, onsets = extract_part_sequence(part)
        parts_data.append({
            'part_index': idx,
            'num_notes': len(pitches),
            'pitches': pitches,
            'durations': durations,
            'onsets': onsets,
        })
    return parts_data


def add_parts_column(df, data_root, flat_structure=False, inplace=False):
    """Adds a 'parts' column to the DataFrame by processing each MIDI file.
    INPUT:df - pandas DataFrame.
          data_root: root directory containing MIDI files.
          flat_structure: if True, files are directly in data_root; else in data_root/composer_lower/genre_lower/.
          inplace: if True, modify the original DataFrame; otherwise return a new DataFrame.
    RETURN: DataFrame with added 'parts' column (or None if inplace=True).
            Rows where processing failed (file not found or error) have parts=None.
    """
    if not inplace:
        df = df.copy()

    def process_row(row):
        fname = row['filename']
        composer = row['composer']
        genre = row['genre']

        if flat_structure:
            midi_path = os.path.join(data_root, fname)
        else:
            midi_path = os.path.join(data_root, composer.lower(), genre.lower(), fname)

        if not os.path.exists(midi_path):
            print(f"File not found: {midi_path}")
            return None

        parts_data = process_midi_file(midi_path)  # returns list of part dicts or None
        return parts_data

    df['parts'] = df.apply(process_row, axis=1)
    print(f"Added 'parts' column. {df['parts'].notna().sum()} / {len(df)} works processed successfully.")

    if not inplace:
        return df
