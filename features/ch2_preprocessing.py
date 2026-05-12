import glob
import os
from music21 import chord, converter, note
from tqdm.auto import tqdm


EPS = 1e-9


def build_lookup(root, ext='mid'):
    """Build a filename-to-path lookup for files with a fixed extension.
    INPUT: root (str): root directory in which files are searched recursively.
           ext (str): file extension without the leading dot.
    OUTPUT: dict[str, str]: dictionary mapping each basename to its full file path.
    """
    lookup = {}
    for fp in glob.glob(os.path.join(root, '**', f'*.{ext}'), recursive=True):
        lookup[os.path.basename(fp)] = fp
    return lookup


def qround(x, step=0.125, ndigits=6):
    """"Quantize a numerical time value to a fixed grid.
    INPUT:  x (float): value to be rounded.
            step (float | None): quantization step. If None, only ordinary rounding is applied.
            ndigits (int): number of decimal digits kept after rounding.
    OUTPUT: float: rounded or quantized value.
    """
    x = float(x)
    if step is None:
        return round(x, ndigits)
    return round(round(x / step) * step, ndigits)


def build_part_note_events(part, quant_step=0.125):
    """Build note-like objects for one part with atomic pitches and full durations.
    INPUT: part (music21.stream.Part) - one score part
           quant_step (float) - quantization step for offsets and durations
    RETURN: list[dict] - note-like events with keys:
            'start' (float) - quantized start time
            'end' (float) - quantized end time
            'items' (list[dict]) - atomic pitch items, each with:
                'pitch' (int) - MIDI pitch
                'full_dur' (float) - full quantized duration of the original note/chord object
    """
    events = []
    elems = list(part.flatten().notesAndRests)

    for el in elems:
        start = qround(el.offset, step=quant_step)
        dur = qround(el.quarterLength, step=quant_step)
        end = qround(start + dur, step=None)

        if dur <= 0:
            continue

        if isinstance(el, note.Note):
            items = [{"pitch": int(el.pitch.midi), "full_dur": float(dur),}]
        elif isinstance(el, chord.Chord):
            items = [{"pitch": int(p.midi), "full_dur": float(dur),} for p in el.pitches]
        else:
            continue

        if len(items) == 0:
            continue

        events.append({
            "start": start,
            "end": end,
            "items": items,
        })

    events = sorted(events, key=lambda x: (x["start"], x["end"]))
    return events


def extract_vertical_events(file_path, quant_step=0.125, min_event_dur=0.125, drop_all_rest=True):
    """Extract vertical events together with per-part atomic pitch items.
    INPUT: file_path (str) - MIDI file path
           quant_step (float) - quantization step for offsets and durations
           min_event_dur (float) - minimum vertical-event duration
           drop_all_rest (bool) - if True, discard all-rest vertical events
    RETURN: list[dict] - vertical events with keys:
            'start' (float), 'end' (float), 'dur' (float),
            'part_pitch_sets' (list[tuple[int]]) - pitch support in each part,
            'part_pitch_items' (list[list[dict]]) - atomic pitch items in each part
    """
    score = converter.parse(file_path)
    parts = list(score.parts)

    if len(parts) == 0:
        raise ValueError("No parts found")

    part_events = [build_part_note_events(p, quant_step=quant_step) for p in parts]

    boundaries = set([0.0])
    for evs in part_events:
        for ev in evs:
            boundaries.add(ev["start"])
            boundaries.add(ev["end"])
    boundaries = sorted(boundaries)

    idxs = [0] * len(parts)
    active = [[] for _ in parts]

    vertical_events = []

    for t0, t1 in zip(boundaries[:-1], boundaries[1:]):
        dur = qround(t1 - t0, step=None)
        if dur <= EPS:
            continue
        if dur < min_event_dur:
            continue

        part_pitch_sets = []
        part_pitch_items = []

        for r, evs in enumerate(part_events):
            while idxs[r] < len(evs) and evs[idxs[r]]["start"] <= t0 + EPS:
                active[r].append(evs[idxs[r]])
                idxs[r] += 1

            active[r] = [ev for ev in active[r] if ev["end"] > t0 + EPS]

            items = []
            for ev in active[r]:
                for item in ev["items"]:
                    items.append({
                        "pitch": int(item["pitch"]),
                        "full_dur": float(item["full_dur"]),
                    })

            pitch_set = tuple(sorted(item["pitch"] for item in items))

            part_pitch_items.append(items)
            part_pitch_sets.append(pitch_set)

        if drop_all_rest and all(len(x) == 0 for x in part_pitch_sets):
            continue

        vertical_events.append({
            "start": t0,
            "end": t1,
            "dur": dur,
            "part_pitch_sets": part_pitch_sets,
            "part_pitch_items": part_pitch_items,
        })

    merged = []
    for ev in vertical_events:
        if not merged:
            merged.append(ev.copy())
            continue

        prev = merged[-1]
        if prev["part_pitch_sets"] == ev["part_pitch_sets"]:
            prev["end"] = ev["end"]
            prev["dur"] = qround(prev["end"] - prev["start"], step=None)
        else:
            merged.append(ev.copy())

    return merged


def build_event_cache(df_meta_filtered, quant_step=0.125, min_event_dur=0.125):
    """Build a cache of vertical events for all files in a metadata table.
    INPUT:  df_meta_filtered (pandas.DataFrame): metadata table with columns filename and filepath.
            quant_step (float): quantization step for offsets and durations.
            min_event_dur (float): minimum duration of a retained vertical event.
    OUTPUT: tuple[dict, list]:
            cache: dictionary mapping filename to extracted vertical events.
            failed: list of pairs (filename, error_message) for files that failed to parse.
    """
    cache = {}
    failed = []

    for _, row in tqdm(df_meta_filtered.iterrows(), total=len(df_meta_filtered)):
        fp = row["filepath"]
        try:
            events = extract_vertical_events(
                fp,
                quant_step=quant_step,
                min_event_dur=min_event_dur,
                drop_all_rest=True
            )
            cache[row["filename"]] = events
        except Exception as e:
            failed.append((row["filename"], str(e)))

    return cache, failed
