"""Song-isolated splits and decoded note metrics, including exact exported timing."""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from transmelody.models.melody_transformer import TICKS_PER_STEP
from transmelody.grid.musical_timeline import samples_at_ticks


def song_split(song_ids: list[str], path: Path, seed: int = 2026,
               validation_ids: list[str] | None = None) -> tuple[list[str], list[str]]:
    ids = sorted(set(song_ids), key=lambda x: (0, int(x)) if x.isdigit() else (1, x))
    if len(ids) < 3:
        raise ValueError("At least three corrected songs are required for song-isolated validation.")
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        val = saved["validation_ids"]
        if validation_ids is not None and set(validation_ids) != set(val):
            raise ValueError("Validation IDs disagree with the saved split; use a new split file for a new experiment.")
    else:
        val = validation_ids if validation_ids is not None else random.Random(seed).sample(ids, min(3, max(1, len(ids) // 5)))
    if not val or len(set(val)) != len(val) or not set(val) < set(ids):
        raise ValueError("Validation songs must exist, be unique, and leave training songs available.")
    train = [sid for sid in ids if sid not in val]
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"version": 1, "seed": seed, "validation_ids": val}, indent=2) + "\n", encoding="utf-8")
    return train, val


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _matched(valid: np.ndarray, distance: np.ndarray) -> int:
    if not valid.size:
        return 0
    # Invalid assignments cost more than all valid distances combined, so
    # cardinality takes precedence over timing preference.
    costs = np.where(valid, np.minimum(distance, 1.0), min(valid.shape) + 2.0)
    rows, columns = linear_sum_assignment(costs)
    return int(valid[rows, columns].sum())


def _reference_note_steps(song: dict):
    grid = song["grid"]
    rate = int(grid["source"]["sample_rate"])
    truth = []
    for start in song["onset"].nonzero().flatten().tolist():
        if not bool(song["loss_mask"][start]):
            continue
        end = start + 1
        # Continuation labels distinguish adjacent same-pitch notes too.
        while end < len(song["pitch"]) and bool(song["continuation"][end - 1]):
            end += 1
        if not bool(song["loss_mask"][end - 1]):
            continue
        if float(samples_at_ticks(grid, end * TICKS_PER_STEP, rate)) > int(grid["source"]["num_samples"]):
            continue
        truth.append((start, end, int(song["pitch"][start]) - 1))
    return truth


def _note_coordinates(notes, song: dict):
    grid = song['grid']
    rate = int(grid['source']['sample_rate'])
    truth = _reference_note_steps(song)
    def coordinates(items):
        if not items:
            return np.zeros((0, 3))
        array = np.array(items, dtype=float)
        array[:, :2] = samples_at_ticks(grid, array[:, :2] * TICKS_PER_STEP, rate) / rate
        return array
    reference = coordinates(truth)
    estimated = coordinates([(n.start_step, n.end_step, n.pitch) for n in notes])
    return reference, estimated


def note_counts(notes, song: dict) -> dict[str, int]:
    reference, estimated = _note_coordinates(notes, song)
    onset_distance = abs(reference[:, None, 0] - estimated[None, :, 0])
    offset_distance = abs(reference[:, None, 1] - estimated[None, :, 1])
    onset_ok = onset_distance <= 0.05
    pitch_ok = reference[:, None, 2] == estimated[None, :, 2]
    offset_tolerance = np.maximum(0.05, 0.2 * (reference[:, 1] - reference[:, 0]))[:, None]
    full = onset_ok & pitch_ok & (offset_distance <= offset_tolerance)
    return {"reference_notes": len(reference), "predicted_notes": len(estimated),
        "onset_matches": _matched(onset_ok, onset_distance),
        "pitch_onset_matches": _matched(onset_ok & pitch_ok, onset_distance),
        "full_note_matches": _matched(full, onset_distance + offset_distance)}


def semitone_aba_counts(notes, song: dict) -> dict[str, int]:
    """Three adjacent pitch/onset matches, with no extra pitch-transition rule."""
    reference, estimated = _note_coordinates(notes, song)
    distance = abs(reference[:, None, 0] - estimated[None, :, 0])
    valid = (distance <= .05) & (reference[:, None, 2] == estimated[None, :, 2])
    matched = set()
    if valid.size:
        rows, cols = linear_sum_assignment(np.where(valid, np.minimum(distance,1.),min(valid.shape)+2.))
        matched = {int(r) for r,c in zip(rows,cols) if valid[r,c]}
    centers = [i for i in range(1,len(reference)-1)
        if reference[i-1,2] == reference[i+1,2] and abs(reference[i,2]-reference[i-1,2]) == 1
        and abs(reference[i,0]-reference[i-1,1]) <= .05
        and abs(reference[i+1,0]-reference[i,1]) <= .05]
    return {'aba_count':len(centers),
        'aba_recovered':sum(all(j in matched for j in (i-1,i,i+1)) for i in centers)}


def passes_pitch_aba_guard(candidate, baseline):
    """A higher overall score alone must not conceal a semitone ABA regression."""
    return (candidate['pitch_onset_f1'] >= baseline['pitch_onset_f1'] and
        all(candidate['songs'][sid]['aba_recovered'] >= old['aba_recovered']
            for sid,old in baseline['songs'].items()))


def note_artifact_counts(notes,song):
    from transmelody.evaluation.compare_melody_midi import edit_patterns
    ref,pred = _note_coordinates(notes,song)
    short_mask = np.array([b-a <= 6 for a,b,_ in _reference_note_steps(song)],dtype=bool)
    short = ref[short_mask]
    distance = abs(short[:,None,0]-pred[None,:,0])
    valid = (distance <= .05) & (short[:,None,2] == pred[None,:,2])
    edits = edit_patterns(ref,pred)
    return {'short_note_count':int(short_mask.sum()),'short_note_matches':_matched(valid,distance),
        'split_reference_notes':edits['split_reference_notes'],
        'mostly_reference_rest_notes':edits['mostly_reference_rest_notes']}


def passes_artifact_guard(candidate,baseline):
    """Do not buy fewer fragments by swallowing genuine short notes or ABA."""
    return (passes_pitch_aba_guard(candidate,baseline)
        and all(candidate['songs'][sid]['short_note_matches'] >= old['short_note_matches']
            for sid,old in baseline['songs'].items())
        and all(sum(s[key] for s in candidate['songs'].values()) <= sum(s[key] for s in baseline['songs'].values())
            for key in ('split_reference_notes','mostly_reference_rest_notes')))


def summarize_note_counts(counts: dict[str, int]) -> dict:
    result = dict(counts)
    denominator = counts["reference_notes"] + counts["predicted_notes"]
    for name in ("onset", "pitch_onset", "full_note"):
        tp = counts[name + "_matches"]
        result[name + "_precision"] = tp / max(1, counts["predicted_notes"])
        result[name + "_recall"] = tp / max(1, counts["reference_notes"])
        result[name + "_f1"] = 2 * tp / max(1, denominator)
    result["missed_notes"] = counts["reference_notes"] - counts["pitch_onset_matches"]
    result["extra_notes"] = counts["predicted_notes"] - counts["pitch_onset_matches"]
    return result
