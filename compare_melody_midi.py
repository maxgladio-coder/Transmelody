"""Read-only score comparison on each MIDI's own tempo map and WAV START.

Assignments diagnose edits, not singer intonation or acoustic delay causality.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import mido
import numpy as np
from scipy.optimize import linear_sum_assignment

from audio_audition import midi_notes_in_wav_time
from melody_evaluation import file_digest, summarize_note_counts


def pairs(valid, distance):
    if not valid.size:
        return []
    cost = np.where(valid, np.minimum(distance, 1.), min(valid.shape) + 2.)
    rows, cols = linear_sum_assignment(cost)
    return [(int(a), int(b)) for a, b in zip(rows, cols) if valid[a, b]]


def edit_patterns(reference, predicted):
    """Timing-only review candidates; not automatic breath/phoneme labels.

    80% overlap and 50 ms tolerances are diagnostics only, never MIDI filters.
    Short genuine notes are retained when present in the reference score.
    """
    ref = np.asarray(reference, dtype=float).reshape(-1,3)
    pred = np.asarray(predicted, dtype=float).reshape(-1,3)
    overlap = np.maximum(0., np.minimum(ref[:,None,1],pred[None,:,1])-
        np.maximum(ref[:,None,0],pred[None,:,0]))
    owners, rests = {}, []
    for j, note in enumerate(pred):
        duration = max(1e-9,note[1]-note[0])
        # Union coverage, so overlapping reference voices cannot double count.
        spans = sorted((max(note[0],r[0]),min(note[1],r[1])) for r in ref
            if min(note[1],r[1]) > max(note[0],r[0]))
        coverage, right = 0., note[0]
        for a,b in spans:
            coverage += max(0.,b-max(a,right))
            right = max(right,b)
        if coverage / duration <= .2:
            rests.append(dict(prediction_index=j,note=note.tolist(),score_overlap_fraction=coverage/duration))
        if len(ref):
            i = int(overlap[:,j].argmax())
            if overlap[i,j] / duration >= .8:
                owners.setdefault(i,[]).append(j)
    split, tails = [], []
    for i, js in owners.items():
        if len(js) > 1:
            split.append(dict(reference_index=i,reference=ref[i].tolist(),prediction_indices=js,
                fragments=pred[js].tolist()))
        for j in js:
            if abs(pred[j,0]-ref[i,0]) <= .05 and pred[j,2] == ref[i,2] and pred[j,1]-ref[i,1] > .05:
                tails.append(dict(reference_index=i,prediction_index=j,
                    reference=ref[i].tolist(),prediction=pred[j].tolist()))
    return dict(split_reference_notes=len(split),split_extra_fragments=sum(len(x['fragments'])-1 for x in split),
        mostly_reference_rest_notes=len(rests),overlong_same_pitch_tails=len(tails),
        split_examples=split,rest_examples=rests,tail_examples=tails,
        caveat='Timing overlap candidates only; no claim that deleted audio is breath or delay. '
            'Pitch substitutions with correct boundaries are not counted as splits.')


def compare(reference, predicted, tolerance=.05):
    ref = np.asarray(reference, dtype=float).reshape(-1, 3)
    pred = np.asarray(predicted, dtype=float).reshape(-1, 3)
    onset = abs(ref[:, None, 0] - pred[None, :, 0])
    offset = abs(ref[:, None, 1] - pred[None, :, 1])
    same_pitch = ref[:, None, 2] == pred[None, :, 2]
    on_ok = onset <= tolerance
    pitch_pairs = pairs(on_ok & same_pitch, onset)
    # Pitch is only a tie-break among essentially identical onset distances.
    onset_pairs = pairs(on_ok, onset + (~same_pitch) * 1e-7)
    full = on_ok & same_pitch & (offset <= np.maximum(.05, .2 * (ref[:, 1] - ref[:, 0]))[:, None])
    metrics = summarize_note_counts(dict(reference_notes=len(ref), predicted_notes=len(pred),
        onset_matches=len(onset_pairs), pitch_onset_matches=len(pitch_pairs),
        full_note_matches=len(pairs(full, onset + offset))))
    mismatch = []
    for i, j in onset_pairs:
        if ref[i, 2] != pred[j, 2]:
            mismatch.append(dict(reference_index=i, prediction_index=j,
                wav_seconds=float(ref[i, 0]), reference_pitch=int(ref[i, 2]),
                predicted_pitch=int(pred[j, 2]), semitone_error=int(pred[j, 2] - ref[i, 2])))
    matched = dict(pitch_pairs)
    aba = []
    for i in range(1, len(ref) - 1):
        a, b, c = ref[i-1:i+2]
        gaps = (b[0]-a[1], c[0]-b[1])
        if a[2] == c[2] and abs(a[2]-b[2]) == 1 and all(abs(g) <= tolerance for g in gaps):
            recovered = all(k in matched for k in (i-1, i, i+1))
            js = [matched.get(k) for k in (i-1, i, i+1)]
            consecutive = recovered and js[1] == js[0]+1 and js[2] == js[1]+1
            context = pred[(pred[:, 0] < c[1]+.1) & (pred[:, 1] > a[0]-.1)].tolist()
            aba.append(dict(reference_indices=[i-1,i,i+1], reference=ref[i-1:i+2].tolist(),
                all_three_pitch_onsets_matched=bool(recovered),
                consecutive_prediction=bool(consecutive), predicted_nearby=context))
    offset_errors = [float(pred[j,1]-ref[i,1]) for i,j in pitch_pairs]
    return dict(metrics=metrics, onset_tolerance_sec=tolerance,
        onset_matched_wrong_pitch=len(mismatch),
        pitch_error_histogram=dict(Counter(str(x['semitone_error']) for x in mismatch)),
        pitch_errors=mismatch, semitone_aba=dict(count=len(aba),
            recovered=sum(x['all_three_pitch_onsets_matched'] for x in aba),
            consecutive=sum(x['consecutive_prediction'] for x in aba), examples=aba),
        matched_pitch_offset_error_sec=dict(median=float(np.median(offset_errors)) if offset_errors else None,
            absolute_median=float(np.median(np.abs(offset_errors))) if offset_errors else None),
        extra_prediction_indices=[j for j in range(len(pred)) if j not in set(matched.values())],
        missing_reference_indices=[i for i in range(len(ref)) if i not in matched])


def metadata(path):
    midi = mido.MidiFile(path)
    tick, events = 0, []
    for msg in mido.merge_tracks(midi.tracks):
        tick += msg.time
        if msg.type in ('set_tempo', 'time_signature', 'marker'):
            events.append(dict(tick=tick, message=msg.dict()))
    return dict(path=str(path.resolve()), sha256=file_digest(path), ppq=midi.ticks_per_beat,
        end_tick=tick, events=events)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reference', type=Path)
    parser.add_argument('prediction', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    ref = [x[:3] for x in midi_notes_in_wav_time(args.reference)]
    pred = [x[:3] for x in midi_notes_in_wav_time(args.prediction)]
    result = compare(ref, pred)
    result.update(reference=metadata(args.reference), prediction=metadata(args.prediction),
        interpretation='No automatic alignment or pitch correction. Actual WAV START event ticks are used. '
        'Missing/extra counts include timing/pitch substitutions; they are not pure deletions/additions. '
        'ABA means adjacent one-semitone MIDI pitches, not inferred scale degrees.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k: result[k] for k in ('metrics','onset_matched_wrong_pitch','pitch_error_histogram')}, indent=2))
    print('Semitone ABA:', result['semitone_aba']['recovered'], '/', result['semitone_aba']['count'])
    print('Report:', args.output)


if __name__ == '__main__':
    main()
