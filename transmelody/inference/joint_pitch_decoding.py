"""Optional global pitch/boundary decoding of existing MIDI-supervised scores.

No raw-F0 thresholds, key snapping, pitch-jump penalties or ABA suppression.
This is a discriminative path score, NOT calibrated posterior inference.
"""
from __future__ import annotations

import numpy as np
import torch

from transmelody.models.melody_transformer import PredictedNote, rhythm_boundary_mask
from transmelody.models.musical_boundary import REST, HOLD, START, END


def decode_pitch_boundary_sequence(outputs, modes, observed, *, pitch_weight=1.):
    if not np.isfinite(pitch_weight) or pitch_weight <= 0:
        raise ValueError('Pitch evidence weight must be finite and positive.')
    if 'boundary_state' not in outputs:
        raise ValueError('Joint sequence decoding needs a trained boundary-state head.')
    observed = observed.cpu().bool()
    length = len(observed)
    if not length:
        return []
    allowed = rhythm_boundary_mask(modes, length)
    nodes = allowed.nonzero().flatten().tolist()
    # End at an observation break even when it falls between musical nodes.
    breaks = (observed[:-1] & ~observed[1:]).nonzero().flatten().add(1).tolist()
    nodes = sorted(set(nodes + breaks + [length]))
    pitch = outputs['pitch'].float().cpu().log_softmax(-1).numpy()
    boundary = outputs['boundary_state'].float().cpu().log_softmax(-1).numpy()
    count = len(nodes) - 1
    parents = np.zeros((count, 129), dtype=np.int16)
    restarts = np.zeros((count, 129), dtype=bool)
    score = np.full(129, -np.inf, dtype=np.float64)
    score[0] = 0.
    for cell, (left, right) in enumerate(zip(nodes, nodes[1:])):
        valid = observed[left:right].numpy()
        # Integrate acoustic support over time (one unit per sixteenth), not
        # one equal vote per entire provisional note. All 128 pitches compete.
        emission = (pitch[left:right][valid].sum(0) / 3 * pitch_weight
            if valid.any() else np.zeros(129))
        local = boundary[left]
        best_previous = int(score.argmax())
        best_active = int(score[1:].argmax()) + 1
        stay_silent = score[0] + local[REST]
        end_note = score[best_active] + local[END]
        new = np.empty_like(score)
        if end_note > stay_silent:
            parents[cell, 0], new[0] = best_active, end_note
        else:
            parents[cell, 0], new[0] = 0, stay_silent
        continuation = score[1:] + local[HOLD]
        start_note = score[best_previous] + local[START]
        restart = start_note > continuation
        parents[cell, 1:] = np.where(restart, best_previous, np.arange(1, 129))
        restarts[cell, 1:] = restart
        new[1:] = np.maximum(start_note, continuation)
        new += emission
        if not observed[left] or not allowed[left]:
            new[1:] = -np.inf
        score = new - np.max(new)
    state = int(score.argmax())
    path, starts = np.zeros(count, dtype=np.int16), np.zeros(count, dtype=bool)
    for cell in range(count - 1, -1, -1):
        path[cell], starts[cell] = state, restarts[cell, state]
        state = int(parents[cell, state])
    notes = []
    onset, active_pitch = None, None

    def close(end):
        if onset is not None and end > onset:
            notes.append(PredictedNote(onset, end, active_pitch,
                float(np.exp(boundary[onset, START])), end - onset))

    for cell, state in enumerate(path):
        if not state:
            close(nodes[cell])
            onset, active_pitch = None, None
        elif starts[cell] or onset is None:
            close(nodes[cell])
            onset, active_pitch = nodes[cell], int(state) - 1
    close(length)
    return notes
