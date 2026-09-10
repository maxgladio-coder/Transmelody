"""Mutually exclusive MIDI boundary states; phonemes never provide the targets."""
from __future__ import annotations

import torch

from melody_transformer import PredictedNote, rhythm_boundary_mask

REST, HOLD, START, END = range(4)


def boundary_targets(batch):
    """State at t; continuation[t-1] means the SAME note crosses that edge.

    Touching same-pitch notes and marked turns are START, not HOLD. Mask
    chunk starts: their preceding label/context is not available in the chunk.
    Accept either a whole song [T] or a training batch [B,T].
    """
    target = torch.zeros_like(batch["onset"], dtype=torch.long)
    previous = torch.zeros_like(batch["continuation"])
    previous[..., 1:] = batch["continuation"][..., :-1]
    target[previous >= .5] = HOLD
    target[batch["offset"] >= .5] = END
    target[batch["onset"] >= .5] = START
    valid = batch["boundary_mask"].bool().clone()
    if "valid" in batch:
        valid &= batch["valid"]
        valid[..., 1:] &= batch["valid"][..., :-1]
    valid[..., 1:] &= batch["continuation_mask"][..., :-1]
    valid[..., 0] = False
    return target, valid


def decode_boundary_states(outputs, modes, observed):
    """One learned choice per legal node; no onset-threshold precedence.

    START closes/reopens even at identical pitch. HOLD never invents a new
    note. REST/END close it. No minimum length, same-pitch merge or ABA veto.
    Start/end grid legality is checked independently, including across bars.
    """
    scores = outputs["boundary_state"].float().cpu().softmax(-1)
    states = scores.argmax(-1)
    observed = observed.cpu().bool()
    allowed = rhythm_boundary_mask(modes, len(states))
    pitch = outputs["pitch"].float().cpu()
    notes = []
    start = None

    def close(end):
        if start is None or end <= start:
            return
        pitch_start = start + 1 if end - start >= 3 else start
        # Same segment pitch estimator as v2; boundary changes are isolated.
        value = int(pitch[pitch_start:end].mean(0)[1:].argmax())
        notes.append(PredictedNote(start, end, value, float(scores[start, START]), end - start))

    for step in range(len(states)):
        if not observed[step]:
            close(step)
            start = None
            continue
        if not allowed[step]:
            continue
        state = int(states[step])
        if state != HOLD:
            close(step)
            start = step if state == START else None
    close(len(states))
    return notes
