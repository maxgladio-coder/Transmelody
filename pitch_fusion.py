"""Pitch-only fusion; predicted note boundaries are immutable."""
from dataclasses import replace
import numpy as np
import torch


def fuse_pitches(notes, old_logits, new_logits, new_weight):
    if not np.isfinite(new_weight) or not 0 <= new_weight <= 1:
        raise ValueError('Fusion weight must be in [0, 1].')
    if old_logits.shape != new_logits.shape or old_logits.ndim != 2 or old_logits.shape[1] != 129:
        raise ValueError('Pitch heads must share the same [time, 129] timeline.')
    if not torch.isfinite(old_logits).all() or not torch.isfinite(new_logits).all():
        raise ValueError('Pitch scores must be finite.')
    for note in notes:
        if not 0 <= note.start_step < note.end_step <= len(old_logits):
            raise ValueError('Predicted note lies outside the pitch timeline.')
    if new_weight == 0:
        return list(notes)
    old = old_logits.float().cpu()[:, 1:].log_softmax(-1)
    new = new_logits.float().cpu()[:, 1:].log_softmax(-1)
    pooled = (1-new_weight) * old + new_weight * new
    return [replace(note,pitch=int(pooled[note.start_step:note.end_step].mean(0).argmax())) for note in notes]
