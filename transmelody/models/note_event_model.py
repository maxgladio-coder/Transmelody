"""Complete-note scores shared by supervised contrastive learning and decoding.

Semi-Markov decoding uses the learned duration distribution. Its final duration
bin is a >= bin, so sustained notes are not forcibly capped or chopped.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from transmelody.models.melody_transformer import PredictedNote, rhythm_boundary_mask
from transmelody.models.musical_boundary import REST, HOLD, START, END

PITCH_WEIGHT = .25
DURATION_WEIGHT = .25


def decode_note_events(outputs, modes, observed, *, pitch_weight=PITCH_WEIGHT,
                       duration_weight=DURATION_WEIGHT, allowed_boundary_mask=None):
    if not np.isfinite([pitch_weight, duration_weight]).all() or pitch_weight <= 0 or duration_weight < 0:
        raise ValueError('Invalid complete-note score weights.')
    observed = observed.cpu().bool().numpy()
    length = len(observed)
    if not length:
        return []
    if allowed_boundary_mask is None:
        allowed = rhythm_boundary_mask(modes, length).numpy()
    else:
        allowed = torch.as_tensor(allowed_boundary_mask).cpu().bool().numpy().copy()
        if allowed.shape != (length,):
            raise ValueError('Boundary candidate mask must match the observed timeline.')
    breaks = np.flatnonzero(observed[:-1] & ~observed[1:]) + 1
    nodes = np.unique(np.r_[np.flatnonzero(allowed), breaks, length])
    count = len(nodes)
    pitch = outputs['pitch'].float().cpu().log_softmax(-1).numpy() * pitch_weight / 3
    if 'note_active_score' in outputs:
        pitch[:,1:] += outputs['note_active_score'].float().cpu().numpy()[:,None]
        pitch[:,0] += outputs['note_rest_score'].float().cpu().numpy()
    pitch[~observed] = 0
    cumulative = np.vstack([np.zeros((1, 129)), np.cumsum(pitch, axis=0, dtype=np.float64)])[nodes]
    boundary = outputs['boundary_state'].float().cpu().log_softmax(-1).numpy()[nodes[:-1]]
    duration = outputs['duration'].float().cpu().log_softmax(-1).numpy()[nodes[:-1]]
    cap = duration.shape[-1]
    hold = np.r_[0., np.cumsum(boundary[:, HOLD], dtype=np.float64)]
    off, active = np.full(count, -np.inf), np.full(count, -np.inf)
    off[0] = 0.
    off_parent = np.zeros(count, dtype=np.int8)
    note_start, note_pitch = np.zeros(count, dtype=np.int32), np.zeros(count, dtype=np.int16)
    begin = np.full(count, -np.inf)
    begin_parent = np.zeros(count, dtype=np.int8)
    short = []
    tail = np.full(128, -np.inf)
    tail_start = np.zeros(128, dtype=np.int32)

    for right in range(1, count):
        left = right - 1
        a, b = nodes[left], nodes[right]
        if observed[a] and allowed[a]:
            begin_parent[left] = int(active[left] > off[left])
            begin[left] = max(active[left], off[left]) + boundary[left, START]
            short.append(left)
        if not observed[a]:
            short.clear()
            tail.fill(-np.inf)
        # Move >=cap starts to an exact running maximum per pitch; arbitrarily
        # long notes retain their own start/backpointer without O(song^2) work.
        while short and b - nodes[short[0]] >= cap:
            start = short.pop(0)
            candidate = (begin[start] - cumulative[start, 1:] - hold[start+1]
                + duration_weight * duration[start, -1])
            better = candidate > tail
            tail[better], tail_start[better] = candidate[better], start
        best = tail + cumulative[right, 1:] + hold[right]
        pitch_index = int(best.argmax())
        active[right], note_start[right], note_pitch[right] = best[pitch_index], tail_start[pitch_index], pitch_index
        if short:
            indices = np.array(short)
            durations = b - nodes[indices]
            candidate = (begin[indices, None] + cumulative[right, None, 1:] - cumulative[indices, 1:]
                + hold[right] - hold[indices+1, None]
                + duration_weight * duration[indices, durations-1, None])
            winner = np.unravel_index(candidate.argmax(), candidate.shape)
            value = candidate[winner]
            if value > active[right]:
                active[right], note_start[right], note_pitch[right] = value, indices[winner[0]], winner[1]
        stay = off[left] + boundary[left, REST]
        end = active[left] + boundary[left, END]
        off_parent[right] = int(end > stay)
        off[right] = max(stay, end) + cumulative[right, 0] - cumulative[left, 0]

    cursor, state, notes = count - 1, int(active[-1] > off[-1]), []
    while cursor > 0:
        if state:
            start = int(note_start[cursor])
            a, b = int(nodes[start]), int(nodes[cursor])
            notes.append(PredictedNote(a, b, int(note_pitch[cursor]),
                float(np.exp(boundary[start, START])), b-a))
            state, cursor = int(begin_parent[start]), start
        else:
            state, cursor = int(off_parent[cursor]), cursor - 1
    return list(reversed(notes))


def note_event_loss(outputs, batch, *, batch_index=0, max_notes=12, semitone_contrasts=False,
                    artifact_contrasts=False, allowed_boundary_mask=None):
    """Compare labeled notes with false splits, merged neighbors and shifted edges.

    This is sampled structured contrastive supervision, not an exact global
    likelihood. Every score uses the same pitch/start/hold/duration terms as the
    decoder. Only fully observed, complete notes participate.
    """
    item = batch_index % len(batch['onset'])
    logits = {k: outputs[k][item].float() for k in ('pitch', 'boundary_state', 'duration')}
    pitch = logits['pitch'].log_softmax(-1)
    state = logits['boundary_state'].log_softmax(-1)
    duration = logits['duration'].log_softmax(-1)
    valid = (batch['valid'][item] & batch['loss_mask'][item]).detach().cpu()
    onset = batch['onset'][item].detach().cpu()
    continuation = batch['continuation'][item].detach().cpu()
    if allowed_boundary_mask is None:
        rhythm = batch['rhythm_mode'][item].detach().cpu()
        allowed = torch.tensor([i % 48 % (4 if int(rhythm[i]) else 3) == 0 for i in range(len(valid))])
    else:
        allowed = torch.as_tensor(allowed_boundary_mask).detach().cpu().bool()
        if allowed.ndim == 2:
            allowed = allowed[item]
        if allowed.shape != valid.shape:
            raise ValueError('Structured-loss candidate mask must match score steps.')
    nodes = allowed.nonzero().flatten().tolist()
    truth = []
    for start in onset.nonzero().flatten().tolist():
        if start == 0 or not valid[start] or not allowed[start]:
            continue
        end = start + 1
        while end < len(valid) and continuation[end-1]:
            end += 1
        if end < len(valid) and allowed[end] and valid[start:end+1].all():
            truth.append((start, end, int(batch['pitch'][item, start])))
    if not truth and not artifact_contrasts:
        return pitch.sum() * 0.
    indices = torch.randperm(len(truth))[:max_notes].tolist()
    aba_centers = []
    if semitone_contrasts:
        for i in range(1, len(truth)-1):
            left, center, right = truth[i-1:i+2]
            if (left[1] == center[0] and center[1] == right[0]
                    and left[2] == right[2] and abs(left[2]-center[2]) == 1):
                aba_centers.append(i)
        # Labels, not an inference-time rule: give rare semitone returns a chance
        # to contribute even when the random note sample mostly contains holds.
        chosen = torch.randperm(len(aba_centers))[:4].tolist()
        aba_centers = [aba_centers[i] for i in chosen]
        indices = sorted(set(indices + aba_centers))
    losses = []

    def score(a, b, note_pitch=None):
        evidence = PITCH_WEIGHT * pitch[a:b, 1:].sum(0) / 3
        acoustic = evidence.max() if note_pitch is None else evidence[note_pitch-1]
        interior = [n for n in nodes if a < n < b]
        activity = outputs['note_active_score'][item,a:b].sum() if 'note_active_score' in outputs else 0.
        return (acoustic + activity + state[a, START] + state[interior, HOLD].sum()
            + DURATION_WEIGHT * duration[a, min(b-a, duration.shape[-1])-1])

    def rest_score(a,b):
        # Compare a completed false note with continuous REST through its end.
        activity = outputs['note_rest_score'][item,a:b].sum() if 'note_rest_score' in outputs else 0.
        return (PITCH_WEIGHT * pitch[a:b,0].sum()/3 + activity
            + state[[n for n in nodes if a <= n <= b],REST].sum())

    for index in indices:
        a, b, p = truth[index]
        correct = score(a, b, p)
        if semitone_contrasts:
            # Equal-note weighting keeps a short B from being drowned out by
            # long A frames. No preferred key, scale, or pitch at inference.
            pooled = pitch[a:b].mean(0)
            alternatives = [q for q in (p-1, p+1) if 1 <= q <= 128]
            losses.append(F.cross_entropy(pooled[None], torch.tensor([p], device=pitch.device)))
            losses.append(F.softplus(pooled[alternatives] - pooled[p]).mean())
            if index in aba_centers:
                left, _, right = truth[index-1:index+2]
                gold = score(*left) + correct + score(*right)
                # Both failure modes: AAA with correct boundaries, or one
                # continuous A. Real sustained A still gets anti-split training.
                losses.append(F.softplus(score(a, b, left[2]) - correct))
                losses.append(F.softplus(score(left[0], right[1], left[2]) - gold))
        splits = [n for n in nodes if a < n < b]
        if splits:
            split = splits[len(splits)//2]
            wrong = score(a, split) + score(split, b)
            losses.append(F.softplus(wrong - correct))
            if artifact_contrasts:
                # Head/tail splits as well as the midpoint. A true short note
                # is a separate labeled note and never appears inside this span.
                for split in sorted(set([splits[0],splits[-1]])-{split}):
                    losses.append(F.softplus(score(a,split)+score(split,b)-correct))
        if index + 1 < len(truth) and truth[index+1][0] == b:
            _, end, next_pitch = truth[index+1]
            correct_pair = correct + score(b, end, next_pitch)
            losses.append(F.softplus(score(a, end) - correct_pair))
            candidates = [n for n in nodes if a < n < end and n != b]
            if candidates:
                shifted = min(candidates, key=lambda n: abs(n-b))
                losses.append(F.softplus(score(a, shifted) + score(shifted, end) - correct_pair))
    if artifact_contrasts:
        # Learn deletions from completely reviewed REST spans, not from low
        # energy or short duration. No assumptions about whether audio is breath.
        pitches = batch['pitch'][item].detach().cpu()
        candidates = [(a,nodes[j+span]) for j,a in enumerate(nodes) for span in (1,2)
            if j+span < len(nodes) and a > 0
            and valid[a-1:nodes[j+span]+1].all()
            and (pitches[a-1:nodes[j+span]+1] == 0).all()]
        if len(candidates) > 32:
            candidates = [candidates[i] for i in np.linspace(0,len(candidates)-1,32,dtype=int)]
        if candidates:
            differences = torch.stack([score(a,b)+state[b,END]-rest_score(a,b) for a,b in candidates])
            losses.extend(F.softplus(differences.topk(min(4,len(candidates))).values).unbind())
    return torch.stack(losses).mean() if losses else pitch.sum() * 0.
