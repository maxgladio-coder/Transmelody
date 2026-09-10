"""Score-event evidence first, grid assignment second. No phoneme veto or ABA rule."""
from __future__ import annotations

import math

import torch

from transmelody.models.melody_transformer import STEPS_PER_BAR, decode_segment_events


def event_peaks(probability, threshold=.55):
    """Select neural-event peaks; equal plateaus have one leftmost representative."""
    p = probability.float().cpu()
    left = torch.cat([p.new_tensor([-1.]), p[:-1]])
    right = torch.cat([p[1:], p.new_tensor([-1.])])
    return ((p >= threshold) & (p > left) & (p >= right)).nonzero().flatten().tolist()


def select_event_grids(outputs, observed, triplet_threshold=.60):
    """Combine learned bar prior with unmasked note-edge placement evidence.

    Scores are heuristic compatibility, not calibrated acoustic probabilities.
    Shared straight/triplet positions carry no discriminating rhythm evidence.
    Each start/end belongs to the bar containing that edge, not its entire note.
    """
    if not .5 <= triplet_threshold <= 1:
        raise ValueError("Triplet threshold must lie in [0.5, 1].")
    observed = observed.cpu().bool()
    onset = outputs["onset"].float().cpu().sigmoid()
    offset = outputs["offset"].float().cpu().sigmoid()
    peaks = {}
    for p in (onset, offset):
        for index in event_peaks(p):
            if observed[index]:
                peaks[index] = max(peaks.get(index, 0.), float(p[index]))
    probabilities = outputs["rhythm"].float().cpu().softmax(-1)
    modes = []
    for start in range(0, len(observed), STEPS_PER_BAR):
        prior = probabilities[start:start + STEPS_PER_BAR].mean(0).clamp_min(1e-6)
        differences, weights = [], []
        for index, weight in peaks.items():
            if not start <= index < start + STEPS_PER_BAR:
                continue
            local = index - start
            distances = [min(local % step, step - local % step) for step in (3, 4)]
            if distances[0] != distances[1]:
                differences.append((distances[0] ** 2 - distances[1] ** 2) / (2 * .5 ** 2))
                weights.append(weight)
        evidence = (sum(d * w for d, w in zip(differences, weights)) / sum(weights)
            * min(3., len(weights) / 3)) if weights else 0.
        # An uncalibrated, imbalanced classifier must not erase contradictory
        # acoustic events with extreme logits. Limit this heuristic prior to 2:1.
        prior_log_odds = torch.log(prior[1] / prior[0]).clamp(-math.log(2), math.log(2))
        score = float(torch.sigmoid(prior_log_odds + evidence))
        modes.append(int(score >= triplet_threshold))
    return torch.tensor(modes, dtype=torch.long)


def project_event_peaks(logits, modes, observed, *, threshold=.55):
    """Assign each neural event once to its nearest legal node in the same bar.

    Keep neighboring off-grid evidence instead of clearing it before matching.
    This never adds a note based on spectral/phoneme thresholds.
    """
    probability = logits.float().cpu().sigmoid()
    result = torch.full_like(probability, -20.)
    for source in event_peaks(probability, threshold):
        if not bool(observed[source]):
            continue
        bar = source // STEPS_PER_BAR
        multiple = 4 if int(modes[bar]) else 3
        local = source % STEPS_PER_BAR
        target = bar * STEPS_PER_BAR + round(local / multiple) * multiple
        # An edge near a bar line may quantize to that shared next-bar line.
        if target >= len(result) or not bool(observed[target]):
            continue
        result[target] = max(float(result[target]), float(logits[source]))
    return result


def decode_musical_events(outputs, observed, *, output_grid="auto", triplet_threshold=.60):
    observed = observed.cpu().bool()
    modes = select_event_grids(outputs, observed, triplet_threshold)
    if output_grid == "1/16":
        modes.zero_()
    if "boundary_state" in outputs:
        from transmelody.models.musical_boundary import decode_boundary_states
        return decode_boundary_states(outputs, modes, observed), modes
    projected = dict(outputs)
    projected["onset"] = project_event_peaks(outputs["onset"], modes, observed)
    projected["offset"] = project_event_peaks(outputs["offset"], modes, observed)
    # These heads still learn MIDI turns/rearticulations, but cannot independently
    # fabricate extra starts or prevent a MIDI-trained same-vowel turn.
    projected["pitch_change"] = torch.full_like(projected["onset"], -20.)
    projected["articulation"] = torch.full_like(projected["onset"], -20.)
    notes = decode_segment_events(projected, rhythm_modes=modes, observation_mask=observed,
        strong_onset_threshold=.55)
    return notes, modes
