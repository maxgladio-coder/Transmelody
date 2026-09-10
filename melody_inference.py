"""Shared overlapping inference and score export for all entry points."""
from __future__ import annotations

from pathlib import Path

import torch

from melody_transformer import (STEPS_PER_BAR, FeatureConfig, decode_rhythm_modes,
    decode_segment_events, save_segment_prediction_midi)
from musical_timeline import canonicalize_grid
from tempo_map import build_tempo_segments, save_tempo_map_midi, tempo_events


@torch.inference_mode()
def predict_outputs(model, features: torch.Tensor, device: torch.device,
                    chunk_steps: int = 8 * STEPS_PER_BAR) -> dict[str, torch.Tensor]:
    if chunk_steps < STEPS_PER_BAR or chunk_steps % STEPS_PER_BAR:
        raise ValueError("Inference chunks must contain a positive whole number of bars.")
    if not len(features):
        raise ValueError("Cannot predict an empty song.")
    stride = max(STEPS_PER_BAR, (chunk_steps // STEPS_PER_BAR // 2) * STEPS_PER_BAR)
    starts = list(range(0, max(1, len(features) - chunk_steps + 1), stride))
    final_start = max(0, len(features) - chunk_steps)
    if starts[-1] != final_start:
        starts.append(final_start)
    totals: dict[str, torch.Tensor] = {}
    counts = torch.zeros(len(features))
    for start in starts:
        chunk = features[start:start + chunk_steps].float().to(device)[None]
        positions = (torch.arange(start, start + chunk.shape[1], device=device) % STEPS_PER_BAR)[None]
        outputs = model(chunk, positions)
        # The center has full left/right context; song endpoints keep nonzero weight.
        weights = torch.hann_window(chunk.shape[1] + 2, periodic=False)[1:-1].clamp_min(0.05)
        end = start + chunk.shape[1]
        counts[start:end] += weights
        for name, values in outputs.items():
            values = values[0].float().cpu()
            if name not in totals:
                totals[name] = torch.zeros((len(features), *values.shape[1:]))
            shape = (-1,) + (1,) * (values.ndim - 1)
            totals[name][start:end] += values * weights.reshape(shape)
    result = {name: value / counts.reshape((-1,) + (1,) * (value.ndim - 1)) for name, value in totals.items()}
    if 'note_active_score' in result:
        # Calibrate on the same combined activity logits that inference sees.
        # Averaging log-sigmoid(chunk logits) is NOT log-sigmoid(mean logits).
        weight = model.config.note_activity_weight / 3
        result['note_active_score'] = torch.nn.functional.logsigmoid(result['activity']) * weight
        result['note_rest_score'] = torch.nn.functional.logsigmoid(-result['activity']) * weight
    return result


def synchronize_prediction_grid(grid: dict, max_error_ms: float = 25.0) -> dict:
    segments = build_tempo_segments(grid, max_anchor_error_ms=max_error_ms)
    return canonicalize_grid(grid, tempo_events(segments))


def decode_prediction(outputs, features: torch.Tensor, config: FeatureConfig,
                      *, output_grid: str = "auto", triplet_threshold: float = 0.60,
                      joint_pitch_weight: float | None = None):
    if joint_pitch_weight is not None:
        from joint_pitch_decoding import decode_pitch_boundary_sequence
        _, modes = decode_prediction(outputs, features, config,
            output_grid=output_grid, triplet_threshold=triplet_threshold)
        return decode_pitch_boundary_sequence(outputs, modes, features[:, config.audio_valid_index] >= .5,
            pitch_weight=joint_pitch_weight), modes
    if "note_event_onset" in outputs:
        from musical_event_decoding import select_event_grids
        from note_event_model import decode_note_events
        observed = features[:, config.audio_valid_index] >= .5
        modes = select_event_grids(outputs, observed, triplet_threshold)
        if output_grid == "1/16":
            modes.zero_()
        return decode_note_events(outputs, modes, observed), modes
    if "musical_event_onset" in outputs:
        from musical_event_decoding import decode_musical_events
        return decode_musical_events(outputs, features[:, config.audio_valid_index] >= .5,
            output_grid=output_grid, triplet_threshold=triplet_threshold)
    modes = decode_rhythm_modes(outputs["rhythm"], total_steps=len(features), triplet_threshold=triplet_threshold)
    if output_grid == "1/16":
        modes.zero_()
    notes = decode_segment_events(outputs, rhythm_modes=modes,
        observation_mask=features[:, config.audio_valid_index] >= 0.5)
    return notes, modes


def export_prediction(midi_path: Path, grid: dict, notes, modes) -> Path:
    if not grid.get("timeline"):
        raise ValueError("Synchronize the grid BEFORE extracting prediction features.")
    segments = build_tempo_segments(grid)
    tempo_path = midi_path.with_name(midi_path.stem.removesuffix("_predicted") + "_tempo_map.mid")
    save_tempo_map_midi(tempo_path, grid, segments)
    save_segment_prediction_midi(midi_path, notes, bpm=float(grid["tempo"]["bpm"]),
        grid_end_tick=int(grid["midi_alignment"]["grid_end_tick"]), rhythm_modes=modes,
        tempo_map_events=tempo_events(segments), audio_start_tick=int(grid["midi_alignment"]["audio_start_tick"]))
    return tempo_path


def rhythm_confidences(outputs) -> list[float]:
    probabilities = outputs["rhythm"].float().softmax(-1)[:, 1]
    return [float(probabilities[start:start + STEPS_PER_BAR].mean())
            for start in range(0, len(probabilities), STEPS_PER_BAR)]
