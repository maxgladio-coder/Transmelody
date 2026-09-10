from __future__ import annotations
from transmelody.paths import PROJECT_ROOT

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

os.environ.setdefault("TORCH_HOME", str(PROJECT_ROOT / ".cache" / "torch"))
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(PROJECT_ROOT / ".cache" / "matplotlib"),
)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf


from beat_this.inference import Audio2Beats  # noqa: E402
from transmelody.config import INTEGER_BPM_SNAP_TOLERANCE, PROJECT_PPQ


@dataclass(frozen=True)
class AudioBundle:
    signal: np.ndarray
    sample_rate: int
    num_samples: int
    source_paths: tuple[Path, ...]


def load_and_mix_audio(paths: Iterable[Path]) -> AudioBundle:
    source_paths = tuple(Path(path).resolve() for path in paths)
    if not source_paths:
        raise ValueError("At least one audio input is required.")

    mixed: np.ndarray | None = None
    sample_rate: int | None = None
    expected_shape: tuple[int, ...] | None = None

    for path in source_paths:
        signal, current_sample_rate = sf.read(path, dtype="float32", always_2d=True)
        if sample_rate is None:
            sample_rate = current_sample_rate
            expected_shape = signal.shape
            mixed = np.zeros(expected_shape, dtype=np.float32)
        elif current_sample_rate != sample_rate:
            raise ValueError(
                f"Sample-rate mismatch: {path} is {current_sample_rate} Hz, "
                f"expected {sample_rate} Hz."
            )
        elif signal.shape != expected_shape:
            raise ValueError(
                f"Audio-shape mismatch: {path} is {signal.shape}, "
                f"expected {expected_shape}."
            )
        mixed += signal

    assert mixed is not None
    assert sample_rate is not None

    mixed /= len(source_paths)
    return AudioBundle(
        signal=mixed,
        sample_rate=sample_rate,
        num_samples=mixed.shape[0],
        source_paths=source_paths,
    )


def robust_tempo(beats_sec: np.ndarray) -> dict[str, float | list[float]]:
    intervals = np.diff(beats_sec)
    intervals = intervals[np.isfinite(intervals) & (intervals > 0.1)]
    if len(intervals) == 0:
        raise ValueError("Not enough valid beats to estimate tempo.")

    bpm_values = 60.0 / intervals
    median_bpm = float(np.median(bpm_values))
    return {
        "bpm": median_bpm,
        "bpm_p10": float(np.percentile(bpm_values, 10)),
        "bpm_p90": float(np.percentile(bpm_values, 90)),
        "median_beat_interval_sec": float(np.median(intervals)),
        "half_double_candidates_bpm": [
            median_bpm / 2.0,
            median_bpm,
            median_bpm * 2.0,
        ],
    }


def snap_near_integer_bpm(
    bpm: float,
    tolerance: float = INTEGER_BPM_SNAP_TOLERANCE,
) -> tuple[float, bool]:
    nearest_integer = round(bpm)
    should_snap = bool(abs(bpm - nearest_integer) <= tolerance)
    return (float(nearest_integer) if should_snap else float(bpm), should_snap)


def half_tempo_evidence(
    beats_sec: np.ndarray,
    downbeats_sec: np.ndarray,
) -> dict[str, float | int | bool]:
    """Detect a stable 2:1 tactus ambiguity using two independent event heads."""

    def support(
        events: np.ndarray,
        *,
        minimum_interval: float,
        maximum_interval: float,
    ) -> tuple[float, int, int, float]:
        intervals = np.diff(events)
        intervals = intervals[
            np.isfinite(intervals)
            & (intervals >= minimum_interval)
            & (intervals <= maximum_interval)
        ]
        if len(intervals) == 0:
            return 0.0, 0, 0, 0.0
        long_mode = float(np.median(intervals))
        short = (
            (intervals >= 0.42 * long_mode)
            & (intervals <= 0.58 * long_mode)
        )
        return (
            float(np.mean(short)),
            int(np.sum(short)),
            len(intervals),
            long_mode,
        )

    beat_support, beat_short, beat_total, beat_mode = support(
        beats_sec,
        minimum_interval=0.20,
        maximum_interval=1.20,
    )
    downbeat_support, downbeat_short, downbeat_total, downbeat_mode = support(
        downbeats_sec,
        minimum_interval=0.50,
        maximum_interval=5.00,
    )
    should_double = bool(
        beat_support >= 0.18
        and downbeat_support >= 0.20
        and beat_short >= 24
        and downbeat_short >= 8
    )
    return {
        "beat_half_interval_support": beat_support,
        "downbeat_half_interval_support": downbeat_support,
        "beat_half_interval_count": beat_short,
        "downbeat_half_interval_count": downbeat_short,
        "beat_interval_count": beat_total,
        "downbeat_interval_count": downbeat_total,
        "beat_long_mode_sec": beat_mode,
        "downbeat_long_mode_sec": downbeat_mode,
        "decision": should_double,
        "method": "dual_beat_downbeat_harmonic_vote_v1",
    }


def double_tempo_pulse_evidence(
    audio: AudioBundle,
    base_bpm: float,
) -> dict[str, float | int | bool | str]:
    """Detect a stable missing-every-other-beat pattern from audio onsets."""
    doubled_bpm = base_bpm * 2.0
    eligible_range = bool(
        70.0 <= base_bpm <= 110.0
        and 150.0 <= doubled_bpm <= 220.0
    )
    if not eligible_range:
        return {
            "base_bpm": float(base_bpm),
            "doubled_bpm": float(doubled_bpm),
            "base_pulse_strength": 0.0,
            "double_pulse_strength": 0.0,
            "double_to_base_ratio": 0.0,
            "high_tempo_weight": 1.05,
            "weighted_double_to_base_ratio": 0.0,
            "high_tempo_window_support": 0.0,
            "num_analysis_windows": 0,
            "eligible_range": False,
            "decision": False,
            "method": "onset_autocorrelation_harmonic_v1",
        }

    import librosa

    analysis_rate = 11025
    hop_length = 256
    mono = np.asarray(audio.signal, dtype=np.float32).mean(axis=1)
    if audio.sample_rate != analysis_rate:
        mono = librosa.resample(
            mono,
            orig_sr=audio.sample_rate,
            target_sr=analysis_rate,
        )
    onset_envelope = librosa.onset.onset_strength(
        y=mono,
        sr=analysis_rate,
        hop_length=hop_length,
    )
    base_lag = int(round(60.0 * analysis_rate / (hop_length * base_bpm)))
    double_lag = int(round(base_lag / 2.0))

    def local_strength(autocorrelation: np.ndarray, lag: int) -> float:
        zero_lag = max(float(autocorrelation[0]), 1e-12)
        left = max(1, lag - 1)
        right = min(len(autocorrelation), lag + 2)
        return float(np.max(autocorrelation[left:right]) / zero_lag)

    maximum_lag = math.ceil(analysis_rate / hop_length * 1.25)
    autocorrelation = librosa.autocorrelate(
        onset_envelope,
        max_size=maximum_lag,
    )
    base_strength = local_strength(autocorrelation, base_lag)
    double_strength = local_strength(autocorrelation, double_lag)
    ratio = double_strength / max(base_strength, 1e-12)
    high_tempo_weight = 1.05
    weighted_ratio = ratio * high_tempo_weight

    window_frames = max(1, round(20.0 * analysis_rate / hop_length))
    hop_frames = max(1, window_frames // 2)
    window_starts = list(
        range(0, max(1, len(onset_envelope) - window_frames + 1), hop_frames)
    )
    final_start = max(0, len(onset_envelope) - window_frames)
    if not window_starts or window_starts[-1] != final_start:
        window_starts.append(final_start)
    supporting_weight = 0.0
    total_weight = 0.0
    for start in window_starts:
        window = onset_envelope[start : start + window_frames]
        if len(window) < 2:
            continue
        window_autocorrelation = librosa.autocorrelate(
            window,
            max_size=maximum_lag,
        )
        window_base = local_strength(window_autocorrelation, base_lag)
        window_double = local_strength(window_autocorrelation, double_lag)
        activity_weight = max(float(np.sum(window)), 1e-12)
        total_weight += activity_weight
        if (
            window_double >= 0.30
            and window_double * high_tempo_weight >= window_base
        ):
            supporting_weight += activity_weight
    window_support = supporting_weight / max(total_weight, 1e-12)
    decision = bool(
        base_strength >= 0.35
        and double_strength >= 0.35
        and ratio >= 0.90
        and weighted_ratio >= 1.0
        and window_support >= 0.55
    )
    return {
        "base_bpm": float(base_bpm),
        "doubled_bpm": float(doubled_bpm),
        "base_pulse_strength": base_strength,
        "double_pulse_strength": double_strength,
        "double_to_base_ratio": ratio,
        "high_tempo_weight": high_tempo_weight,
        "weighted_double_to_base_ratio": weighted_ratio,
        "high_tempo_window_support": window_support,
        "num_analysis_windows": len(window_starts),
        "eligible_range": True,
        "decision": decision,
        "method": "windowed_onset_autocorrelation_high_tempo_weight_v2",
    }


def seconds_to_tick_from_beat_anchors(
    seconds: float,
    beat_entries: list[dict],
) -> int:
    """Map seconds to ticks using the same local beat anchors as the grid."""
    if len(beat_entries) < 2:
        raise ValueError("At least two beat anchors are required.")

    anchor_seconds = np.asarray(
        [beat["sec"] for beat in beat_entries],
        dtype=np.float64,
    )
    anchor_ticks = np.asarray(
        [beat["tick"] for beat in beat_entries],
        dtype=np.float64,
    )

    right = int(np.searchsorted(anchor_seconds, seconds, side="right"))
    if right <= 1:
        left, right = 0, 1
    elif right >= len(anchor_seconds):
        left, right = len(anchor_seconds) - 2, len(anchor_seconds) - 1
    else:
        left = right - 1

    interval_sec = anchor_seconds[right] - anchor_seconds[left]
    if interval_sec <= 0:
        raise ValueError("Beat anchors must be strictly increasing.")
    fraction = (seconds - anchor_seconds[left]) / interval_sec
    tick = anchor_ticks[left] + fraction * (
        anchor_ticks[right] - anchor_ticks[left]
    )
    return int(round(float(tick)))


def nearest_beat_indices(
    beats_sec: np.ndarray,
    downbeats_sec: np.ndarray,
    tolerance_sec: float = 0.08,
) -> np.ndarray:
    indices: list[int] = []
    for downbeat in downbeats_sec:
        index = int(np.argmin(np.abs(beats_sec - downbeat)))
        if abs(float(beats_sec[index] - downbeat)) <= tolerance_sec:
            if not indices or index != indices[-1]:
                indices.append(index)
    return np.asarray(indices, dtype=np.int64)


def infer_beats_per_bar(downbeat_beat_indices: np.ndarray) -> int:
    gaps = np.diff(downbeat_beat_indices)
    gaps = gaps[(gaps >= 2) & (gaps <= 12)]
    if len(gaps) == 0:
        return 4
    values, counts = np.unique(gaps, return_counts=True)
    return int(values[np.argmax(counts)])


def regularize_downbeats(
    downbeats_sec: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    if len(downbeats_sec) < 2:
        raise ValueError("At least two downbeat candidates are required.")

    raw_intervals = np.diff(downbeats_sec)
    bar_duration_sec = float(np.median(raw_intervals[raw_intervals > 0]))
    phase_tolerance = 0.22 * bar_duration_sec

    # A tracker can emit one false downbeat before settling into a stable
    # sequence. Find the earliest candidate followed by three bar-consistent
    # intervals instead of anchoring unconditionally to candidate zero.
    interval_multiples = np.maximum(
        1,
        np.rint(raw_intervals / bar_duration_sec),
    )
    interval_residuals = np.abs(
        raw_intervals - interval_multiples * bar_duration_sec
    )
    stable_intervals = interval_residuals <= phase_tolerance
    required_run = min(3, len(stable_intervals))
    anchor_index = 0
    for index in range(len(stable_intervals) - required_run + 1):
        if bool(np.all(stable_intervals[index : index + required_run])):
            anchor_index = index
            break
    selected = [float(downbeats_sec[anchor_index])]
    rejected: list[float] = [
        float(candidate)
        for candidate in downbeats_sec[:anchor_index]
    ]

    for candidate in downbeats_sec[anchor_index + 1 :]:
        delta = float(candidate - selected[-1])
        multiple = max(1, int(round(delta / bar_duration_sec)))
        expected_delta = multiple * bar_duration_sec
        if abs(delta - expected_delta) > phase_tolerance:
            rejected.append(float(candidate))
            continue

        previous = selected[-1]
        for step in range(1, multiple):
            selected.append(previous + delta * step / multiple)
        selected.append(float(candidate))

    return (
        np.asarray(selected, dtype=np.float64),
        np.asarray(rejected, dtype=np.float64),
        bar_duration_sec,
    )


def subdivide_downbeats(
    downbeats_sec: np.ndarray,
    beats_per_bar: int,
    *,
    grid_origin_sec: float,
    song_duration_sec: float,
) -> np.ndarray:
    first_downbeat_sec = float(downbeats_sec[0])
    prefix_bar_count = int(
        round((first_downbeat_sec - grid_origin_sec) / np.median(np.diff(downbeats_sec)))
    )
    prefix = np.linspace(
        grid_origin_sec,
        first_downbeat_sec,
        prefix_bar_count + 1,
        dtype=np.float64,
    )
    bar_anchors = np.concatenate([prefix[:-1], downbeats_sec])

    beat_times: list[float] = []
    for left, right in zip(bar_anchors[:-1], bar_anchors[1:]):
        beat_times.extend(
            float(left + (right - left) * subdivision / beats_per_bar)
            for subdivision in range(beats_per_bar)
        )

    final_bar_duration = float(np.median(np.diff(downbeats_sec)))
    final_beat_duration = final_bar_duration / beats_per_bar
    last_downbeat = float(downbeats_sec[-1])
    beat_times.extend(
        last_downbeat + subdivision * final_beat_duration
        for subdivision in range(beats_per_bar)
    )
    while beat_times[-1] < song_duration_sec:
        beat_times.append(beat_times[-1] + final_beat_duration)

    return np.asarray(beat_times, dtype=np.float64)


def build_grid(
    audio: AudioBundle,
    beats_sec: np.ndarray,
    downbeats_sec: np.ndarray,
    *,
    model_name: str,
    ppq: int,
    forced_beats_per_bar: int | None,
) -> dict:
    if len(beats_sec) < 2:
        raise ValueError("Beat tracker returned fewer than two beats.")

    raw_tempo = robust_tempo(beats_sec)
    downbeat_indices = nearest_beat_indices(beats_sec, downbeats_sec)
    if len(downbeat_indices) == 0:
        raise ValueError("Beat tracker did not return a usable downbeat.")

    beats_per_bar = forced_beats_per_bar or infer_beats_per_bar(downbeat_indices)
    regular_downbeats, rejected_downbeats, median_bar_sec = regularize_downbeats(
        downbeats_sec
    )
    first_downbeat_sec = float(regular_downbeats[0])
    median_interval_sec = median_bar_sec / beats_per_bar

    # Keep all real audio samples non-negative in MIDI time. If the first
    # detected downbeat occurs after sample zero, create enough virtual bars
    # before it instead of physically padding the audio.
    prepend_bars = (
        max(1, int(math.ceil(first_downbeat_sec / median_bar_sec)))
        if first_downbeat_sec > 0.05
        else 0
    )
    grid_origin_sec = first_downbeat_sec - prepend_bars * median_bar_sec
    grid_origin_sample = int(round(grid_origin_sec * audio.sample_rate))
    song_duration_sec = audio.num_samples / audio.sample_rate
    canonical_beats = subdivide_downbeats(
        regular_downbeats,
        beats_per_bar,
        grid_origin_sec=grid_origin_sec,
        song_duration_sec=song_duration_sec,
    )

    beat_entries: list[dict] = []
    for beat_index, beat_sec in enumerate(canonical_beats):
        if beat_sec < 0 or beat_sec > song_duration_sec:
            continue
        tick = beat_index * ppq
        is_grid_downbeat = beat_index % beats_per_bar == 0
        nearest_model_beat = float(beats_sec[np.argmin(np.abs(beats_sec - beat_sec))])
        nearest_model_downbeat = float(
            downbeats_sec[np.argmin(np.abs(downbeats_sec - beat_sec))]
        )
        beat_entries.append(
            {
                "beat_index": beat_index,
                "sample": int(round(float(beat_sec) * audio.sample_rate)),
                "sec": round(float(beat_sec), 6),
                "tick": int(tick),
                "bar_index": int(beat_index // beats_per_bar),
                "beat_in_bar": int(beat_index % beats_per_bar),
                "is_downbeat": is_grid_downbeat,
                "model_beat_candidate_sec": (
                    round(nearest_model_beat, 6)
                    if abs(nearest_model_beat - beat_sec) <= 0.12
                    else None
                ),
                "model_downbeat_candidate_sec": (
                    round(nearest_model_downbeat, 6)
                    if abs(nearest_model_downbeat - beat_sec) <= 0.12
                    else None
                ),
            }
        )

    # The canonical grid can follow small tempo changes, so endpoint mapping
    # must use its local anchors too. A single median interval can otherwise
    # disagree with the beat list by several bars on a long song.
    audio_end_tick = seconds_to_tick_from_beat_anchors(
        song_duration_sec,
        beat_entries,
    )
    ticks_per_bar = beats_per_bar * ppq
    grid_end_tick = int(math.ceil(audio_end_tick / ticks_per_bar) * ticks_per_bar)
    trailing_padding_ticks = grid_end_tick - audio_end_tick
    trailing_padding_sec = trailing_padding_ticks / ppq * median_interval_sec
    audio_start_tick = seconds_to_tick_from_beat_anchors(0.0, beat_entries)
    regular_tempo = robust_tempo(
        np.asarray([beat["sec"] for beat in beat_entries], dtype=np.float64)
    )
    regression_ticks = np.asarray(
        [beat["tick"] for beat in beat_entries],
        dtype=np.float64,
    )
    regression_seconds = np.asarray(
        [beat["sec"] for beat in beat_entries],
        dtype=np.float64,
    )
    seconds_per_tick, _ = np.polyfit(
        regression_ticks,
        regression_seconds,
        1,
    )
    global_regression_bpm = 60.0 / (seconds_per_tick * ppq)
    project_bpm, integer_snap_applied = snap_near_integer_bpm(
        global_regression_bpm
    )

    grid = {
        "schema_version": "song_grid_v1.0",
        "source": {
            "files": [str(path) for path in audio.source_paths],
            "mix_strategy": "equal_gain_sum",
            "sample_rate": audio.sample_rate,
            "num_samples": audio.num_samples,
            "duration_sec": round(audio.num_samples / audio.sample_rate, 6),
        },
        "model": {
            "name": "Beat This!",
            "checkpoint": model_name,
            "dbn_postprocessing": False,
            "raw_num_beat_candidates": len(beats_sec),
            "raw_num_downbeat_candidates": len(downbeats_sec),
        },
        "tempo": {
            **regular_tempo,
            "bpm": project_bpm,
            "median_interval_bpm": regular_tempo["bpm"],
            "raw_median_bpm": raw_tempo["bpm"],
            "global_regression_bpm": float(global_regression_bpm),
            "suggested_project_bpm": project_bpm,
            "integer_snap_applied": integer_snap_applied,
            "integer_snap_tolerance_bpm": INTEGER_BPM_SNAP_TOLERANCE,
            "mode": "beat_anchors",
        },
        "meter": {
            "beats_per_bar": beats_per_bar,
            "time_signature": [beats_per_bar, 4],
            "source": "manual" if forced_beats_per_bar else "inferred_from_downbeats",
            "median_bar_duration_sec": median_bar_sec,
        },
        "midi_alignment": {
            "ppq": ppq,
            "prepend_bars": prepend_bars,
            "grid_origin_sample": grid_origin_sample,
            "grid_origin_sec": round(grid_origin_sec, 6),
            "audio_start_tick": audio_start_tick,
            "first_detected_downbeat_sample": int(
                round(first_downbeat_sec * audio.sample_rate)
            ),
            "first_detected_downbeat_sec": round(first_downbeat_sec, 6),
            "first_detected_downbeat_tick": prepend_bars * beats_per_bar * ppq,
            "audio_end_tick": audio_end_tick,
            "grid_end_tick": grid_end_tick,
            "song_end_tick": grid_end_tick,
            "trailing_padding_ticks": trailing_padding_ticks,
            "trailing_padding_sec": round(trailing_padding_sec, 6),
            "grid_end_virtual_sec": round(
                song_duration_sec + trailing_padding_sec,
                6,
            ),
        },
        "summary": {
            "num_beats": len(beat_entries),
            "num_downbeats": int(sum(entry["is_downbeat"] for entry in beat_entries)),
            "num_model_downbeat_candidates": len(downbeats_sec),
            "num_rejected_downbeat_candidates": len(rejected_downbeats),
            "total_complete_bars": int(grid_end_tick // ticks_per_bar),
        },
        "model_events": {
            "beat_candidates_sec": [round(float(value), 6) for value in beats_sec],
            "downbeat_candidates_sec": [
                round(float(value), 6) for value in downbeats_sec
            ],
            "rejected_downbeat_candidates_sec": [
                round(float(value), 6) for value in rejected_downbeats
            ],
        },
        "beats": beat_entries,
        "warnings": (
            [
                {
                    "type": "downbeat_grid_disagreement",
                    "message": (
                        "Some model downbeat candidates conflict with the dominant "
                        f"{beats_per_bar}/4 bar spacing. They are preserved in "
                        "model_events but not used as grid downbeats."
                    ),
                    "num_rejected_candidates": len(rejected_downbeats),
                }
            ]
            if len(rejected_downbeats)
            else []
        ),
    }
    if ppq == PROJECT_PPQ and beats_per_bar == 4:
        from transmelody.grid.tempo_region_grid import apply_detected_tempo_regions
        from transmelody.grid.tempo_region_grid import multiply_grid_tempo

        grid = apply_detected_tempo_regions(grid)
        evidence = half_tempo_evidence(beats_sec, downbeats_sec)
        pulse_evidence = double_tempo_pulse_evidence(
            audio,
            global_regression_bpm,
        )
        evidence["pulse_evidence"] = pulse_evidence
        evidence["decision"] = bool(
            evidence["decision"]
            or pulse_evidence["decision"]
        )
        evidence["method"] = "interval_and_onset_harmonic_vote_v2"
        grid["tempo"]["half_tempo_evidence"] = evidence
        if evidence["decision"] and not grid["tempo"].get("regions"):
            doubled_bpm, doubled_snap = snap_near_integer_bpm(
                global_regression_bpm * 2.0
            )
            grid = multiply_grid_tempo(
                grid,
                2,
                target_bpm=doubled_bpm,
            )
            grid["tempo"]["integer_snap_applied"] = doubled_snap
            grid["tempo"]["half_tempo_evidence"] = evidence
    return grid


def write_beats_tsv(path: Path, grid: dict) -> None:
    lines = [
        "sec\tsample\ttick\tbar_index\tbeat_in_bar\tis_downbeat"
        "\tmodel_beat_candidate_sec\tmodel_downbeat_candidate_sec"
    ]
    lines.extend(
        "\t".join(
            [
                f"{beat['sec']:.6f}",
                str(beat["sample"]),
                str(beat["tick"]),
                str(beat["bar_index"]),
                str(beat["beat_in_bar"]),
                "1" if beat["is_downbeat"] else "0",
                (
                    f"{beat['model_beat_candidate_sec']:.6f}"
                    if beat["model_beat_candidate_sec"] is not None
                    else ""
                ),
                (
                    f"{beat['model_downbeat_candidate_sec']:.6f}"
                    if beat["model_downbeat_candidate_sec"] is not None
                    else ""
                ),
            ]
        )
        for beat in grid["beats"]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_click_preview(
    path: Path,
    audio: AudioBundle,
    grid: dict,
    click_gain: float = 0.22,
) -> None:
    preview = audio.signal.copy()
    peak = float(np.max(np.abs(preview)))
    if peak > 0:
        preview *= 0.72 / peak

    sample_rate = audio.sample_rate
    click_length = int(round(0.045 * sample_rate))
    time = np.arange(click_length, dtype=np.float32) / sample_rate
    decay = np.exp(-time * 55.0).astype(np.float32)

    for beat in grid["beats"]:
        start = beat["sample"]
        if start < 0 or start >= audio.num_samples:
            continue
        frequency = 1760.0 if beat["is_downbeat"] else 1100.0
        click = np.sin(2.0 * np.pi * frequency * time).astype(np.float32) * decay
        end = min(start + click_length, audio.num_samples)
        preview[start:end] += click_gain * click[: end - start, None]

    preview = np.clip(preview, -1.0, 1.0)
    sf.write(path, preview, sample_rate, subtype="PCM_16")


def make_alignment_plot(path: Path, audio: AudioBundle, grid: dict) -> None:
    mono = audio.signal.mean(axis=1)
    duration_sec = audio.num_samples / audio.sample_rate
    block_size = max(1, int(audio.sample_rate * 0.05))
    usable = len(mono) // block_size * block_size
    envelope = np.max(
        np.abs(mono[:usable].reshape(-1, block_size)),
        axis=1,
    )
    envelope_time = (
        np.arange(len(envelope), dtype=np.float64) * block_size / audio.sample_rate
    )

    beats = grid["beats"]
    beat_times = np.asarray([beat["sec"] for beat in beats], dtype=np.float64)
    downbeat_times = np.asarray(
        [beat["sec"] for beat in beats if beat["is_downbeat"]],
        dtype=np.float64,
    )
    local_bpm = 60.0 / np.diff(beat_times)

    figure, axes = plt.subplots(
        2,
        1,
        figsize=(16, 7),
        sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.0]},
    )
    axes[0].plot(envelope_time, envelope, color="#3d5066", linewidth=0.7)
    axes[0].vlines(
        beat_times,
        0,
        max(float(envelope.max()), 1e-6),
        color="#77b5d9",
        alpha=0.18,
        linewidth=0.5,
        label="Beat",
    )
    axes[0].vlines(
        downbeat_times,
        0,
        max(float(envelope.max()), 1e-6),
        color="#e45756",
        alpha=0.75,
        linewidth=0.8,
        label="Downbeat",
    )
    axes[0].set_ylabel("Peak envelope")
    axes[0].set_title("Global beat/downbeat alignment")
    axes[0].legend(loc="upper right")

    axes[1].plot(beat_times[1:], local_bpm, color="#4c78a8", linewidth=0.8)
    axes[1].axhline(
        grid["tempo"]["bpm"],
        color="#f58518",
        linewidth=1.2,
        label=f"Median {grid['tempo']['bpm']:.2f} BPM",
    )
    axes[1].set_ylim(
        max(0.0, float(np.percentile(local_bpm, 1)) - 5.0),
        float(np.percentile(local_bpm, 99)) + 5.0,
    )
    axes[1].set_ylabel("Local BPM")
    axes[1].set_xlabel("Audio time (seconds)")
    axes[1].set_xlim(0, duration_sec)
    axes[1].grid(alpha=0.2)
    axes[1].legend(loc="upper right")

    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def analyze(
    input_paths: list[Path],
    output_dir: Path,
    *,
    model_name: str,
    device: str,
    float16: bool,
    ppq: int,
    beats_per_bar: int | None,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    audio = load_and_mix_audio(input_paths)

    tracker = Audio2Beats(
        checkpoint_path=model_name,
        device=device,
        float16=float16,
        dbn=False,
    )
    beats_sec, downbeats_sec = tracker(audio.signal, audio.sample_rate)
    beats_sec = np.asarray(beats_sec, dtype=np.float64)
    downbeats_sec = np.asarray(downbeats_sec, dtype=np.float64)

    grid = build_grid(
        audio,
        beats_sec,
        downbeats_sec,
        model_name=model_name,
        ppq=ppq,
        forced_beats_per_bar=beats_per_bar,
    )

    (output_dir / "grid.json").write_text(
        json.dumps(grid, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_beats_tsv(output_dir / "beats.tsv", grid)
    make_alignment_plot(output_dir / "alignment_plot.png", audio, grid)
    make_click_preview(output_dir / "click_preview.wav", audio, grid)
    return grid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a global song grid from one file or aligned stems."
    )
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("output/grid"))
    parser.add_argument("--model", default="final0")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--float16", action="store_true")
    parser.add_argument("--ppq", type=int, default=PROJECT_PPQ)
    parser.add_argument("--beats-per-bar", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    grid = analyze(
        args.inputs,
        args.output_dir,
        model_name=args.model,
        device=args.device,
        float16=args.float16,
        ppq=args.ppq,
        beats_per_bar=args.beats_per_bar,
    )
    print(
        f"Wrote {args.output_dir}: "
        f"{grid['tempo']['bpm']:.2f} BPM, "
        f"{grid['summary']['num_beats']} beats, "
        f"{grid['summary']['num_downbeats']} downbeats."
    )


if __name__ == "__main__":
    main()
