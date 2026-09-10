from pathlib import Path

import numpy as np

from transmelody.grid.grid_initializer import (
    AudioBundle,
    build_grid,
    double_tempo_pulse_evidence,
    half_tempo_evidence,
    infer_beats_per_bar,
    nearest_beat_indices,
    seconds_to_tick_from_beat_anchors,
    snap_near_integer_bpm,
)
from transmelody.config import PROJECT_PPQ


def make_audio(duration_sec: float = 12.0, sample_rate: int = 1000) -> AudioBundle:
    num_samples = int(duration_sec * sample_rate)
    return AudioBundle(
        signal=np.zeros((num_samples, 2), dtype=np.float32),
        sample_rate=sample_rate,
        num_samples=num_samples,
        source_paths=(Path("test.wav"),),
    )


def test_downbeats_map_to_nearest_beats() -> None:
    beats = np.arange(0.5, 8.5, 0.5)
    downbeats = np.asarray([0.51, 2.49, 4.52, 6.5])
    indices = nearest_beat_indices(beats, downbeats)
    assert indices.tolist() == [0, 4, 8, 12]
    assert infer_beats_per_bar(indices) == 4


def test_near_integer_bpm_is_snapped() -> None:
    assert snap_near_integer_bpm(176.995) == (177.0, True)
    assert snap_near_integer_bpm(176.7) == (176.7, False)


def test_dual_harmonic_interval_vote_detects_half_tempo() -> None:
    beats = np.concatenate(
        [[0.0], np.cumsum([0.66] * 80 + [0.33] * 25)]
    )
    downbeats = np.concatenate(
        [[0.0], np.cumsum([2.66] * 40 + [1.33] * 25)]
    )

    evidence = half_tempo_evidence(beats, downbeats)

    assert evidence["decision"] is True
    assert evidence["beat_half_interval_support"] >= 0.18
    assert evidence["downbeat_half_interval_support"] >= 0.20


def test_one_head_alone_does_not_trigger_half_tempo() -> None:
    beats = np.concatenate(
        [[0.0], np.cumsum([0.66] * 80 + [0.33] * 25)]
    )
    downbeats = np.arange(0.0, 70.0, 2.66)

    evidence = half_tempo_evidence(beats, downbeats)

    assert evidence["decision"] is False


def test_audio_pulses_detect_stable_missing_every_other_beat() -> None:
    sample_rate = 11025
    duration_sec = 24.0
    signal = np.zeros((round(duration_sec * sample_rate), 2), dtype=np.float32)
    fast_bpm = 60.0 * sample_rate / (256 * 14)
    fast_interval = 60.0 / fast_bpm
    click_length = round(0.03 * sample_rate)
    click_time = np.arange(click_length) / sample_rate
    click = (
        np.sin(2.0 * np.pi * 1000.0 * click_time)
        * np.hanning(click_length)
    ).astype(np.float32)
    for pulse_sec in np.arange(0.0, duration_sec, fast_interval):
        start = round(pulse_sec * sample_rate)
        end = min(len(signal), start + click_length)
        signal[start:end] = click[: end - start, None]
    audio = AudioBundle(
        signal=signal,
        sample_rate=sample_rate,
        num_samples=len(signal),
        source_paths=(Path("pulse.wav"),),
    )

    evidence = double_tempo_pulse_evidence(audio, fast_bpm / 2.0)

    assert evidence["eligible_range"] is True
    assert evidence["double_to_base_ratio"] >= 0.94
    assert evidence["high_tempo_window_support"] >= 0.55
    assert evidence["decision"] is True


def test_audio_pulse_double_check_is_disabled_for_fast_base_tempo() -> None:
    evidence = double_tempo_pulse_evidence(make_audio(), 180.0)

    assert evidence["eligible_range"] is False
    assert evidence["decision"] is False


def test_endpoint_tick_uses_local_beat_anchors() -> None:
    anchors = [
        {"sec": 0.0, "tick": 0},
        {"sec": 0.5, "tick": 480},
        {"sec": 0.9, "tick": 960},
        {"sec": 1.3, "tick": 1440},
    ]

    assert seconds_to_tick_from_beat_anchors(1.5, anchors) == 1680
    assert seconds_to_tick_from_beat_anchors(-0.25, anchors) == -240


def test_virtual_bar_keeps_audio_start_at_positive_tick() -> None:
    audio = make_audio()
    beats = np.arange(0.5, 11.6, 0.5)
    downbeats = np.arange(0.5, 11.0, 2.0)

    grid = build_grid(
        audio,
        beats,
        downbeats,
        model_name="test",
        ppq=PROJECT_PPQ,
        forced_beats_per_bar=4,
    )

    alignment = grid["midi_alignment"]
    assert alignment["prepend_bars"] == 1
    assert alignment["grid_origin_sample"] == -1500
    assert alignment["audio_start_tick"] == 1440
    assert alignment["grid_end_tick"] % (4 * PROJECT_PPQ) == 0
    assert alignment["grid_end_tick"] >= alignment["audio_end_tick"]
    assert (
        alignment["trailing_padding_ticks"]
        == alignment["grid_end_tick"] - alignment["audio_end_tick"]
    )
    assert grid["beats"][0]["tick"] == 1440
    assert grid["beats"][0]["bar_index"] == 0
    assert grid["beats"][0]["beat_in_bar"] == 3


def test_spurious_downbeats_do_not_break_global_bar_phase() -> None:
    audio = make_audio()
    beats = np.arange(0.5, 11.6, 0.5)
    downbeats = np.asarray([0.5, 2.5, 4.5, 5.5, 6.5, 8.5, 10.5])

    grid = build_grid(
        audio,
        beats,
        downbeats,
        model_name="test",
        ppq=PROJECT_PPQ,
        forced_beats_per_bar=4,
    )

    grid_downbeats = [
        index for index, beat in enumerate(grid["beats"]) if beat["is_downbeat"]
    ]
    assert grid_downbeats == [1, 5, 9, 13, 17, 21]
    assert grid["summary"]["num_rejected_downbeat_candidates"] == 1
    assert grid["warnings"][0]["type"] == "downbeat_grid_disagreement"
