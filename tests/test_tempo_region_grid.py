import numpy as np

from tempo_region_grid import (
    choose_transition_bar_count,
    detect_two_tempo_regions,
    multiply_grid_tempo,
)
from tempo_map import build_tempo_segments


def synthetic_grid(first_bpm: float, second_bpm: float) -> dict:
    change = 30.48
    duration = 90.0
    first_beats = np.arange(0.72, change, 60.0 / first_bpm)
    second_beats = np.arange(change, duration, 60.0 / second_bpm)
    first_downbeats = np.arange(0.72, change - 1.0, 240.0 / first_bpm)
    second_downbeats = np.arange(change, duration, 240.0 / second_bpm)
    return {
        "source": {"duration_sec": duration},
        "model_events": {
            "beat_candidates_sec": np.concatenate(
                [first_beats, second_beats]
            ).tolist(),
            "downbeat_candidates_sec": np.concatenate(
                [first_downbeats, second_downbeats]
            ).tolist(),
        },
    }


def test_detects_two_stable_tempo_regions() -> None:
    detected = detect_two_tempo_regions(synthetic_grid(140.0, 195.0))
    assert detected is not None
    assert detected["first_bpm"] == 140.0
    assert detected["second_bpm"] == 195.0
    assert abs(detected["change_sec"] - 30.48) < 0.05
    assert detected["transition_bars"] == 1


def test_half_double_ambiguity_is_not_a_tempo_change() -> None:
    assert detect_two_tempo_regions(synthetic_grid(90.0, 180.0)) is None


def test_transition_bar_count_prefers_an_intermediate_bpm() -> None:
    assert choose_transition_bar_count(9.18, 140.0, 195.0, 4) == 6


def test_half_tempo_grid_is_retimed_and_exports_one_integer_section() -> None:
    beats = [
        {
            "tick": index * 480,
            "sec": index * (60.0 / 90.0),
            "sample": round(index * (60.0 / 90.0) * 44100),
        }
        for index in range(9)
    ]
    grid = {
        "schema_version": "song_grid_v1.0",
        "source": {"duration_sec": 16.0 / 3.0, "sample_rate": 44100},
        "tempo": {
            "bpm": 90.0,
            "global_regression_bpm": 90.0,
            "median_interval_bpm": 90.0,
            "raw_median_bpm": 90.0,
            "integer_snap_applied": True,
            "integer_snap_tolerance_bpm": 0.1,
        },
        "meter": {"beats_per_bar": 4, "median_bar_duration_sec": 8.0 / 3.0},
        "midi_alignment": {
            "prepend_bars": 0,
            "grid_origin_sec": 0.0,
            "audio_start_tick": 0,
            "first_detected_downbeat_sec": 0.0,
            "first_detected_downbeat_tick": 0,
            "audio_end_tick": 3840,
            "grid_end_tick": 3840,
            "song_end_tick": 3840,
            "trailing_padding_ticks": 0,
        },
        "summary": {"total_complete_bars": 2},
        "model_events": {
            "beat_candidates_sec": [beat["sec"] for beat in beats],
            "downbeat_candidates_sec": [0.0, 8.0 / 3.0],
        },
        "beats": beats,
        "warnings": [],
    }

    doubled = multiply_grid_tempo(grid, 2, target_bpm=180.0)
    sections = build_tempo_segments(doubled)

    assert doubled["midi_alignment"]["grid_end_tick"] == 7680
    assert doubled["summary"]["total_complete_bars"] == 4
    assert doubled["tempo"]["bpm"] == 180.0
    assert len(sections) == 1
    assert sections[0].bpm == 180.0
