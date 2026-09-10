"""One tick/sample mapping shared by feature extraction, labels and MIDI export."""
from __future__ import annotations

import copy
import math

import mido
import numpy as np

from project_settings import PROJECT_PPQ


TIMELINE_VERSION = "midi_tempo_timeline_v1"


def normalize_tempo_events(events: list[tuple[int, int]]) -> list[tuple[int, int]]:
    by_tick = {int(tick): int(tempo) for tick, tempo in events}
    if not by_tick or min(by_tick) != 0 or any(t < 0 or v <= 0 for t, v in by_tick.items()):
        raise ValueError("Tempo map must begin at tick 0 and contain positive tempos.")
    result = []
    for tick, tempo in sorted(by_tick.items()):
        if not result or result[-1][1] != tempo:
            result.append((tick, tempo))
    return result


def tempo_axes(events: list[tuple[int, int]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    events = normalize_tempo_events(events)
    ticks = np.array([e[0] for e in events], dtype=np.float64)
    rates = np.array([e[1] for e in events], dtype=np.float64) / (1e6 * PROJECT_PPQ)
    seconds = np.concatenate(([0.0], np.cumsum(np.diff(ticks) * rates[:-1])))
    return ticks, seconds, rates


def tick_seconds(values, events: list[tuple[int, int]]) -> np.ndarray:
    ticks, seconds, rates = tempo_axes(events)
    values = np.asarray(values, dtype=np.float64)
    indices = np.searchsorted(ticks, values, side="right") - 1
    indices = np.maximum(indices, 0)
    return seconds[indices] + (values - ticks[indices]) * rates[indices]


def seconds_ticks(values, events: list[tuple[int, int]]) -> np.ndarray:
    ticks, seconds, rates = tempo_axes(events)
    values = np.asarray(values, dtype=np.float64)
    indices = np.maximum(np.searchsorted(seconds, values, side="right") - 1, 0)
    return ticks[indices] + (values - seconds[indices]) / rates[indices]


def grid_tempo_events(grid: dict) -> list[tuple[int, int]]:
    return [(int(e["tick"]), int(e["tempo"])) for e in grid["timeline"]["tempo_events"]]


def samples_at_ticks(grid: dict, ticks, sample_rate: int) -> np.ndarray:
    if grid.get("timeline", {}).get("version") != TIMELINE_VERSION:
        raise ValueError("Grid has not been synchronized to the exported MIDI timeline.")
    events = grid_tempo_events(grid)
    start = int(grid["midi_alignment"]["audio_start_tick"])
    return (tick_seconds(ticks, events) - tick_seconds(start, events)) * sample_rate


def canonicalize_grid(
    grid: dict,
    events: list[tuple[int, int]],
    *,
    audio_start_tick: int | None = None,
    minimum_end_tick: int = 0,
    source: str = "prediction_tempo_map",
) -> dict:
    """Preserve score coordinates; regenerate time coordinates from exact MIDI tempos."""
    events = normalize_tempo_events(events)
    result = copy.deepcopy(grid)
    alignment = result["midi_alignment"]
    start = int(alignment["audio_start_tick"] if audio_start_tick is None else audio_start_tick)
    if start < 0:
        raise ValueError("WAV START cannot precede MIDI tick 0.")
    rate = int(result["source"]["sample_rate"])
    duration = int(result["source"]["num_samples"]) / rate
    bar_ticks = int(result["meter"]["beats_per_bar"]) * PROJECT_PPQ
    origin = -float(tick_seconds(start, events))
    audio_end = float(seconds_ticks(duration - origin, events))
    end = math.ceil(max(audio_end, minimum_end_tick, int(alignment["grid_end_tick"])) / bar_ticks) * bar_ticks
    old_beats = result.get("detected_beats", result.get("beats", []))
    result["detected_beats"] = copy.deepcopy(old_beats)
    if old_beats:
        old_ticks = [b["tick"] for b in old_beats]
        errors = np.array([b["sec"] for b in old_beats]) - (tick_seconds(old_ticks, events) + origin)
        residual = {"max_ms": float(abs(errors).max() * 1000), "median_ms": float(np.median(abs(errors)) * 1000)}
    else:
        residual = {}
    result["timeline"] = {
        "version": TIMELINE_VERSION,
        "source": source,
        "tempo_events": [{"tick": t, "tempo": v} for t, v in events],
        "detected_anchor_residual": residual,
    }
    end_sec = float(tick_seconds(end, events)) + origin
    alignment.update(
        ppq=PROJECT_PPQ, audio_start_tick=start, grid_origin_sec=origin,
        grid_origin_sample=round(origin * rate), audio_end_tick=round(audio_end),
        grid_end_tick=end, song_end_tick=end, grid_end_virtual_sec=end_sec,
        trailing_padding_ticks=end - round(audio_end), trailing_padding_sec=end_sec - duration,
    )
    result["beats"] = []
    for index, tick in enumerate(range(0, end + 1, PROJECT_PPQ)):
        sec = float(tick_seconds(tick, events)) + origin
        # Include virtual endpoints so interpolation cannot invent a different slope.
        result["beats"].append({
            "tick": tick, "sec": sec, "sample": round(sec * rate),
            "beat_index": index, "bar_index": tick // bar_ticks,
            "beat_in_bar": index % (bar_ticks // PROJECT_PPQ),
            "is_downbeat": tick % bar_ticks == 0,
        })
    result.setdefault("summary", {}).update(total_complete_bars=end // bar_ticks,
        num_beats=len(result["beats"]), num_downbeats=sum(b["is_downbeat"] for b in result["beats"]))
    # MIDI stores integer microseconds/quarter. Display that representational
    # rounding cleanly; all timing calculations still use the exact integers.
    result["tempo"]["bpm"] = round(float(mido.tempo2bpm(events[0][1])), 3)
    return result
