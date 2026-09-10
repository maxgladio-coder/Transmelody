from __future__ import annotations

import argparse
import copy
import json
import math
from dataclasses import dataclass
from pathlib import Path

import mido
import numpy as np

from transmelody.config import PROJECT_PPQ


@dataclass(frozen=True)
class RegionAnchor:
    start_tick: int
    end_tick: int
    start_sec: float
    end_sec: float
    kind: str

    @property
    def bpm(self) -> float:
        quarters = (self.end_tick - self.start_tick) / PROJECT_PPQ
        return 60.0 * quarters / (self.end_sec - self.start_sec)


def choose_transition_bar_count(
    duration_sec: float,
    first_bpm: float,
    second_bpm: float,
    beats_per_bar: int,
) -> int:
    """Choose a small integer bar count whose bridge BPM sits between regions."""
    if duration_sec <= 0:
        raise ValueError("Transition duration must be positive.")
    low_bpm, high_bpm = sorted((first_bpm, second_bpm))
    minimum = max(
        1,
        int(math.ceil(duration_sec * low_bpm / (60.0 * beats_per_bar))),
    )
    maximum = max(
        minimum,
        int(math.floor(duration_sec * high_bpm / (60.0 * beats_per_bar))),
    )
    target_bpm = (first_bpm + second_bpm) / 2.0
    candidates = range(minimum, maximum + 1)
    return min(
        candidates,
        key=lambda bars: abs(
            60.0 * beats_per_bar * bars / duration_sec - target_bpm
        ),
    )


def _window_tempo(
    beats_sec: np.ndarray,
    start_sec: float,
    end_sec: float,
) -> float | None:
    selected = beats_sec[(beats_sec >= start_sec) & (beats_sec <= end_sec)]
    if len(selected) < 6:
        return None
    intervals = np.diff(selected)
    intervals = intervals[(intervals >= 0.20) & (intervals <= 0.80)]
    if len(intervals) < 4:
        return None
    return float(60.0 / np.median(intervals))


def _stable_downbeat_chain_start(
    downbeats_sec: np.ndarray,
    *,
    target_bpm: float,
    earliest_sec: float,
    required_bars: int = 4,
) -> float | None:
    target_bar_sec = 240.0 / target_bpm
    tolerance = 0.16 * target_bar_sec
    for candidate in downbeats_sec[downbeats_sec >= earliest_sec]:
        current = float(candidate)
        stable = True
        for _ in range(required_bars):
            future = downbeats_sec[downbeats_sec > current + 0.45 * target_bar_sec]
            if len(future) == 0:
                stable = False
                break
            next_candidate = float(
                future[np.argmin(np.abs(future - (current + target_bar_sec)))]
            )
            if abs(next_candidate - (current + target_bar_sec)) > tolerance:
                stable = False
                break
            current = next_candidate
        if stable:
            return float(candidate)
    return None


def _refine_bpm_from_downbeat_chain(
    downbeats_sec: np.ndarray,
    *,
    start_sec: float,
    approximate_bpm: float,
    maximum_bars: int = 32,
) -> float:
    target_bar_sec = 240.0 / approximate_bpm
    tolerance = 0.18 * target_bar_sec
    selected = [start_sec]
    current = start_sec
    for _ in range(maximum_bars):
        future = downbeats_sec[downbeats_sec > current + 0.45 * target_bar_sec]
        if len(future) == 0:
            break
        next_candidate = float(
            future[np.argmin(np.abs(future - (current + target_bar_sec)))]
        )
        if abs(next_candidate - (current + target_bar_sec)) > tolerance:
            break
        selected.append(next_candidate)
        current = next_candidate
    if len(selected) < 5:
        return approximate_bpm
    indices = np.arange(len(selected), dtype=np.float64)
    seconds_per_bar, _ = np.polyfit(indices, np.asarray(selected), 1)
    return float(240.0 / seconds_per_bar)


def detect_two_tempo_regions(source_grid: dict) -> dict | None:
    """Detect a clear A→B tempo jump while ignoring short tracker jitter."""
    beats = np.asarray(
        source_grid["model_events"]["beat_candidates_sec"],
        dtype=np.float64,
    )
    downbeats = np.asarray(
        source_grid["model_events"]["downbeat_candidates_sec"],
        dtype=np.float64,
    )
    duration = float(source_grid["source"]["duration_sec"])
    if len(beats) < 24 or len(downbeats) < 10 or duration < 30.0:
        return None

    edge_window = min(20.0, duration * 0.2)
    first_tempo = _window_tempo(beats, 0.0, edge_window)
    second_tempo = _window_tempo(beats, max(0.0, duration - edge_window), duration)
    if first_tempo is None or second_tempo is None:
        return None
    first_bpm = float(round(first_tempo))
    second_bpm = float(round(second_tempo))
    relative_change = abs(second_bpm - first_bpm) / max(first_bpm, second_bpm)
    if relative_change < 0.12:
        return None
    tempo_ratio = max(first_bpm, second_bpm) / min(first_bpm, second_bpm)
    # A near-exact 2:1 observation is usually a beat-level half/double
    # ambiguity, not an engineering tempo-region change.
    if 1.85 <= tempo_ratio <= 2.15:
        return None

    window_size = 5.0
    centers = np.arange(window_size / 2.0, duration, window_size)
    local_tempos = [
        _window_tempo(
            beats,
            max(0.0, center - window_size / 2.0),
            min(duration, center + window_size / 2.0),
        )
        for center in centers
    ]
    closer_to_second = [
        tempo is not None
        and abs(tempo - second_bpm) < abs(tempo - first_bpm)
        for tempo in local_tempos
    ]
    approximate_change_sec: float | None = None
    for index in range(len(closer_to_second) - 1):
        if closer_to_second[index] and closer_to_second[index + 1]:
            approximate_change_sec = max(0.0, centers[index] - window_size / 2.0)
            break
    if approximate_change_sec is None:
        return None

    change_sec = _stable_downbeat_chain_start(
        downbeats,
        target_bpm=second_bpm,
        earliest_sec=max(edge_window, approximate_change_sec - 2.0),
    )
    first_downbeat_sec = _stable_downbeat_chain_start(
        downbeats,
        target_bpm=first_bpm,
        earliest_sec=0.0,
    )
    if change_sec is None or first_downbeat_sec is None:
        return None
    first_bpm = float(
        round(
            _refine_bpm_from_downbeat_chain(
                downbeats,
                start_sec=first_downbeat_sec,
                approximate_bpm=first_bpm,
            )
        )
    )
    second_bpm = float(
        round(
            _refine_bpm_from_downbeat_chain(
                downbeats,
                start_sec=change_sec,
                approximate_bpm=second_bpm,
            )
        )
    )
    if change_sec - first_downbeat_sec < 4.0 * 240.0 / first_bpm:
        return None
    return {
        "first_downbeat_sec": first_downbeat_sec,
        "first_bpm": first_bpm,
        "transition_start_sec": None,
        "change_sec": change_sec,
        "transition_bars": 1,
        "second_bpm": second_bpm,
    }


def apply_detected_tempo_regions(source_grid: dict) -> dict:
    detection = detect_two_tempo_regions(source_grid)
    if detection is None:
        return source_grid
    grid, _ = build_three_region_grid(source_grid, **detection)
    grid["tempo"]["detection"] = {
        "method": "stable_edge_regions_and_downbeat_chain",
        **detection,
    }
    return grid


def multiply_grid_tempo(
    source_grid: dict,
    factor: int,
    *,
    target_bpm: float | None = None,
) -> dict:
    """Resolve a half-tempo tracker result without changing audio time."""
    if factor < 2:
        raise ValueError("Tempo multiplication factor must be at least 2.")
    beats_per_bar = int(source_grid["meter"]["beats_per_bar"])
    ticks_per_bar = beats_per_bar * PROJECT_PPQ
    original_bpm = float(source_grid["tempo"]["bpm"])
    bpm = float(target_bpm if target_bpm is not None else original_bpm * factor)
    first_downbeat_sec = float(
        source_grid["midi_alignment"]["first_detected_downbeat_sec"]
    )
    first_downbeat_tick = (
        int(source_grid["midi_alignment"]["first_detected_downbeat_tick"])
        * factor
    )
    grid_origin_sec = (
        first_downbeat_sec
        - first_downbeat_tick / PROJECT_PPQ * 60.0 / bpm
    )
    song_duration_sec = float(source_grid["source"]["duration_sec"])
    audio_start_tick = int(
        round((0.0 - grid_origin_sec) * bpm / 60.0 * PROJECT_PPQ)
    )
    audio_end_tick = int(
        round(
            (song_duration_sec - grid_origin_sec)
            * bpm
            / 60.0
            * PROJECT_PPQ
        )
    )
    new_end_tick = int(
        math.ceil(audio_end_tick / ticks_per_bar) * ticks_per_bar
    )
    raw_beats = np.asarray(
        source_grid["model_events"]["beat_candidates_sec"],
        dtype=np.float64,
    )
    raw_downbeats = np.asarray(
        source_grid["model_events"]["downbeat_candidates_sec"],
        dtype=np.float64,
    )

    beat_entries = []
    for tick in range(0, new_end_tick, PROJECT_PPQ):
        sec = grid_origin_sec + tick / PROJECT_PPQ * 60.0 / bpm
        if sec < 0 or sec > song_duration_sec:
            continue
        sample = int(round(sec * source_grid["source"]["sample_rate"]))
        beat_index = tick // PROJECT_PPQ
        beat_in_bar = beat_index % beats_per_bar
        nearest_beat = float(raw_beats[np.argmin(np.abs(raw_beats - sec))])
        nearest_downbeat = float(
            raw_downbeats[np.argmin(np.abs(raw_downbeats - sec))]
        )
        beat_entries.append(
            {
                "beat_index": beat_index,
                "sample": sample,
                "sec": round(sec, 6),
                "tick": tick,
                "bar_index": beat_index // beats_per_bar,
                "beat_in_bar": beat_in_bar,
                "is_downbeat": beat_in_bar == 0,
                "model_beat_candidate_sec": (
                    round(nearest_beat, 6)
                    if abs(nearest_beat - sec) <= 0.12
                    else None
                ),
                "model_downbeat_candidate_sec": (
                    round(nearest_downbeat, 6)
                    if beat_in_bar == 0
                    and abs(nearest_downbeat - sec) <= 0.16
                    else None
                ),
            }
        )

    result = copy.deepcopy(source_grid)
    result["schema_version"] = "song_grid_v1.1_tempo_multiplier"
    result["tempo"] = {
        "bpm": bpm,
        "suggested_project_bpm": bpm,
        "global_regression_bpm": float(
            source_grid["tempo"].get("global_regression_bpm", original_bpm)
        )
        * factor,
        "median_interval_bpm": float(
            source_grid["tempo"].get("median_interval_bpm", original_bpm)
        )
        * factor,
        "raw_median_bpm": float(
            source_grid["tempo"].get("raw_median_bpm", original_bpm)
        )
        * factor,
        "integer_snap_applied": float(bpm).is_integer(),
        "integer_snap_tolerance_bpm": source_grid["tempo"].get(
            "integer_snap_tolerance_bpm",
            0.1,
        ),
        "half_double_candidates_bpm": [bpm / 2.0, bpm, bpm * 2.0],
        "mode": "manual_tempo_multiplier",
        "tempo_multiplier": factor,
        "source_detected_bpm": original_bpm,
    }
    result["meter"]["median_bar_duration_sec"] = (
        float(source_grid["meter"]["median_bar_duration_sec"]) / factor
    )
    alignment = result["midi_alignment"]
    grid_end_virtual_sec = (
        grid_origin_sec + new_end_tick / PROJECT_PPQ * 60.0 / bpm
    )
    alignment.update(
        {
            "prepend_bars": first_downbeat_tick // ticks_per_bar,
            "grid_origin_sec": round(grid_origin_sec, 6),
            "grid_origin_sample": int(
                round(grid_origin_sec * source_grid["source"]["sample_rate"])
            ),
            "audio_start_tick": audio_start_tick,
            "first_detected_downbeat_tick": first_downbeat_tick,
            "audio_end_tick": audio_end_tick,
            "grid_end_tick": new_end_tick,
            "song_end_tick": new_end_tick,
            "trailing_padding_ticks": new_end_tick - audio_end_tick,
            "trailing_padding_sec": round(
                grid_end_virtual_sec - song_duration_sec,
                6,
            ),
            "grid_end_virtual_sec": round(grid_end_virtual_sec, 6),
        }
    )
    result["summary"].update(
        {
            "num_beats": len(beat_entries),
            "num_downbeats": sum(
                int(entry["is_downbeat"]) for entry in beat_entries
            ),
            "total_complete_bars": new_end_tick // ticks_per_bar,
            "tempo_multiplier": factor,
        }
    )
    result["beats"] = beat_entries
    result.setdefault("warnings", []).append(
        {
            "type": "manual_tempo_multiplier",
            "message": (
                f"Detected {original_bpm:.3f} BPM was interpreted at x{factor} "
                f"as {bpm:.3f} BPM."
            ),
        }
    )
    return result


def seconds_at_tick(tick: float, regions: list[RegionAnchor]) -> float:
    for region in regions:
        if tick <= region.end_tick:
            fraction = (tick - region.start_tick) / (
                region.end_tick - region.start_tick
            )
            return region.start_sec + fraction * (
                region.end_sec - region.start_sec
            )
    final = regions[-1]
    seconds_per_tick = 60.0 / (final.bpm * PROJECT_PPQ)
    return final.end_sec + (tick - final.end_tick) * seconds_per_tick


def tick_at_seconds(seconds: float, regions: list[RegionAnchor]) -> int:
    for region in regions:
        if seconds <= region.end_sec:
            fraction = (seconds - region.start_sec) / (
                region.end_sec - region.start_sec
            )
            return int(
                round(
                    region.start_tick
                    + fraction * (region.end_tick - region.start_tick)
                )
            )
    final = regions[-1]
    ticks_per_second = final.bpm * PROJECT_PPQ / 60.0
    return int(round(final.end_tick + (seconds - final.end_sec) * ticks_per_second))


def build_three_region_grid(
    source_grid: dict,
    *,
    first_downbeat_sec: float,
    first_bpm: float,
    transition_start_sec: float | None,
    change_sec: float,
    transition_bars: int,
    second_bpm: float,
) -> tuple[dict, list[RegionAnchor]]:
    beats_per_bar = int(source_grid["meter"]["beats_per_bar"])
    ticks_per_bar = beats_per_bar * PROJECT_PPQ
    nominal_first_bar_sec = 60.0 * beats_per_bar / first_bpm
    grid_origin_sec = first_downbeat_sec - nominal_first_bar_sec

    if transition_bars <= 0:
        if transition_start_sec is None:
            raise ValueError(
                "Automatic transition-bar count needs --transition-start-sec."
            )
        transition_bars = choose_transition_bar_count(
            change_sec - transition_start_sec,
            first_bpm,
            second_bpm,
            beats_per_bar,
        )
    if transition_start_sec is None:
        change_bar = int(
            round((change_sec - grid_origin_sec) / nominal_first_bar_sec)
        )
        transition_start_bar = max(1, change_bar - transition_bars)
        transition_start_sec = (
            grid_origin_sec + transition_start_bar * nominal_first_bar_sec
        )
    else:
        transition_start_bar = int(
            round((transition_start_sec - grid_origin_sec) / nominal_first_bar_sec)
        )
    transition_start_tick = transition_start_bar * ticks_per_bar
    change_tick = transition_start_tick + transition_bars * ticks_per_bar
    regions = [
        RegionAnchor(
            start_tick=0,
            end_tick=transition_start_tick,
            start_sec=grid_origin_sec,
            end_sec=transition_start_sec,
            kind="stable",
        ),
        RegionAnchor(
            start_tick=transition_start_tick,
            end_tick=change_tick,
            start_sec=transition_start_sec,
            end_sec=change_sec,
            kind="transition",
        ),
    ]
    second_bar_sec = 60.0 * beats_per_bar / second_bpm
    song_duration_sec = float(source_grid["source"]["duration_sec"])
    remaining_bars = max(
        1,
        int(math.ceil((song_duration_sec - change_sec) / second_bar_sec)),
    )
    grid_end_tick = change_tick + remaining_bars * ticks_per_bar
    grid_end_sec = change_sec + remaining_bars * second_bar_sec
    regions.append(
        RegionAnchor(
            start_tick=change_tick,
            end_tick=grid_end_tick,
            start_sec=change_sec,
            end_sec=grid_end_sec,
            kind="stable",
        )
    )

    beat_entries = []
    raw_beats = np.asarray(
        source_grid["model_events"]["beat_candidates_sec"],
        dtype=np.float64,
    )
    raw_downbeats = np.asarray(
        source_grid["model_events"]["downbeat_candidates_sec"],
        dtype=np.float64,
    )
    beat_index = 0
    for tick in range(0, grid_end_tick, PROJECT_PPQ):
        sec = seconds_at_tick(tick, regions)
        if sec < 0 or sec > song_duration_sec:
            beat_index += 1
            continue
        nearest_beat = float(raw_beats[np.argmin(np.abs(raw_beats - sec))])
        nearest_downbeat = float(
            raw_downbeats[np.argmin(np.abs(raw_downbeats - sec))]
        )
        beat_in_bar = beat_index % beats_per_bar
        beat_entries.append(
            {
                "beat_index": beat_index,
                "sample": int(round(sec * source_grid["source"]["sample_rate"])),
                "sec": round(sec, 6),
                "tick": tick,
                "bar_index": beat_index // beats_per_bar,
                "beat_in_bar": beat_in_bar,
                "is_downbeat": beat_in_bar == 0,
                "model_beat_candidate_sec": (
                    round(nearest_beat, 6)
                    if abs(nearest_beat - sec) <= 0.12
                    else None
                ),
                "model_downbeat_candidate_sec": (
                    round(nearest_downbeat, 6)
                    if beat_in_bar == 0 and abs(nearest_downbeat - sec) <= 0.16
                    else None
                ),
            }
        )
        beat_index += 1

    audio_start_tick = tick_at_seconds(0.0, regions)
    audio_end_tick = tick_at_seconds(song_duration_sec, regions)
    result = json.loads(json.dumps(source_grid))
    result["schema_version"] = "song_grid_v1.1_tempo_regions"
    result["tempo"] = {
        "bpm": float(first_bpm),
        "suggested_project_bpm": float(first_bpm),
        "mode": "piecewise_constant_regions",
        "num_regions": len(regions),
        "regions": [
            {
                "start_tick": region.start_tick,
                "end_tick": region.end_tick,
                "start_sec": round(region.start_sec, 6),
                "end_sec": round(region.end_sec, 6),
                "bpm": region.bpm,
                "kind": region.kind,
            }
            for region in regions
        ],
    }
    result["meter"]["median_bar_duration_sec"] = float(
        np.median(
            [
                (region.end_sec - region.start_sec)
                / ((region.end_tick - region.start_tick) / ticks_per_bar)
                for region in regions
            ]
        )
    )
    result["midi_alignment"].update(
        {
            "prepend_bars": 1,
            "grid_origin_sec": round(grid_origin_sec, 6),
            "grid_origin_sample": int(
                round(grid_origin_sec * source_grid["source"]["sample_rate"])
            ),
            "audio_start_tick": audio_start_tick,
            "first_detected_downbeat_sec": round(first_downbeat_sec, 6),
            "first_detected_downbeat_sample": int(
                round(first_downbeat_sec * source_grid["source"]["sample_rate"])
            ),
            "first_detected_downbeat_tick": ticks_per_bar,
            "audio_end_tick": audio_end_tick,
            "grid_end_tick": grid_end_tick,
            "song_end_tick": grid_end_tick,
            "trailing_padding_ticks": grid_end_tick - audio_end_tick,
            "trailing_padding_sec": round(grid_end_sec - song_duration_sec, 6),
            "grid_end_virtual_sec": round(grid_end_sec, 6),
        }
    )
    result["summary"].update(
        {
            "num_beats": len(beat_entries),
            "num_downbeats": sum(
                int(entry["is_downbeat"]) for entry in beat_entries
            ),
            "total_complete_bars": grid_end_tick // ticks_per_bar,
            "num_tempo_regions": len(regions),
        }
    )
    result["beats"] = beat_entries
    result["warnings"] = [
        warning
        for warning in result.get("warnings", [])
        if warning.get("type") != "downbeat_grid_disagreement"
    ]
    result["warnings"].append(
        {
            "type": "tempo_region_override",
            "message": (
                "Global-tempo regularization was replaced by stable tempo "
                "regions plus one transition block."
            ),
        }
    )
    return result, regions


def save_region_tempo_midi(
    output_path: Path,
    grid: dict,
    regions: list[RegionAnchor],
) -> None:
    midi = mido.MidiFile(ticks_per_beat=PROJECT_PPQ)
    track = mido.MidiTrack()
    events: list[tuple[int, int, mido.MetaMessage]] = [
        (
            0,
            0,
            mido.MetaMessage(
                "time_signature",
                numerator=int(grid["meter"]["beats_per_bar"]),
                denominator=4,
            ),
        )
    ]
    for index, region in enumerate(regions, start=1):
        events.extend(
            [
                (
                    region.start_tick,
                    1,
                    mido.MetaMessage(
                        "set_tempo",
                        tempo=int(round(mido.bpm2tempo(region.bpm))),
                    ),
                ),
                (
                    region.start_tick,
                    2,
                    mido.MetaMessage(
                        "marker",
                        text=(
                            f"Tempo region {index}: {region.kind}, "
                            f"{region.bpm:.3f} BPM"
                        ),
                    ),
                ),
            ]
        )
    events.append(
        (
            int(grid["midi_alignment"]["audio_start_tick"]),
            3,
            mido.MetaMessage("marker", text="AUDIO START (place WAV here)"),
        )
    )
    previous_tick = 0
    for tick, _, message in sorted(events, key=lambda item: (item[0], item[1])):
        track.append(message.copy(time=tick - previous_tick))
        previous_tick = tick
    track.append(
        mido.MetaMessage(
            "end_of_track",
            time=max(
                0,
                int(grid["midi_alignment"]["grid_end_tick"]) - previous_tick,
            ),
        )
    )
    midi.tracks.append(track)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    midi.save(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild a grid as stable tempo regions with a transition block."
    )
    parser.add_argument("source_grid", type=Path)
    parser.add_argument("output_grid", type=Path)
    parser.add_argument("tempo_midi", type=Path)
    parser.add_argument("--first-downbeat-sec", type=float, required=True)
    parser.add_argument("--first-bpm", type=float, required=True)
    parser.add_argument(
        "--transition-start-sec",
        type=float,
        help=(
            "Start of the bridge block. Omit it to place the requested number "
            "of transition bars immediately before the detected change."
        ),
    )
    parser.add_argument("--change-sec", type=float, required=True)
    parser.add_argument(
        "--transition-bars",
        type=int,
        default=1,
        help="Number of bridge bars; defaults to one engineering transition bar.",
    )
    parser.add_argument("--second-bpm", type=float, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = json.loads(args.source_grid.read_text(encoding="utf-8"))
    grid, regions = build_three_region_grid(
        source,
        first_downbeat_sec=args.first_downbeat_sec,
        first_bpm=args.first_bpm,
        transition_start_sec=args.transition_start_sec,
        change_sec=args.change_sec,
        transition_bars=args.transition_bars,
        second_bpm=args.second_bpm,
    )
    args.output_grid.write_text(
        json.dumps(grid, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    save_region_tempo_midi(args.tempo_midi, grid, regions)
    print(f"Saved grid: {args.output_grid}")
    print(f"Saved Tempo Map: {args.tempo_midi}")
    for region in regions:
        print(
            f"  {region.kind}: ticks {region.start_tick}-{region.end_tick}, "
            f"seconds {region.start_sec:.3f}-{region.end_sec:.3f}, "
            f"{region.bpm:.3f} BPM"
        )


if __name__ == "__main__":
    main()
