from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

import mido
import numpy as np

from project_settings import PROJECT_PPQ


@dataclass(frozen=True)
class TempoSegment:
    start_tick: int
    end_tick: int
    start_bar: float
    end_bar: float
    bpm: float
    max_anchor_error_ms: float
    rms_anchor_error_ms: float


def _grid_anchors(grid: dict) -> tuple[np.ndarray, np.ndarray]:
    origin_sec = float(grid["midi_alignment"]["grid_origin_sec"])
    end_tick = int(grid["midi_alignment"]["grid_end_tick"])
    end_sec = float(grid["midi_alignment"]["grid_end_virtual_sec"]) - origin_sec
    anchors: dict[int, float] = {0: 0.0, end_tick: end_sec}
    for beat in grid["beats"]:
        tick = int(beat["tick"])
        if 0 <= tick <= end_tick:
            anchors[tick] = float(beat["sec"]) - origin_sec
    ticks = np.asarray(sorted(anchors), dtype=np.int64)
    seconds = np.asarray([anchors[int(tick)] for tick in ticks], dtype=np.float64)
    return ticks, seconds


def _bar_boundaries(grid: dict, anchor_ticks: np.ndarray) -> np.ndarray:
    end_tick = int(grid["midi_alignment"]["grid_end_tick"])
    ticks = {0, end_tick}
    ticks.update(
        int(beat["tick"])
        for beat in grid["beats"]
        if beat.get("is_downbeat") and 0 < int(beat["tick"]) < end_tick
    )
    # A missing downbeat must not make the final segment span an arbitrary
    # number of bars. Add nominal bar lines only where interpolation is safe.
    bar_ticks = int(grid["meter"]["beats_per_bar"]) * PROJECT_PPQ
    ticks.update(range(0, end_tick + 1, bar_ticks))
    return np.asarray(
        sorted(tick for tick in ticks if anchor_ticks[0] <= tick <= anchor_ticks[-1]),
        dtype=np.int64,
    )


def build_tempo_segments(
    grid: dict,
    *,
    max_anchor_error_ms: float = 25.0,
) -> list[TempoSegment]:
    """Find the fewest bar-boundary constant-tempo sections within tolerance."""
    if max_anchor_error_ms <= 0:
        raise ValueError("max_anchor_error_ms must be positive.")
    tempo = grid["tempo"]
    if grid.get("timeline"):
        from musical_timeline import grid_tempo_events
        events = grid_tempo_events(grid)
        end = int(grid["midi_alignment"]["grid_end_tick"])
        bar_ticks = int(grid["meter"]["beats_per_bar"]) * PROJECT_PPQ
        return [TempoSegment(tick, min(end, events[i + 1][0] if i + 1 < len(events) else end),
            tick / bar_ticks, min(end, events[i + 1][0] if i + 1 < len(events) else end) / bar_ticks,
            float(mido.tempo2bpm(value)), 0.0, 0.0)
            for i, (tick, value) in enumerate(events) if tick < end]
    if tempo.get("regions") or tempo.get("manual_override_bpm"):
        from musical_timeline import canonicalize_grid
        regions = tempo.get("regions")
        events = ([(int(r["start_tick"]), mido.bpm2tempo(float(r["bpm"]))) for r in regions]
                  if regions else [(0, mido.bpm2tempo(float(tempo["manual_override_bpm"])))])
        return build_tempo_segments(canonicalize_grid(grid, events), max_anchor_error_ms=max_anchor_error_ms)
    if (
        not tempo.get("regions")
        and tempo.get("integer_snap_applied")
        and float(tempo["bpm"]).is_integer()
    ):
        bpm = float(tempo["bpm"])
        end_tick = int(grid["midi_alignment"]["grid_end_tick"])
        origin_sec = float(grid["midi_alignment"]["grid_origin_sec"])
        beat_ticks = np.asarray(
            [beat["tick"] for beat in grid["beats"]],
            dtype=np.float64,
        )
        beat_seconds = np.asarray(
            [beat["sec"] for beat in grid["beats"]],
            dtype=np.float64,
        )
        expected = origin_sec + beat_ticks / PROJECT_PPQ * 60.0 / bpm
        errors = beat_seconds - expected
        beats_per_bar = int(grid["meter"]["beats_per_bar"])
        constant_segments = [
            TempoSegment(
                start_tick=0,
                end_tick=end_tick,
                start_bar=0.0,
                end_bar=end_tick / (beats_per_bar * PROJECT_PPQ),
                bpm=bpm,
                max_anchor_error_ms=float(
                    np.max(np.abs(errors), initial=0.0) * 1000.0
                ),
                rms_anchor_error_ms=float(
                    np.sqrt(np.mean(np.square(errors))) * 1000.0
                ),
            )
        ]
        if constant_segments[0].max_anchor_error_ms <= max_anchor_error_ms:
            return constant_segments
    anchor_ticks, anchor_seconds = _grid_anchors(grid)
    boundaries = _bar_boundaries(grid, anchor_ticks)
    boundary_seconds = np.interp(boundaries, anchor_ticks, anchor_seconds)
    tolerance_sec = max_anchor_error_ms / 1000.0
    beats_per_bar = int(grid["meter"]["beats_per_bar"])
    bar_ticks = beats_per_bar * PROJECT_PPQ

    @lru_cache(maxsize=None)
    def segment_fit(start_index: int, end_index: int) -> tuple[float, float, float]:
        start_tick = int(boundaries[start_index])
        end_tick = int(boundaries[end_index])
        start_sec = float(boundary_seconds[start_index])
        end_sec = float(boundary_seconds[end_index])
        duration = end_sec - start_sec
        if duration <= 0 or end_tick <= start_tick:
            return math.inf, math.inf, 0.0
        mask = (anchor_ticks >= start_tick) & (anchor_ticks <= end_tick)
        expected = start_sec + (
            (anchor_ticks[mask] - start_tick)
            / (end_tick - start_tick)
            * duration
        )
        errors = anchor_seconds[mask] - expected
        max_error = float(np.max(np.abs(errors), initial=0.0))
        rms_error = float(np.sqrt(np.mean(np.square(errors)))) if len(errors) else 0.0
        quarters = (end_tick - start_tick) / PROJECT_PPQ
        bpm = 60.0 * quarters / duration
        return max_error, rms_error, bpm

    count = len(boundaries)
    best_segments = [math.inf] * count
    best_rms = [math.inf] * count
    previous = [-1] * count
    best_segments[0] = 0
    best_rms[0] = 0.0

    for end_index in range(1, count):
        for start_index in range(end_index):
            max_error, rms_error, bpm = segment_fit(start_index, end_index)
            if max_error > tolerance_sec or not 30.0 <= bpm <= 300.0:
                continue
            candidate_segments = best_segments[start_index] + 1
            candidate_rms = best_rms[start_index] + rms_error
            if (
                candidate_segments < best_segments[end_index]
                or (
                    candidate_segments == best_segments[end_index]
                    and candidate_rms < best_rms[end_index]
                )
            ):
                best_segments[end_index] = candidate_segments
                best_rms[end_index] = candidate_rms
                previous[end_index] = start_index

    if previous[-1] < 0:
        raise ValueError(
            "Could not fit a bar-boundary tempo map. The detected grid likely "
            "contains a misplaced beat. Review the anchors or provide a confirmed --bpm "
            "to predict_test_audio.py; integer snapping will not silently bypass the error limit."
        )

    ranges: list[tuple[int, int]] = []
    end_index = count - 1
    while end_index > 0:
        start_index = previous[end_index]
        ranges.append((start_index, end_index))
        end_index = start_index
    ranges.reverse()

    segments = []
    for start_index, end_index in ranges:
        max_error, rms_error, bpm = segment_fit(start_index, end_index)
        start_tick = int(boundaries[start_index])
        end_tick = int(boundaries[end_index])
        segments.append(
            TempoSegment(
                start_tick=start_tick,
                end_tick=end_tick,
                start_bar=start_tick / bar_ticks,
                end_bar=end_tick / bar_ticks,
                bpm=bpm,
                max_anchor_error_ms=max_error * 1000.0,
                rms_anchor_error_ms=rms_error * 1000.0,
            )
        )
    return segments


def tempo_events(segments: list[TempoSegment]) -> list[tuple[int, int]]:
    return [
        (segment.start_tick, int(round(mido.bpm2tempo(segment.bpm))))
        for segment in segments
    ]


def build_tempo_track(
    grid: dict,
    segments: list[TempoSegment],
) -> mido.MidiTrack:
    end_tick = int(grid["midi_alignment"]["grid_end_tick"])
    beats_per_bar = int(grid["meter"]["beats_per_bar"])
    track = mido.MidiTrack()
    events: list[tuple[int, int, mido.MetaMessage]] = [
        (
            0,
            0,
            mido.MetaMessage(
                "time_signature",
                numerator=beats_per_bar,
                denominator=4,
            ),
        )
    ]
    for index, segment in enumerate(segments, start=1):
        events.extend(
            [
                (
                    segment.start_tick,
                    1,
                    mido.MetaMessage(
                        "set_tempo",
                        tempo=int(round(mido.bpm2tempo(segment.bpm))),
                    ),
                ),
                (
                    segment.start_tick,
                    2,
                    mido.MetaMessage(
                        "marker",
                        text=(
                            f"Tempo section {index}: {segment.bpm:.3f} BPM, "
                            f"bars {segment.start_bar:.2f}-{segment.end_bar:.2f}"
                        ),
                    ),
                ),
            ]
        )
    audio_start_tick = int(grid["midi_alignment"]["audio_start_tick"])
    events.append(
        (
            audio_start_tick,
            3,
            mido.MetaMessage(
                "marker",
                text=(
                    "WAV START (align audio sample 0 here): "
                    f"tick {audio_start_tick}"
                ),
            ),
        )
    )
    previous_tick = 0
    for tick, _, message in sorted(events, key=lambda item: (item[0], item[1])):
        track.append(message.copy(time=tick - previous_tick))
        previous_tick = tick
    track.append(
        mido.MetaMessage(
            "end_of_track",
            time=max(0, end_tick - previous_tick),
        )
    )
    return track


def save_tempo_map_midi(
    output_path: Path,
    grid: dict,
    segments: list[TempoSegment],
) -> None:
    midi = mido.MidiFile(ticks_per_beat=PROJECT_PPQ)
    track = build_tempo_track(grid, segments)
    midi.tracks.append(track)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    midi.save(output_path)


def save_midi_with_tempo_map(
    source_midi_path: Path,
    output_path: Path,
    grid: dict,
    segments: list[TempoSegment],
) -> None:
    source = mido.MidiFile(source_midi_path)
    if source.ticks_per_beat != PROJECT_PPQ:
        raise ValueError(
            f"Source MIDI PPQ is {source.ticks_per_beat}; expected {PROJECT_PPQ}."
        )
    tempo_track = build_tempo_track(grid, segments)
    if source.tracks:
        source.tracks[0] = tempo_track
    else:
        source.tracks.append(tempo_track)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    source.save(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compress beat anchors into a bar-boundary MIDI Tempo Map."
    )
    parser.add_argument("grid", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--max-error-ms", type=float, default=25.0)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--source-midi", type=Path)
    parser.add_argument("--combined-output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    grid = json.loads(args.grid.read_text(encoding="utf-8"))
    segments = build_tempo_segments(
        grid,
        max_anchor_error_ms=args.max_error_ms,
    )
    save_tempo_map_midi(args.output, grid, segments)
    if bool(args.source_midi) != bool(args.combined_output):
        raise ValueError(
            "--source-midi and --combined-output must be supplied together."
        )
    if args.source_midi:
        save_midi_with_tempo_map(
            args.source_midi,
            args.combined_output,
            grid,
            segments,
        )
    report = {
        "grid": str(args.grid.resolve()),
        "tempo_map_midi": str(args.output.resolve()),
        "combined_midi": (
            str(args.combined_output.resolve())
            if args.combined_output
            else None
        ),
        "max_error_limit_ms": args.max_error_ms,
        "audio_start_tick": int(grid["midi_alignment"]["audio_start_tick"]),
        "segments": [asdict(segment) for segment in segments],
    }
    report_path = args.report or args.output.with_suffix(".json")
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Saved {args.output} with {len(segments)} tempo sections.")
    for segment in segments:
        print(
            f"  bars {segment.start_bar:.2f}-{segment.end_bar:.2f}: "
            f"{segment.bpm:.3f} BPM, max error "
            f"{segment.max_anchor_error_ms:.1f} ms"
        )


if __name__ == "__main__":
    main()
