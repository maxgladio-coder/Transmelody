from __future__ import annotations

import argparse
import math
from collections import defaultdict, deque
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import mido

from transmelody.config import DEFAULT_BEATS_PER_BAR, PROJECT_PPQ


GRID_STEPS: dict[str, Fraction] = {
    "1/4": Fraction(1, 1),
    "1/8": Fraction(1, 2),
    "1/8T": Fraction(1, 3),
    "1/16": Fraction(1, 4),
    "1/16T": Fraction(1, 6),
    "1/32": Fraction(1, 8),
}


@dataclass(frozen=True)
class QuantizeConfig:
    bpm: float | None = None
    grid: str = "1/16"
    quantize_note_ends: bool = True
    minimum_length_steps: int = 1
    repair_monophonic_overlaps: bool = True
    bridge_short_gaps: bool = True
    maximum_gap_steps: int = 1
    complete_final_bar: bool = True
    beats_per_bar: int | None = None
    target_ppq: int = PROJECT_PPQ
    minimum_input_velocity: int = 40
    output_velocity: int = 100


@dataclass
class Event:
    message: mido.Message | mido.MetaMessage
    absolute_tick: int
    original_order: int


@dataclass
class Note:
    track_index: int
    channel: int
    pitch: int
    start_event: Event
    end_event: Event
    original_start_tick: int
    original_end_tick: int
    new_start_tick: int
    new_end_tick: int


@dataclass(frozen=True)
class QuantizeStats:
    input_notes: int
    notes: int
    deleted_low_velocity_notes: int
    moved_onsets: int
    moved_ends: int
    repaired_overlaps: int
    bridged_gaps: int
    unmatched_note_ons: int
    unmatched_note_offs: int
    max_adjustment_ticks: int
    max_adjustment_ms: float
    source_ticks_per_beat: int
    ticks_per_beat: int
    grid_step_ticks: float
    original_end_tick: int
    output_end_tick: int
    trailing_padding_ticks: int
    time_signature: tuple[int, int]
    input_bpm: float
    output_bpm: float
    tempo_replaced: bool


def grid_step_ticks(ticks_per_beat: int, grid: str) -> Fraction:
    if grid not in GRID_STEPS:
        raise ValueError(f"Unsupported grid: {grid}")
    return ticks_per_beat * GRID_STEPS[grid]


def round_fraction_to_int(value: Fraction) -> int:
    whole, remainder = divmod(value.numerator, value.denominator)
    if remainder * 2 >= value.denominator:
        whole += 1
    return whole


def snap_tick(tick: int, step: Fraction) -> int:
    grid_position = Fraction(tick, 1) / step
    snapped_position = round_fraction_to_int(grid_position)
    return max(0, round_fraction_to_int(step * snapped_position))


def track_to_absolute_events(track: mido.MidiTrack) -> list[Event]:
    absolute_tick = 0
    events: list[Event] = []
    for order, message in enumerate(track):
        absolute_tick += int(message.time)
        events.append(
            Event(
                message=message.copy(time=0),
                absolute_tick=absolute_tick,
                original_order=order,
            )
        )
    return events


def rescale_event_ticks(
    tracks: list[list[Event]],
    *,
    source_ppq: int,
    target_ppq: int,
) -> None:
    if source_ppq == target_ppq:
        return
    for events in tracks:
        for event in events:
            event.absolute_tick = round_fraction_to_int(
                Fraction(event.absolute_tick * target_ppq, source_ppq)
            )


def pair_notes(
    tracks: list[list[Event]],
) -> tuple[list[Note], int, int]:
    notes: list[Note] = []
    unmatched_note_offs = 0
    unmatched_note_ons = 0

    for track_index, events in enumerate(tracks):
        active: dict[tuple[int, int], deque[Event]] = defaultdict(deque)

        for event in events:
            message = event.message
            if message.type == "note_on" and message.velocity > 0:
                active[(message.channel, message.note)].append(event)
                continue

            is_note_off = message.type == "note_off" or (
                message.type == "note_on" and message.velocity == 0
            )
            if not is_note_off:
                continue

            key = (message.channel, message.note)
            if not active[key]:
                unmatched_note_offs += 1
                continue

            start_event = active[key].popleft()
            notes.append(
                Note(
                    track_index=track_index,
                    channel=message.channel,
                    pitch=message.note,
                    start_event=start_event,
                    end_event=event,
                    original_start_tick=start_event.absolute_tick,
                    original_end_tick=event.absolute_tick,
                    new_start_tick=start_event.absolute_tick,
                    new_end_tick=event.absolute_tick,
                )
            )

        unmatched_note_ons += sum(len(queue) for queue in active.values())

    return notes, unmatched_note_ons, unmatched_note_offs


def filter_and_normalize_velocity(
    tracks: list[list[Event]],
    notes: list[Note],
    *,
    minimum_input_velocity: int,
    output_velocity: int,
) -> tuple[list[Note], int]:
    deleted_notes = [
        note
        for note in notes
        if note.start_event.message.velocity < minimum_input_velocity
    ]
    deleted_event_ids = {
        id(event)
        for note in deleted_notes
        for event in (note.start_event, note.end_event)
    }
    if deleted_event_ids:
        for events in tracks:
            events[:] = [
                event for event in events if id(event) not in deleted_event_ids
            ]

    retained_notes = [
        note
        for note in notes
        if note.start_event.message.velocity >= minimum_input_velocity
    ]
    for note in retained_notes:
        note.start_event.message = note.start_event.message.copy(
            velocity=output_velocity
        )
    return retained_notes, len(deleted_notes)


def quantize_notes(
    notes: list[Note],
    *,
    step: Fraction,
    quantize_note_ends: bool,
    minimum_length_steps: int,
    repair_monophonic_overlaps: bool,
    bridge_short_gaps: bool,
    maximum_gap_steps: int,
) -> tuple[int, int, int, int, int]:
    minimum_ticks = max(
        1,
        round_fraction_to_int(step * max(1, minimum_length_steps)),
    )
    for note in notes:
        note.new_start_tick = snap_tick(note.original_start_tick, step)
        if quantize_note_ends:
            note.new_end_tick = snap_tick(note.original_end_tick, step)
        else:
            note.new_end_tick = (
                note.original_end_tick
                + note.new_start_tick
                - note.original_start_tick
            )

        note.new_end_tick = max(
            note.new_end_tick,
            note.new_start_tick + minimum_ticks,
        )
    repaired_overlaps = 0
    bridged_gaps = 0
    if repair_monophonic_overlaps or bridge_short_gaps:
        groups: dict[tuple[int, int], list[Note]] = defaultdict(list)
        for note in notes:
            groups[(note.track_index, note.channel)].append(note)

        for group in groups.values():
            group.sort(
                key=lambda note: (
                    note.new_start_tick,
                    note.original_start_tick,
                    note.pitch,
                )
            )
            for current, following in zip(group[:-1], group[1:]):
                if (
                    repair_monophonic_overlaps
                    and current.new_end_tick > following.new_start_tick
                    and following.new_start_tick > current.new_start_tick
                ):
                    current.new_end_tick = following.new_start_tick
                    repaired_overlaps += 1

                gap_ticks = following.new_start_tick - current.new_end_tick
                maximum_gap_ticks = round_fraction_to_int(
                    step * max(0, maximum_gap_steps)
                )
                if (
                    bridge_short_gaps
                    and 0 < gap_ticks <= maximum_gap_ticks
                ):
                    current.new_end_tick = following.new_start_tick
                    bridged_gaps += 1

    moved_onsets = sum(
        note.new_start_tick != note.original_start_tick for note in notes
    )
    moved_ends = sum(
        note.new_end_tick != note.original_end_tick for note in notes
    )
    max_adjustment_ticks = max(
        (
            adjustment
            for note in notes
            for adjustment in (
                abs(note.new_start_tick - note.original_start_tick),
                abs(note.new_end_tick - note.original_end_tick),
            )
        ),
        default=0,
    )

    for note in notes:
        note.start_event.absolute_tick = note.new_start_tick
        note.end_event.absolute_tick = note.new_end_tick

    return (
        moved_onsets,
        moved_ends,
        repaired_overlaps,
        bridged_gaps,
        max_adjustment_ticks,
    )


def infer_time_signature(
    tracks: list[list[Event]],
) -> tuple[int, int]:
    for events in tracks:
        for event in events:
            if event.message.type == "time_signature":
                return event.message.numerator, event.message.denominator
    return DEFAULT_BEATS_PER_BAR, 4


def event_priority(event: Event) -> tuple[int, int]:
    message = event.message
    if message.type == "set_tempo":
        return 0, event.original_order
    if message.type == "note_off" or (
        message.type == "note_on" and message.velocity == 0
    ):
        return 1, event.original_order
    if message.type == "note_on":
        return 2, event.original_order
    if message.type == "end_of_track":
        return 9, event.original_order
    return 3, event.original_order


def rebuild_track(
    events: list[Event],
    *,
    minimum_end_tick: int = 0,
) -> mido.MidiTrack:
    non_end_events = [
        event for event in events if event.message.type != "end_of_track"
    ]
    original_end_tick = max(
        (
            event.absolute_tick
            for event in events
            if event.message.type == "end_of_track"
        ),
        default=0,
    )
    final_tick = max(
        [
            minimum_end_tick,
            original_end_tick,
            *(event.absolute_tick for event in non_end_events),
        ],
        default=0,
    )
    non_end_events.sort(
        key=lambda event: (
            event.absolute_tick,
            *event_priority(event),
        )
    )

    track = mido.MidiTrack()
    previous_tick = 0
    for event in non_end_events:
        delta = event.absolute_tick - previous_tick
        track.append(event.message.copy(time=delta))
        previous_tick = event.absolute_tick
    track.append(mido.MetaMessage("end_of_track", time=final_tick - previous_tick))
    return track


def quantize_midi(
    midi: mido.MidiFile,
    config: QuantizeConfig,
) -> QuantizeStats:
    if config.bpm is not None and config.bpm <= 0:
        raise ValueError("BPM must be greater than zero.")
    if config.minimum_length_steps < 1:
        raise ValueError("Minimum note length must be at least one grid step.")
    if config.beats_per_bar is not None and config.beats_per_bar < 1:
        raise ValueError("Beats per bar must be at least one.")
    if config.maximum_gap_steps < 0:
        raise ValueError("Maximum short gap must not be negative.")
    if config.target_ppq < 1:
        raise ValueError("Target PPQ must be greater than zero.")
    if not 0 <= config.minimum_input_velocity <= 127:
        raise ValueError("Minimum input velocity must be between 0 and 127.")
    if not 1 <= config.output_velocity <= 127:
        raise ValueError("Output velocity must be between 1 and 127.")

    source_ticks_per_beat = midi.ticks_per_beat
    tracks = [track_to_absolute_events(track) for track in midi.tracks]
    time_signature = infer_time_signature(tracks)
    original_tempos = [
        event.message.tempo
        for events in tracks
        for event in events
        if event.message.type == "set_tempo"
    ]
    default_tempo = mido.bpm2tempo(120)
    input_tempo = original_tempos[0] if original_tempos else default_tempo
    input_bpm = float(mido.tempo2bpm(input_tempo))
    rescale_event_ticks(
        tracks,
        source_ppq=source_ticks_per_beat,
        target_ppq=config.target_ppq,
    )
    midi.ticks_per_beat = config.target_ppq
    step = grid_step_ticks(midi.ticks_per_beat, config.grid)
    original_end_tick = max(
        (
            event.absolute_tick
            for events in tracks
            for event in events
        ),
        default=0,
    )
    notes, unmatched_note_ons, unmatched_note_offs = pair_notes(tracks)
    input_note_count = len(notes)
    notes, deleted_low_velocity_notes = filter_and_normalize_velocity(
        tracks,
        notes,
        minimum_input_velocity=config.minimum_input_velocity,
        output_velocity=config.output_velocity,
    )

    (
        moved_onsets,
        moved_ends,
        repaired_overlaps,
        bridged_gaps,
        max_adjustment_ticks,
    ) = (
        quantize_notes(
            notes,
            step=step,
            quantize_note_ends=config.quantize_note_ends,
            minimum_length_steps=config.minimum_length_steps,
            repair_monophonic_overlaps=config.repair_monophonic_overlaps,
            bridge_short_gaps=config.bridge_short_gaps,
            maximum_gap_steps=config.maximum_gap_steps,
        )
    )

    tempo_replaced = config.bpm is not None
    if tempo_replaced:
        for events in tracks:
            events[:] = [
                event for event in events if event.message.type != "set_tempo"
            ]
        output_tempo = mido.bpm2tempo(config.bpm)
        tempo_message = mido.MetaMessage(
            "set_tempo",
            tempo=output_tempo,
            time=0,
        )
        tracks[0].append(Event(tempo_message, absolute_tick=0, original_order=-1))
    else:
        output_tempo = input_tempo
    event_end_tick = max(
        (
            event.absolute_tick
            for events in tracks
            for event in events
            if event.message.type != "end_of_track"
        ),
        default=0,
    )
    natural_end_tick = max(original_end_tick, event_end_tick)
    if config.beats_per_bar is not None:
        ticks_per_bar = midi.ticks_per_beat * config.beats_per_bar
    else:
        numerator, denominator = time_signature
        ticks_per_bar = round_fraction_to_int(
            Fraction(
                midi.ticks_per_beat * numerator * 4,
                denominator,
            )
        )
    output_end_tick = (
        int(math.ceil(natural_end_tick / ticks_per_bar) * ticks_per_bar)
        if config.complete_final_bar
        else natural_end_tick
    )
    midi.tracks[:] = [
        rebuild_track(events, minimum_end_tick=output_end_tick)
        for events in tracks
    ]

    output_bpm = float(mido.tempo2bpm(output_tempo))
    milliseconds_per_tick = 60_000.0 / (output_bpm * midi.ticks_per_beat)
    return QuantizeStats(
        input_notes=input_note_count,
        notes=len(notes),
        deleted_low_velocity_notes=deleted_low_velocity_notes,
        moved_onsets=moved_onsets,
        moved_ends=moved_ends,
        repaired_overlaps=repaired_overlaps,
        bridged_gaps=bridged_gaps,
        unmatched_note_ons=unmatched_note_ons,
        unmatched_note_offs=unmatched_note_offs,
        max_adjustment_ticks=max_adjustment_ticks,
        max_adjustment_ms=max_adjustment_ticks * milliseconds_per_tick,
        source_ticks_per_beat=source_ticks_per_beat,
        ticks_per_beat=midi.ticks_per_beat,
        grid_step_ticks=float(step),
        original_end_tick=original_end_tick,
        output_end_tick=output_end_tick,
        trailing_padding_ticks=output_end_tick - natural_end_tick,
        time_signature=time_signature,
        input_bpm=input_bpm,
        output_bpm=output_bpm,
        tempo_replaced=tempo_replaced,
    )


def quantize_midi_file(
    input_path: Path,
    output_path: Path,
    config: QuantizeConfig,
) -> QuantizeStats:
    input_path = Path(input_path)
    output_path = Path(output_path)
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Output path must be different from input path.")

    midi = mido.MidiFile(input_path)
    stats = quantize_midi(midi, config)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    midi.save(output_path)
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Snap MIDI notes to a musical grid.")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--bpm",
        type=float,
        help="Replace the MIDI tempo. Omit to preserve the original tempo.",
    )
    parser.add_argument("--grid", choices=GRID_STEPS, default="1/16")
    parser.add_argument("--onsets-only", action="store_true")
    parser.add_argument("--minimum-length-steps", type=int, default=1)
    parser.add_argument("--keep-overlaps", action="store_true")
    parser.add_argument("--keep-short-gaps", action="store_true")
    parser.add_argument("--maximum-gap-steps", type=int, default=1)
    parser.add_argument("--keep-partial-final-bar", action="store_true")
    parser.add_argument("--beats-per-bar", type=int)
    parser.add_argument("--minimum-velocity", type=int, default=40)
    parser.add_argument("--output-velocity", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats = quantize_midi_file(
        args.input,
        args.output,
        QuantizeConfig(
            bpm=args.bpm,
            grid=args.grid,
            quantize_note_ends=not args.onsets_only,
            minimum_length_steps=args.minimum_length_steps,
            repair_monophonic_overlaps=not args.keep_overlaps,
            bridge_short_gaps=not args.keep_short_gaps,
            maximum_gap_steps=args.maximum_gap_steps,
            complete_final_bar=not args.keep_partial_final_bar,
            beats_per_bar=args.beats_per_bar,
            minimum_input_velocity=args.minimum_velocity,
            output_velocity=args.output_velocity,
        ),
    )
    print(
        f"Quantized {stats.notes} notes; deleted "
        f"{stats.deleted_low_velocity_notes} low-velocity notes; moved "
        f"{stats.moved_onsets} onsets and {stats.moved_ends} ends. "
        f"Wrote {args.output}."
    )


if __name__ == "__main__":
    main()
