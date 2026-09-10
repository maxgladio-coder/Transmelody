from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import mido
import numpy as np
import soundfile as sf

from grid_initializer import build_grid, load_and_mix_audio
from beat_this.inference import Audio2Beats
from midi_grid_quantizer import pair_notes, track_to_absolute_events
from project_settings import DEFAULT_BEATS_PER_BAR, PROJECT_PPQ


@dataclass(frozen=True)
class SongFiles:
    song_id: str
    inst_audio: Path
    vocal_audio: Path
    vocal_midi: Path


def index_by_suffix(
    folder: Path,
    *,
    suffix: str | tuple[str, ...],
    extensions: set[str],
) -> dict[str, Path]:
    result: dict[str, Path] = {}
    suffixes = (suffix,) if isinstance(suffix, str) else suffix
    suffixes = tuple(sorted(suffixes, key=len, reverse=True))
    for path in sorted(folder.iterdir()):
        if not path.is_file() or path.suffix.lower() not in extensions:
            continue
        stem = path.stem
        matched_suffix = next(
            (
                candidate
                for candidate in suffixes
                if stem.lower().endswith(candidate.lower())
            ),
            None,
        )
        if matched_suffix is None:
            continue
        song_id = stem[: -len(matched_suffix)].strip()
        if not song_id:
            continue
        if song_id in result:
            raise ValueError(f"Duplicate song id {song_id!r} in {folder}")
        result[song_id] = path.resolve()
    return result


def discover_dataset(dataset_root: Path) -> tuple[list[SongFiles], list[dict]]:
    dataset_root = dataset_root.resolve()
    folder_specs = {
        "inst_audio": ("_inst", {".wav", ".flac"}),
        "vocal_audio": (("_vocals", "_vocal"), {".wav", ".flac"}),
        "vocal_mid": ("_vocal", {".mid", ".midi"}),
    }
    indices: dict[str, dict[str, Path]] = {}
    issues: list[dict] = []

    for folder_name, (suffix, extensions) in folder_specs.items():
        folder = dataset_root / folder_name
        if not folder.is_dir():
            raise FileNotFoundError(f"Missing dataset folder: {folder}")
        indices[folder_name] = index_by_suffix(
            folder,
            suffix=suffix,
            extensions=extensions,
        )

    all_ids = sorted(
        set().union(*(index.keys() for index in indices.values())),
        key=lambda song_id: (
            0 if song_id.isdigit() else 1,
            int(song_id) if song_id.isdigit() else song_id.lower(),
        ),
    )
    songs: list[SongFiles] = []
    for song_id in all_ids:
        missing = [
            folder_name
            for folder_name, index in indices.items()
            if song_id not in index
        ]
        if missing:
            issues.append(
                {
                    "song_id": song_id,
                    "type": "missing_pair",
                    "missing": missing,
                }
            )
            continue
        songs.append(
            SongFiles(
                song_id=song_id,
                inst_audio=indices["inst_audio"][song_id],
                vocal_audio=indices["vocal_audio"][song_id],
                vocal_midi=indices["vocal_mid"][song_id],
            )
        )
    return songs, issues


def absolute_meta(tracks: list[list], message_type: str) -> list[tuple[int, object]]:
    return [
        (event.absolute_tick, event.message)
        for track in tracks
        for event in track
        if event.message.type == message_type
    ]


def interpolate_with_extrapolation(
    values: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    if len(x) < 2:
        raise ValueError("At least two Grid anchors are required.")
    result = np.interp(values, x, y)

    before = values < x[0]
    if np.any(before):
        slope = (y[1] - y[0]) / (x[1] - x[0])
        result[before] = y[0] + (values[before] - x[0]) * slope

    after = values > x[-1]
    if np.any(after):
        slope = (y[-1] - y[-2]) / (x[-1] - x[-2])
        result[after] = y[-1] + (values[after] - x[-1]) * slope
    return result


def validate_audio_pair(song: SongFiles) -> tuple[object, object, list[dict]]:
    inst_info = sf.info(song.inst_audio)
    vocal_info = sf.info(song.vocal_audio)
    warnings: list[dict] = []

    if inst_info.samplerate != vocal_info.samplerate:
        raise ValueError(
            f"Audio sample-rate mismatch: {inst_info.samplerate} vs "
            f"{vocal_info.samplerate}"
        )
    if inst_info.frames != vocal_info.frames:
        raise ValueError(
            f"Audio length mismatch: {inst_info.frames} vs {vocal_info.frames}"
        )
    if inst_info.channels != vocal_info.channels:
        warnings.append(
            {
                "type": "audio_channel_mismatch",
                "inst_channels": inst_info.channels,
                "vocal_channels": vocal_info.channels,
            }
        )
    return inst_info, vocal_info, warnings


def midi_metadata(midi_path: Path) -> dict:
    midi = mido.MidiFile(midi_path)
    if midi.ticks_per_beat != PROJECT_PPQ:
        raise ValueError(
            f"MIDI PPQ is {midi.ticks_per_beat}; expected {PROJECT_PPQ}. "
            "Run the MIDI quantizer first."
        )

    tracks = [track_to_absolute_events(track) for track in midi.tracks]
    notes, unmatched_note_ons, unmatched_note_offs = pair_notes(tracks)
    if unmatched_note_ons or unmatched_note_offs:
        raise ValueError(
            "MIDI has unmatched note events: "
            f"{unmatched_note_ons} note_on, {unmatched_note_offs} note_off"
        )

    tempos = absolute_meta(tracks, "set_tempo")
    signatures = absolute_meta(tracks, "time_signature")
    if not tempos:
        raise ValueError("MIDI has no tempo event.")
    bpm = float(mido.tempo2bpm(tempos[0][1].tempo))
    tempo_events = [
        {
            "tick": int(tick),
            "tempo": int(message.tempo),
            "bpm": float(mido.tempo2bpm(message.tempo)),
        }
        for tick, message in sorted(tempos, key=lambda item: item[0])
    ]
    if signatures:
        time_signature = [
            signatures[0][1].numerator,
            signatures[0][1].denominator,
        ]
    else:
        time_signature = [DEFAULT_BEATS_PER_BAR, 4]
    if time_signature != [4, 4] or any(
        [msg.numerator, msg.denominator] != [4, 4] for _, msg in signatures
    ):
        raise ValueError("The melody model currently supports constant 4/4 meter only.")
    starts = {int(tick) for tick, msg in absolute_meta(tracks, "marker")
              if msg.text.startswith("WAV START")}
    if len(starts) > 1:
        raise ValueError("Conflicting WAV START markers in MIDI.")

    end_tick = max(
        (
            event.absolute_tick
            for track in tracks
            for event in track
            if event.message.type == "end_of_track"
        ),
        default=0,
    )
    return {
        "notes": notes,
        "bpm": bpm,
        "tempo_mode": (
            "tempo_map"
            if len({event["bpm"] for event in tempo_events}) > 1
            else "constant"
        ),
        "tempo_events": tempo_events,
        "time_signature": time_signature,
        "end_tick": end_tick,
        "audio_start_tick": next(iter(starts), None),
    }


def build_note_labels(
    *,
    song: SongFiles,
    grid: dict,
    midi_data: dict,
    sample_rate: int,
    num_samples: int,
) -> dict:
    beat_ticks = np.asarray(
        [beat["tick"] for beat in grid["beats"]],
        dtype=np.float64,
    )
    beat_samples = np.asarray(
        [beat["sample"] for beat in grid["beats"]],
        dtype=np.float64,
    )
    notes = midi_data["notes"]
    start_ticks = np.asarray(
        [note.original_start_tick for note in notes],
        dtype=np.float64,
    )
    end_ticks = np.asarray(
        [note.original_end_tick for note in notes],
        dtype=np.float64,
    )
    start_samples = np.rint(
        interpolate_with_extrapolation(start_ticks, beat_ticks, beat_samples)
    ).astype(np.int64)
    end_samples = np.rint(
        interpolate_with_extrapolation(end_ticks, beat_ticks, beat_samples)
    ).astype(np.int64)
    if grid.get("timeline"):
        from musical_timeline import samples_at_ticks
        start_samples = np.rint(samples_at_ticks(grid, start_ticks, sample_rate)).astype(np.int64)
        end_samples = np.rint(samples_at_ticks(grid, end_ticks, sample_rate)).astype(np.int64)

    labels: list[dict] = []
    for index, (note, start_sample, end_sample) in enumerate(
        zip(notes, start_samples, end_samples)
    ):
        fully_observable = start_sample >= 0 and end_sample <= num_samples
        intersects_audio = end_sample > 0 and start_sample < num_samples
        labels.append(
            {
                "id": index,
                "track_index": note.track_index,
                "channel": note.channel,
                "pitch": note.pitch,
                "velocity": note.start_event.message.velocity,
                "midi_time": {
                    "start_tick": note.original_start_tick,
                    "end_tick": note.original_end_tick,
                    "duration_ticks": (
                        note.original_end_tick - note.original_start_tick
                    ),
                    "start_beat": note.original_start_tick / PROJECT_PPQ,
                    "end_beat": note.original_end_tick / PROJECT_PPQ,
                },
                "audio_time": {
                    "start_sample": int(start_sample),
                    "end_sample": int(end_sample),
                    "duration_samples": int(end_sample - start_sample),
                    "start_sec": round(start_sample / sample_rate, 6),
                    "end_sec": round(end_sample / sample_rate, 6),
                },
                "training": {
                    "intersects_audio": bool(intersects_audio),
                    "fully_observable": bool(fully_observable),
                    "loss_start_sample": int(max(0, start_sample)),
                    "loss_end_sample": int(min(num_samples, end_sample)),
                },
            }
        )

    step_ticks = PROJECT_PPQ // 4
    shared_grid_ticks = PROJECT_PPQ // 12
    off_grid_onsets = sum(
        note.original_start_tick % step_ticks != 0 for note in notes
    )
    off_grid_ends = sum(
        note.original_end_tick % step_ticks != 0 for note in notes
    )
    off_shared_grid_onsets = sum(
        note.original_start_tick % shared_grid_ticks != 0 for note in notes
    )
    off_shared_grid_ends = sum(
        note.original_end_tick % shared_grid_ticks != 0 for note in notes
    )
    return {
        "schema_version": "melody_training_labels_v1.0",
        "song_id": song.song_id,
        "source": {
            "inst_audio": str(song.inst_audio),
            "vocal_audio": str(song.vocal_audio),
            "vocal_midi": str(song.vocal_midi),
            "sample_rate": sample_rate,
            "num_samples": num_samples,
            "ppq": PROJECT_PPQ,
        },
        "alignment": grid["midi_alignment"],
        "reference_music": {
            "bpm": midi_data["bpm"],
            "tempo_mode": midi_data["tempo_mode"],
            "tempo_events": midi_data["tempo_events"],
            "time_signature": midi_data["time_signature"],
            "midi_end_tick": midi_data["end_tick"],
        },
        "summary": {
            "num_notes": len(labels),
            "num_fully_observable_notes": sum(
                label["training"]["fully_observable"] for label in labels
            ),
            "num_partially_observable_notes": sum(
                label["training"]["intersects_audio"]
                and not label["training"]["fully_observable"]
                for label in labels
            ),
            "num_unobservable_notes": sum(
                not label["training"]["intersects_audio"] for label in labels
            ),
            "off_grid_onsets_1_16": off_grid_onsets,
            "off_grid_ends_1_16": off_grid_ends,
            "off_supported_grid_onsets_1_48": off_shared_grid_onsets,
            "off_supported_grid_ends_1_48": off_shared_grid_ends,
        },
        "notes": labels,
    }


def relative_path(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def write_json_if_changed(path: Path, payload: dict) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if not path.exists() or path.read_text(encoding="utf-8") != text:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


def process_song(
    song: SongFiles,
    *,
    dataset_root: Path,
    grid_path: Path,
    labels_path: Path,
    tracker: Audio2Beats | None,
    model_name: str,
    force: bool,
    beats_per_bar: int,
) -> tuple[dict, Audio2Beats | None]:
    _, vocal_info, warnings = validate_audio_pair(song)
    midi_data = midi_metadata(song.vocal_midi)

    if grid_path.exists() and not force:
        grid = json.loads(grid_path.read_text(encoding="utf-8"))
        if (grid["source"]["sample_rate"] != vocal_info.samplerate
                or grid["source"]["num_samples"] != vocal_info.frames):
            raise ValueError("Cached Grid rate/length does not match current audio; review alignment before rebuilding.")
    else:
        if tracker is None:
            tracker = Audio2Beats(
                checkpoint_path=model_name,
                device="cuda",
                float16=True,
                dbn=False,
            )
        audio = load_and_mix_audio([song.inst_audio])
        beats_sec, downbeats_sec = tracker(audio.signal, audio.sample_rate)
        grid = build_grid(
            audio,
            np.asarray(beats_sec, dtype=np.float64),
            np.asarray(downbeats_sec, dtype=np.float64),
            model_name=model_name,
            ppq=PROJECT_PPQ,
            forced_beats_per_bar=beats_per_bar,
        )
        grid_path.parent.mkdir(parents=True, exist_ok=True)
        grid_path.write_text(
            json.dumps(grid, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    from musical_timeline import canonicalize_grid
    # The corrected DAW tempo map is the authority for score time. Preserve
    # the existing WAV marker unless the reviewed MIDI explicitly supplies one.
    grid = canonicalize_grid(
        grid,
        [(e["tick"], e["tempo"]) for e in midi_data["tempo_events"]],
        audio_start_tick=midi_data["audio_start_tick"],
        minimum_end_tick=max((n.original_end_tick for n in midi_data["notes"]), default=0),
        source="reviewed_midi",
    )
    write_json_if_changed(grid_path, grid)
    labels = build_note_labels(
        song=song,
        grid=grid,
        midi_data=midi_data,
        sample_rate=vocal_info.samplerate,
        num_samples=vocal_info.frames,
    )
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    write_json_if_changed(labels_path, labels)

    grid_regions = grid["tempo"].get("regions")
    if grid_regions:
        midi_tempos_by_tick = {
            int(event["tick"]): float(event["bpm"])
            for event in midi_data["tempo_events"]
        }
        disagreements = []
        for region in grid_regions:
            tick = int(region["start_tick"])
            midi_bpm = midi_tempos_by_tick.get(tick)
            if midi_bpm is None or abs(midi_bpm - float(region["bpm"])) > 1.0:
                disagreements.append(
                    {
                        "tick": tick,
                        "grid_bpm": float(region["bpm"]),
                        "midi_bpm": midi_bpm,
                    }
                )
        if disagreements:
            warnings.append(
                {
                    "type": "tempo_map_disagreement",
                    "regions": disagreements,
                }
            )
    else:
        grid_bpm = float(
            grid["tempo"].get(
                "global_regression_bpm",
                grid["tempo"]["bpm"],
            )
        )
        bpm_error = abs(grid_bpm - midi_data["bpm"])
        if bpm_error > 1.0:
            warnings.append(
                {
                    "type": "tempo_disagreement",
                    "grid_bpm": grid_bpm,
                    "midi_bpm": midi_data["bpm"],
                }
            )
    if grid["midi_alignment"]["grid_end_tick"] != midi_data["end_tick"]:
        warnings.append(
            {
                "type": "end_tick_disagreement",
                "grid_end_tick": grid["midi_alignment"]["grid_end_tick"],
                "midi_end_tick": midi_data["end_tick"],
            }
        )
    if labels["summary"]["off_supported_grid_onsets_1_48"]:
        warnings.append(
            {
                "type": "off_supported_rhythm_grid_midi_onsets",
                "count": labels["summary"]["off_supported_grid_onsets_1_48"],
            }
        )
    velocity_values = [
        note.start_event.message.velocity for note in midi_data["notes"]
    ]
    non_normalized_velocities = sum(
        velocity != 100 for velocity in velocity_values
    )
    low_velocity_notes = sum(velocity < 40 for velocity in velocity_values)
    if non_normalized_velocities:
        warnings.append(
            {
                "type": "velocity_not_normalized",
                "count": non_normalized_velocities,
                "expected_velocity": 100,
                "low_velocity_notes_below_40": low_velocity_notes,
            }
        )

    sorted_notes = sorted(
        midi_data["notes"],
        key=lambda note: (
            note.original_start_tick,
            note.original_end_tick,
        ),
    )
    short_gap_threshold = PROJECT_PPQ // 4
    short_gaps = sum(
        0 < following.original_start_tick - current.original_end_tick
        <= short_gap_threshold
        for current, following in zip(sorted_notes[:-1], sorted_notes[1:])
    )
    if short_gaps:
        warnings.append(
            {
                "type": "short_midi_gaps",
                "count": short_gaps,
                "maximum_gap_ticks": short_gap_threshold,
            }
        )

    manifest_entry = {
        "song_id": song.song_id,
        "inst_audio": relative_path(song.inst_audio, dataset_root),
        "vocal_audio": relative_path(song.vocal_audio, dataset_root),
        "vocal_midi": relative_path(song.vocal_midi, dataset_root),
        "grid": relative_path(grid_path, dataset_root),
        "labels": relative_path(labels_path, dataset_root),
        "sample_rate": vocal_info.samplerate,
        "num_samples": vocal_info.frames,
        "duration_sec": vocal_info.duration,
        "ppq": PROJECT_PPQ,
        "bpm": midi_data["bpm"],
        "time_signature": midi_data["time_signature"],
        "num_notes": labels["summary"]["num_notes"],
        "warnings": warnings,
    }
    return manifest_entry, tracker


def prepare_dataset(
    dataset_root: Path,
    *,
    model_name: str = "final0",
    force: bool = False,
    beats_per_bar: int = DEFAULT_BEATS_PER_BAR,
) -> dict:
    dataset_root = dataset_root.resolve()
    songs, discovery_issues = discover_dataset(dataset_root)
    grid_dir = dataset_root / "grid_cache"
    labels_dir = dataset_root / "note_cache"
    tracker: Audio2Beats | None = None
    manifest: list[dict] = []
    errors: list[dict] = []

    for song in songs:
        print(f"[prepare] {song.song_id}")
        try:
            entry, tracker = process_song(
                song,
                dataset_root=dataset_root,
                grid_path=grid_dir / f"{song.song_id}_grid.json",
                labels_path=labels_dir / f"{song.song_id}_notes.json",
                tracker=tracker,
                model_name=model_name,
                force=force,
                beats_per_bar=beats_per_bar,
            )
            manifest.append(entry)
        except Exception as error:
            errors.append(
                {
                    "song_id": song.song_id,
                    "type": type(error).__name__,
                    "message": str(error),
                }
            )
            print(f"[error] {song.song_id}: {error}")

    (dataset_root / "manifest.jsonl").write_text(
        "".join(
            json.dumps(entry, ensure_ascii=False) + "\n" for entry in manifest
        ),
        encoding="utf-8",
    )
    report = {
        "schema_version": "dataset_prepare_report_v1.0",
        "dataset_root": str(dataset_root),
        "project_ppq": PROJECT_PPQ,
        "num_discovered_songs": len(songs),
        "num_prepared_songs": len(manifest),
        "num_errors": len(errors),
        "discovery_issues": discovery_issues,
        "errors": errors,
        "songs": manifest,
    }
    (dataset_root / "prepare_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch-build Grid and sample-aligned melody labels."
    )
    parser.add_argument(
        "dataset_root",
        type=Path,
        nargs="?",
        default=Path("dataset/melody_dataset"),
    )
    parser.add_argument("--model", default="final0")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--beats-per-bar",
        type=int,
        default=DEFAULT_BEATS_PER_BAR,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = prepare_dataset(
        args.dataset_root,
        model_name=args.model,
        force=args.force,
        beats_per_bar=args.beats_per_bar,
    )
    print(
        f"Prepared {report['num_prepared_songs']} / "
        f"{report['num_discovered_songs']} songs; "
        f"{report['num_errors']} errors."
    )
    if report["num_errors"] or report["discovery_issues"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
