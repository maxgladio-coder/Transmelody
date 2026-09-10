from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from beat_this.inference import Audio2Beats
from transmelody.grid.grid_initializer import build_grid, load_and_mix_audio
from transmelody.models.melody_transformer import (
    RHYTHM_NAMES,
    RHYTHM_STRAIGHT,
    RHYTHM_TRIPLET,
    STEPS_PER_BAR,
    FeatureConfig,
    MelodyTransformer,
    ModelConfig,
    extract_grid_features,
)
from transmelody.config import DEFAULT_BEATS_PER_BAR, PROJECT_PPQ
from transmelody.grid.tempo_map import (
    build_tempo_segments,
)
from transmelody.grid.tempo_region_grid import multiply_grid_tempo
from transmelody.inference.melody_inference import (predict_outputs, decode_prediction, export_prediction,
    synchronize_prediction_grid, rhythm_confidences)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate review MIDI from paired, unlabeled test stems."
    )
    parser.add_argument("--input", type=Path, default=Path("test audio"))
    parser.add_argument("--output", type=Path, default=Path("test_output"))
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("output/melody_transformer/final.pt"),
    )
    parser.add_argument("--model", default="final0")
    parser.add_argument("--beats-per-bar", type=int, default=DEFAULT_BEATS_PER_BAR)
    parser.add_argument("--chunk-bars", type=int, default=None, help="Defaults to checkpoint training context (8 bars for legacy models).")
    parser.add_argument("--triplet-confidence", type=float, default=None)
    parser.add_argument(
        "--output-grid",
        choices=("auto", "1/16"),
        default="auto",
        help="auto detects straight-sixteenth or eighth-triplet grid per bar.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Process only the first N naturally sorted stem pairs.",
    )
    parser.add_argument(
        "--song-id",
        help="Process only this song ID, including a pair already moved into the dataset.",
    )
    parser.add_argument("--force-grid", action="store_true")
    parser.add_argument('--no-pitch-fusion',action='store_true',help='Explicitly bypass the active fusion policy for this prediction.')
    parser.add_argument(
        "--bpm",
        type=float,
        help="Confirm a constant BPM and rebuild one shared feature/export timeline; retain detection anchors for diagnostics.",
    )
    parser.add_argument(
        "--tempo-map-max-error-ms",
        type=float,
        default=25.0,
        help="Maximum beat-anchor error within each constant-tempo section.",
    )
    return parser.parse_args()


def discover_pairs(root: Path) -> list[tuple[str, Path, Path]]:
    if (root / "test_inst").is_dir():
        inst_dir = root / "test_inst"
        vocal_dir = root / "test_vocal"
    else:
        inst_dir = root / "inst_audio"
        vocal_dir = root / "vocal_audio"
    inst = {
        path.name.removesuffix("_inst.wav"): path
        for path in inst_dir.glob("*_inst.wav")
    }
    vocal = {
        path.name.removesuffix("_vocal.wav"): path
        for path in vocal_dir.glob("*_vocal.wav")
    }
    if set(inst) != set(vocal):
        raise ValueError(
            f"Stem pairing mismatch; inst-only={sorted(set(inst)-set(vocal))}, "
            f"vocal-only={sorted(set(vocal)-set(inst))}"
        )
    if not inst:
        raise ValueError(f"No paired WAV stems found under {root}.")
    def song_key(song_id: str) -> tuple[int, int | str]:
        return (
            0 if song_id.isdigit() else 1,
            int(song_id) if song_id.isdigit() else song_id.casefold(),
        )

    return [
        (song_id, inst[song_id], vocal[song_id])
        for song_id in sorted(inst, key=song_key)
    ]


def main() -> None:
    args = parse_args()
    if args.beats_per_bar != 4:
        raise ValueError("The melody model currently supports 4/4 meter only.")
    if args.triplet_confidence is not None and not 0.5 <= args.triplet_confidence <= 1.0:
        raise ValueError("Triplet confidence must be between 0.5 and 1.")
    input_root = args.input.resolve()
    output_root = args.output.resolve()
    grid_dir = output_root / "grid_cache"
    diagnostic_dir = output_root / "diagnostics"
    grid_dir.mkdir(parents=True, exist_ok=True)
    diagnostic_dir.mkdir(parents=True, exist_ok=True)

    pairs = discover_pairs(input_root)
    if args.song_id is not None:
        pairs = [pair for pair in pairs if pair[0] == args.song_id]
        if not pairs:
            raise ValueError(f"Song ID {args.song_id!r} was not found under {input_root}.")
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be at least 1.")
        pairs = pairs[: args.limit]
    for song_id, inst_path, vocal_path in pairs:
        inst_info = sf.info(inst_path)
        vocal_info = sf.info(vocal_path)
        if (
            inst_info.samplerate != vocal_info.samplerate
            or inst_info.frames != vocal_info.frames
        ):
            raise ValueError(f"Audio mismatch for {song_id}.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    feature_config = FeatureConfig(**checkpoint["feature_config"])
    model = MelodyTransformer(ModelConfig(**checkpoint["model_config"])).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    from transmelody.inference.fusion_runtime import PitchFusionRuntime
    fusion = PitchFusionRuntime(args.checkpoint,device,disabled=args.no_pitch_fusion)
    chunk_bars = args.chunk_bars or checkpoint.get("training_args", {}).get("bars_per_chunk", 8)
    triplet_threshold = (args.triplet_confidence if args.triplet_confidence is not None
                         else checkpoint.get("training_args", {}).get("triplet_confidence", 0.60))
    tracker: Audio2Beats | None = None
    report = []

    for song_id, inst_path, vocal_path in pairs:
        print(f"[grid] {song_id}", flush=True)
        grid_path = grid_dir / f"{song_id}_grid.json"
        grid_was_rebuilt = not grid_path.exists() or args.force_grid
        source_signature = {p.name: [p.stat().st_size, p.stat().st_mtime_ns] for p in (inst_path, vocal_path)}
        if not grid_was_rebuilt:
            grid = json.loads(grid_path.read_text(encoding="utf-8"))
            previous_signature = grid.get("source", {}).get("file_signature")
            info = sf.info(inst_path)
            if ((previous_signature is not None and previous_signature != source_signature)
                or grid["source"]["sample_rate"] != info.samplerate
                or grid["source"]["num_samples"] != info.frames):
                raise ValueError(f"Source audio changed for {song_id}; use --force-grid after reviewing alignment.")
        if grid_was_rebuilt:
            if tracker is None:
                tracker = Audio2Beats(
                    checkpoint_path=args.model,
                    device=str(device),
                    float16=device.type == "cuda",
                    dbn=False,
                )
            audio = load_and_mix_audio([inst_path])
            beats_sec, downbeats_sec = tracker(audio.signal, audio.sample_rate)
            grid = build_grid(
                audio,
                np.asarray(beats_sec, dtype=np.float64),
                np.asarray(downbeats_sec, dtype=np.float64),
                model_name=args.model,
                ppq=PROJECT_PPQ,
                forced_beats_per_bar=args.beats_per_bar,
            )
        if args.bpm is not None:
            if args.bpm <= 0:
                raise ValueError("--bpm must be positive.")
            detected_bpm = float(grid["tempo"]["bpm"])
            # A previous export timeline must not shadow a new user override.
            grid.pop("timeline", None)
            ratio = args.bpm / detected_bpm
            integer_factor = int(round(ratio))
            if integer_factor >= 2 and abs(ratio - integer_factor) <= 0.05:
                grid = multiply_grid_tempo(
                    grid,
                    integer_factor,
                    target_bpm=float(args.bpm),
                )
            else:
                grid["tempo"]["bpm"] = float(args.bpm)
                grid["tempo"]["suggested_project_bpm"] = float(args.bpm)
                grid["tempo"]["manual_override_bpm"] = float(args.bpm)
                grid["tempo"]["mode"] = "manual_project_bpm_with_beat_anchors"
                grid["tempo"].pop("regions", None)
            grid["tempo"]["manual_override_bpm"] = float(args.bpm)
        grid = synchronize_prediction_grid(grid, args.tempo_map_max_error_ms)
        grid["source"]["file_signature"] = source_signature
        grid_path.write_text(json.dumps(grid, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        print(f"[predict] {song_id}", flush=True)
        features, _, _ = extract_grid_features(
            vocal_path,
            grid,
            feature_config,
            device,
        )
        outputs = predict_outputs(
            model,
            features,
            device,
            chunk_bars * STEPS_PER_BAR,
        )
        predicted_notes, rhythm_modes = decode_prediction(outputs, features, feature_config,
            output_grid=args.output_grid, triplet_threshold=triplet_threshold)
        predicted_notes, fusion_report = fusion.apply(predicted_notes,outputs,vocal_path,grid)
        midi_path = output_root / f"{song_id}_predicted.mid"
        grid_end_tick = int(grid["midi_alignment"]["grid_end_tick"])
        audio_start_tick = int(grid["midi_alignment"]["audio_start_tick"])
        tempo_segments = build_tempo_segments(
            grid,
            max_anchor_error_ms=args.tempo_map_max_error_ms,
        )
        tempo_map_path = export_prediction(midi_path, grid, predicted_notes, rhythm_modes)
        np.savez_compressed(
            diagnostic_dir / f"{song_id}_prediction.npz",
            pitch_logits=outputs["pitch"].numpy(),
            onset_probability=torch.sigmoid(outputs["onset"]).numpy(),
            offset_probability=torch.sigmoid(outputs["offset"]).numpy(),
            activity_probability=torch.sigmoid(outputs["activity"]).numpy(),
            continuation_probability=torch.sigmoid(
                outputs["continuation"]
            ).numpy(),
            duration_logits=outputs["duration"].numpy(),
            boundary_state_probability=(outputs['boundary_state'].softmax(-1).numpy()
                if 'boundary_state' in outputs else np.empty((0, 4), dtype=np.float32)),
            articulation_probability=(
                torch.sigmoid(outputs["articulation"]).numpy()
                if "articulation" in outputs
                else np.zeros(len(features), dtype=np.float32)
            ),
            pitch_change_probability=(
                torch.sigmoid(outputs["pitch_change"]).numpy()
                if "pitch_change" in outputs
                else np.zeros(len(features), dtype=np.float32)
            ),
            rhythm_logits=outputs["rhythm"].numpy(),
            rhythm_modes=rhythm_modes.numpy(),
            pronunciation_probability=(torch.sigmoid(outputs["pronunciation"]).numpy()
                if "pronunciation" in outputs else np.zeros(len(features), dtype=np.float32)),
            note_start_steps=np.asarray(
                [note.start_step for note in predicted_notes],
                dtype=np.int32,
            ),
            note_end_steps=np.asarray(
                [note.end_step for note in predicted_notes],
                dtype=np.int32,
            ),
            note_pitches=np.asarray(
                [note.pitch for note in predicted_notes],
                dtype=np.int16,
            ),
        )
        report.append(
            {
                "song_id": song_id,
                "instrumental": str(inst_path),
                "vocal": str(vocal_path),
                "midi": str(midi_path),
                "tempo_map_midi": str(tempo_map_path),
                "bpm": float(grid["tempo"]["bpm"]),
                "audio_start_tick": audio_start_tick,
                "tempo_resolution": grid["tempo"].get(
                    "half_tempo_evidence"
                ),
                "bars": grid_end_tick / (PROJECT_PPQ * args.beats_per_bar),
                "num_predicted_notes": len(predicted_notes),
                "num_single_sixteenth_notes": sum(
                    note.end_step - note.start_step == 3
                    for note in predicted_notes
                ),
                "num_single_triplet_eighth_notes": sum(
                    note.end_step - note.start_step == 4
                    for note in predicted_notes
                ),
                "output_grid": args.output_grid,
                "triplet_confidence_threshold": triplet_threshold,
                "triplet_probability_by_bar": rhythm_confidences(outputs),
                "timeline": grid["timeline"],
                "checkpoint": str(args.checkpoint.resolve()),
                "pitch_fusion": fusion_report,
                "pitch_logits_source": "base model before optional pitch-only fusion; note_pitches are final",
                "model_schema": checkpoint.get("schema_version"),
                "decoder": "complete_note_segments" if 'note_event_onset' in outputs else
                    "joint_boundary_states" if 'boundary_state' in outputs else "legacy_or_event_peaks",
                "tempo_sections": [
                    {
                        "start_bar": segment.start_bar,
                        "end_bar": segment.end_bar,
                        "bpm": segment.bpm,
                        "max_anchor_error_ms": segment.max_anchor_error_ms,
                    }
                    for segment in tempo_segments
                ],
                "rhythm_mode_by_bar": [
                    RHYTHM_NAMES[int(mode)]
                    for mode in rhythm_modes.tolist()
                ],
                "num_straight_bars": int(
                    (rhythm_modes == RHYTHM_STRAIGHT).sum()
                ),
                "num_triplet_bars": int(
                    (rhythm_modes == RHYTHM_TRIPLET).sum()
                ),
            }
        )
        print(
            f"[done] {midi_path.name}: {grid['tempo']['bpm']:.2f} BPM, "
            f"{report[-1]['bars']:.0f} bars, "
            f"rhythm={report[-1]['num_straight_bars']} straight/"
            f"{report[-1]['num_triplet_bars']} triplet",
            flush=True,
        )

    (output_root / "prediction_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Generated {len(report)} MIDI files in {output_root}.")


if __name__ == "__main__":
    main()
