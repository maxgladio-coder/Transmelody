from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from transmelody.models.melody_transformer import (
    STEPS_PER_BAR,
    FeatureConfig,
    build_song_cache,
    MelodyTransformer,
    ModelConfig,
    read_manifest,
)
from transmelody.inference.melody_inference import predict_outputs, decode_prediction, export_prediction
from transmelody.evaluation.melody_evaluation import note_counts, summarize_note_counts, file_digest, note_artifact_counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run melody Transformer inference.")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--dataset", type=Path, default=Path("dataset/melody_dataset"))
    parser.add_argument("--output", type=Path, default=Path("output/melody_transformer/predictions"))
    parser.add_argument("--chunk-bars", type=int, default=None)
    parser.add_argument("--triplet-confidence", type=float, default=None)
    parser.add_argument("--output-grid", choices=("auto", "1/16"), default="auto")
    parser.add_argument("--song-id", help="Predict only this dataset song.")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--pitch-context", type=Path, help="Optional separate pitch-only context checkpoint.")
    parser.add_argument('--no-pitch-fusion',action='store_true',help='Explicitly bypass the active fusion policy for this prediction.')
    parser.add_argument("--joint-pitch-weight", type=float,
        help="Opt-in joint pitch/boundary sequence decoding; weight selected on development songs.")
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = MelodyTransformer(ModelConfig(**checkpoint["model_config"])).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    from transmelody.inference.fusion_runtime import PitchFusionRuntime
    fusion = PitchFusionRuntime(args.checkpoint,device,disabled=args.no_pitch_fusion)
    if fusion.policy and args.pitch_context:
        raise ValueError('Use --no-pitch-fusion when explicitly evaluating a separate pitch-context refiner.')
    refiner = None
    if args.pitch_context:
        from transmelody.models.pitch_context import load_refiner
        refiner, refiner_checkpoint = load_refiner(args.pitch_context, device, file_digest(args.checkpoint))

    cache_dir = args.cache_dir or args.checkpoint.parent / "feature_cache"
    manifest = {song["song_id"]: song for song in read_manifest(args.dataset)}
    if args.song_id:
        if args.song_id not in manifest:
            raise ValueError(f"Song {args.song_id} is not in the prepared dataset.")
        manifest = {args.song_id: manifest[args.song_id]}
    feature_config = FeatureConfig(**checkpoint["feature_config"])
    chunk_steps = (args.chunk_bars or checkpoint.get("training_args", {}).get("bars_per_chunk", 8)) * STEPS_PER_BAR
    triplet_threshold = (args.triplet_confidence if args.triplet_confidence is not None
                         else checkpoint.get("training_args", {}).get("triplet_confidence", 0.60))
    args.output.mkdir(parents=True, exist_ok=True)

    for song_id, song in manifest.items():
        cache_path = build_song_cache(args.dataset.resolve(), song, cache_dir, feature_config, device)
        cache = torch.load(cache_path, map_location="cpu", weights_only=False)
        song_id = cache["song_id"]
        features = cache["features"].float()
        outputs = predict_outputs(model, features, device, chunk_steps)
        predicted_notes, rhythm_modes = decode_prediction(outputs, features, feature_config,
            output_grid=args.output_grid, triplet_threshold=triplet_threshold,
            joint_pitch_weight=args.joint_pitch_weight)
        predicted_notes, fusion_report = fusion.apply(predicted_notes,outputs,cache['source_audio'],cache['grid'])
        pitch_context_report = None
        refined_pitch_scores = np.empty((0, 128), dtype=np.float32)
        if refiner is not None:
            from transmelody.models.pitch_context import note_inputs, refine_logits, replace_note_pitches, tonality_report
            raw_notes = predicted_notes
            logits, note_features, starts, durations = note_inputs(outputs, raw_notes)
            corrected = refine_logits(refiner, logits, note_features, device)
            predicted_notes = replace_note_pitches(raw_notes, corrected)
            refined_pitch_scores = corrected.softmax(-1).numpy()
            pitch_context_report = {"checkpoint": str(args.pitch_context.resolve()),
                "epoch": refiner_checkpoint['epoch'], "boundaries_unchanged": True,
                "metrics_before": summarize_note_counts(note_counts(raw_notes, cache)),
                "changes": [{"note_index_0based": i, "start_step": a.start_step,
                    "from_pitch": a.pitch, "to_pitch": b.pitch}
                    for i, (a, b) in enumerate(zip(raw_notes, predicted_notes)) if a.pitch != b.pitch],
                "tonality_from_raw_acoustic_scores": tonality_report(logits, starts, durations,
                    total_bars=(len(features) + 47) // 48)}
        grid = cache["grid"]
        output_path = args.output / f"{song_id}_predicted.mid"
        export_prediction(output_path, grid, predicted_notes, rhythm_modes)
        report = {"song_id": song_id, "checkpoint": str(args.checkpoint.resolve()),
            "pitch_context": pitch_context_report,
            "pitch_fusion": fusion_report,
            "decoder": "joint_pitch_boundary_sequence" if args.joint_pitch_weight is not None else
                "complete_note_segments" if "note_event_onset" in outputs else
                "joint_boundary_states" if "boundary_state" in outputs else
                "acoustic_event_peaks" if "musical_event_onset" in outputs else "legacy_segments",
            "joint_pitch_weight": args.joint_pitch_weight,
            "checkpoint_epoch": checkpoint.get("epoch"), "schema_version": checkpoint.get("schema_version"),
            "note_count": len(predicted_notes), "ppq": 480,
            "bpm_start": grid["tempo"]["bpm"], "tempo_events": grid["timeline"]["tempo_events"],
            "audio_start_tick": grid["midi_alignment"]["audio_start_tick"],
            "triplet_bars_1based": (rhythm_modes.nonzero().flatten() + 1).tolist(),
            "validation_song": song_id in checkpoint.get("run_metadata", {}).get("validation_ids", []),
            "metrics_against_score": summarize_note_counts(note_counts(predicted_notes, cache)),
            "artifact_metrics_against_score": note_artifact_counts(predicted_notes,cache),
            "metric_note": "Score matching at 50 ms onset tolerance, not human-verified acoustic onset accuracy. "
                "Dataset inference uses the existing reviewed MIDI tempo map and WAV START."}
        if "musical_event_onset" in outputs:
            legacy_notes, _ = decode_prediction({k: v for k, v in outputs.items() if k not in ("musical_event_onset", "note_event_onset")},
                features, feature_config, output_grid=args.output_grid, triplet_threshold=triplet_threshold)
            report["same_outputs_legacy_decoder"] = summarize_note_counts(note_counts(legacy_notes, cache))
            report["comparison_note"] = "Same new model outputs, different decoder only. Not a comparison with the old trained model."
        if "boundary_state" in outputs:
            if args.joint_pitch_weight is not None:
                local_notes, _ = decode_prediction(outputs, features, feature_config,
                    output_grid=args.output_grid, triplet_threshold=triplet_threshold)
                report["same_outputs_local_boundary_decoder"] = summarize_note_counts(note_counts(local_notes, cache))
            event_notes, _ = decode_prediction({k: v for k, v in outputs.items() if k not in ("boundary_state", "note_event_onset")},
                features, feature_config, output_grid=args.output_grid, triplet_threshold=triplet_threshold)
            report["same_outputs_v2_event_decoder"] = summarize_note_counts(note_counts(event_notes, cache))
            report["boundary_state_classes"] = ["rest", "hold", "start", "end"]
            report["boundary_note"] = ("Onset/offset/continuation NPZ scores are auxiliary evidence; "
                "v3 note boundaries use the mutually exclusive boundary_state_probability at legal grid nodes.")
            if args.joint_pitch_weight is not None:
                report["boundary_note"] = ("Global joint pitch/boundary path decoding on legal grid nodes. "
                    "Local boundary argmax no longer unconditionally fixes the note spans. "
                    "No raw-F0 or reference-pitch input is used to produce the MIDI.")
            if "note_event_onset" in outputs and args.joint_pitch_weight is None:
                report["boundary_note"] = ("Complete-note semi-Markov decoding jointly scores pitch, boundary, "
                    "and learned duration. The final duration bin is unbounded; no forced maximum note length.")
        output_path.with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        np.savez_compressed(
            args.output / f"{song_id}_prediction.npz",
            onset_probability=torch.sigmoid(outputs["onset"]).numpy(),
            offset_probability=torch.sigmoid(outputs["offset"]).numpy(),
            activity_probability=torch.sigmoid(outputs["activity"]).numpy(),
            continuation_probability=torch.sigmoid(
                outputs["continuation"]
            ).numpy(),
            rhythm_logits=outputs["rhythm"].numpy(),
            rhythm_modes=rhythm_modes.numpy(),
            refined_note_pitch_probability=refined_pitch_scores,
            boundary_state_probability=(outputs["boundary_state"].softmax(-1).numpy()
                if "boundary_state" in outputs else np.empty((0, 4), dtype=np.float32)),
            pronunciation_probability=(torch.sigmoid(outputs["pronunciation"]).numpy()
                if "pronunciation" in outputs else np.zeros(len(features), dtype=np.float32)),
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
        print(f"[prediction] {manifest[song_id]['vocal_audio']} -> {output_path}; notes={len(predicted_notes)}; "
            f"BPM(start)={grid['tempo']['bpm']}; tempo sections={len(grid['timeline']['tempo_events'])}")


if __name__ == "__main__":
    main()
