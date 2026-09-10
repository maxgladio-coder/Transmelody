from __future__ import annotations
from transmelody.paths import PACKAGE_ROOT, PROJECT_ROOT

import argparse
import json
import random
import os
import shutil
from datetime import datetime, timezone
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, WeightedRandomSampler

from transmelody.models.melody_transformer import (
    NUM_RHYTHM_CLASSES,
    STEPS_PER_BAR,
    FeatureConfig,
    GridChunkDataset,
    MelodyTransformer,
    ModelConfig,
    build_feature_caches,
    read_manifest,
)
from transmelody.models.articulation_encoder import clear_articulation_encoder_cache
from transmelody.inference.melody_inference import predict_outputs, decode_prediction
from transmelody.evaluation.melody_evaluation import (song_split, note_counts, summarize_note_counts, file_digest,
    semitone_aba_counts, passes_pitch_aba_guard, note_artifact_counts, passes_artifact_guard)
from transmelody.models.musical_boundary import boundary_targets
from transmelody.models.note_event_model import note_event_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the grid-aware melody Transformer.")
    parser.add_argument("dataset", type=Path, nargs="?", default=Path("dataset/melody_dataset"))
    parser.add_argument("--output", type=Path, default=Path("output/melody_transformer"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--bars-per-chunk", type=int, default=8)
    parser.add_argument("--stride-bars", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--split", type=Path, help="Persistent whole-song validation split JSON.")
    parser.add_argument("--validation-ids", help="Comma-separated IDs, used when creating a new split.")
    parser.add_argument("--validate-every", type=int, default=5)
    parser.add_argument("--triplet-confidence", type=float, default=0.60)
    parser.add_argument("--no-pronunciation-context", action="store_true", help="Ablation: disable the learned speech-span branch.")
    parser.add_argument("--no-sequence-position", action="store_true", help="Ablation: keep legacy bar-only positions.")
    parser.add_argument("--no-publish", action="store_true", help="Keep this experimental run out of the workflow's final.pt.")
    parser.add_argument("--no-musical-event-context", action="store_true", help="Ablation: legacy score onset decoder without local acoustic attention.")
    parser.add_argument("--musical-event-context", action="store_true", help="Opt into experimental acoustic musical-onset learning and balanced triplet sampling.")
    parser.add_argument("--cache-dir", type=Path, help="Reuse feature caches across isolated experiments.")
    parser.add_argument("--note-event-training", action="store_true",
        help="Train complete-note contrasts and use pitch/boundary/duration semi-Markov decoding.")
    parser.add_argument("--init-checkpoint", type=Path,
        help="Initialize weights only; unchanged initial model competes as epoch0 on the same validation set.")
    parser.add_argument("--semitone-training", action="store_true",
        help="Supervised note-balanced adjacent-pitch/ABA contrasts and explicit no-note pitch supervision.")
    parser.add_argument("--pitch-head-only", action="store_true",
        help="Freeze the initialized acoustic/boundary network and adapt only the pitch classifier.")
    parser.add_argument("--semitone-guard", action="store_true",
        help="Reject updates that regress pitch-onset F1 or any development song's semitone ABA recall.")
    parser.add_argument('--artifact-training', action='store_true',
        help='Train false head/tail split and reviewed-rest false-note contrasts.')
    parser.add_argument('--artifact-guard',action='store_true',
        help='Also protect short-note matches and reject increased split/reference-rest artifacts.')
    parser.add_argument('--note-activity-weight', type=float,
        help='Soft melody-activity score; omitted inherits the initializer, otherwise defaults to zero.')
    return parser.parse_args()


def binary_precision_recall(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
) -> tuple[float, float]:
    predicted = (torch.sigmoid(logits) >= 0.5) & valid
    truth = (target >= 0.5) & valid
    true_positive = (predicted & truth).sum().item()
    return (
        true_positive / max(1, predicted.sum().item()),
        true_positive / max(1, truth.sum().item()),
    )


def masked_binary_loss(logits, targets, mask, pos_weight=None):
    if not mask.any():
        return logits.sum() * 0.0
    return F.binary_cross_entropy_with_logits(logits[mask], targets[mask], pos_weight=pos_weight)


def supervised_pitch_loss(logits, targets, valid, *, supervise_rest=False):
    """Active-note CE plus separately balanced no-note CE when requested.

    Class zero is consumed by the complete-note decoder, not the activity head.
    Long instrumental sections must not outweigh all sung notes.
    """
    active = valid & (targets > 0)
    rest = valid & (targets == 0)
    loss = F.cross_entropy(logits[active], targets[active]) if active.any() else logits.sum() * 0.
    if supervise_rest and rest.any():
        loss = loss + .25 * F.cross_entropy(logits[rest], targets[rest])
    return loss


def freeze_except_pitch_head(model):
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith('pitch_head.'))


def metrics(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
) -> dict[str, float]:
    valid = batch["valid"] & batch["loss_mask"]
    boundary_valid = batch["valid"] & batch["boundary_mask"]
    event_mask = batch["duration_mask"] & valid
    active_mask = (batch["activity"] >= 0.5) & valid
    pitch_predicted = outputs["pitch"].argmax(dim=-1)
    duration_predicted = outputs["duration"].argmax(dim=-1)
    pitch_accuracy = (
        (pitch_predicted[active_mask] == batch["pitch"][active_mask])
        .float()
        .mean()
        .item()
        if active_mask.any()
        else 0.0
    )
    duration_accuracy = (
        (duration_predicted[event_mask] == batch["duration"][event_mask])
        .float()
        .mean()
        .item()
        if event_mask.any()
        else 0.0
    )
    onset_precision, onset_recall = binary_precision_recall(
        outputs["onset"],
        batch["onset"],
        boundary_valid,
    )
    offset_precision, offset_recall = binary_precision_recall(
        outputs["offset"],
        batch["offset"],
        boundary_valid,
    )
    continuation_precision, continuation_recall = binary_precision_recall(
        outputs["continuation"],
        batch["continuation"],
        batch["valid"] & batch["continuation_mask"],
    )
    pitch_change_precision, pitch_change_recall = binary_precision_recall(
        outputs["pitch_change"],
        batch["pitch_change"],
        boundary_valid,
    )
    articulation_valid = batch["articulation_mask"] & valid
    articulation_precision, articulation_recall = binary_precision_recall(
        outputs["articulation"],
        batch["articulation_boundary"],
        articulation_valid,
    )
    activity_predicted = (torch.sigmoid(outputs["activity"]) >= 0.5) & valid
    activity_true = (batch["activity"] >= 0.5) & valid
    activity_accuracy = (
        (activity_predicted[valid] == activity_true[valid])
        .float()
        .mean()
        .item()
    )
    rhythm_mask = batch["rhythm_mask"] & valid
    rhythm_accuracy = (
        (
            outputs["rhythm"].argmax(dim=-1)[rhythm_mask]
            == batch["rhythm_mode"][rhythm_mask]
        )
        .float()
        .mean()
        .item()
        if rhythm_mask.any()
        else 0.0
    )
    return {
        "event_pitch_accuracy": pitch_accuracy,
        "duration_accuracy": duration_accuracy,
        "activity_accuracy": activity_accuracy,
        "rhythm_accuracy": rhythm_accuracy,
        "onset_precision": onset_precision,
        "onset_recall": onset_recall,
        "offset_precision": offset_precision,
        "offset_recall": offset_recall,
        "continuation_precision": continuation_precision,
        "continuation_recall": continuation_recall,
        "pitch_change_precision": pitch_change_precision,
        "pitch_change_recall": pitch_change_recall,
        "articulation_precision": articulation_precision,
        "articulation_recall": articulation_recall,
    }


def main() -> None:
    args = parse_args()
    initial = torch.load(args.init_checkpoint,map_location='cpu',weights_only=False) if args.init_checkpoint else None
    if args.note_activity_weight is None:
        args.note_activity_weight = initial['model_config'].get('note_activity_weight',0.) if initial else 0.
    if not np.isfinite(args.note_activity_weight) or args.note_activity_weight < 0:
        raise ValueError('Note activity weight must be finite and nonnegative.')
    if (args.artifact_training or args.note_activity_weight) and not args.note_event_training:
        raise ValueError('Artifact/activity scoring requires --note-event-training.')
    if args.semitone_training and not args.note_event_training:
        raise ValueError("Semitone training requires --note-event-training.")
    if args.pitch_head_only and not args.init_checkpoint:
        raise ValueError("Pitch-only adaptation requires a trained --init-checkpoint.")
    if args.semitone_guard and not args.init_checkpoint:
        raise ValueError("Semitone regression guard requires an --init-checkpoint baseline.")
    if args.artifact_guard and not args.init_checkpoint:
        raise ValueError('Artifact guard requires an --init-checkpoint baseline.')
    if args.note_event_training:
        if args.no_musical_event_context:
            raise ValueError("Complete-note training requires the musical-event network.")
        args.musical_event_context = True
    if min(args.epochs, args.batch_size, args.bars_per_chunk, args.stride_bars, args.validate_every) < 1:
        raise ValueError("Epochs, batch size, chunk/stride bars and validation interval must be positive.")
    if not 0.5 <= args.triplet_confidence <= 1.0:
        raise ValueError("Triplet confidence must be in [0.5, 1].")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.set_float32_matmul_precision("high")
    print(f"Device: {device}")

    args.output.mkdir(parents=True, exist_ok=True)
    run_dir = args.output / "runs" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir.mkdir(parents=True)
    feature_config = FeatureConfig(use_articulation_features=True)
    cache_paths = build_feature_caches(
        args.dataset.resolve(),
        args.cache_dir or args.output / "feature_cache",
        feature_config,
        device,
    )
    clear_articulation_encoder_cache()
    chunk_steps = args.bars_per_chunk * STEPS_PER_BAR
    stride_steps = args.stride_bars * STEPS_PER_BAR
    ids = [path.name.removesuffix("_grid_features.pt") for path in cache_paths]
    train_ids, validation_ids = song_split(ids, args.split or args.dataset / "training_split.json", args.seed,
        args.validation_ids.split(",") if args.validation_ids else None)
    dataset = GridChunkDataset([p for p, sid in zip(cache_paths, ids) if sid in train_ids], chunk_steps, stride_steps)
    validation_songs = [torch.load(p, map_location="cpu", weights_only=False)
                        for p, sid in zip(cache_paths, ids) if sid in validation_ids]
    print(f"Song split: train={train_ids}; validation={validation_ids}", flush=True)
    manifest_rows = read_manifest(args.dataset)
    data_provenance = {row["song_id"]: {key: file_digest(args.dataset / row[key])
        for key in ("vocal_midi", "vocal_audio", "inst_audio", "grid") if key in row}
        for row in manifest_rows}
    source_provenance = {p.relative_to(PACKAGE_ROOT).as_posix(): file_digest(p) for p in PACKAGE_ROOT.rglob("*.py")}
    for sid in ids:
        label_path = args.dataset / "pronunciation_labels" / f"{sid}.json"
        if label_path.exists():
            data_provenance.setdefault(sid, {})["pronunciation_labels"] = file_digest(label_path)
    run_metadata = {"train_ids": train_ids, "validation_ids": validation_ids,
        "data_sha256": data_provenance, "source_sha256": source_provenance,
        "reviewed_pronunciation_frames": {s["song_id"]: s.get("reviewed_pronunciation_frames", 0)
            for s in dataset.songs + validation_songs},
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}}
    (run_dir / "run.json").write_text(json.dumps(run_metadata, indent=2), encoding="utf-8")
    chunk_weights = []
    for song_index, start in dataset.index:
        song = dataset.songs[song_index]
        triplet = (song["rhythm_mode"][start:start + chunk_steps] == 1) & song["rhythm_mask"][start:start + chunk_steps]
        chunk_weights.append(4.0 if triplet.any() and args.musical_event_context and not args.no_musical_event_context else 1.0)
    run_metadata["sampler"] = {"enabled": args.musical_event_context and not args.no_musical_event_context,
        "triplet_chunk_weight": 4.0 if args.musical_event_context and not args.no_musical_event_context else 1.0,
        "triplet_chunks": sum(w > 1 for w in chunk_weights), "total_chunks": len(chunk_weights)}
    (run_dir / "run.json").write_text(json.dumps(run_metadata, indent=2), encoding="utf-8")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=(WeightedRandomSampler(chunk_weights, len(chunk_weights), replacement=True,
            generator=torch.Generator().manual_seed(args.seed))
            if args.musical_event_context and not args.no_musical_event_context else None),
        shuffle=not (args.musical_event_context and not args.no_musical_event_context),
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    model_config = ModelConfig(
        input_dim=feature_config.input_dim,
        use_articulation_heads=True,
        use_sequence_position=not args.no_sequence_position,
        use_pronunciation_context=not args.no_pronunciation_context,
        articulation_feature_start=feature_config.articulation_start_index,
        use_musical_event_context=args.musical_event_context and not args.no_musical_event_context,
        musical_event_version=3,
        use_note_event_model=args.note_event_training,
        note_activity_weight=args.note_activity_weight,
        acoustic_bins=feature_config.n_mels,
        acoustic_frames=len(feature_config.context_offsets),
    )
    model = MelodyTransformer(model_config).to(device)
    if args.init_checkpoint:
        initial_train = set(initial.get('run_metadata', {}).get('train_ids', []))
        if not initial_train or initial_train & set(validation_ids):
            raise ValueError('Initializer trained on validation songs or has no train provenance.')
        if initial['feature_config'] != asdict(feature_config):
            raise ValueError('Initializer feature configuration differs from this training run.')
        model.load_state_dict(initial['model_state'], strict=True)
        run_metadata['initializer'] = {'path': str(args.init_checkpoint.resolve()),
            'sha256': file_digest(args.init_checkpoint), 'epoch': initial.get('epoch'),
            'note': 'Weights only; optimizer is fresh. Initial validation baseline participates in selection.'}
        del initial
    if args.pitch_head_only:
        freeze_except_pitch_head(model)
    run_metadata['trainable_parameters'] = [name for name, p in model.named_parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate,
        weight_decay=1e-2,
    )
    scaler = GradScaler("cuda", enabled=device.type == "cuda")

    def positive_weight(name: str, cap: float = 30.0, mask_name: str = "boundary_mask") -> torch.Tensor:
        all_steps = sum(int(song[mask_name].sum()) for song in dataset.songs)
        positives = sum(float(song[name][song[mask_name]].sum()) for song in dataset.songs)
        value = min(cap, (all_steps - positives) / max(1.0, positives))
        return torch.tensor(value, device=device)

    onset_pos_weight = positive_weight("onset")
    offset_pos_weight = positive_weight("offset")
    activity_pos_weight = positive_weight("activity", cap=5.0, mask_name="loss_mask")
    continuation_pos_weight = positive_weight("continuation", cap=5.0, mask_name="continuation_mask")
    pitch_change_pos_weight = positive_weight("pitch_change")
    articulation_pos_weight = positive_weight(
        "articulation_boundary",
        cap=30.0,
    )
    rhythm_counts = torch.zeros(NUM_RHYTHM_CLASSES, dtype=torch.float32)
    for song in dataset.songs:
        confident = song["rhythm_mask"] & song["loss_mask"]
        rhythm_counts += torch.bincount(
            song["rhythm_mode"][confident],
            minlength=NUM_RHYTHM_CLASSES,
        ).float()
    rhythm_class_weight = torch.ones(
        NUM_RHYTHM_CLASSES,
        dtype=torch.float32,
    )
    present = rhythm_counts > 0
    if present.any():
        rhythm_class_weight[present] = (
            rhythm_counts[present].sum()
            / present.sum()
            / rhythm_counts[present]
        ).clamp(max=10.0)
    rhythm_class_weight = rhythm_class_weight.to(device)
    state_counts = torch.zeros(4)
    for song in dataset.songs:
        target, mask = boundary_targets(song)
        state_counts += torch.bincount(target[mask], minlength=4).float()
    # Mild class balancing, unlike independent high positive-weight onset BCE.
    state_weights = (state_counts.sum() / (4 * state_counts.clamp_min(1))).sqrt().clamp(max=5).to(device)
    run_metadata["boundary_states"] = {"classes": ["rest", "hold", "start", "end"],
        "enabled": model_config.use_musical_event_context,
        "training_counts": state_counts.tolist(), "class_weights": state_weights.tolist()}
    (run_dir / "run.json").write_text(json.dumps(run_metadata, indent=2), encoding="utf-8")
    print(
        "Rhythm supervision: "
        f"straight={int(rhythm_counts[0] / STEPS_PER_BAR)} bars, "
        f"triplet={int(rhythm_counts[1] / STEPS_PER_BAR)} bars"
    )
    if rhythm_counts[1] == 0:
        print(
            "Warning: no triplet/shuffle bars are labeled yet; "
            "automatic shuffle detection needs at least one corrected song."
        )
    history = []
    best_score = -1.0

    def evaluate_current_model():
        model.eval()
        combined = {key: 0 for key in ("reference_notes", "predicted_notes", "onset_matches", "pitch_onset_matches", "full_note_matches")}
        per_song = {}
        for song in validation_songs:
            outputs = predict_outputs(model, song["features"], device, chunk_steps)
            notes, _ = decode_prediction(outputs, song["features"], feature_config, triplet_threshold=args.triplet_confidence)
            counts = note_counts(notes, song)
            for key, value in counts.items():
                combined[key] += value
            per_song[song["song_id"]] = {**summarize_note_counts(counts), **semitone_aba_counts(notes, song),
                **note_artifact_counts(notes,song)}
        return {**summarize_note_counts(combined), "songs": per_song}

    def make_checkpoint(epoch):
        return {
            "schema_version": ("complete_note_melody_transformer_checkpoint_v13" if args.note_event_training
                else "joint_boundary_melody_transformer_checkpoint_v12"),
            "epoch": epoch, "model_state": model.state_dict(), "model_config": asdict(model_config),
            "feature_config": asdict(feature_config), "training_args": vars(args), "history": history,
            "run_metadata": run_metadata, "optimizer_state": optimizer.state_dict(), "scaler_state": scaler.state_dict()}

    if args.init_checkpoint:
        baseline = evaluate_current_model()
        best_score = baseline['full_note_f1']
        history.append({'epoch': 0, 'validation': baseline, 'unmodified_initializer': True})
        torch.save(make_checkpoint(0), run_dir / 'best.pt')
        print(f'Initializer baseline: complete-note F1={best_score:.4f}; included in checkpoint selection.', flush=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        if args.pitch_head_only:
            # Frozen dropout must not perturb the features used for calibration.
            model.eval()
        totals: dict[str, float] = {
            "loss": 0.0,
            "boundary_state_loss": 0.0,
            "note_event_loss": 0.0,
            "event_pitch_accuracy": 0.0,
            "duration_accuracy": 0.0,
            "activity_accuracy": 0.0,
            "rhythm_accuracy": 0.0,
            "onset_precision": 0.0,
            "onset_recall": 0.0,
            "offset_precision": 0.0,
            "offset_recall": 0.0,
            "continuation_precision": 0.0,
            "continuation_recall": 0.0,
            "pitch_change_precision": 0.0,
            "pitch_change_recall": 0.0,
            "articulation_precision": 0.0,
            "articulation_recall": 0.0,
        }
        for batch_index, batch in enumerate(loader):
            batch = {
                key: value.to(device, non_blocking=True)
                for key, value in batch.items()
            }
            padding_valid = batch["valid"]
            valid = padding_valid & batch["loss_mask"]
            boundary_mask = padding_valid & batch["boundary_mask"]
            continuation_mask = padding_valid & batch["continuation_mask"]
            event_mask = batch["duration_mask"] & valid
            active_mask = (batch["activity"] >= 0.5) & valid
            rhythm_mask = batch["rhythm_mask"] & valid
            articulation_mask = batch["articulation_mask"] & valid
            if not valid.any():
                continue
            optimizer.zero_grad(set_to_none=True)
            with autocast("cuda", enabled=device.type == "cuda"):
                outputs = model(
                    batch["features"],
                    batch["bar_position"],
                    padding_mask=~padding_valid,
                )
                pitch_loss = supervised_pitch_loss(outputs["pitch"], batch["pitch"], valid,
                    supervise_rest=args.semitone_training)
                if event_mask.any():
                    duration_loss = F.cross_entropy(
                        outputs["duration"][event_mask],
                        batch["duration"][event_mask],
                    )
                else:
                    duration_loss = outputs["duration"].sum() * 0.0
                onset_loss = masked_binary_loss(
                    outputs["onset"], batch["onset"], boundary_mask,
                    pos_weight=onset_pos_weight,
                )
                offset_loss = masked_binary_loss(
                    outputs["offset"], batch["offset"], boundary_mask,
                    pos_weight=offset_pos_weight,
                )
                activity_loss = masked_binary_loss(
                    outputs["activity"], batch["activity"], valid,
                    pos_weight=activity_pos_weight,
                )
                continuation_loss = masked_binary_loss(
                    outputs["continuation"], batch["continuation"], continuation_mask,
                    pos_weight=continuation_pos_weight,
                )
                pitch_change_loss = masked_binary_loss(
                    outputs["pitch_change"], batch["pitch_change"], boundary_mask,
                    pos_weight=pitch_change_pos_weight,
                )
                if articulation_mask.any():
                    articulation_loss = F.binary_cross_entropy_with_logits(
                        outputs["articulation"][articulation_mask],
                        batch["articulation_boundary"][articulation_mask],
                        pos_weight=articulation_pos_weight,
                    )
                else:
                    articulation_loss = outputs["articulation"].sum() * 0.0
                if rhythm_mask.any():
                    rhythm_loss = F.cross_entropy(
                        outputs["rhythm"][rhythm_mask],
                        batch["rhythm_mode"][rhythm_mask],
                        weight=rhythm_class_weight,
                    )
                else:
                    rhythm_loss = outputs["rhythm"].sum() * 0.0
                pronunciation_loss = outputs["pitch"].sum() * 0.0
                state_loss = outputs["pitch"].sum() * 0.0
                complete_note_loss = outputs["pitch"].sum() * 0.0
                if "boundary_state" in outputs:
                    state_target, state_mask = boundary_targets(batch)
                    if state_mask.any():
                        state_loss = F.cross_entropy(outputs["boundary_state"][state_mask],
                            state_target[state_mask], weight=state_weights)
                if args.note_event_training:
                    complete_note_loss = note_event_loss(outputs, batch, batch_index=batch_index,
                        semitone_contrasts=args.semitone_training, artifact_contrasts=args.artifact_training)
                pronunciation_mask = (batch["pronunciation_weight"] > 0) & articulation_mask
                if "pronunciation" in outputs and pronunciation_mask.any():
                    pronunciation_loss = (F.binary_cross_entropy_with_logits(
                        outputs["pronunciation"][pronunciation_mask],
                        batch["pronunciation_boundary"][pronunciation_mask], reduction="none")
                        * batch["pronunciation_weight"][pronunciation_mask]).mean()
                loss = (
                    pitch_loss
                    + 0.20 * duration_loss
                    + 0.80 * onset_loss
                    + 0.35 * offset_loss
                    + 0.25 * activity_loss
                    + 0.60 * continuation_loss
                    + 0.30 * rhythm_loss
                    + 0.60 * pitch_change_loss
                    + 0.60 * articulation_loss
                    + pronunciation_loss
                    + state_loss
                    + .20 * complete_note_loss
                )
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss; this run will not replace the working checkpoint.")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            detached_outputs = {
                key: value.detach()
                for key, value in outputs.items()
            }
            batch_metrics = metrics(
                detached_outputs,
                batch,
            )
            totals["loss"] += loss.item()
            totals["boundary_state_loss"] += state_loss.item()
            totals["note_event_loss"] += complete_note_loss.item()
            for key, value in batch_metrics.items():
                totals[key] += value

        epoch_result = {
            "epoch": epoch,
            **{key: value / len(loader) for key, value in totals.items()},
        }
        history.append(epoch_result)
        validation = None
        if epoch == 1 or epoch % args.validate_every == 0 or epoch == args.epochs:
            validation = evaluate_current_model()
            epoch_result["validation"] = validation
            print(f"validation: note_F1={validation['full_note_f1']:.4f}, pitch_onset_F1={validation['pitch_onset_f1']:.4f}, "
                  f"missed={validation['missed_notes']}, extra={validation['extra_notes']}", flush=True)
        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            print(
                f"epoch {epoch:03d} "
                f"loss={epoch_result['loss']:.4f} "
                f"joint_boundary={epoch_result['boundary_state_loss']:.4f} "
                f"note_event={epoch_result['note_event_loss']:.4f} "
                f"event_pitch={epoch_result['event_pitch_accuracy']:.3f} "
                f"duration={epoch_result['duration_accuracy']:.3f} "
                f"activity={epoch_result['activity_accuracy']:.3f} "
                f"rhythm={epoch_result['rhythm_accuracy']:.3f} "
                f"onset_P/R={epoch_result['onset_precision']:.3f}/"
                f"{epoch_result['onset_recall']:.3f} "
                f"offset_P/R={epoch_result['offset_precision']:.3f}/"
                f"{epoch_result['offset_recall']:.3f} "
                f"continue_P/R={epoch_result['continuation_precision']:.3f}/"
                f"{epoch_result['continuation_recall']:.3f} "
                f"pitch_change_P/R="
                f"{epoch_result['pitch_change_precision']:.3f}/"
                f"{epoch_result['pitch_change_recall']:.3f} "
                f"articulation_P/R="
                f"{epoch_result['articulation_precision']:.3f}/"
                f"{epoch_result['articulation_recall']:.3f}",
                flush=True,
            )

        checkpoint = make_checkpoint(epoch)
        torch.save(checkpoint, run_dir / "latest.pt")
        if (validation is not None and validation["full_note_f1"] > best_score
                and (not args.semitone_guard or passes_pitch_aba_guard(validation, baseline))
                and (not args.artifact_guard or passes_artifact_guard(validation,baseline))):
            best_score = validation["full_note_f1"]
            torch.save(checkpoint, run_dir / "best.pt")
        (run_dir / "training_history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")

    if not args.no_publish:
        if args.note_event_training:
            from transmelody.training.model_release import activate_checkpoint
            activate_checkpoint(run_dir / 'best.pt', args.output)
            print(f"Saved run: {run_dir}; selected validation note F1={best_score:.4f}")
            return
        existing = args.output / "final.pt"
        if existing.exists():
            shutil.copy2(existing, run_dir / "previous_workflow_model.pt")
        temporary = args.output / ".final_pending.pt"
        shutil.copy2(run_dir / "best.pt", temporary)
        os.replace(temporary, existing)
        shutil.copy2(run_dir / "training_history.json", args.output / "training_history.json")
    print(f"Saved run: {run_dir}; selected validation note F1={best_score:.4f}")


if __name__ == "__main__":
    main()
