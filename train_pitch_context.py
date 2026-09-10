"""Train a separate pitch-only refiner with a frozen acoustic/boundary model."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F

from melody_transformer import MelodyTransformer, ModelConfig, FeatureConfig
from melody_inference import predict_outputs, decode_prediction
from melody_evaluation import file_digest, note_counts, summarize_note_counts
from pitch_context import (PitchContextRefiner, note_inputs, context_features, refine_logits,
    replace_note_pitches)


def pitch_targets(notes, song):
    """Supervision only. Ambiguous merged/misaligned segments are masked, not relabeled.

    Features/segmentation/tonality never read this target. Validation uses the
    original reference event metrics, not the training overlap label policy.
    """
    targets, masks = [], []
    for note in notes:
        labels = song['pitch'][note.start_step:note.end_step]
        valid = song['loss_mask'][note.start_step:note.end_step]
        counts = torch.bincount(labels[valid], minlength=129)
        counts[0] = 0
        winner = int(counts.argmax())
        targets.append(max(0, winner - 1))
        masks.append(winner > 0 and float(counts[winner]) >= .8 * len(labels))
    return torch.tensor(targets), torch.tensor(masks, dtype=torch.bool)


def augment(logits, targets, mask):
    """Transpose plus occasional uncertain semitone/octave acoustic confusions.

    The score target, including chromatic tones, is preserved except transposition.
    Tonal features are recomputed from the noisy INPUT, never the clean target.
    """
    x, y = logits.clone(), targets.clone()
    shift = int(torch.randint(-5, 7, (1,)))
    if mask.any() and ((y[mask] + shift >= 0) & (y[mask] + shift < 128)).all():
        x = x.roll(shift, -1)
        if shift > 0:
            x[:, :shift] = -30
        elif shift < 0:
            x[:, shift:] = -30
        y = (y + shift).clamp(0, 127)
    corrupt = (torch.rand(len(x)) < .20) & mask
    for i in corrupt.nonzero().flatten().tolist():
        interval = int(torch.tensor([-12, -2, -1, 1, 2, 12])[torch.randint(6, (1,))])
        alternative = int(y[i]) + interval
        if 0 <= alternative < 128:
            maximum = x[i].max().clone()
            x[i, y[i]] = maximum - torch.rand(()) * .8
            x[i, alternative] = maximum + torch.rand(()) * .8
    return x, y


def evaluate(model, songs, device):
    combined = {k: 0 for k in ('reference_notes', 'predicted_notes', 'onset_matches',
        'pitch_onset_matches', 'full_note_matches')}
    report = {}
    for song in songs:
        scores = refine_logits(model, song['logits'], song['features'], device)
        notes = replace_note_pitches(song['notes'], scores)
        counts = note_counts(notes, song['reference'])
        for k, v in counts.items():
            combined[k] += v
        report[song['id']] = summarize_note_counts(counts)
        report[song['id']]['changed_pitches'] = sum(a.pitch != b.pitch for a, b in zip(song['notes'], notes))
    return {**summarize_note_counts(combined), 'songs': report}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('base_checkpoint', type=Path)
    parser.add_argument('--cache-dir', type=Path, default=Path('output/melody_transformer/feature_cache'))
    parser.add_argument('--output', type=Path, default=Path('output/pitch_context_experiment'))
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--seed', type=int, default=2026)
    args = parser.parse_args()
    if args.epochs < 1:
        raise ValueError('Epochs must be positive.')
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base = torch.load(args.base_checkpoint, map_location='cpu', weights_only=False)
    train_ids = base['run_metadata']['train_ids']
    val_ids = base['run_metadata']['validation_ids']
    if set(train_ids) & set(val_ids) or not train_ids or not val_ids:
        raise ValueError('Need disjoint whole-song train/development sets from the base model.')
    base_sha = file_digest(args.base_checkpoint)
    run = args.output / 'runs' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    run.mkdir(parents=True)
    acoustic = MelodyTransformer(ModelConfig(**base['model_config'])).to(device).eval()
    acoustic.load_state_dict(base['model_state'], strict=True)
    config = FeatureConfig(**base['feature_config'])
    train, val = [], []
    cache_hashes = {}
    for sid in train_ids + val_ids:
        cache_path = args.cache_dir / f'{sid}_grid_features.pt'
        cache_hashes[sid] = file_digest(cache_path)
        song = torch.load(cache_path, map_location='cpu', weights_only=False)
        outputs = predict_outputs(acoustic, song['features'], device,
            base.get('training_args', {}).get('bars_per_chunk', 8) * 48)
        notes, _ = decode_prediction(outputs, song['features'], config,
            triplet_threshold=base.get('training_args', {}).get('triplet_confidence', .6))
        logits, features, starts, durations = note_inputs(outputs, notes)
        targets, mask = pitch_targets(notes, song)
        item = {'id': sid, 'logits': logits, 'features': features, 'starts': starts,
            'durations': durations, 'notes': notes, 'targets': targets, 'mask': mask}
        if sid in val_ids:
            item['reference'] = {k: song[k] for k in ('grid', 'onset', 'pitch', 'loss_mask', 'continuation')}
        (train if sid in train_ids else val).append(item)
        print(f'[frozen acoustic] {sid}: {len(notes)} notes; usable labels={int(mask.sum())}', flush=True)
    del acoustic, base
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    model = PitchContextRefiner().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.01)
    model.eval()
    baseline = evaluate(model, val, device)
    history = [{'epoch': 0, 'validation': baseline}]
    best_score = baseline['pitch_onset_f1']
    provenance = {'base_checkpoint': str(args.base_checkpoint.resolve()), 'base_sha256': base_sha,
        'train_ids': train_ids, 'validation_ids': val_ids, 'cache_sha256': cache_hashes,
        'seed': args.seed, 'source_sha256': {name: file_digest(Path(__file__).with_name(name))
            for name in ('pitch_context.py', 'train_pitch_context.py')},
        'selection': 'Development pitch+onset F1, includes identity baseline epoch0; never publishes automatically.',
        'tonality': 'Unsupervised major/natural-minor profile/HMM INPUT, not supervised key annotations.'}

    def save(epoch):
        torch.save({'schema': 'note_pitch_context_v1', 'config': asdict(model.config),
            'model_state': model.state_dict(), 'base_sha256': base_sha, 'epoch': epoch,
            'provenance': provenance, 'baseline': baseline}, run / 'best.pt')

    save(0)
    print(f'Identity baseline pitch+onset F1={best_score:.4f}', flush=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        chunks = []
        for song in train:
            x, y = augment(song['logits'], song['targets'], song['mask'])
            f = context_features(x, song['starts'], song['durations'])
            for start in range(0, max(1, len(x) - 15), 32):
                end = min(len(x), start + 64)
                if end > start and song['mask'][start:end].any():
                    chunks.append((x[start:end], f[start:end], y[start:end], song['mask'][start:end]))
        total, batches = 0., 0
        order = torch.randperm(len(chunks)).tolist()
        for offset in range(0, len(order), 16):
            selected = [chunks[i] for i in order[offset:offset+16]]
            length = max(len(c[0]) for c in selected)
            x, f = torch.zeros(len(selected), length, 128), torch.zeros(len(selected), length, 169)
            y, mask = torch.zeros(len(selected), length, dtype=torch.long), torch.zeros(len(selected), length, dtype=torch.bool)
            valid = mask.clone()
            for i, (cx, cf, cy, cm) in enumerate(selected):
                n = len(cx)
                x[i, :n], f[i, :n], y[i, :n], mask[i, :n] = cx, cf, cy, cm
                valid[i, :n] = True
            x, f, y, mask, valid = [t.to(device) for t in (x, f, y, mask, valid)]
            optimizer.zero_grad(set_to_none=True)
            pred = model(x, f, padding_mask=~valid)
            loss = F.cross_entropy(pred[mask], y[mask]) + .01 * (pred[valid] - x[valid]).square().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite context loss; base model is untouched.')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            total += float(loss)
            batches += 1
        row = {'epoch': epoch, 'loss': total / max(1, batches)}
        if epoch % 5 == 0 or epoch == 1 or epoch == args.epochs:
            model.eval()
            result = evaluate(model, val, device)
            row['validation'] = result
            if result['pitch_onset_f1'] > best_score:
                best_score = result['pitch_onset_f1']
                save(epoch)
            print(f'epoch {epoch}: loss={row["loss"]:.4f}; pitch+onset F1={result["pitch_onset_f1"]:.4f}; '
                f'full F1={result["full_note_f1"]:.4f}', flush=True)
        history.append(row)
        (run / 'training_history.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
    (run / 'run.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
    print(f'Saved {run}; best pitch+onset F1={best_score:.4f}; base checkpoint unchanged.', flush=True)


if __name__ == '__main__':
    main()
