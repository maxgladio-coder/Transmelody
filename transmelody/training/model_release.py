"""Verified, backed-up activation of a complete-note workflow checkpoint."""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import torch

from transmelody.models.melody_transformer import MelodyTransformer, ModelConfig
from transmelody.evaluation.melody_evaluation import file_digest


def activate_checkpoint(source: Path, output: Path):
    source, output = source.resolve(), output.resolve()
    checkpoint = torch.load(source, map_location='cpu', weights_only=False)
    config = ModelConfig(**checkpoint['model_config'])
    if not (config.use_note_event_model and config.use_musical_event_context and config.musical_event_version >= 3):
        raise ValueError('This release helper requires the complete-note architecture/decoder.')
    model = MelodyTransformer(config)
    model.load_state_dict(checkpoint['model_state'], strict=True)
    meta = checkpoint['run_metadata']
    train, validation = set(meta['train_ids']), meta['validation_ids']
    if not train or not validation or train & set(validation):
        raise ValueError('Invalid training/development provenance.')
    selected = [row['validation'] for row in checkpoint['history']
        if row['epoch'] == checkpoint['epoch'] and 'validation' in row]
    if not selected:
        raise ValueError('Checkpoint lacks validation for its selected epoch.')
    split = {'version': 1, 'seed': checkpoint['training_args'].get('seed', 2026), 'validation_ids': validation}
    split_path = output / 'training_split.json'
    if split_path.exists() and set(json.loads(split_path.read_text(encoding='utf-8'))['validation_ids']) != set(validation):
        raise ValueError('Existing workflow validation split differs; explicit migration is required.')
    output.mkdir(parents=True, exist_ok=True)
    final = output / 'final.pt'
    backup = None
    if final.exists():
        digest = file_digest(final)
        backup = output / 'backups' / f'final_{digest[:16]}.pt'
        backup.parent.mkdir(exist_ok=True)
        if backup.exists() and file_digest(backup) != digest:
            raise ValueError('Backup filename collision; existing backup will not be overwritten.')
        if not backup.exists():
            shutil.copy2(final, backup)
        if file_digest(backup) != digest:
            raise IOError('Backup verification failed; current model remains active.')
    pending = output / '.note_event_pending.pt'
    shutil.copy2(source, pending)
    if file_digest(pending) != file_digest(source):
        raise IOError('Checkpoint copy verification failed; current model remains active.')
    report = {'checkpoint_source': str(source), 'checkpoint_sha256': file_digest(source),
        'previous_checkpoint_backup': str(backup) if backup else None, 'decoder': 'complete_note_segments',
        'selected_epoch': checkpoint['epoch'], 'validation': selected[0],
        'train_ids': meta['train_ids'], 'validation_ids': validation,
        'queue_action': 'None: no song prediction, audio move, reference MIDI or registry write.'}
    blend = meta.get('checkpoint_interpolation')
    if blend:
        report['model_update'] = {'type':'checkpoint_interpolation',
            'adapted_weight':blend['selected_alpha'], 'base_sha256':blend['base_sha256'],
            'adapted_sha256':blend['adapted_sha256'], 'selection_note':blend['rule']}
    report['note_activity_weight'] = config.note_activity_weight
    policy_path = output/'inference_policy.json'
    if policy_path.exists():
        policy = json.loads(policy_path.read_text(encoding='utf-8'))
        if policy.get('enabled') and Path(policy['base_checkpoint']).resolve() == final.resolve():
            report['inference_policy'] = {'type':'pitch_only_fusion',
                'policy_file':str(policy_path), 'new_pitch_weight':policy['new_pitch_weight'],
                'auxiliary_checkpoint_sha256':policy['assets']['score_model']['sha256'],
                'validation_scope':'This release validates the base, not the frozen-auxiliary ensemble.',
                'ensemble_needs_revalidation':file_digest(source) != policy['reviewed_base_sha256']}
    if meta.get('decoder_calibration'):
        report['decoder_calibration'] = meta['decoder_calibration']
    # Persist the immutable split before activating; a crash here leaves the
    # former model intact and cannot accidentally reshuffle validation songs.
    if not split_path.exists():
        split_path.write_text(json.dumps(split, indent=2), encoding='utf-8')
    os.replace(pending, final)
    (output / 'active_model.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    history = source.parent / 'training_history.json'
    if history.exists() and history.resolve() != (output / 'training_history.json').resolve():
        shutil.copy2(history, output / 'training_history.json')
    return report


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--output', type=Path, default=Path('output/melody_transformer'))
    args = parser.parse_args()
    print(json.dumps(activate_checkpoint(args.checkpoint, args.output), indent=2))
