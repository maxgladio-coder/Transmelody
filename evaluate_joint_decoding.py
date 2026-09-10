"""Compare fixed global score weights on development songs, keeping the model frozen."""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import torch

from melody_transformer import FeatureConfig, MelodyTransformer, ModelConfig
from melody_inference import predict_outputs, decode_prediction
from melody_evaluation import file_digest, note_counts, summarize_note_counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--cache-dir', type=Path, default=Path('output/melody_transformer/feature_cache'))
    parser.add_argument('--output', type=Path, default=Path('output/joint_pitch_experiment'))
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    config = FeatureConfig(**checkpoint['feature_config'])
    model = MelodyTransformer(ModelConfig(**checkpoint['model_config'])).to(device).eval()
    model.load_state_dict(checkpoint['model_state'], strict=True)
    ids = checkpoint['run_metadata']['validation_ids']
    weights = [None, .25, .5, 1., 2.]
    results = {str(w): {'weight': w, 'songs': {}, 'counts': {k: 0 for k in
        ('reference_notes', 'predicted_notes', 'onset_matches', 'pitch_onset_matches', 'full_note_matches')}} for w in weights}
    hashes = {}
    for sid in ids:
        path = args.cache_dir / f'{sid}_grid_features.pt'
        hashes[sid] = file_digest(path)
        song = torch.load(path, map_location='cpu', weights_only=False)
        outputs = predict_outputs(model, song['features'], device,
            checkpoint.get('training_args', {}).get('bars_per_chunk', 8) * 48)
        for weight in weights:
            notes, _ = decode_prediction(outputs, song['features'], config,
                triplet_threshold=checkpoint.get('training_args', {}).get('triplet_confidence', .6),
                joint_pitch_weight=weight)
            counts = note_counts(notes, song)
            result = results[str(weight)]
            result['songs'][sid] = summarize_note_counts(counts)
            for k, v in counts.items():
                result['counts'][k] += v
            print(sid, 'weight', weight, 'notes', len(notes), 'fullF1',
                round(result['songs'][sid]['full_note_f1'], 4), flush=True)
    for result in results.values():
        result['combined'] = summarize_note_counts(result['counts'])
    best = max(results.values(), key=lambda r: r['combined']['full_note_f1'])
    run = args.output / 'runs' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    run.mkdir(parents=True)
    report = {'checkpoint': str(args.checkpoint.resolve()), 'checkpoint_sha256': file_digest(args.checkpoint),
        'development_ids': ids, 'cache_sha256': hashes,
        'source_sha256': {name: file_digest(Path(__file__).with_name(name)) for name in
            ('joint_pitch_decoding.py', 'melody_inference.py', 'evaluate_joint_decoding.py')},
        'selection': 'Combined complete-note F1 on existing development songs; baseline included. '
            '18 was used to diagnose and design this change, so this is not independent test performance.',
        'results': list(results.values()), 'selected_weight': best['weight']}
    (run / 'evaluation.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('Selected', best['weight'], 'combined', best['combined'], 'report', run, flush=True)


if __name__ == '__main__':
    main()
