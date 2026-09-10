"""Evaluate frozen candidates against reviewed MIDI, without moving the queue."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

import torch

from audio_audition import midi_notes_in_wav_time
from compare_melody_midi import compare
from melody_evaluation import file_digest
from melody_inference import predict_outputs, decode_prediction, export_prediction
from melody_transformer import FeatureConfig, ModelConfig, MelodyTransformer
from musical_timeline import tick_seconds, grid_tempo_events


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', action='append', required=True, help='NAME=PATH')
    parser.add_argument('--output', type=Path, default=Path('output/semitone_review/evaluation'))
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    report = {'selection_songs': ['4','11','18'], 'review_song': '19',
        'caveat': '19 is training data after this review, not an independent test. '
        '4/11/18 are repeatedly used development songs. Inference uses reviewed tempo/offset only, '
        'not reference note boundaries or pitches.', 'models': {}}
    for value in args.checkpoint:
        name, path = value.split('=', 1)
        if not name or Path(name).name != name or name in ('.','..'):
            raise ValueError('Model name must be a single safe directory component.')
        path = Path(path)
        cp = torch.load(path, map_location='cpu', weights_only=False)
        config = FeatureConfig(**cp['feature_config'])
        model = MelodyTransformer(ModelConfig(**cp['model_config'])).to(device).eval()
        model.load_state_dict(cp['model_state'], strict=True)
        folder = args.output / name
        folder.mkdir(parents=True, exist_ok=True)
        record = {'checkpoint': str(path.resolve()), 'sha256': file_digest(path),
            'epoch': cp['epoch'], 'train_ids': cp['run_metadata']['train_ids'], 'songs': {}}
        for sid in ['4','11','18','19']:
            cache_path = Path('output/melody_transformer/feature_cache') / f'{sid}_grid_features.pt'
            song = torch.load(cache_path, map_location='cpu', weights_only=False)
            if song['feature_config'] != asdict(config):
                raise ValueError('Feature cache configuration mismatch.')
            outputs = predict_outputs(model, song['features'], device,
                cp['training_args'].get('bars_per_chunk',8)*48)
            notes, modes = decode_prediction(outputs, song['features'], config,
                triplet_threshold=cp['training_args'].get('triplet_confidence',.6))
            midi = folder / f'{sid}_predicted.mid'
            export_prediction(midi, song['grid'], notes, modes)
            reference_path = Path('dataset/melody_dataset/vocal_mid') / f'{sid}_vocal.mid'
            # Early dataset MIDI files predate WAV START markers. Use their
            # existing reviewed cache alignment, never fit an offset to notes.
            origin = float(tick_seconds(song['grid']['midi_alignment']['audio_start_tick'],
                grid_tempo_events(song['grid'])))
            reference = [x[:3] for x in midi_notes_in_wav_time(reference_path, audio_start_seconds=origin)]
            predicted = [x[:3] for x in midi_notes_in_wav_time(midi)]
            result = compare(reference,predicted)
            result['reference_sha256'] = file_digest(reference_path)
            result['cache_sha256'] = file_digest(cache_path)
            # Oracle boundaries diagnose pitch vs segmentation, never exported.
            truth, correct = 0, 0
            for start in song['onset'].nonzero().flatten().tolist():
                end = start+1
                while end < len(song['pitch']) and song['continuation'][end-1]:
                    end += 1
                if not song['loss_mask'][start:end].all():
                    continue
                p = int(outputs['pitch'][start:end,1:].mean(0).argmax())+1
                truth += 1
                correct += p == int(song['pitch'][start])
            result['diagnostic_only_reference_boundaries_pitch'] = {'correct':correct, 'total':truth}
            record['songs'][sid] = result
            print(name,sid,'fullF1',round(result['metrics']['full_note_f1'],4),
                'ABA',result['semitone_aba']['recovered'],'/',result['semitone_aba']['count'],flush=True)
        report['models'][name] = record
        del model, cp
    report['source_sha256'] = {p: file_digest(Path(p)) for p in
        ('evaluate_semitone_update.py','compare_melody_midi.py','note_event_model.py','melody_inference.py')}
    (args.output / 'comparison.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')


if __name__ == '__main__':
    main()
