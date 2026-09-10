"""A/B evaluation of the isolated score model, without publishing or queue writes."""
import argparse
import json
from pathlib import Path

import torch
import numpy as np
import soundfile as sf

import transmelody.models.score_transcriber_v2 as v2
from transmelody.evaluation.melody_evaluation import (note_counts, summarize_note_counts, semitone_aba_counts,
    note_artifact_counts, file_digest, _reference_note_steps)
from transmelody.models.melody_transformer import FeatureConfig, ModelConfig, MelodyTransformer
from transmelody.inference.melody_inference import predict_outputs as old_outputs, decode_prediction, export_prediction


def stats(notes, song):
    return {**summarize_note_counts(note_counts(notes, song)), **semitone_aba_counts(notes, song),
            **note_artifact_counts(notes, song)}


def oracle_pitch(outputs, song):
    """Diagnostic only: given reference spans, is pitch evidence sufficient?

    Never serialize these oracle notes, select checkpoints with this metric, or
    call this automatic transcription accuracy.
    """
    truth = _reference_note_steps(song)
    correct = sum(int(outputs['pitch'][a:b, 1:].float().log_softmax(-1).mean(0).argmax()) == p for a,b,p in truth)
    return {'reference_span_pitch_correct': correct, 'reference_span_pitch_total': len(truth),
            'reference_span_pitch_accuracy': correct / max(1,len(truth))}


def render_previews(folder, sid, song):
    from transmelody.audio.audio_audition import midi_notes_in_wav_time, render_audition
    from transmelody.grid.musical_timeline import samples_at_ticks
    wave, rate = sf.read(v2.ROOT / f'dataset/melody_dataset/vocal_audio/{sid}_vocal.wav',
                         dtype='float32', always_2d=True)
    wave = wave.mean(1)
    truth = _reference_note_steps(song)
    first = float(samples_at_ticks(song['grid'], truth[0][0]*40, rate)) / rate if truth else 0
    start, end = max(0., first - 1), min(len(wave)/rate, max(0., first-1) + 40)
    for variant in ('baseline', 'new'):
        notes = midi_notes_in_wav_time(folder / f'{sid}_{variant}.mid')
        mix = render_audition(wave, rate, start, end, notes=notes, vocal_gain=.60, cue_gain=.24)
        sf.write(folder / f'{sid}_{variant}_preview.wav', mix, 44100, subtype='PCM_16')
    return {'start_seconds_in_original_wav': start, 'end_seconds_in_original_wav': end,
            'selection': 'First reference vocal onset minus 1 s, then 40 s; same segment/gain for both models.'}


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=v2.DEFAULT_OUTPUT)
    parser.add_argument('--previews', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    new_cp = torch.load(args.output / 'best.pt', map_location='cpu', weights_only=False)
    new = v2.ScoreModel(**new_cp['model_config']).to(device).eval()
    new.load_state_dict(new_cp['model_state'])
    baseline_path = v2.ROOT / 'output/melody_transformer/final.pt'
    cp = torch.load(baseline_path, map_location='cpu', weights_only=False)
    old = MelodyTransformer(ModelConfig(**cp['model_config'])).to(device).eval()
    old.load_state_dict(cp['model_state'])
    fc = FeatureConfig(**cp['feature_config'])
    dev = new_cp['metadata']['development_ids']
    report = {'development_ids': dev, 'training_example_ids': ['19'],
        'baseline_sha256': file_digest(baseline_path), 'candidate_sha256': file_digest(args.output / 'best.pt'),
        'candidate_epoch': new_cp['epoch'], 'variants': {k: {'songs': {}} for k in ('production', 'new', 'new_ce_unbiased')},
        'note': 'Fixed reviewed timeline; 19 is training-fit only; CE-unbiased subtracts log training class weights, not a tuned threshold.'}
    folder = args.output / 'samples'; folder.mkdir(exist_ok=True)
    output_cache = args.output / 'evaluation_outputs'; output_cache.mkdir(exist_ok=True)
    for sid in [*dev, '19']:
        song = v2.load_song(Path(new_cp['metadata'].get('cache', args.output / 'cache')), sid)
        legacy = torch.load(v2.ROOT / f'output/melody_transformer/feature_cache/{sid}_grid_features.pt',
            map_location='cpu', weights_only=False)
        if not torch.equal(song['pitch'], legacy['pitch']) or song['grid']['timeline']['tempo_events'] != legacy['grid']['timeline']['tempo_events']:
            raise ValueError('Baseline cache labels/timeline are stale; rebuild before comparing.')
        baseline = old_outputs(old, legacy['features'], device, cp['training_args'].get('bars_per_chunk', 8) * 48)
        notes, modes = decode_prediction(baseline, legacy['features'], fc,
            triplet_threshold=cp['training_args'].get('triplet_confidence', .6))
        report['variants']['production']['songs'][sid] = {**stats(notes, song), **oracle_pitch(baseline, song)}
        export_prediction(folder / f'{sid}_baseline.mid', song['grid'], notes, modes)
        outputs = v2.predict_outputs(new, song, device)
        torch.save(outputs, output_cache / f'{sid}.pt')
        for variant in ('new', 'new_ce_unbiased'):
            out = dict(outputs)
            if variant == 'new_ce_unbiased':
                out['boundary_state'] = out['boundary_state'] - torch.tensor([1., 1., 2., 2.]).log()
            notes = v2.decode(out, song)
            result = {**stats(notes, song), **oracle_pitch(out, song)}
            report['variants'][variant]['songs'][sid] = result
            export_prediction(folder / f'{sid}_{variant}.mid', song['grid'], notes, None)
            print(sid, variant, result, flush=True)
        if args.previews and sid in ('18', '19'):
            report.setdefault('previews', {})[sid] = render_previews(folder, sid, song)
    keys = ('reference_notes', 'predicted_notes', 'onset_matches', 'pitch_onset_matches', 'full_note_matches')
    for row in report['variants'].values():
        row['development'] = summarize_note_counts({k: sum(row['songs'][s][k] for s in dev) for k in keys})
        for k in ('aba_count', 'aba_recovered', 'short_note_count', 'short_note_matches',
                  'split_reference_notes', 'mostly_reference_rest_notes'):
            row['development'][k] = sum(row['songs'][s][k] for s in dev)
    v2.save_json(args.output / 'comparison.json', report)
    print(json.dumps({k:v['development'] for k,v in report['variants'].items()}), flush=True)


if __name__ == '__main__':
    main()
