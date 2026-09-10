"""Experimental pitch-only fusion: immutable production spans, two pitch heads.

No retraining, production publishing, reference-boundary snapping or queue work.
The new network supplies pitch evidence ONLY, not activity, rhythm or boundaries.
"""
from __future__ import annotations

from pitch_fusion import fuse_pitches
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

import score_transcriber_v2 as v2
from melody_transformer import FeatureConfig, ModelConfig, MelodyTransformer
from melody_inference import predict_outputs, decode_prediction, export_prediction
from melody_evaluation import file_digest, summarize_note_counts
from evaluate_score_transcriber_v2 import stats


def aggregate(songs, ids):
    keys = ('reference_notes', 'predicted_notes', 'onset_matches', 'pitch_onset_matches', 'full_note_matches')
    result = summarize_note_counts({k: sum(songs[s][k] for s in ids) for k in keys})
    for k in ('aba_count', 'aba_recovered', 'short_note_count', 'short_note_matches',
              'split_reference_notes', 'mostly_reference_rest_notes', 'changed_pitches'):
        result[k] = sum(songs[s][k] for s in ids)
    return result


def eligible(row, baseline, ids):
    a, b = row['development'], baseline['development']
    return (a['full_note_f1'] >= b['full_note_f1'] and a['pitch_onset_f1'] >= b['pitch_onset_f1']
        and all(row['songs'][s][k] >= baseline['songs'][s][k]
                for s in ids for k in ('aba_recovered', 'short_note_matches')))


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output = v2.ROOT / 'output/pitch_fusion_review'
    output.mkdir(exist_ok=True)
    production_path = v2.ROOT / 'output/melody_transformer/final.pt'
    new_path = v2.DEFAULT_OUTPUT / 'structured/best.pt'
    hashes = {'production': file_digest(production_path), 'new': file_digest(new_path)}
    cp = torch.load(production_path, map_location='cpu', weights_only=False)
    nc = torch.load(new_path, map_location='cpu', weights_only=False)
    ids = nc['metadata']['development_ids']
    if ids != cp['run_metadata']['validation_ids']:
        raise ValueError('Models must share the same development split.')
    fc = FeatureConfig(**cp['feature_config'])
    old = MelodyTransformer(ModelConfig(**cp['model_config'])).to(device).eval()
    old.load_state_dict(cp['model_state'], strict=True)
    new = v2.ScoreModel(**nc['model_config']).to(device).eval()
    new.load_state_dict(nc['model_state'], strict=True)
    weights = (0., .25, .5, .75, 1.)
    rows = [{'new_weight': w, 'songs': {}} for w in weights]
    material = {}
    for sid in ids + ['19']:
        song = v2.load_song(Path(nc['metadata']['cache']), sid)
        legacy = torch.load(v2.ROOT / f'output/melody_transformer/feature_cache/{sid}_grid_features.pt',
                            map_location='cpu', weights_only=False)
        if song['midi_sha256'] != file_digest(v2.ROOT / f'dataset/melody_dataset/vocal_mid/{sid}_vocal.mid'):
            raise ValueError('Reference score changed; rebuild experimental cache.')
        if song['audio_sha256'] != file_digest(v2.ROOT / f'dataset/melody_dataset/vocal_audio/{sid}_vocal.wav'):
            raise ValueError('Audio changed; rebuild experimental cache.')
        if (not torch.equal(song['pitch'], legacy['pitch'])
                or song['grid']['timeline']['tempo_events'] != legacy['grid']['timeline']['tempo_events']
                or song['grid']['midi_alignment']['audio_start_tick'] != legacy['grid']['midi_alignment']['audio_start_tick']):
            raise ValueError('The two inputs are not synchronized to the same score timeline.')
        old_out = predict_outputs(old, legacy['features'], device, cp['training_args'].get('bars_per_chunk',8)*48)
        baseline, modes = decode_prediction(old_out, legacy['features'], fc,
            triplet_threshold=cp['training_args'].get('triplet_confidence',.6))
        new_out = v2.predict_outputs(new, song, device)
        variants = {}
        for row in rows:
            fused = fuse_pitches(baseline, old_out['pitch'], new_out['pitch'], row['new_weight'])
            assert [(n.start_step,n.end_step) for n in fused] == [(n.start_step,n.end_step) for n in baseline]
            row['songs'][sid] = {**stats(fused,song),
                'changed_pitches': sum(n.pitch != b.pitch for n,b in zip(fused,baseline))}
            variants[row['new_weight']] = fused
        material[sid] = (song, baseline, modes, variants)
        print('[song]',sid,[(r['new_weight'],round(r['songs'][sid]['full_note_f1'],4),
              r['songs'][sid]['changed_pitches']) for r in rows],flush=True)
    for row in rows:
        row['development'] = aggregate(row['songs'],ids)
        row['passes_guard'] = eligible(row, rows[0], ids)
    guarded = max((r for r in rows if r['passes_guard']),
        key=lambda r: (r['development']['full_note_f1'],r['development']['pitch_onset_f1'],-r['new_weight']))
    # User requested an audition experiment, not a production release. Export
    # the best nonzero candidate honestly even if a per-song guard rejects it;
    # never label the unchanged zero-weight baseline as a fused sample.
    chosen = max((r for r in rows if r['new_weight'] > 0),
        key=lambda r: (r['development']['full_note_f1'],r['development']['pitch_onset_f1'],-r['new_weight']))
    report = {'development_ids':ids, 'training_example_ids':['19'], 'checkpoint_sha256':hashes,
        'source_sha256':file_digest(Path(__file__)), 'selected_new_weight':chosen['new_weight'],
        'guarded_new_weight':guarded['new_weight'], 'audition_candidate_passes_guard':chosen['passes_guard'],
        'published':False,
        'selection':'Audition: highest development full-note F1 among nonzero weights, pitch-onset F1 tie-break. '
                    'Separately report per-song ABA/short-note guards; failed candidates are NOT published. '
                    'Song19 not used for selection. All original predicted note spans are immutable.',
        'limitations':'Historically reused development songs; fixed reviewed tempo/WAV alignment. '
                      'No independent test or automatic BPM evaluation. Model scores are not calibrated probabilities.',
        'results':rows, 'exports':{}}
    for sid,(song,baseline,modes,variants) in material.items():
        notes = variants[chosen['new_weight']]
        path = output / f'{sid}_fused_predicted.mid'
        export_prediction(path,song['grid'],notes,modes)
        export_prediction(output/f'{sid}_baseline.mid',song['grid'],baseline,modes)
        report['exports'][sid] = {'file':str(path), 'note_count':len(notes),
            'bpm':song['grid']['tempo']['bpm'], 'tempo_events':song['grid']['timeline']['tempo_events'],
            'wav_start_tick':song['grid']['midi_alignment']['audio_start_tick'],
            'note_boundaries_unchanged':True,
            'pitch_changes':[{'start_tick':n.start_step*40,'end_tick':n.end_step*40,'old':b.pitch,'fused':n.pitch}
                             for n,b in zip(notes,baseline) if n.pitch != b.pitch]}
    # Same pure-synth 40-second segment and instrument/gain as the prior A/B.
    from audio_audition import midi_notes_in_wav_time, render_audition
    segment = json.loads((v2.DEFAULT_OUTPUT/'structured/preview_segments.json').read_text(encoding='utf-8'))['18']
    start,end = segment['start_seconds_in_original_wav'],segment['end_seconds_in_original_wav']
    silence = np.zeros(int(np.ceil(end*44100)),dtype=np.float32)
    for tag, filename in [('baseline','18_baseline.mid'),('fused','18_fused_predicted.mid')]:
        notes = midi_notes_in_wav_time(output/filename)
        audio = render_audition(silence,44100,start,end,notes=notes,vocal_gain=0,cue_gain=.6,clicks=False)
        assert len(audio) == 40*44100 and np.isfinite(audio).all()
        sf.write(output/f'18_{tag}_midi_only.wav',audio,44100,subtype='PCM_16')
    report['preview_segment'] = segment
    if file_digest(production_path) != hashes['production']:
        raise RuntimeError('Production checkpoint changed during experiment.')
    v2.save_json(output/'comparison.json',report)
    print('[selected]',chosen['new_weight'],json.dumps(chosen['development']),flush=True)


if __name__ == '__main__':
    main()
