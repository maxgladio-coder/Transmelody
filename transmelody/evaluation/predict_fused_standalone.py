"""Pitch-only fusion from a standalone baseline prediction, without workflow actions.

Run predict_test_audio.py for the requested pair first, using an isolated output
directory. This command reuses its grid, note spans and pitch logits. No reference
MIDI, manifest, registry, training cache, or queue operation is required.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import json

import numpy as np
import soundfile as sf
import torch

from transmelody.audio.audio_audition import midi_notes_in_wav_time, render_audition
from transmelody.evaluation.melody_evaluation import file_digest
from transmelody.inference.melody_inference import export_prediction
from transmelody.models.melody_transformer import PredictedNote
from transmelody.grid.musical_timeline import samples_at_ticks
from transmelody.evaluation.pitch_fusion_review import fuse_pitches
import transmelody.models.score_transcriber_v2 as v2


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-output', type=Path, required=True)
    parser.add_argument('--song-id', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    source = args.baseline_output.resolve()
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    sid = args.song_id
    baseline_report = next(r for r in json.loads((source/'prediction_report.json').read_text(encoding='utf-8'))
                           if str(r['song_id']) == sid)
    if baseline_report.get('pitch_fusion',{}).get('enabled'):
        raise ValueError('Baseline is already fused; regenerate baseline with --no-pitch-fusion.')
    selection = json.loads((v2.ROOT/'output/pitch_fusion_review/comparison.json').read_text(encoding='utf-8'))
    cp_path = v2.DEFAULT_OUTPUT/'structured/best.pt'
    old_path = Path(baseline_report['checkpoint'])
    if (file_digest(cp_path) != selection['checkpoint_sha256']['new']
            or file_digest(old_path) != selection['checkpoint_sha256']['production']):
        raise ValueError('Model versions differ from the evaluated pitch fusion pair.')
    cp = torch.load(cp_path,map_location='cpu',weights_only=False)
    if sid in cp['metadata']['train_ids'] + cp['metadata']['development_ids']:
        raise ValueError('This standalone comparison expects a song outside the new model training/development sets.')
    vocal, inst = Path(baseline_report['vocal']), Path(baseline_report['instrumental'])
    initial_hashes = {str(p):file_digest(p) for p in (vocal,inst,cp_path,old_path)}
    grid = json.loads((source/'grid_cache'/f'{sid}_grid.json').read_text(encoding='utf-8'))
    signature = {p.name:[p.stat().st_size,p.stat().st_mtime_ns] for p in (inst,vocal)}
    if signature != grid['source']['file_signature']:
        raise ValueError('Audio changed since baseline grid generation.')
    with np.load(source/'diagnostics'/f'{sid}_prediction.npz') as data:
        logits = torch.from_numpy(data['pitch_logits'].copy())
        modes = torch.from_numpy(data['rhythm_modes'].copy())
        original = [PredictedNote(int(a),int(b),int(p),float(data['onset_probability'][int(a)]),int(b-a))
            for a,b,p in zip(data['note_start_steps'],data['note_end_steps'],data['note_pitches'])]
    info = sf.info(vocal)
    if info.frames != grid['source']['num_samples'] or info.samplerate != grid['source']['sample_rate']:
        raise ValueError('Audio length/sample rate differs from the shared grid.')
    print('[rmvpe]',sid,flush=True)
    extractor = v2.EvidenceExtractor(device)
    audio = extractor.extract(vocal)
    extractor_hashes = extractor.fingerprint
    del extractor
    if device.type == 'cuda': torch.cuda.empty_cache()
    # This input has NO reference notes or targets, only audio and query times.
    song = {'acoustic':torch.cat((audio['salience'],(audio['mel']+5)/5),dim=-1).half(),
        'query_frames':torch.from_numpy(samples_at_ticks(grid,np.arange(len(logits))*40,16000)/160).float(),
        'grid':grid}
    model = v2.ScoreModel(**cp['model_config']).to(device).eval()
    model.load_state_dict(cp['model_state'],strict=True)
    new_outputs = v2.predict_outputs(model,song,device)
    weight = float(selection['selected_new_weight'])
    fused = fuse_pitches(original,logits,new_outputs['pitch'],weight)
    assert [(n.start_step,n.end_step) for n in original] == [(n.start_step,n.end_step) for n in fused]
    midi = out/f'{sid}_fused_predicted.mid'
    tempo = export_prediction(midi,grid,fused,modes)
    old_midi = source/f'{sid}_predicted.mid'
    old_seconds = midi_notes_in_wav_time(old_midi)
    new_seconds = midi_notes_in_wav_time(midi)
    assert [(a,b,v) for a,b,p,v in old_seconds] == [(a,b,v) for a,b,p,v in new_seconds]
    # Choose the same initial predicted-vocal segment for both, without labels.
    first = min((a for a,b,p,v in old_seconds if b > 0),default=0.)
    start = max(0.,first-1.)
    end = min(info.duration,start+40.)
    silence = np.zeros(int(np.ceil(end*44100)),dtype=np.float32)
    for tag,notes in [('baseline',old_seconds),('fused',new_seconds)]:
        preview = render_audition(silence,44100,start,end,notes=notes,vocal_gain=0,cue_gain=.6,clicks=False)
        assert np.isfinite(preview).all()
        sf.write(out/f'{sid}_{tag}_midi_only.wav',preview,44100,subtype='PCM_16')
    changes = [{'start_tick':a.start_step*40,'end_tick':a.end_step*40,'old_pitch':a.pitch,'fused_pitch':b.pitch}
        for a,b in zip(original,fused) if a.pitch != b.pitch]
    report = {'song_id':sid,'baseline_midi':str(old_midi),'fused_midi':str(midi),'tempo_map_midi':str(tempo),
        'new_pitch_weight':weight,'pitch_changes':changes,'num_changed_pitches':len(changes),
        'num_notes':len(fused),'boundaries_unchanged':True,'ppq':480,
        'bpm':grid['tempo']['bpm'],'tempo_sections':baseline_report['tempo_sections'],
        'tempo_events':grid['timeline']['tempo_events'],'wav_start_tick':grid['midi_alignment']['audio_start_tick'],
        'complete_bars':grid['midi_alignment']['grid_end_tick']//1920,
        'preview':{'start_seconds_in_original_wav':start,'end_seconds_in_original_wav':end,
            'selection':'First baseline predicted vocal onset minus 1 second, at most 40 seconds; no reference labels.'},
        'source_hashes':initial_hashes,'extractor_hashes':extractor_hashes,'reference_midi_used':False,
        'workflow_action':'None. No train, queue, registry, source MIDI/audio or active-model changes.',
        'note':'Unlabeled comparison: no accuracy claims; estimated tempo map shared by both outputs.'}
    assert all(file_digest(Path(p))==h for p,h in initial_hashes.items())
    v2.save_json(out/'comparison.json',report)
    print('[done]',json.dumps({k:report[k] for k in ('song_id','bpm','num_notes','num_changed_pitches','complete_bars','preview')}),flush=True)


if __name__=='__main__':
    main()
