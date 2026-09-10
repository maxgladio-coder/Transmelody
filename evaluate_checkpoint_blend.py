"""Bounded development-only checkpoint interpolation with an ABA regression guard."""
from copy import deepcopy
import json
from pathlib import Path

import torch

from compare_melody_midi import compare
from melody_evaluation import note_counts, summarize_note_counts, file_digest
from melody_inference import predict_outputs, decode_prediction
from melody_transformer import MelodyTransformer, ModelConfig, FeatureConfig
from musical_timeline import samples_at_ticks


@torch.inference_mode()
def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('base', type=Path)
    parser.add_argument('adapted', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    a = torch.load(args.base, map_location='cpu', weights_only=False)
    b = torch.load(args.adapted, map_location='cpu', weights_only=False)
    for key in ('model_config','feature_config'):
        if a[key] != b[key]:
            raise ValueError('Cannot blend different architectures/features.')
    ids = a['run_metadata']['validation_ids']
    if set(ids) != set(b['run_metadata']['validation_ids']) or any(
            set(ids) & set(cp['run_metadata']['train_ids']) for cp in (a,b)):
        raise ValueError('Incompatible or contaminated development split.')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = FeatureConfig(**a['feature_config'])
    model = MelodyTransformer(ModelConfig(**a['model_config'])).to(device).eval()
    songs = [torch.load(Path('output/melody_transformer/feature_cache') / f'{sid}_grid_features.pt',
        map_location='cpu', weights_only=False) for sid in ids]
    rows = []
    for alpha in (0., .1, .25, .5):
        state = {k: torch.lerp(v, b['model_state'][k], alpha) if v.is_floating_point() else v
            for k,v in a['model_state'].items()}
        model.load_state_dict(state, strict=True)
        counts, per_song, aba_count, aba_recovered = {}, {}, 0, 0
        for song in songs:
            outputs = predict_outputs(model, song['features'], device,
                a['training_args'].get('bars_per_chunk',8)*48)
            notes,_ = decode_prediction(outputs,song['features'],config,
                triplet_threshold=a['training_args'].get('triplet_confidence',.6))
            current = note_counts(notes,song)
            for k,v in current.items():
                counts[k] = counts.get(k,0)+v
            truth = []
            for start in song['onset'].nonzero().flatten().tolist():
                end = start+1
                while end < len(song['pitch']) and song['continuation'][end-1]:
                    end+=1
                if song['loss_mask'][start:end].all():
                    truth.append((start,end,int(song['pitch'][start])-1))
            def coords(items):
                sr = song['grid']['source']['sample_rate']
                return [(*list(samples_at_ticks(song['grid'],[x[0]*40,x[1]*40],sr)/sr),x[2]) for x in items]
            diagnostic = compare(coords(truth),coords([(n.start_step,n.end_step,n.pitch) for n in notes]))
            aba = diagnostic['semitone_aba']
            aba_count += aba['count']
            aba_recovered += aba['recovered']
            per_song[song['song_id']] = {**summarize_note_counts(current),
                'aba_count':aba['count'],'aba_recovered':aba['recovered']}
        validation = {**summarize_note_counts(counts),'songs':per_song,
            'aba_count':aba_count,'aba_recovered':aba_recovered}
        rows.append({'alpha':alpha,'validation':validation})
        print('alpha',alpha,'fullF1',validation['full_note_f1'],
            'pitchF1',validation['pitch_onset_f1'],'ABA',aba_recovered,'/',aba_count,flush=True)
    baseline = rows[0]['validation']
    eligible = [r for r in rows if r['validation']['full_note_f1'] >= baseline['full_note_f1']
        and r['validation']['pitch_onset_f1'] >= baseline['pitch_onset_f1']
        and all(r['validation']['songs'][sid]['aba_recovered'] >= baseline['songs'][sid]['aba_recovered']
            for sid in ids)]
    best = max(eligible,key=lambda r:r['validation']['full_note_f1'])
    args.output.mkdir(parents=True,exist_ok=True)
    report = {'base_sha256':file_digest(args.base),'adapted_sha256':file_digest(args.adapted),
        'development_ids':ids,'results':rows,'selected_alpha':best['alpha'],
        'rule':'Select highest complete-note F1, with no aggregate pitch-onset F1 regression '
        'or per-development-song semitone ABA recall regression. No song19 metrics used.',
        'source_sha256':file_digest(Path(__file__))}
    if best['alpha']:
        cp = deepcopy(b)
        cp['model_state'] = {k:torch.lerp(v,b['model_state'][k],best['alpha']) if v.is_floating_point() else v
            for k,v in a['model_state'].items()}
        cp['epoch'] = 0
        cp['history'] = [{'epoch':0,'validation':best['validation'],'checkpoint_interpolation':True}]
        cp['optimizer_state'] = {}  # Not resumable optimizer state after interpolation.
        cp['run_metadata']['checkpoint_interpolation'] = report
        cp['run_metadata']['train_ids'] = sorted(set(a['run_metadata']['train_ids']) |
            set(b['run_metadata']['train_ids']),key=int)
        torch.save(cp,args.output/'best.pt')
    (args.output/'evaluation.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print('Selected alpha:',best['alpha'],flush=True)


if __name__ == '__main__':
    main()
