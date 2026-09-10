"""Bounded, soft activity calibration with short-note and ABA regression checks."""
from transmelody.paths import module_path
from copy import deepcopy
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from transmelody.evaluation.compare_melody_midi import compare, edit_patterns
from transmelody.evaluation.melody_evaluation import (_note_coordinates, note_counts, summarize_note_counts,
    semitone_aba_counts, passes_pitch_aba_guard, file_digest)
from transmelody.models.melody_transformer import FeatureConfig, ModelConfig, MelodyTransformer
from transmelody.inference.melody_inference import predict_outputs, decode_prediction, export_prediction


def evaluate(notes,song):
    ref,pred = _note_coordinates(notes,song)
    result = {**summarize_note_counts(note_counts(notes,song)),**semitone_aba_counts(notes,song)}
    diagnostic = compare(ref,pred)
    short = []
    for start in song['onset'].nonzero().flatten().tolist():
        end=start+1
        while end < len(song['pitch']) and song['continuation'][end-1]:
            end+=1
        if end-start <= 6 and song['loss_mask'][start:end].all():
            short.append((start,end,int(song['pitch'][start])-1))
    # <= half a quarter-note beat includes real eighths/sixteenths/triplet eighths.
    # Match with tempo-aware timing, not a fixed millisecond duration cutoff.
    from transmelody.models.melody_transformer import PredictedNote
    _,short_ref = _note_coordinates([PredictedNote(a,b,p,1.,b-a) for a,b,p in short],song)
    short_comparison = compare(short_ref,pred)
    result['short_note_count'] = len(short)
    result['short_note_matches'] = short_comparison['metrics']['pitch_onset_matches']
    result['artifacts'] = edit_patterns(ref,pred)
    return result


@torch.inference_mode()
def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint',type=Path)
    parser.add_argument('--output',type=Path,default=Path('output/note_artifact_review'))
    parser.add_argument('--weights',type=float,nargs='+',default=[0.,.125,.25,.5])
    args=parser.parse_args()
    cp=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    config=FeatureConfig(**cp['feature_config'])
    model=MelodyTransformer(ModelConfig(**cp['model_config'])).eval().cuda()
    model.load_state_dict(cp['model_state'],strict=True)
    ids=cp['run_metadata']['validation_ids']
    rows={w:{'weight':w,'songs':{}} for w in args.weights}
    all_songs, all_outputs = {}, {}
    for sid in [*ids,'19']:
        song=torch.load(Path('output/melody_transformer/feature_cache')/f'{sid}_grid_features.pt',
            map_location='cpu',weights_only=False)
        outputs=predict_outputs(model,song['features'],torch.device('cuda'),cp['training_args'].get('bars_per_chunk',8)*48)
        all_songs[sid],all_outputs[sid]=song,outputs
        for w,row in rows.items():
            out={k:v for k,v in outputs.items() if k not in ('note_active_score','note_rest_score')}
            if w:
                out['note_active_score']=F.logsigmoid(out['activity'])*w/3
                out['note_rest_score']=F.logsigmoid(-out['activity'])*w/3
            notes,modes=decode_prediction(out,song['features'],config,
                triplet_threshold=cp['training_args'].get('triplet_confidence',.6))
            stats=evaluate(notes,song)
            row['songs'][sid]=stats
            print(sid,w,'F1',round(stats['full_note_f1'],4),'ABA',stats['aba_recovered'],
                'short',stats['short_note_matches'],'split',stats['artifacts']['split_reference_notes'],
                'rest',stats['artifacts']['mostly_reference_rest_notes'],flush=True)
    keys=('reference_notes','predicted_notes','onset_matches','pitch_onset_matches','full_note_matches')
    for row in rows.values():
        row['validation']={**summarize_note_counts({k:sum(row['songs'][sid][k] for sid in ids) for k in keys}),
            'songs':{sid:row['songs'][sid] for sid in ids}}
    base_weight=cp['model_config'].get('note_activity_weight',0.)
    base=rows[base_weight]['validation']
    eligible=[row for row in rows.values() if row['validation']['full_note_f1'] >= base['full_note_f1']
        and passes_pitch_aba_guard(row['validation'],base)
        and all(row['songs'][sid]['short_note_matches'] >= base['songs'][sid]['short_note_matches'] for sid in ids)
        and sum(row['songs'][sid]['artifacts']['split_reference_notes'] for sid in ids)
            <= sum(base['songs'][sid]['artifacts']['split_reference_notes'] for sid in ids)
        and sum(row['songs'][sid]['artifacts']['mostly_reference_rest_notes'] for sid in ids)
            <= sum(base['songs'][sid]['artifacts']['mostly_reference_rest_notes'] for sid in ids)]
    selected=max(eligible,key=lambda row:row['validation']['full_note_f1'])
    args.output.mkdir(parents=True,exist_ok=True)
    report={'checkpoint_sha256':file_digest(args.checkpoint),'development_ids':ids,
        'selected_weight':selected['weight'],'results':list(rows.values()),
        'selection':'Full-note/pitch-onset F1 must not regress. Each development song must retain short-note '
            'and semitone ABA matches. Aggregate split and reference-rest-note counts must not rise. '
            'Song19 is a reviewed training song and is not used for selection.',
        'source_sha256':{p:file_digest(module_path(p)) for p in ('evaluate_note_artifacts.py','note_event_model.py','melody_transformer.py')}}
    if selected['weight'] != base_weight:
        chosen=deepcopy(cp)
        chosen['model_config']['note_activity_weight']=selected['weight']
        chosen['training_args']['note_activity_weight']=selected['weight']
        chosen['epoch']=0
        chosen['history']=[{'epoch':0,'validation':selected['validation'],'decoder_calibration':True}]
        chosen['run_metadata']['decoder_calibration']={k:v for k,v in report.items() if k!='results'}
        torch.save(chosen,args.output/'best.pt')
    # Sample is always the selected candidate, never a label-snapped oracle.
    w=selected['weight']; out=dict(all_outputs['19']); song=all_songs['19']
    out.pop('note_active_score',None); out.pop('note_rest_score',None)
    if w:
        out['note_active_score']=F.logsigmoid(out['activity'])*w/3
        out['note_rest_score']=F.logsigmoid(-out['activity'])*w/3
    notes,modes=decode_prediction(out,song['features'],config,
        triplet_threshold=cp['training_args'].get('triplet_confidence',.6))
    export_prediction(args.output/'19_predicted.mid',song['grid'],notes,modes)
    (args.output/'evaluation.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print('Selected weight',w,flush=True)


if __name__=='__main__':
    main()
