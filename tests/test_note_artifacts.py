import torch
import torch.nn.functional as F

from transmelody.evaluation.compare_melody_midi import edit_patterns
from transmelody.models.melody_transformer import _event_labels_from_notes, MelodyTransformer, ModelConfig
from transmelody.models.note_event_model import decode_note_events, note_event_loss
from transmelody.models.musical_boundary import START,HOLD,REST,END


def test_review_patterns_separate_splits_from_rest_insertions_and_real_shorts():
    ref=[(0,1,60),(2,2.1,61)]
    out=edit_patterns(ref,[(0,.1,59),(.1,1,60),(1.5,1.6,60),(2,2.1,61)])
    assert out['split_reference_notes']==1
    assert out['mostly_reference_rest_notes']==1


def test_edge_split_is_supervised_even_when_middle_split_looks_unlikely():
    target=_event_labels_from_notes({'notes':[{'pitch':60,'midi_time':{'start_tick':120,'end_tick':1320}}]},1920)
    batch={k:v[None] for k,v in target.items()}
    batch['valid']=batch['loss_mask']=torch.ones(1,48,dtype=torch.bool)
    out={'pitch':torch.zeros(1,48,129),'boundary_state':torch.zeros(1,48,4),'duration':torch.zeros(1,48,192)}
    out['pitch'][:,:,61]=8
    out['boundary_state'][:,:,HOLD]=8
    out['boundary_state'][0,3,START]=12
    out['boundary_state'][0,6,START]=20  # false short head split
    out={k:v.requires_grad_() for k,v in out.items()}
    loss=note_event_loss(out,batch,artifact_contrasts=True)
    loss.backward()
    assert out['boundary_state'].grad[0,6,START]>0
    assert out['boundary_state'].grad[0,6,HOLD]<0


def test_empty_reviewed_span_trains_rest_instead_of_hallucinated_pitch():
    target=_event_labels_from_notes({'notes':[]},1920)
    batch={k:v[None] for k,v in target.items()}
    batch['valid']=batch['loss_mask']=torch.ones(1,48,dtype=torch.bool)
    out={k:torch.zeros(*shape,requires_grad=True) for k,shape in
        [('pitch',(1,48,129)),('boundary_state',(1,48,4)),('duration',(1,48,192))]}
    loss=note_event_loss(out,batch,artifact_contrasts=True)
    assert loss>0
    loss.backward()
    assert out['pitch'].grad[:,:,0].sum()<0
    assert out['boundary_state'].grad[:,:,REST].sum()<0
    batch['loss_mask'].zero_()
    assert note_event_loss(out,batch,artifact_contrasts=True)==0


def test_soft_activity_is_not_a_short_note_or_semitone_veto():
    length=12
    out={'pitch':torch.full((length,129),-8.),'boundary_state':torch.full((length,4),-8.),
        'duration':torch.zeros(length,192)}
    out['boundary_state'][:,HOLD]=5
    for start,pitch in [(0,60),(3,61),(6,60)]:
        end=6 if start==3 else 3 if start==0 else 12
        out['pitch'][start:end,pitch+1]=8
        out['boundary_state'][start,START]=20
    # Even low activity is only soft evidence; strong note evidence can win.
    out['note_active_score']=F.logsigmoid(torch.full((length,),-2.))*.5/3
    out['note_rest_score']=F.logsigmoid(torch.full((length,),2.))*.5/3
    notes=decode_note_events(out,torch.tensor([0]),torch.ones(length,dtype=torch.bool))
    assert [(n.start_step,n.end_step,n.pitch) for n in notes]==[(0,3,60),(3,6,61),(6,12,60)]


def test_activity_score_reaches_activity_head_in_structured_loss():
    target=_event_labels_from_notes({'notes':[]},1920)
    batch={k:v[None] for k,v in target.items()}
    batch['valid']=batch['loss_mask']=torch.ones(1,48,dtype=torch.bool)
    activity=torch.full((1,48),3.,requires_grad=True)
    out={'pitch':torch.zeros(1,48,129,requires_grad=True),'boundary_state':torch.zeros(1,48,4,requires_grad=True),
        'duration':torch.zeros(1,48,192,requires_grad=True),
        'note_active_score':F.logsigmoid(activity)*.5/3,'note_rest_score':F.logsigmoid(-activity)*.5/3}
    note_event_loss(out,batch,artifact_contrasts=True).backward()
    assert activity.grad.sum()>0


def test_artifact_guard_rejects_loss_of_real_short_notes():
    from copy import deepcopy
    from transmelody.evaluation.melody_evaluation import passes_artifact_guard
    base={'pitch_onset_f1':.8,'songs':{'18':{'aba_recovered':4,'short_note_matches':169,
        'split_reference_notes':9,'mostly_reference_rest_notes':5}}}
    proposed=deepcopy(base)
    proposed['songs']['18']['split_reference_notes']=5
    proposed['songs']['18']['mostly_reference_rest_notes']=3
    assert passes_artifact_guard(proposed,base)
    proposed['songs']['18']['short_note_matches']=168
    assert not passes_artifact_guard(proposed,base)


def test_overlapping_activity_calibration_uses_combined_logits():
    from types import SimpleNamespace
    from transmelody.inference.melody_inference import predict_outputs
    class Model:
        config=SimpleNamespace(note_activity_weight=.5)
        def __call__(self,features,positions):
            activity=torch.linspace(-6,6,features.shape[1])[None]
            return {'activity':activity,'note_active_score':F.logsigmoid(activity)*.5/3,
                'note_rest_score':F.logsigmoid(-activity)*.5/3}
    out=predict_outputs(Model(),torch.ones(72,1),torch.device('cpu'),48)
    assert torch.allclose(out['note_active_score'],F.logsigmoid(out['activity'])*.5/3)
    assert torch.allclose(out['note_rest_score'],F.logsigmoid(-out['activity'])*.5/3)
