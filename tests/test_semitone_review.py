import torch

from compare_melody_midi import compare
from melody_transformer import _event_labels_from_notes
from note_event_model import note_event_loss, decode_note_events
from musical_boundary import HOLD, START
from train_melody import supervised_pitch_loss


def test_aba_comparison_distinguishes_wrong_pitch_from_missing_boundary():
    ref = [(0, .5, 60), (.5, 1., 61), (1., 1.5, 60)]
    out = compare(ref, [(a,b,60) for a,b,_ in ref])
    assert out['metrics']['onset_matches'] == 3
    assert out['onset_matched_wrong_pitch'] == 1
    assert out['semitone_aba']['count'] == 1
    assert out['semitone_aba']['recovered'] == 0
    assert compare(ref, ref)['semitone_aba']['consecutive'] == 1
    assert compare(ref, [])['metrics']['onset_matches'] == 0
    assert compare([], [])['semitone_aba']['count'] == 0


def test_rest_supervision_reaches_pitch_head_but_not_padding():
    logits = torch.zeros(1, 3, 129, requires_grad=True)
    target, valid = torch.tensor([[61, 0, 0]]), torch.tensor([[True, True, False]])
    supervised_pitch_loss(logits, target, valid, supervise_rest=True).backward()
    assert logits.grad[0, 1, 0] < 0
    assert logits.grad[0, 0, 61] < 0
    assert logits.grad[0, 2].abs().sum() == 0


def test_semitone_aba_is_not_vetoed_by_decoder():
    out = {'pitch': torch.full((18,129), -10.),
        'boundary_state': torch.full((18,4), -10.), 'duration': torch.zeros(18,192)}
    out['boundary_state'][:, HOLD] = 8
    for start, pitch in [(0,60), (6,61), (12,60)]:
        out['pitch'][start:start+6,pitch+1] = 10
        out['boundary_state'][start, START] = 20
    notes = decode_note_events(out, torch.tensor([0]), torch.ones(18,dtype=torch.bool))
    assert [(n.start_step,n.end_step,n.pitch) for n in notes] == [(0,6,60),(6,12,61),(12,18,60)]


def test_supervised_aba_loss_penalizes_constant_middle_pitch():
    targets = _event_labels_from_notes({'notes': [
        {'pitch': p, 'midi_time': {'start_tick': a, 'end_tick': b}}
        for a,b,p in [(120,360,60),(360,600,61),(600,840,60)]]},1920)
    batch = {k:v[None] for k,v in targets.items()}
    batch['valid'] = batch['loss_mask'] = torch.ones(1,48,dtype=torch.bool)
    out = {'pitch':torch.zeros(1,48,129), 'boundary_state':torch.zeros(1,48,4),
        'duration':torch.zeros(1,48,192)}
    out['pitch'][:,:,61] = 8  # Incorrect AAA, even though boundaries can be right.
    out = {k:v.requires_grad_() for k,v in out.items()}
    loss = note_event_loss(out,batch,semitone_contrasts=True,max_notes=0)
    assert loss > 0
    loss.backward()
    assert out['pitch'].grad[0,9:15,62].sum() < 0
    assert out['pitch'].grad[0,9:15,61].sum() > 0
    assert all(v.grad.abs().sum() > 0 for v in out.values())


def test_pitch_only_adaptation_does_not_update_boundary_parameters():
    from melody_transformer import MelodyTransformer, ModelConfig
    from train_melody import freeze_except_pitch_head
    model = MelodyTransformer(ModelConfig(input_dim=8, d_model=8, nhead=2, num_layers=1,
        dim_feedforward=16))
    freeze_except_pitch_head(model)
    assert {name for name,p in model.named_parameters() if p.requires_grad} == {
        'pitch_head.weight', 'pitch_head.bias'}
    before = {name:p.detach().clone() for name,p in model.named_parameters()}
    model.eval()
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=.1)
    outputs = model(torch.randn(1,6,8), torch.arange(6)[None])
    outputs['pitch'][...,61].sum().backward()
    optimizer.step()
    for name,p in model.named_parameters():
        if not name.startswith('pitch_head.'):
            assert torch.equal(p,before[name])
    assert not torch.equal(model.pitch_head.weight,before['pitch_head.weight'])


def test_validation_guard_does_not_hide_one_song_aba_regression():
    from melody_evaluation import passes_pitch_aba_guard, semitone_aba_counts
    from melody_transformer import PredictedNote
    from musical_timeline import canonicalize_grid
    baseline = {'pitch_onset_f1':.8,'songs':{'4':{'aba_recovered':3},'11':{'aba_recovered':10}}}
    candidate = {'pitch_onset_f1':.82,'songs':{'4':{'aba_recovered':2},'11':{'aba_recovered':12}}}
    assert not passes_pitch_aba_guard(candidate,baseline)
    candidate['songs']['4']['aba_recovered'] = 3
    assert passes_pitch_aba_guard(candidate,baseline)
    targets = _event_labels_from_notes({'notes': [
        {'pitch':p,'midi_time':{'start_tick':a,'end_tick':b}}
        for a,b,p in [(120,360,60),(360,600,61),(600,840,60)]]},1920)
    grid = canonicalize_grid({'source':{'sample_rate':16000,'num_samples':32000},
        'meter':{'beats_per_bar':4},'tempo':{'bpm':120},'beats':[],
        'midi_alignment':{'audio_start_tick':0,'grid_end_tick':1920}},[(0,500000)])
    song = {**targets,'grid':grid,'loss_mask':torch.ones(48,dtype=torch.bool)}
    notes = [PredictedNote(a,b,p,1.,b-a) for a,b,p in [(3,9,60),(9,15,61),(15,21,60)]]
    assert semitone_aba_counts(notes,song) == {'aba_count':1,'aba_recovered':1}
