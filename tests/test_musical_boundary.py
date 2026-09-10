import torch
import torch.nn.functional as F

from transmelody.models.melody_transformer import ModelConfig, MelodyTransformer
from transmelody.models.musical_boundary import REST, HOLD, START, END, boundary_targets, decode_boundary_states
from transmelody.models.musical_event_decoding import decode_musical_events


def fixture(length=48):
    outputs = {k: torch.full((length,), 10.) for k in
        ('onset', 'offset', 'activity', 'continuation', 'articulation', 'pitch_change')}
    outputs['rhythm'] = torch.tensor([[10., -10.]] * length)
    outputs['boundary_state'] = torch.full((length, 4), -10.)
    outputs['boundary_state'][:, HOLD] = 10.
    outputs['pitch'] = torch.full((length, 129), -10.)
    outputs['pitch'][:, 61] = 10.
    return outputs


def set_state(out, step, state):
    out['boundary_state'][step].fill_(-10.)
    out['boundary_state'][step, state] = 10.


def test_joint_hold_wins_over_conflicting_auxiliary_peaks():
    out = fixture(12)
    set_state(out, 0, START)
    notes, _ = decode_musical_events(out, torch.ones(12, dtype=torch.bool), output_grid='1/16')
    assert [(n.start_step, n.end_step) for n in notes] == [(0, 12)]


def test_joint_start_preserves_same_pitch_and_semitone_aba():
    out = fixture(12)
    for step in (0, 3, 6, 9):
        set_state(out, step, START)
    out['pitch'][3:6, 61] = -10.
    out['pitch'][3:6, 62] = 10.
    notes = decode_boundary_states(out, torch.tensor([0]), torch.ones(12, dtype=torch.bool))
    assert [n.pitch for n in notes] == [60, 61, 60, 60]
    assert [n.start_step for n in notes] == [0, 3, 6, 9]


def test_rest_end_and_orphan_hold_do_not_create_notes():
    out = fixture(24)
    for step, state in ((0, REST), (3, START), (9, END), (15, START), (21, REST)):
        set_state(out, step, state)
    notes = decode_boundary_states(out, torch.tensor([0]), torch.ones(24, dtype=torch.bool))
    assert [(n.start_step, n.end_step) for n in notes] == [(3, 9), (15, 21)]


def test_edges_use_their_own_bar_grid():
    out = fixture(96)
    set_state(out, 45, START)  # straight bar
    set_state(out, 51, END)    # illegal in next triplet bar
    set_state(out, 52, END)    # legal triplet endpoint
    notes = decode_boundary_states(out, torch.tensor([0, 1]), torch.ones(96, dtype=torch.bool))
    assert [(n.start_step, n.end_step) for n in notes] == [(45, 52)]


def test_no_notes_across_unobserved_audio():
    out = fixture(12)
    set_state(out, 0, START)
    set_state(out, 9, START)
    valid = torch.ones(12, dtype=torch.bool)
    valid[6:] = False
    notes = decode_boundary_states(out, torch.tensor([0]), valid)
    assert [(n.start_step, n.end_step) for n in notes] == [(0, 6)]


def test_targets_shift_continuation_and_preserve_rearticulation():
    batch = {k: torch.zeros(1, 10) for k in ('onset', 'offset', 'continuation')}
    batch.update({k: torch.ones(1, 10, dtype=torch.bool)
        for k in ('valid', 'boundary_mask', 'continuation_mask')})
    batch['onset'][0, [1, 4]] = 1
    batch['offset'][0, [4, 7]] = 1
    batch['continuation'][0, [1, 2, 4, 5]] = 1
    target, valid = boundary_targets(batch)
    assert target.tolist() == [[REST, START, HOLD, HOLD, START, HOLD, HOLD, END, REST, REST]]
    assert not valid[0, 0] and valid[0, 1:].all()
    batch['valid'][0, 5:] = False
    assert not boundary_targets(batch)[1][0, 5:].any()


def test_state_supervision_reaches_both_branches_and_checkpoint_reloads():
    config = ModelConfig(input_dim=53, d_model=16, nhead=2, num_layers=1,
        dim_feedforward=32, dropout=0., use_musical_event_context=True,
        musical_event_version=3, acoustic_bins=4)
    model = MelodyTransformer(config)
    out = model(torch.randn(2, 12, 53), torch.arange(12)[None].repeat(2, 1))
    F.cross_entropy(out['boundary_state'].reshape(-1, 4), torch.arange(24) % 4).backward()
    for layer in (model.boundary_state_head[0], model.event_temporal, model.continuation_head,
                  model.event_query, model.event_onset_head):
        assert layer.weight.grad.abs().sum() > 0
    MelodyTransformer(config).load_state_dict(model.state_dict(), strict=True)
