import torch

from melody_transformer import _event_labels_from_notes
from musical_boundary import HOLD, START
from joint_pitch_decoding import decode_pitch_boundary_sequence
from note_event_model import decode_note_events, note_event_loss


def fixture(length=48, cap=192):
    out = {'pitch': torch.full((length, 129), -8.),
        'boundary_state': torch.full((length, 4), -8.),
        'duration': torch.zeros(length, cap)}
    out['pitch'][:, 61] = 8
    out['boundary_state'][:, HOLD] = 8
    out['boundary_state'][0, START] = 20
    return out


def test_zero_duration_weight_matches_joint_markov_path():
    out = fixture()
    out['pitch'][12:24, 61] = -8
    out['pitch'][12:24, 62] = 8
    modes, observed = torch.tensor([0]), torch.ones(48, dtype=torch.bool)
    expected = decode_pitch_boundary_sequence(out, modes, observed, pitch_weight=.25)
    actual = decode_note_events(out, modes, observed, duration_weight=0)
    assert actual == expected


def test_learned_duration_rejects_extra_same_pitch_split():
    out = fixture(12)
    out['boundary_state'][6, START] = 9  # locally beats HOLD by 1
    out['duration'].fill_(-20)
    out['duration'][0, 11] = 20
    modes, observed = torch.tensor([0]), torch.ones(12, dtype=torch.bool)
    assert len(decode_note_events(out, modes, observed, duration_weight=0)) == 2
    notes = decode_note_events(out, modes, observed)
    assert [(n.start_step, n.end_step) for n in notes] == [(0, 12)]


def test_tail_bin_never_forces_maximum_note_length():
    out = fixture(480, cap=12)
    out['duration'][:, -1] = 10
    notes = decode_note_events(out, torch.zeros(10, dtype=torch.long), torch.ones(480, dtype=torch.bool))
    assert [(n.start_step, n.end_step) for n in notes] == [(0, 480)]


def test_unobserved_gap_cannot_be_crossed():
    out = fixture(24, cap=6)
    valid = torch.ones(24, dtype=torch.bool)
    valid[8:16] = False
    out['boundary_state'][18, START] = 20
    notes = decode_note_events(out, torch.tensor([0]), valid)
    assert all(n.end_step <= 8 or n.start_step >= 16 for n in notes)
    assert not decode_note_events(out, torch.tensor([0]), torch.zeros_like(valid))


def test_structured_event_loss_reaches_pitch_boundary_and_duration():
    target = _event_labels_from_notes({'notes': [
        {'pitch': 60, 'midi_time': {'start_tick': 120, 'end_tick': 360}},
        {'pitch': 61, 'midi_time': {'start_tick': 360, 'end_tick': 600}}]}, 1920)
    batch = {k: v[None] for k, v in target.items()}
    batch['valid'] = batch['loss_mask'] = torch.ones(1, 48, dtype=torch.bool)
    outputs = {k: v[None].clone().normal_().requires_grad_() for k, v in fixture().items()}
    loss = note_event_loss(outputs, batch)
    assert torch.isfinite(loss) and loss > 0
    loss.backward()
    assert all(v.grad.abs().sum() > 0 for v in outputs.values())


def test_partial_or_unobserved_notes_do_not_enter_event_loss():
    target = _event_labels_from_notes({'notes': [
        {'pitch': 60, 'midi_time': {'start_tick': 0, 'end_tick': 1920}}]}, 1920)
    batch = {k: v[None] for k, v in target.items()}
    batch['valid'] = batch['loss_mask'] = torch.ones(1, 48, dtype=torch.bool)
    outputs = {k: v[None].requires_grad_() for k, v in fixture().items()}
    assert note_event_loss(outputs, batch) == 0


def test_shared_prediction_dispatches_complete_notes_automatically():
    from melody_transformer import FeatureConfig
    from melody_inference import decode_prediction
    out = fixture()
    out.update(onset=torch.full((48,), -10.), offset=torch.full((48,), -10.),
        rhythm=torch.tensor([[10., -10.]] * 48), note_event_onset=torch.zeros(48))
    config = FeatureConfig(n_mels=4, context_offsets=(0,))
    features = torch.ones(48, config.input_dim)
    notes, modes = decode_prediction(out, features, config)
    assert notes == decode_note_events(out, modes, torch.ones(48, dtype=torch.bool))
