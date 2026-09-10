import torch
import pytest

from score_transcriber_v2 import ScoreModel, union_grid, pitch_kernel, score_loss
from note_event_model import decode_note_events


def test_pitch_evidence_preserves_semitone_candidates():
    kernel = pitch_kernel()
    assert kernel.shape == (360, 128)
    assert torch.allclose(kernel.sum(-1), torch.ones(360), atol=1e-5)
    # Published RMVPE mapping begins at 31.7 Hz, NOT an exact MIDI integer.
    # Every five 20-cent bins advances a semitone.
    assert kernel[183].argmax() == 60
    assert kernel[188].argmax() == 61
    assert kernel[182, 60] > 0 and kernel[182, 61] > 0


def test_union_keeps_both_rhythms_in_one_bar():
    assert union_grid(torch.arange(12)).nonzero().flatten().tolist() == [0, 3, 4, 6, 8, 9]


def ideal_outputs(spans, length):
    pitch = torch.full((length, 129), -20.)
    boundary = torch.full((length, 4), -20.)
    boundary[:, 0] = 20
    pitch[:, 0] = 20
    for a, b, p in spans:
        pitch[a:b] = -20; pitch[a:b, p+1] = 20
        boundary[a:b] = -20; boundary[a:b, 1] = 20
        boundary[a] = -20; boundary[a, 2] = 20
        if b < length:
            boundary[b] = -20; boundary[b, 3] = 20
    return {'pitch': pitch, 'boundary_state': boundary, 'duration': torch.zeros(length, 192)}


@pytest.mark.parametrize('spans,length', [
    ([(0, 3, 60), (3, 6, 61), (6, 12, 60)], 12),
    ([(0, 3, 60), (3, 12, 61), (12, 16, 62), (16, 20, 63), (20, 24, 64)], 24),
    ([(0, 480, 60)], 480),
    ([(12, 24, 60), (36, 48, 60)], 48),
])
def test_joint_decode_short_aba_mixed_rhythm_long_notes_and_rests(spans, length):
    notes = decode_note_events(ideal_outputs(spans, length), None, torch.ones(length, dtype=torch.bool),
        allowed_boundary_mask=union_grid(torch.arange(length)))
    assert [(n.start_step, n.end_step, n.pitch) for n in notes] == spans


def test_candidate_mask_shape_checked():
    with pytest.raises(ValueError, match='candidate mask'):
        decode_note_events(ideal_outputs([], 12), None, torch.ones(12, dtype=torch.bool),
            allowed_boundary_mask=torch.ones(13))


def test_model_inputs_do_not_read_score_labels_and_alignments_receive_gradients():
    torch.manual_seed(1)
    model = ScoreModel(width=32, layers=1).eval()
    n = 48
    batch = {'acoustic': torch.rand(1, 200, 488), 'query': torch.linspace(25, 175, n)[None],
        'audio_length': torch.tensor([200]), 'position': torch.arange(n)[None],
        'valid': torch.ones(1, n, dtype=torch.bool)}
    a = model(batch)
    batch.update(pitch=torch.full((1,n), 61), onset=torch.ones(1,n), duration=torch.ones(1,n))
    b = model(batch)
    assert all(torch.equal(a[k], b[k]) for k in a)
    sum(x.square().mean() for x in a.values()).backward()
    assert model.query_score.weight.grad.abs().sum() > 0
    assert model.prior_gain.grad.abs() > 0


def test_empty_audio_label_chunk_loss_is_finite():
    from melody_transformer import _event_labels_from_notes
    labels = _event_labels_from_notes({'notes': []}, 1920)
    batch = {k: v[None] for k, v in labels.items()}
    for k in ('valid', 'loss_mask', 'boundary_mask', 'continuation_mask'):
        batch[k] = torch.ones(1, 48, dtype=torch.bool)
    batch['position'] = torch.arange(48)[None]
    outputs = {k: v[None].requires_grad_() for k,v in ideal_outputs([], 48).items()}
    outputs['activity'] = torch.zeros(1,48,requires_grad=True)
    loss = score_loss(outputs, batch)
    assert torch.isfinite(loss)
    loss.backward()


def test_structured_supervision_uses_union_without_target_bar_rhythm():
    from melody_transformer import _event_labels_from_notes
    from note_event_model import note_event_loss
    labels = _event_labels_from_notes({'notes': [
        {'pitch': 60, 'midi_time': {'start_tick': 120, 'end_tick': 480}},
        {'pitch': 61, 'midi_time': {'start_tick': 480, 'end_tick': 640}},
        {'pitch': 60, 'midi_time': {'start_tick': 640, 'end_tick': 960}}]}, 1920)
    batch = {k: v[None] for k,v in labels.items() if k != 'rhythm_mode'}
    batch['valid'] = batch['loss_mask'] = torch.ones(1,48,dtype=torch.bool)
    outputs = {k: torch.randn_like(v)[None].requires_grad_() for k,v in ideal_outputs([],48).items()}
    loss = note_event_loss(outputs, batch, allowed_boundary_mask=union_grid(torch.arange(48)),
        semitone_contrasts=True, artifact_contrasts=True)
    assert torch.isfinite(loss) and loss > 0
    loss.backward()
    assert all(v.grad.abs().sum() > 0 for v in outputs.values())


def test_inference_works_without_any_midi_note_labels():
    from score_transcriber_v2 import predict
    song = {'acoustic': torch.rand(200,488), 'query_frames': torch.linspace(-20,220,48),
        'grid': {'source': {'num_samples': 32000, 'sample_rate': 16000}}}
    model = ScoreModel(width=32,layers=1)
    notes = predict(model, song, torch.device('cpu'), steps=24)
    assert all(n.start_step >= 4 and n.end_step <= 44 for n in notes)
