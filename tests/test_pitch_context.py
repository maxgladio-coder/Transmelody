from dataclasses import replace

import torch

from melody_transformer import PredictedNote
from pitch_context import (PitchContextRefiner, PitchContextConfig, context_features, refine_logits,
    replace_note_pitches, tonal_context, tonality_report, note_inputs)
from train_pitch_context import pitch_targets, augment


def tokens(pitches):
    x = torch.full((len(pitches), 128), -10.)
    x[torch.arange(len(pitches)), torch.tensor(pitches)] = 10.
    starts = torch.arange(len(pitches)) * 12
    duration = torch.full_like(starts, 12)
    return x, context_features(x, starts, duration), starts, duration


def test_identity_initialization_and_overlapping_inference():
    x, f, _, _ = tokens([60, 61, 60] * 24)
    model = PitchContextRefiner().eval()
    result = refine_logits(model, x, f, torch.device('cpu'))
    assert torch.allclose(result, x, atol=1e-5)


def test_strong_chromatic_and_octave_evidence_cannot_be_pulled_to_key():
    x, f, _, _ = tokens([60, 61, 60, 72])
    model = PitchContextRefiner().eval()
    with torch.no_grad():
        model.residual.bias.fill_(-100)
        model.residual.bias[60] = 100
    result = model(x[None], f[None])[0]
    assert result.argmax(-1).tolist() == [60, 61, 60, 72]
    assert (result - x).abs().max() <= model.config.max_residual


def test_context_can_change_ambiguous_pitch_without_moving_boundaries():
    x, f, _, _ = tokens([60, 61, 60])
    x[1, 60], x[1, 61] = 9.5, 10.
    model = PitchContextRefiner().eval()
    with torch.no_grad():
        model.residual.bias[60] = 10
    result = model(x[None], f[None])[0]
    notes = [PredictedNote(i * 12, (i + 1) * 12, p, .9, 12) for i, p in enumerate([60, 61, 60])]
    new = replace_note_pitches(notes, result)
    assert new[1].pitch == 60
    assert [replace(n, pitch=0) for n in notes] == [replace(n, pitch=0) for n in new]


def test_tonal_hypotheses_can_change_region_and_preserve_uncertainty():
    x, _, start, duration = tokens([60, 64, 67, 60] * 12 + [66, 70, 73, 66] * 12)
    posterior, _, _, _ = tonal_context(x, start, duration)
    assert posterior[3].argmax().item() % 12 == 0
    assert posterior[-4].argmax().item() % 12 == 6
    assert torch.allclose(posterior.sum(-1), torch.ones(len(posterior)), atol=1e-5)
    empty = tonality_report(torch.empty(0, 128), torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long))
    assert empty['regions'][0]['key_candidate'] == 'uncertain'
    assert not empty['modulation_candidates']
    padded = tonality_report(x, start, duration, total_bars=30)
    assert padded['regions'][-1]['key_candidate'] == 'uncertain'
    assert padded['regions'][-1]['end_bar_1based'] == 30


def test_note_crossing_bar_contributes_to_both_bars():
    x, _, _, _ = tokens([60])
    _, _, mass, _ = tonal_context(x, torch.tensor([42]), torch.tensor([18]))
    assert mass.tolist() == [.5, 1.]


def test_pooling_matches_existing_pitch_and_does_not_read_reference():
    notes = [PredictedNote(0, 4, 5, .9, 4)]
    out = {'pitch': torch.randn(4, 129)}
    logits, f, _, _ = note_inputs(out, notes)
    assert torch.equal(logits[0], out['pitch'][1:4].mean(0)[1:])
    assert torch.equal(f, note_inputs(out, [replace(notes[0], pitch=100)])[1])


def test_mixed_pitch_segments_are_masked_for_training():
    notes = [PredictedNote(0, 4, 60, .9, 4), PredictedNote(4, 8, 60, .9, 4)]
    y, mask = pitch_targets(notes, {'pitch': torch.tensor([61, 61, 61, 61, 61, 61, 63, 63]),
        'loss_mask': torch.ones(8, dtype=torch.bool)})
    assert y[0] == 60 and mask.tolist() == [True, False]


def test_training_gradients_are_finite_and_augmentation_preserves_shapes():
    x, f, _, _ = tokens([60, 61, 60, 72])
    model = PitchContextRefiner(PitchContextConfig(d_model=32, layers=1))
    torch.nn.functional.cross_entropy(model(x[None], f[None])[0], torch.tensor([61, 60, 61, 71])).backward()
    assert model.residual.weight.grad.abs().sum() > 0
    assert torch.isfinite(model.residual.weight.grad).all()
    a, y = augment(x, torch.tensor([60, 61, 60, 72]), torch.ones(4, dtype=torch.bool))
    assert a.shape == x.shape and (y[1:] - y[:-1]).tolist() == [1, -1, 12]
