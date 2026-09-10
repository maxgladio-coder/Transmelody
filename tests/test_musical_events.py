import torch

from transmelody.models.melody_transformer import ModelConfig, MelodyTransformer
from transmelody.models.musical_event_decoding import event_peaks, select_event_grids, project_event_peaks, decode_musical_events


def fixture(length=48):
    outputs = {key: torch.full((length,), -10.) for key in ("onset", "offset", "pitch_change", "articulation", "pronunciation")}
    outputs.update(activity=torch.full((length,), 10.), continuation=torch.full((length,), 10.),
        rhythm=torch.tensor([[1., 0.]] * length), pitch=torch.full((length, 129), -10.))
    outputs["pitch"][:, 61] = 10.
    return outputs


def test_off_grid_neural_peak_is_assigned_instead_of_erased():
    output = fixture()
    output["onset"][2] = 10.
    projected = project_event_peaks(output["onset"], torch.tensor([0]), torch.ones(48, dtype=torch.bool))
    assert event_peaks(projected.sigmoid()) == [3]
    assert event_peaks(output["onset"].sigmoid()) == [2]


def test_real_same_vowel_semitone_aba_and_same_pitch_rearticulation_survive():
    output = fixture(12)
    output["onset"][[0, 3, 6, 9]] = 10.
    output["pitch"][4:6, 61] = -10.
    output["pitch"][4:6, 62] = 10.
    notes, _ = decode_musical_events(output, torch.ones(12, dtype=torch.bool), output_grid="1/16")
    assert [n.start_step for n in notes] == [0, 3, 6, 9]
    assert [n.pitch for n in notes] == [60, 61, 60, 60]


def test_phoneme_and_pitch_change_heads_alone_cannot_cut_long_note():
    output = fixture(12)
    output["onset"][0] = 10.
    output["articulation"][[3, 6, 9]] = 10.
    output["pitch_change"][[3, 6, 9]] = 10.
    output["pronunciation"].fill_(10.)
    notes, _ = decode_musical_events(output, torch.ones(12, dtype=torch.bool), output_grid="1/16")
    assert len(notes) == 1 and notes[0].end_step == 12


def test_rhythm_uses_unmasked_events_and_shared_positions_do_not_choose_triplets():
    output = fixture()
    output["onset"][[4, 8, 16, 20, 28, 32]] = 10.
    assert select_event_grids(output, torch.ones(48, dtype=torch.bool)).tolist() == [1]
    output["onset"].fill_(-10.)
    output["onset"][[0, 12, 24, 36]] = 10.
    assert select_event_grids(output, torch.ones(48, dtype=torch.bool)).tolist() == [0]


def test_unobserved_frames_cannot_create_projected_events():
    output = fixture()
    output["onset"][2] = 10.
    valid = torch.ones(48, dtype=torch.bool)
    valid[2] = False
    assert not event_peaks(project_event_peaks(output["onset"], torch.tensor([0]), valid).sigmoid())


def test_midi_event_loss_reaches_local_acoustic_attention():
    config = ModelConfig(input_dim=13 * 4 + 1, d_model=16, nhead=2, num_layers=1,
        dim_feedforward=32, dropout=0., use_musical_event_context=True, musical_event_version=2, acoustic_bins=4)
    model = MelodyTransformer(config)
    output = model(torch.randn(2, 12, config.input_dim), torch.arange(12)[None].repeat(2, 1))
    (output["onset"].square().mean() + output["pitch"].square().mean()).backward()
    assert model.event_query.weight.grad.abs().sum() > 0
    assert model.event_frame_projection[1].weight.grad.abs().sum() > 0
    assert model.event_temporal.weight.grad.abs().sum() > 0
    assert torch.equal(output["musical_event_onset"], output["onset"])


def test_onset_is_invariant_to_bar_embedding_position_and_phoneme_gate():
    config = ModelConfig(input_dim=13 * 4 + 1, d_model=16, nhead=2, num_layers=1,
        dim_feedforward=32, dropout=0., use_musical_event_context=True, musical_event_version=2, acoustic_bins=4)
    model = MelodyTransformer(config).eval()
    features = torch.randn(1, 12, config.input_dim)
    a = model(features, torch.arange(12)[None])["onset"]
    b = model(features, ((torch.arange(12) + 7) % 48)[None])["onset"]
    assert torch.equal(a, b)


def test_first_experiment_checkpoint_layout_remains_loadable():
    config = ModelConfig(input_dim=53, d_model=16, nhead=2, num_layers=1,
        dim_feedforward=32, dropout=0., use_musical_event_context=True, acoustic_bins=4)
    original = MelodyTransformer(config)
    reloaded = MelodyTransformer(config)
    reloaded.load_state_dict(original.state_dict(), strict=True)
    assert not hasattr(reloaded, "event_temporal")
