import json
from types import SimpleNamespace

import mido
import numpy as np
import pytest
import torch

from melody_transformer import (FeatureConfig, ModelConfig, MelodyTransformer,
    STEPS_PER_BAR, _event_labels_from_notes, decode_segment_events,
    decode_rhythm_modes, soft_pronunciation_context)
from melody_inference import predict_outputs, export_prediction
from melody_evaluation import song_split, note_counts, summarize_note_counts
from musical_timeline import canonicalize_grid, samples_at_ticks, tick_seconds, seconds_ticks
from tempo_map import build_tempo_segments


def grid_fixture():
    return {"source": {"sample_rate": 44100, "num_samples": 44100 * 8, "duration_sec": 8.0},
        "meter": {"beats_per_bar": 4}, "tempo": {"bpm": 120.0},
        "midi_alignment": {"audio_start_tick": 240, "grid_origin_sec": -0.25,
            "grid_end_tick": 9600, "grid_end_virtual_sec": 9.75},
        "beats": [{"tick": i * 480, "sec": i * 0.5 - 0.25,
                   "sample": round((i * 0.5 - 0.25) * 44100)} for i in range(21)]}


def test_tempo_map_and_feature_samples_agree_at_nonbeat_changes(tmp_path):
    events = [(0, mido.bpm2tempo(140)), (2080, mido.bpm2tempo(195))]
    grid = canonicalize_grid(grid_fixture(), events)
    path = tmp_path / "example_predicted.mid"
    export_prediction(path, grid, [], torch.zeros(5, dtype=torch.long))
    midi = mido.MidiFile(path)
    tempo_events = []
    tick = 0
    markers = []
    for message in midi.tracks[0]:
        tick += message.time
        if message.type == "set_tempo":
            tempo_events.append((tick, message.tempo))
        if message.type == "marker" and message.text.startswith("WAV START"):
            markers.append(tick)
    assert tempo_events == events
    assert markers == [240]
    queries = np.arange(0, grid["midi_alignment"]["grid_end_tick"], 40)
    expected = (tick_seconds(queries, tempo_events) - tick_seconds(markers[0], tempo_events)) * 44100
    assert np.max(abs(samples_at_ticks(grid, queries, 44100) - expected)) < 1e-6
    assert (tmp_path / "example_tempo_map.mid").exists()
    assert np.allclose(seconds_ticks(tick_seconds(queries, events), events), queries)


def test_integer_bpm_does_not_bypass_anchor_error_limit():
    grid = grid_fixture()
    grid["tempo"]["integer_snap_applied"] = True
    grid["midi_alignment"]["grid_origin_sec"] = 0.0
    grid["midi_alignment"]["grid_end_virtual_sec"] = 9600 / 480 * 60 / 119
    for b in grid["beats"]:
        b["sec"] = b["tick"] / 480 * 60 / 119
    result = build_tempo_segments(grid, max_anchor_error_ms=25)
    assert result[0].bpm == pytest.approx(119)
    assert all(s.max_anchor_error_ms <= 25 for s in result)


def test_low_confidence_triplet_is_not_promoted():
    logits = torch.tensor([[0.0, 0.1]] * STEPS_PER_BAR)
    assert decode_rhythm_modes(logits, triplet_threshold=0.6).tolist() == [0]


def outputs_fixture():
    output = {k: torch.full((12,), -10.0) for k in ("onset", "offset", "activity", "continuation", "pitch_change")}
    output["pitch"] = torch.full((12, 129), -10.0)
    output["pitch"][:, 61] = 10.0
    return output


def test_pitch_change_cannot_start_offgrid_or_in_silence():
    outputs = outputs_fixture()
    outputs["pitch_change"][[1, 3]] = 10.0
    assert decode_segment_events(outputs) == []
    outputs["activity"].fill_(10)
    notes = decode_segment_events(outputs)
    assert notes and all(n.start_step % 3 == 0 for n in notes)


def test_unrecorded_padding_cannot_invent_notes():
    outputs = outputs_fixture()
    outputs["onset"][[0, 3, 6]] = 10.0
    outputs["activity"].fill_(10)
    notes = decode_segment_events(outputs, observation_mask=torch.arange(12) >= 3)
    assert notes and min(n.start_step for n in notes) == 3


def test_cross_bar_note_can_start_straight_and_end_triplet():
    outputs = {k: torch.full((96,), -10.0) for k in ("onset", "offset", "activity", "continuation", "pitch_change")}
    outputs["pitch"] = torch.full((96, 129), -10.0)
    outputs["pitch"][:, 61] = 10
    outputs["activity"].fill_(10)
    outputs["continuation"].fill_(10)
    outputs["onset"][45] = 10
    outputs["continuation"][51] = -10
    outputs["offset"][52] = 10
    notes = decode_segment_events(outputs, rhythm_modes=torch.tensor([0, 1]))
    assert [(n.start_step, n.end_step) for n in notes] == [(45, 52)]


def test_sequence_encoding_distinguishes_bar_order_and_legacy_loads():
    torch.manual_seed(2026)
    model = MelodyTransformer(ModelConfig(input_dim=4, d_model=16, nhead=2,
        num_layers=1, dim_feedforward=32, dropout=0, use_sequence_position=True)).eval()
    features = torch.randn(1, 144, 4)
    positions = (torch.arange(144) % 48)[None]
    order = torch.cat([torch.arange(48, 96), torch.arange(48), torch.arange(96, 144)])
    with torch.no_grad():
        original = model(features, positions)["pitch"]
        permuted = model(features[:, order], positions)["pitch"]
    assert (permuted - original[:, order]).abs().max() > 0.01
    legacy_config = ModelConfig(input_dim=4, d_model=16, nhead=2, num_layers=1, dim_feedforward=32)
    legacy = MelodyTransformer(legacy_config)
    MelodyTransformer(legacy_config).load_state_dict(legacy.state_dict())


def test_speech_span_context_uses_boundaries_without_hard_partition():
    features = torch.tensor([[[0.0], [0.0], [10.0], [10.0]]])
    continuous = soft_pronunciation_context(features, torch.zeros(1, 4), 4)
    separated = soft_pronunciation_context(features, torch.tensor([[0.0, 0.0, 0.99, 0.0]]), 4)
    assert continuous[0, 1, 0] == 5
    assert 0 < separated[0, 1, 0] < 1


def test_pronunciation_network_backpropagates_from_midi_heads():
    config = FeatureConfig(n_mels=4, context_offsets=(0,), use_articulation_features=True)
    model = MelodyTransformer(ModelConfig(input_dim=config.input_dim, d_model=16, nhead=2,
        num_layers=1, dim_feedforward=32, use_sequence_position=True,
        use_articulation_heads=True, use_pronunciation_context=True,
        articulation_feature_start=config.articulation_start_index))
    x = torch.randn(2, 48, config.input_dim)
    outputs = model(x, torch.arange(48)[None].expand(2, -1))
    loss = outputs["pitch"].square().mean() + outputs["onset"].square().mean() + outputs["continuation"].square().mean()
    loss.backward()
    grad = model.pronunciation_head.weight.grad
    assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0


def test_overlap_inference_covers_tail_and_matches_pointwise_model():
    class Pointwise(torch.nn.Module):
        def forward(self, features, positions):
            return {"test": features[..., 0], "vector": features[..., :2]}
    x = torch.randn(213, 3)
    result = predict_outputs(Pointwise(), x, torch.device("cpu"), 96)
    assert torch.allclose(result["test"], x[:, 0], atol=1e-6)
    assert torch.allclose(result["vector"], x[:, :2], atol=1e-6)


def test_validation_split_stays_fixed_as_songs_are_added(tmp_path):
    path = tmp_path / "split.json"
    train, val = song_split([str(i) for i in range(1, 19)], path)
    train2, val2 = song_split([str(i) for i in range(1, 20)], path)
    assert val == val2 and not set(train) & set(val)
    assert "19" in train2
    with pytest.raises(ValueError):
        song_split(train2, path)


def test_note_metric_counts_duplicate_predictions_as_extra():
    grid = canonicalize_grid(grid_fixture(), [(0, 500000)], audio_start_tick=0)
    targets = _event_labels_from_notes({"notes": [{"pitch": 60, "midi_time": {"start_tick": 0, "end_tick": 480}}]}, 9600)
    song = {**targets, "grid": grid, "loss_mask": torch.ones(240, dtype=torch.bool)}
    predicted = [SimpleNamespace(start_step=0, end_step=12, pitch=60)] * 2
    result = summarize_note_counts(note_counts(predicted, song))
    assert result["full_note_matches"] == 1 and result["extra_notes"] == 1
    assert result["full_note_precision"] == 0.5


def test_damaged_audio_preserves_original_and_marks_recovery(tmp_path, monkeypatch):
    import soundfile as sf
    import normalize_audio_format as module
    source = tmp_path / "song.flac"
    sf.write(source, np.zeros(1000), 44100, format="FLAC")
    before = source.read_bytes()
    def fail(*args):
        raise sf.LibsndfileError(1)
    def recover(path, temporary, gain, frames):
        sf.write(temporary, np.zeros(frames), 44100, subtype="PCM_16")
        return "damaged frame"
    monkeypatch.setattr(module, "_convert_with_soundfile", fail)
    monkeypatch.setattr(module, "_convert_with_ffmpeg", recover)
    with pytest.raises(RuntimeError, match="alignment review"):
        module.convert_file(source, 1, target_path=source.with_suffix(".wav"), remove_source=True)
    assert source.read_bytes() == before and not source.with_suffix(".wav").exists()
    assert json.loads((tmp_path / "song.recovered.json").read_text())["internal_alignment_verified"] is False
