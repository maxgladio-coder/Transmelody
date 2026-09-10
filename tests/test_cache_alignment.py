import json

import mido
import numpy as np
import soundfile as sf
import torch

import transmelody.models.melody_transformer as module
from transmelody.grid.musical_timeline import samples_at_ticks


def test_cache_uses_current_midi_and_masks_unrecorded_notes(tmp_path, monkeypatch):
    for name in ("vocal_audio", "inst_audio", "vocal_mid", "grid_cache", "note_cache"):
        (tmp_path / name).mkdir()
    for name, suffix in (("vocal_audio", "vocal"), ("inst_audio", "inst")):
        sf.write(tmp_path / name / f"1_{suffix}.wav", np.zeros(16000), 16000)
    midi = mido.MidiFile(ticks_per_beat=480)
    midi.tracks.append(mido.MidiTrack([
        mido.MetaMessage("set_tempo", tempo=500000),
        mido.Message("note_on", note=60, velocity=100, time=0),
        mido.Message("note_off", note=60, time=960),
        mido.Message("note_on", note=64, velocity=100, time=480),
        mido.Message("note_off", note=64, time=360),
        mido.MetaMessage("end_of_track", time=120),
    ]))
    midi.save(tmp_path / "vocal_mid/1_vocal.mid")
    grid = {"source": {"sample_rate": 16000, "num_samples": 16000},
        "meter": {"beats_per_bar": 4}, "tempo": {"bpm": 120}, "beats": [],
        "midi_alignment": {"audio_start_tick": 480, "grid_end_tick": 1920}}
    (tmp_path / "grid_cache/1_grid.json").write_text(json.dumps(grid))
    # Intentionally stale/empty: MIDI must take precedence.
    (tmp_path / "note_cache/1_notes.json").write_text("{}")
    song = {"song_id": "1", "vocal_audio": "vocal_audio/1_vocal.wav",
        "inst_audio": "inst_audio/1_inst.wav", "vocal_midi": "vocal_mid/1_vocal.mid",
        "grid": "grid_cache/1_grid.json", "labels": "note_cache/1_notes.json"}
    config = module.FeatureConfig(n_mels=4, context_offsets=(0,))
    def extract(path, canonical, feature_config, device):
        steps = canonical["midi_alignment"]["grid_end_tick"] // 40
        samples = samples_at_ticks(canonical, np.arange(steps) * 40, 16000)
        features = torch.zeros(steps, config.input_dim)
        features[:, config.audio_valid_index] = torch.from_numpy((samples >= 0) & (samples < 16000))
        return features.half(), 16000, 16000
    monkeypatch.setattr(module, "extract_grid_features", extract)
    path = module.build_song_cache(tmp_path, song, tmp_path / "features", config, torch.device("cpu"))
    cache = torch.load(path, map_location="cpu", weights_only=False)
    assert cache["onset"].nonzero().flatten().tolist() == [0, 36]
    assert cache["loss_mask"].nonzero().flatten().tolist() == list(range(12, 36))
    assert not cache["duration_mask"].any()
    assert not cache["boundary_mask"][12]
    assert cache["pitch"][0] == 61 and cache["pitch"][36] == 65
    assert cache["grid"]["timeline"]["source"] == "reviewed_midi"
    assert len(cache["source_fingerprint"]) == 4

    # Reviewed labels are optional, audio-relative, and override only reviewed coverage.
    from transmelody.lyrics.pronunciation_labels import SCHEMA, digest
    label_dir = tmp_path / "pronunciation_labels"
    label_dir.mkdir()
    label_path = label_dir / "1.json"
    reviewed = {"schema_version": SCHEMA, "status": "reviewed", "reviewer": "test", "song_id": "1",
        "time_reference": "original_wav_seconds", "audio_duration": 1.,
        "audio_sha256": digest(tmp_path / song["vocal_audio"]), "reviewed_regions": [[0., .6]],
        "units": [{"start": .2, "end": .4, "label": "a"}], "phones": []}
    label_path.write_text(json.dumps(reviewed))
    config = module.FeatureConfig(n_mels=4, context_offsets=(0,), use_articulation_features=True)
    module.build_song_cache(tmp_path, song, tmp_path / "features", config, torch.device("cpu"))
    with_labels = torch.load(path, map_location="cpu", weights_only=False)
    assert len(with_labels["source_fingerprint"]) == 5
    assert with_labels["reviewed_pronunciation_frames"] > 0
    assert with_labels["pronunciation_boundary"][17] > .9  # WAV .208 s, not score .708 s
    assert with_labels["pronunciation_weight"][17] == 1
    assert abs(float(with_labels["pronunciation_weight"][30]) - .1) < 1e-6  # not reviewed
    assert with_labels["pronunciation_weight"][0] == 0  # missing recording head
    # Same-size, same-mtime label edits still invalidate the cache.
    import os
    stat = label_path.stat()
    reviewed["units"][0]["start"] = .3
    label_path.write_text(json.dumps(reviewed))
    os.utime(label_path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    module.build_song_cache(tmp_path, song, tmp_path / "features", config, torch.device("cpu"))
    edited = torch.load(path, map_location="cpu", weights_only=False)
    assert edited["pronunciation_boundary"][17] < .01
    # Removing review labels must not leave stale strong supervision in the cache.
    label_path.unlink()
    module.build_song_cache(tmp_path, song, tmp_path / "features", config, torch.device("cpu"))
    removed = torch.load(path, map_location="cpu", weights_only=False)
    assert len(removed["source_fingerprint"]) == 4
    assert removed["reviewed_pronunciation_frames"] == 0
