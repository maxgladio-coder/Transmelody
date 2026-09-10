import json
import sys
import pytest

import torch

import transmelody.training.train_melody as train_melody
from transmelody.models.melody_transformer import FeatureConfig, _event_labels_from_notes
from transmelody.grid.musical_timeline import canonicalize_grid


@pytest.mark.parametrize('complete_notes', [False, True])
def test_train_validate_select_checkpoint_without_publishing(tmp_path, monkeypatch, complete_notes):
    root = tmp_path / "dataset"
    (root / "vocal_mid").mkdir(parents=True)
    output = tmp_path / "models"
    output.mkdir()
    (output / "final.pt").write_bytes(b"previous working model")
    config = FeatureConfig(use_articulation_features=True)
    (root / "manifest.jsonl").write_text("".join(json.dumps({"song_id": sid,
        "vocal_midi": f"vocal_mid/{sid}_vocal.mid"}) + "\n" for sid in ("1", "2", "3")), encoding="utf-8")
    paths = []
    for sid in ("1", "2", "3"):
        (root / "vocal_mid" / f"{sid}_vocal.mid").write_bytes(b"fixture score")
        targets = _event_labels_from_notes({"notes": [{"pitch": 60,
            "midi_time": {"start_tick": 120, "end_tick": 600}}]}, 1920)
        features = torch.randn(48, config.input_dim).half()
        features[:, config.audio_valid_index] = 1
        grid = canonicalize_grid({"source": {"sample_rate": 16000, "num_samples": 32000},
            "meter": {"beats_per_bar": 4}, "tempo": {"bpm": 120},
            "midi_alignment": {"audio_start_tick": 0, "grid_end_tick": 1920}, "beats": []}, [(0, 500000)])
        song = {**targets, "features": features, "song_id": sid, "grid": grid,
            "loss_mask": torch.ones(48, dtype=torch.bool),
            "boundary_mask": torch.ones(48, dtype=torch.bool),
            "continuation_mask": torch.ones(48, dtype=torch.bool),
            "articulation_mask": torch.ones(48, dtype=torch.bool),
            "pronunciation_boundary": torch.rand(48) * 0.1,
            "pronunciation_weight": torch.full((48,), 0.1)}
        path = tmp_path / f"{sid}_grid_features.pt"
        torch.save(song, path)
        paths.append(path)
    monkeypatch.setattr(train_melody, "build_feature_caches", lambda *args: paths)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(sys, "argv", ["train_melody.py", str(root), "--output", str(output),
        "--epochs", "1", "--bars-per-chunk", "1", "--stride-bars", "1",
        "--validation-ids", "3", "--no-publish", "--musical-event-context"]
        + (["--note-event-training", "--semitone-training", "--artifact-training",
            "--note-activity-weight", "0.5"] if complete_notes else []))
    train_melody.main()
    assert (output / "final.pt").read_bytes() == b"previous working model"
    run = next((output / "runs").iterdir())
    checkpoint = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
    assert checkpoint["model_config"]["use_pronunciation_context"]
    assert checkpoint["model_config"]["use_sequence_position"]
    assert checkpoint["model_config"]["musical_event_version"] == 3
    assert "boundary_state_head.0.weight" in checkpoint["model_state"]
    assert checkpoint["model_config"]["use_musical_event_context"]
    assert checkpoint["model_config"]["use_note_event_model"] == complete_notes
    assert checkpoint["run_metadata"]["validation_ids"] == ["3"]
    assert set(checkpoint["run_metadata"]["train_ids"]) == {"1", "2"}
    assert "validation" in checkpoint["history"][0]
    assert "optimizer_state" in checkpoint
    assert json.loads((root / "training_split.json").read_text())["validation_ids"] == ["3"]
    if complete_notes:
        assert checkpoint['model_config']['note_activity_weight'] == .5
        before_runs = set((output / 'runs').iterdir())
        continued_args = sys.argv.copy()
        index = continued_args.index('--note-activity-weight')
        del continued_args[index:index+2]
        monkeypatch.setattr(sys, 'argv', continued_args + ['--init-checkpoint', str(run / 'best.pt'),
            '--learning-rate', '0', '--pitch-head-only', '--semitone-guard','--artifact-guard'])
        train_melody.main()
        continued = (set((output / 'runs').iterdir()) - before_runs).pop()
        retained = torch.load(continued / 'best.pt', map_location='cpu', weights_only=False)
        assert retained['epoch'] == 0  # no-update baseline cannot lose to later equal scores
        assert retained['history'][0]['unmodified_initializer']
        assert retained['model_config']['note_activity_weight'] == .5
        assert (output / 'final.pt').read_bytes() == b'previous working model'


def test_queue_training_keeps_note_event_architecture_and_validation_split():
    from transmelody.workflow.melody_queue_workflow import workflow_training_command, TRAINING_SPLIT
    command = workflow_training_command(30)
    assert '--note-event-training' in command and '--musical-event-context' in command
    assert command[command.index('--split')+1] == str(TRAINING_SPLIT)
    assert '--init-checkpoint' in command
    assert '--semitone-guard' in command
    assert '--artifact-training' in command and '--artifact-guard' in command
