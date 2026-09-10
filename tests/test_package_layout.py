import importlib.util
from pathlib import Path
import sys

import pytest

from transmelody import __main__ as cli
from transmelody.paths import PROJECT_ROOT, PACKAGE_ROOT, module_path


def test_project_root_does_not_move_with_source_modules():
    from transmelody.inference.fusion_runtime import ROOT as fusion_root
    from transmelody.models.articulation_encoder import PROJECT_ROOT as speech_root
    from transmelody.models.score_transcriber_v2 import ROOT as pitch_root
    from transmelody.workflow.melody_queue_workflow import DATASET_ROOT, CHECKPOINT
    assert PROJECT_ROOT == Path(__file__).resolve().parents[1]
    assert fusion_root == speech_root == pitch_root == PROJECT_ROOT
    assert DATASET_ROOT == PROJECT_ROOT / 'dataset/melody_dataset'
    assert CHECKPOINT == PROJECT_ROOT / 'output/melody_transformer/final.pt'
    assert PACKAGE_ROOT == PROJECT_ROOT / 'transmelody'


def test_all_commands_and_aliases_resolve_to_real_modules():
    assert set(cli.ALIASES.values()) <= set(cli.COMMANDS)
    for name, module in cli.COMMANDS.items():
        assert importlib.util.find_spec(module) is not None
        assert module_path(name + '.py').is_file()


@pytest.mark.parametrize('command', ['predict', 'predict_test_audio', 'predict_test_audio.py'])
def test_cli_forwards_options_without_executing_a_prediction(monkeypatch, command):
    calls = []
    monkeypatch.setattr(sys, 'argv', ['transmelody', command, '--song-id', '20', '--bpm', '176'])
    monkeypatch.setattr(cli.runpy, 'run_module', lambda name, **kwargs: calls.append((name, sys.argv.copy(), kwargs)))
    cli.main()
    assert calls == [('transmelody.inference.predict_test_audio',
                      ['transmelody.inference.predict_test_audio', '--song-id', '20', '--bpm', '176'],
                      {'run_name': '__main__'})]


def test_workflow_subprocesses_use_module_invocation():
    from transmelody.workflow.melody_queue_workflow import workflow_training_command
    from transmelody.ui.melody_queue_workflow_ui import WORKFLOW
    command = workflow_training_command(1)
    assert command[:3] == [sys.executable, '-m', 'transmelody.training.train_melody']
    assert WORKFLOW == 'transmelody.workflow.melody_queue_workflow'


def test_existing_double_click_launchers_use_the_package():
    text = (PROJECT_ROOT / 'launch_melody_queue.cmd').read_text()
    assert '-m transmelody.ui.melody_queue_workflow_ui' in text
    assert 'cd /d "%~dp0"' in text
