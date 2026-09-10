import json
from dataclasses import asdict

import pytest
import torch

from melody_transformer import MelodyTransformer, ModelConfig, FeatureConfig
from model_release import activate_checkpoint


def checkpoint(path):
    config = ModelConfig(input_dim=53, d_model=16, nhead=2, num_layers=1, dim_feedforward=32,
        use_musical_event_context=True, musical_event_version=3, acoustic_bins=4, use_note_event_model=True)
    torch.save({'model_config': asdict(config), 'model_state': MelodyTransformer(config).state_dict(),
        'feature_config': asdict(FeatureConfig(n_mels=4)), 'epoch': 0,
        'training_args': {'seed': 2026}, 'run_metadata': {'train_ids': ['1', '2'], 'validation_ids': ['3']},
        'history': [{'epoch': 0, 'validation': {'full_note_f1': .7}}]}, path)


def test_activation_verifies_model_backs_up_previous_and_preserves_split(tmp_path):
    source = tmp_path / 'candidate.pt'
    checkpoint(source)
    output = tmp_path / 'models'
    output.mkdir()
    (output / 'final.pt').write_bytes(b'previous model')
    report = activate_checkpoint(source, output)
    assert (output / 'final.pt').read_bytes() == source.read_bytes()
    assert next((output / 'backups').glob('*.pt')).read_bytes() == b'previous model'
    assert json.loads((output / 'training_split.json').read_text())['validation_ids'] == ['3']
    assert report['selected_epoch'] == 0 and report['decoder'] == 'complete_note_segments'


def test_split_mismatch_aborts_before_changing_model(tmp_path):
    source = tmp_path / 'candidate.pt'
    checkpoint(source)
    output = tmp_path / 'models'
    output.mkdir()
    (output / 'final.pt').write_bytes(b'previous model')
    (output / 'training_split.json').write_text(json.dumps({'validation_ids': ['4']}))
    with pytest.raises(ValueError, match='split differs'):
        activate_checkpoint(source, output)
    assert (output / 'final.pt').read_bytes() == b'previous model'


def test_new_base_release_preserves_user_fusion_policy(tmp_path):
    source=tmp_path/'candidate.pt'; checkpoint(source)
    output=tmp_path/'models'; output.mkdir()
    final=output/'final.pt'; final.write_bytes(b'old')
    policy={'enabled':True,'base_checkpoint':str(final),'new_pitch_weight':.75,
        'reviewed_base_sha256':'previous-reviewed-digest',
        'assets':{'score_model':{'sha256':'frozen-auxiliary'}}}
    policy_path=output/'inference_policy.json'
    policy_path.write_text(json.dumps(policy),encoding='utf-8')
    before=policy_path.read_bytes()
    report=activate_checkpoint(source,output)
    assert policy_path.read_bytes()==before
    assert report['inference_policy']['ensemble_needs_revalidation'] is True
    assert report['inference_policy']['new_pitch_weight']==.75
