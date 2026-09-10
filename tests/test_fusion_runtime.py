import json
from pathlib import Path

import pytest
import torch

from fusion_runtime import read_policy, PitchFusionRuntime, persist_policy, update_active_summary
from melody_evaluation import file_digest


def fixture(tmp_path):
    base=tmp_path/'final.pt'; base.write_bytes(b'base')
    aux=tmp_path/'aux.pt'; aux.write_bytes(b'auxiliary')
    policy={'schema':'pitch_fusion_policy_v1','enabled':True,'base_checkpoint':str(base),
        'reviewed_base_sha256':file_digest(base),'new_pitch_weight':.75,
        'assets':{'score_model':{'path':'aux.pt','sha256':file_digest(aux)}}}
    persist_policy(tmp_path/'inference_policy.json',policy)
    return base,aux,policy


def test_active_policy_is_scoped_to_specific_base(tmp_path):
    base,aux,policy=fixture(tmp_path)
    assert read_policy(base)==policy
    assert read_policy(tmp_path/'experimental.pt') is None
    assert read_policy(base,disabled=True) is None


def test_corrupt_asset_fails_instead_of_silently_reverting(tmp_path):
    base,aux,policy=fixture(tmp_path)
    aux.write_bytes(b'changed')
    with pytest.raises(ValueError,match='missing/changed'):
        read_policy(base)
    assert read_policy(base,disabled=True) is None


def test_disabled_runtime_does_not_load_networks_or_audio(tmp_path):
    base,aux,policy=fixture(tmp_path)
    runtime=PitchFusionRuntime(base,torch.device('cpu'),disabled=True)
    notes=[object()]
    out,report=runtime.apply(notes,{},Path('missing.wav'),{})
    assert out is notes and report=={'enabled':False}
    assert runtime.model is runtime.extractor is None


def test_base_training_preserves_fusion_and_flags_unreviewed_pair(tmp_path,capsys):
    base,aux,policy=fixture(tmp_path)
    base.write_bytes(b'updated base')
    runtime=PitchFusionRuntime(base,torch.device('cpu'))
    assert runtime.policy is not None
    assert runtime.base_digest != policy['reviewed_base_sha256']
    assert 'needs fresh validation' in capsys.readouterr().out


def test_policy_disable_backs_up_and_can_be_restored(tmp_path):
    base,aux,policy=fixture(tmp_path)
    disabled={**policy,'enabled':False}
    persist_policy(tmp_path/'inference_policy.json',disabled)
    assert read_policy(base) is None
    backup=next((tmp_path/'backups').glob('inference_policy_*.json'))
    assert json.loads(backup.read_text(encoding='utf-8'))==policy


def test_inference_clis_offer_base_only_override(monkeypatch):
    import infer_melody,predict_test_audio,sys
    monkeypatch.setattr(sys,'argv',['infer_melody.py','final.pt','--no-pitch-fusion'])
    assert infer_melody.parse_args().no_pitch_fusion
    monkeypatch.setattr(sys,'argv',['predict_test_audio.py','--no-pitch-fusion'])
    assert predict_test_audio.parse_args().no_pitch_fusion


def test_active_summary_records_enable_and_disable_without_changing_base(tmp_path):
    base,aux,policy=fixture(tmp_path)
    summary=tmp_path/'active_model.json'
    summary.write_text(json.dumps({'checkpoint_sha256':file_digest(base)}),encoding='utf-8')
    digest=file_digest(base)
    update_active_summary(tmp_path,policy)
    assert json.loads(summary.read_text())['inference_policy']['type']=='pitch_only_fusion'
    update_active_summary(tmp_path,{**policy,'enabled':False})
    assert json.loads(summary.read_text())['inference_policy']['type']=='base_only'
    assert file_digest(base)==digest
