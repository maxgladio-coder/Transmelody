import json
from pathlib import Path

import pytest
from fusion_runtime import read_policy
from melody_evaluation import file_digest


def test_relative_policy_survives_relocation(tmp_path, monkeypatch):
    folder = tmp_path / 'moved' / 'model'
    folder.mkdir(parents=True)
    base = folder / 'final.pt'
    aux = folder / 'aux.pt'
    base.write_bytes(b'base')
    aux.write_bytes(b'aux')
    policy = {'schema': 'pitch_fusion_policy_v1', 'enabled': True,
              'base_checkpoint': 'final.pt', 'new_pitch_weight': .75,
              'assets': {'score_model': {'path': 'aux.pt', 'sha256': file_digest(aux)}}}
    (folder / 'inference_policy.json').write_text(json.dumps(policy))
    monkeypatch.chdir(tmp_path)
    assert read_policy(base) == policy
    assert read_policy(folder / 'other.pt') is None


def test_portable_registry_roundtrip_and_max_id(tmp_path):
    from registry_io import write, read, append, update
    from openpyxl import load_workbook
    path = tmp_path / 'registry.xlsx'
    write(path, [])
    append(path, [{'id': '1', 'title': 'かな'}, {'id': '9', 'title': '=not_a_formula'}])
    update(path, [{'id': '9', 'status': '待校对', 'notes': '人工修改'}])
    assert read(path)[1]['status'] == '待校对'
    wb = load_workbook(path, data_only=False)
    assert wb.active['K3'].value == 10
    assert wb.active['B3'].data_type == 's'
    wb.close()
    with pytest.raises(ValueError):
        append(path, [{'id': '9'}])
    assert len(read(path)) == 2


def test_release_weights_are_safe_primitives_and_tensors():
    import torch
    root = Path(__file__).resolve().parents[1]
    for name in ('transmelody_base.pt', 'transmelody_pitch.pt'):
        cp = torch.load(root / 'models' / name, map_location='cpu', weights_only=True)
        assert cp['release']['model_tensors_unchanged']
        assert 'optimizer_state' not in cp and 'run_metadata' not in cp
