"""Install checked release weights and download the third-party RMVPE weight."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def checked_copy(src, dst, digest):
    if sha(src) != digest:
        raise ValueError(f'Checksum mismatch: {src}')
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        if sha(dst) != digest:
            raise FileExistsError(f'Refusing to overwrite changed/trained model: {dst}')
        return
    shutil.copy2(src, dst)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rmvpe-file', type=Path, help='Use an existing trusted rmvpe.pt instead of downloading it.')
    args = parser.parse_args()
    manifest = json.loads((ROOT / 'models/manifest.json').read_text())
    output = ROOT / 'output/melody_transformer'
    runtime = output / 'fusion_assets/rmvpe_runtime'
    policy_path = output / 'inference_policy.json'
    if policy_path.exists():
        old = json.loads(policy_path.read_text())
        if old.get('release') != 'Transmelody-0.1.0':
            raise FileExistsError('Existing non-release inference policy; review it before setup.')
    for source, dest in [('transmelody_base.pt', output / 'final.pt'),
                         ('transmelody_pitch.pt', output / 'fusion_assets/score_model.pt')]:
        checked_copy(ROOT / 'models' / source, dest, manifest['models'][source]['sha256'])
    for name in ['rmvpe.py', 'tools/cuda_graph.py', 'LICENSE']:
        checked_copy(ROOT / 'third_party/rmvpe' / name, runtime / name, manifest['rmvpe'][name])
    weight = runtime / 'rmvpe.pt'
    if args.rmvpe_file:
        checked_copy(args.rmvpe_file.resolve(), weight, manifest['rmvpe']['rmvpe.pt'])
    elif not weight.exists():
        pending = weight.with_suffix('.download')
        url = 'https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.pt'
        print('Downloading third-party RMVPE weight from:', url, flush=True)
        try:
            urllib.request.urlretrieve(url, pending)
            if sha(pending) != manifest['rmvpe']['rmvpe.pt']:
                raise ValueError('RMVPE download differs from the tested weight; not installed.')
            pending.replace(weight)
        finally:
            pending.unlink(missing_ok=True)
    if sha(weight) != manifest['rmvpe']['rmvpe.pt']:
        raise ValueError('Existing RMVPE weight checksum mismatch.')
    assets = {'score_model': {'path': 'fusion_assets/score_model.pt',
               'sha256': manifest['models']['transmelody_pitch.pt']['sha256']}}
    assets.update({name: {'path': 'fusion_assets/rmvpe_runtime/' + name, 'sha256': value}
                   for name, value in manifest['rmvpe'].items()})
    policy = {'schema': 'pitch_fusion_policy_v1', 'release': 'Transmelody-0.1.0',
              'enabled': True, 'base_checkpoint': 'final.pt', 'new_pitch_weight': 0.75,
              'reviewed_base_sha256': manifest['models']['transmelody_base.pt']['sha256'],
              'assets': assets, 'runtime_directory': 'fusion_assets/rmvpe_runtime',
              'known_regression': 'One repeatedly-used development song has semitone ABA recovery 4/17 -> 3/17.',
              'training_policy': 'Base updates only; auxiliary pitch model stays frozen.'}
    policy_path.write_text(json.dumps(policy, indent=2) + '\n', encoding='utf-8')
    from fusion_runtime import read_policy
    assert read_policy(output / 'final.pt')['enabled']
    from registry_io import write
    registry = ROOT / 'dataset/song_registry.xlsx'
    if not registry.exists():
        write(registry, [])
    for folder in ['original audio', 'original stem', 'test audio/test_inst', 'test audio/test_vocal',
                   'dataset/melody_dataset/inst_audio', 'dataset/melody_dataset/vocal_audio',
                   'dataset/melody_dataset/vocal_mid']:
        (ROOT / folder).mkdir(parents=True, exist_ok=True)
    print('Transmelody fusion active: base 25% + auxiliary 75%. No songs or labels were installed.')


if __name__ == '__main__':
    main()
