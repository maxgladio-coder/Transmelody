"""Explicit, reversible production policy for the user-approved pitch ensemble.

final.pt remains the trainable base. The auxiliary score model and RMVPE assets
are pinned independently. Training the base does not silently disable fusion;
subsequent predictions flag that the new pair has not been re-evaluated.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil

import numpy as np
import soundfile as sf
import torch

from melody_evaluation import file_digest
from musical_timeline import samples_at_ticks
from pitch_fusion import fuse_pitches

ROOT = Path(__file__).resolve().parent
POLICY_NAME = 'inference_policy.json'


def read_policy(checkpoint, disabled=False):
    checkpoint = Path(checkpoint).resolve()
    if disabled:
        return None
    path = checkpoint.parent/POLICY_NAME
    if not path.exists():
        return None
    policy = json.loads(path.read_text(encoding='utf-8'))
    policy_base = Path(policy['base_checkpoint'])
    if not policy_base.is_absolute():
        policy_base = path.parent / policy_base
    if not policy.get('enabled') or checkpoint != policy_base.resolve():
        return None
    if policy.get('schema') != 'pitch_fusion_policy_v1':
        raise ValueError('Unsupported inference policy; use --no-pitch-fusion to explicitly run the base.')
    if not 0 < float(policy['new_pitch_weight']) <= 1:
        raise ValueError('Invalid fusion policy weight.')
    for item in policy['assets'].values():
        asset = checkpoint.parent/item['path']
        if not asset.is_file() or file_digest(asset) != item['sha256']:
            raise ValueError(f'Fusion asset missing/changed: {asset}. No silent base-only fallback.')
    return policy


class PitchFusionRuntime:
    def __init__(self, checkpoint, device, disabled=False):
        self.checkpoint = Path(checkpoint).resolve()
        self.device = device
        self.policy = read_policy(self.checkpoint,disabled)
        self.extractor = self.model = None
        self.base_digest = file_digest(self.checkpoint) if self.policy else None
        if self.policy:
            print(f"[fusion] ACTIVE: base pitch 25% + auxiliary pitch 75%; base note boundaries preserved",flush=True)
            if self.base_digest != self.policy['reviewed_base_sha256']:
                print('[fusion] Base checkpoint updated since ensemble review; auxiliary remains frozen. '
                      'This model pair needs fresh validation.',flush=True)

    @torch.inference_mode()
    def apply(self, notes, outputs, audio_path, grid):
        if not self.policy:
            return notes, {'enabled':False}
        import score_transcriber_v2 as v2
        folder = self.checkpoint.parent
        if self.extractor is None:
            self.extractor = v2.EvidenceExtractor(self.device,folder/self.policy['runtime_directory'])
            aux = folder/self.policy['assets']['score_model']['path']
            cp = torch.load(aux,map_location='cpu',weights_only=False)
            self.model = v2.ScoreModel(**cp['model_config']).to(self.device).eval()
            self.model.load_state_dict(cp['model_state'],strict=True)
        info = sf.info(audio_path)
        if info.frames != grid['source']['num_samples'] or info.samplerate != grid['source']['sample_rate']:
            raise ValueError('Fusion input and base grid differ in audio length/sample rate.')
        evidence = self.extractor.extract(audio_path)
        song = {'acoustic':torch.cat((evidence['salience'],(evidence['mel']+5)/5),dim=-1).half(),
            'query_frames':torch.from_numpy(samples_at_ticks(grid,np.arange(len(outputs['pitch']))*40,16000)/160).float(),
            'grid':grid}
        aux_outputs = v2.predict_outputs(self.model,song,self.device)
        fused = fuse_pitches(notes,outputs['pitch'],aux_outputs['pitch'],self.policy['new_pitch_weight'])
        assert [(n.start_step,n.end_step) for n in fused] == [(n.start_step,n.end_step) for n in notes]
        report = {'enabled':True,'new_pitch_weight':self.policy['new_pitch_weight'],
            'boundaries_unchanged':True,'base_checkpoint_sha256':self.base_digest,
            'auxiliary_checkpoint_sha256':self.policy['assets']['score_model']['sha256'],
            'base_updated_since_fusion_review':self.base_digest != self.policy['reviewed_base_sha256'],
            'changed_pitches':sum(a.pitch != b.pitch for a,b in zip(notes,fused)),
            'auxiliary_training':'frozen; normal workflow training updates the base only',
            'activation':'user-approved provisional mainline; prior ABA regression remains documented'}
        print(f"[fusion] {report['changed_pitches']} pitch changes; {len(fused)} note spans unchanged",flush=True)
        return fused,report


def persist_policy(path, policy):
    path = Path(path)
    if path.exists():
        backup = path.parent/'backups'/f'inference_policy_{file_digest(path)[:16]}.json'
        backup.parent.mkdir(exist_ok=True)
        if not backup.exists(): shutil.copy2(path,backup)
    pending = path.with_suffix('.pending.json')
    pending.write_text(json.dumps(policy,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    os.replace(pending,path)


def update_active_summary(output, policy):
    path = Path(output)/'active_model.json'
    if not path.exists():
        return
    previous=json.loads(path.read_text(encoding='utf-8'))
    backup=path.parent/'backups'/f'active_model_{file_digest(path)[:16]}.json'
    backup.parent.mkdir(exist_ok=True)
    if not backup.exists(): shutil.copy2(path,backup)
    previous['inference_policy']={'type':'pitch_only_fusion' if policy['enabled'] else 'base_only',
        'enabled':policy['enabled'],'policy_file':str(path.parent/POLICY_NAME),
        'new_pitch_weight':policy['new_pitch_weight'] if policy['enabled'] else 0.,
        'auxiliary_checkpoint_sha256':policy['assets']['score_model']['sha256'],
        'validation_scope':'Existing checkpoint validation describes the base; ensemble comparison is separate.',
        'known_regression':policy.get('known_regression'),
        'activation':policy.get('authorization')}
    pending=path.with_suffix('.pending.json')
    pending.write_text(json.dumps(previous,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    os.replace(pending,path)


def activate(output):
    import score_transcriber_v2 as v2
    output = Path(output).resolve()
    base = output/'final.pt'
    source = v2.DEFAULT_OUTPUT/'structured/best.pt'
    review = json.loads((ROOT/'output/pitch_fusion_review/comparison.json').read_text(encoding='utf-8'))
    if file_digest(base) != review['checkpoint_sha256']['production'] or file_digest(source) != review['checkpoint_sha256']['new']:
        raise ValueError('Requested release differs from the reviewed fusion pair.')
    cp = torch.load(source,map_location='cpu',weights_only=False)
    model = v2.ScoreModel(**cp['model_config']); model.load_state_dict(cp['model_state'],strict=True)
    release = output/'fusion_assets'; release.mkdir(exist_ok=True)
    sources = {'score_model':(source,release/'score_model.pt')}
    for name in ('rmvpe.py','rmvpe.pt','tools/cuda_graph.py','LICENSE'):
        sources[name] = (v2.RUNTIME/name,release/'rmvpe_runtime'/name)
    assets = {}
    for key,(src,dst) in sources.items():
        digest = file_digest(src)
        dst.parent.mkdir(parents=True,exist_ok=True)
        if dst.exists() and file_digest(dst) != digest:
            raise ValueError(f'Pinned fusion asset collision: {dst}')
        if not dst.exists(): shutil.copy2(src,dst)
        if file_digest(dst) != digest: raise IOError('Pinned asset copy failed.')
        assets[key] = {'path':str(dst.relative_to(output)), 'sha256':digest}
    backup = output/'backups'/f'final_{file_digest(base)[:16]}.pt'
    backup.parent.mkdir(exist_ok=True)
    if not backup.exists(): shutil.copy2(base,backup)
    if file_digest(base) != file_digest(backup): raise IOError('Base backup verification failed.')
    policy = {'schema':'pitch_fusion_policy_v1','enabled':True,'base_checkpoint':str(base),
        'new_pitch_weight':.75,'reviewed_base_sha256':file_digest(base),'assets':assets,
        'runtime_directory':str((release/'rmvpe_runtime').relative_to(output)),
        'activated_utc':datetime.now(timezone.utc).isoformat(),
        'authorization':'User explicitly requested provisional fusion mainline after audition.',
        'known_regression':'Development song18 semitone ABA 4/17 -> 3/17; activation is a user override, not a guard pass.',
        'training_policy':'Keep base trainable, auxiliary frozen; preserve ensemble policy and flag unreviewed base updates.',
        'rollback':'python fusion_runtime.py disable (or --no-pitch-fusion for one prediction)',
        'queue_action':'none'}
    persist_policy(output/POLICY_NAME,policy)
    read_policy(base)  # Validate the published policy and pinned assets.
    update_active_summary(output,policy)
    return policy


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('activate','disable','status'))
    parser.add_argument('--output',type=Path,default=ROOT/'output/melody_transformer')
    args = parser.parse_args()
    if args.action=='activate': result=activate(args.output)
    elif args.action=='disable':
        path=args.output/POLICY_NAME
        result=json.loads(path.read_text(encoding='utf-8')); result['enabled']=False
        persist_policy(path,result)
        update_active_summary(args.output,result)
    else: result=read_policy(args.output/'final.pt')
    print(json.dumps(result,ensure_ascii=False,indent=2))
