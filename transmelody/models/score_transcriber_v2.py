"""Isolated experiment: frozen RMVPE evidence -> learned editable score.

No production checkpoint, registry, source MIDI or audio is modified. Acoustic
caches contain only audio-derived values; reviewed MIDI supplies training labels
and the fixed evaluation timeline, never inference pitch/boundary inputs.
"""
from __future__ import annotations
from transmelody.paths import PROJECT_ROOT

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import soundfile as sf
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchaudio

from transmelody.models.melody_transformer import _event_labels_from_notes, read_manifest
from transmelody.grid.musical_timeline import canonicalize_grid, samples_at_ticks
from transmelody.models.musical_boundary import boundary_targets
from transmelody.models.note_event_model import decode_note_events, note_event_loss
from transmelody.evaluation.melody_evaluation import (file_digest, note_counts, summarize_note_counts,
    semitone_aba_counts, note_artifact_counts, song_split)

ROOT = PROJECT_ROOT
DEFAULT_OUTPUT = ROOT / 'output/score_transcriber_v2'
RUNTIME = ROOT / '.cache/rmvpe_runtime'
SCHEMA = 'continuous_rmvpe_score_v1'
TARGETS = ('pitch', 'onset', 'offset', 'activity', 'duration', 'duration_mask',
           'continuation', 'loss_mask', 'boundary_mask', 'continuation_mask')


def union_grid(positions):
    """Candidate endpoints, not a forced per-bar rhythm label."""
    return (positions % 3 == 0) | (positions % 4 == 0)


def pitch_kernel():
    cents = 20 * torch.arange(360, dtype=torch.float32) + 1997.3794084376191
    midi = 69 + 12 * torch.log2(10 * torch.pow(2., cents / 1200) / 440)
    # Smooth mapping keeps adjacent semitone alternatives. No argmax F0 target.
    weights = torch.exp(-.5 * ((midi[:, None] - torch.arange(128)[None]) / .35) ** 2)
    return weights / weights.sum(1, keepdim=True).clamp_min(1e-8)


class EvidenceExtractor:
    def __init__(self, device, runtime_path=RUNTIME):
        runtime_path = Path(runtime_path)
        sys.path.insert(0, str(runtime_path))
        spec = importlib.util.spec_from_file_location('soc_rmvpe_runtime', runtime_path / 'rmvpe.py')
        runtime = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runtime)
        # Bypass RVC application config/device policy and optional CUDA graphs.
        self.model = runtime.E2E(4, 1, (2, 2)).to(device).eval()
        self.model.load_state_dict(torch.load(runtime_path / 'rmvpe.pt', map_location='cpu', weights_only=True))
        self.mel = runtime.MelSpectrogram(False, 128, 16000, 1024, 160, None, 30, 8000).to(device)
        self.device = device
        self.fingerprint = {p: file_digest(runtime_path / p) for p in ('rmvpe.py', 'rmvpe.pt', 'tools/cuda_graph.py')}

    @torch.inference_mode()
    def extract(self, path):
        audio, rate = sf.read(path, dtype='float32', always_2d=True)
        audio = torch.from_numpy(audio.mean(axis=1))
        if rate != 16000:
            audio = torchaudio.functional.resample(audio, rate, 16000)
        audio = audio.to(self.device)
        # Global STFT preserves exact 160-sample frame centres across chunks.
        mel = self.mel(audio[None])
        n = mel.shape[-1]
        scores = torch.empty(n, 360, dtype=torch.float16)
        for start in range(0, n, 800):
            end = min(n, start + 800)
            left, right = max(0, start - 128), min(n, end + 128)
            block = mel[..., left:right]
            padded = F.pad(block, (0, (-block.shape[-1]) % 32))
            with torch.autocast(self.device.type, enabled=self.device.type == 'cuda', dtype=torch.float16):
                hidden = self.model(padded)
            scores[start:end] = hidden[0, start-left:end-left].half().cpu()
        return {'salience': scores, 'mel': mel[0].T.half().cpu(),
                'frame_seconds': .01, 'num_audio_samples_16k': len(audio)}


def prepare(output, device, ids=None):
    dataset = ROOT / 'dataset/melody_dataset'
    cache = output / 'cache'
    cache.mkdir(parents=True, exist_ok=True)
    extractor = None
    from transmelody.training.prepare_dataset import midi_metadata, build_note_labels, SongFiles
    for row in read_manifest(dataset):
        sid = str(row['song_id'])
        if ids and sid not in ids:
            continue
        audio, midi = dataset / row['vocal_audio'], dataset / row['vocal_midi']
        acoustic_path = cache / f'{sid}_audio.pt'
        digest = file_digest(audio)
        # Pretrained asset identity participates in cache invalidation.
        asset_hash = file_digest(RUNTIME / 'rmvpe.pt')
        source_hash = file_digest(RUNTIME / 'rmvpe.py')
        fresh = False
        if acoustic_path.exists():
            old = torch.load(acoustic_path, map_location='cpu', weights_only=True)
            fresh = (old.get('schema') == SCHEMA and old.get('audio_sha256') == digest
                     and old.get('weight_sha256') == asset_hash and old.get('runtime_sha256') == source_hash)
        if not fresh:
            if extractor is None:
                extractor = EvidenceExtractor(device)
            values = extractor.extract(audio)
            values.update(schema=SCHEMA, audio_sha256=digest, weight_sha256=asset_hash,
                          runtime_sha256=source_hash, source_audio=str(audio))
            torch.save(values, acoustic_path)
        metadata = midi_metadata(midi)
        grid = canonicalize_grid(json.loads((dataset / row['grid']).read_text(encoding='utf-8')),
            [(e['tick'], e['tempo']) for e in metadata['tempo_events']],
            audio_start_tick=metadata['audio_start_tick'],
            minimum_end_tick=max((n.original_end_tick for n in metadata['notes']), default=0),
            source='reviewed_midi_fixed_timeline_experiment')
        info = sf.info(audio)
        labels = build_note_labels(song=SongFiles(sid, dataset / row['inst_audio'], audio, midi),
            grid=grid, midi_data=metadata, sample_rate=info.samplerate, num_samples=info.frames)
        targets = _event_labels_from_notes(labels, grid['midi_alignment']['grid_end_tick'])
        seconds = samples_at_ticks(grid, np.arange(len(targets['pitch'])) * 40, info.samplerate) / info.samplerate
        observed = torch.from_numpy((seconds >= 0) & (seconds < info.duration))
        targets['loss_mask'] = observed
        targets['boundary_mask'] = observed & torch.cat((torch.tensor([False]), observed[:-1]))
        targets['continuation_mask'] = observed & torch.cat((observed[1:], torch.tensor([False])))
        targets['duration_mask'] &= observed
        for note in labels['notes']:
            if not note['training']['fully_observable']:
                targets['duration_mask'][note['midi_time']['start_tick'] // 40] = False
        # MIDI-derived features are prohibited: only query TIMES enter the model.
        torch.save({**targets, 'grid': grid, 'song_id': sid,
            'query_frames': torch.from_numpy(seconds * 100).float(),
            'midi_sha256': file_digest(midi), 'audio_sha256': digest}, cache / f'{sid}_score.pt')
        print(f'[cache] {sid}: {len(seconds)} score steps, audio {"cached" if fresh else "extracted"}', flush=True)


def load_song(cache, sid):
    song = torch.load(cache / f'{sid}_score.pt', map_location='cpu', weights_only=False)
    audio = torch.load(cache / f'{sid}_audio.pt', map_location='cpu', weights_only=True)
    if song['audio_sha256'] != audio['audio_sha256']:
        raise ValueError('Audio and score cache fingerprints disagree.')
    # Fixed global scaling, no song labels or validation statistics involved.
    song['acoustic'] = torch.cat((audio['salience'], (audio['mel'] + 5) / 5), dim=-1).half()
    return song


def chunk(song, start, steps=384):
    end = min(len(song['query_frames']), start + steps)
    q = song['query_frames'][start:end]
    left = max(0, min(len(song['acoustic']) - 1, math.floor(float(q[0])) - 48))
    right = min(len(song['acoustic']), max(left + 1, math.ceil(float(q[-1])) + 49))
    item = {'acoustic': song['acoustic'][left:right].float(), 'query': q - left,
            'position': torch.arange(start, end), 'valid': torch.ones(end-start, dtype=torch.bool)}
    item.update({k: song[k][start:end] for k in TARGETS if k in song})
    return item


class ScoreDataset(Dataset):
    def __init__(self, songs, steps=384, stride=192):
        self.songs, self.steps = songs, steps
        self.index = []
        for i, song in enumerate(songs):
            last = max(0, len(song['pitch']) - steps)
            self.index.extend((i, s) for s in sorted(set([*range(0, last+1, stride), last])))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        song, start = self.index[i]
        return chunk(self.songs[song], start, self.steps)


def collate(items):
    result = {}
    for key in items[0]:
        result[key] = nn.utils.rnn.pad_sequence([x[key] for x in items], batch_first=True)
    result['audio_length'] = torch.tensor([len(x['acoustic']) for x in items])
    return result


class ScoreModel(nn.Module):
    def __init__(self, width=128, layers=3):
        super().__init__()
        self.config = {'width': width, 'layers': layers}
        self.projection = nn.Sequential(nn.LayerNorm(488), nn.Linear(488, width), nn.GELU())
        self.temporal = nn.Sequential(nn.Conv1d(width, width, 5, padding=2), nn.GELU(),
                                      nn.Conv1d(width, width, 5, padding=2))
        self.offset = nn.Parameter(torch.randn(49, width) * .02)
        self.query_score = nn.Linear(width, 2)
        self.fuse = nn.Linear(width * 2, width)
        self.bar_position = nn.Embedding(48, width)
        encoder = nn.TransformerEncoderLayer(width, 4, width * 3, dropout=.15,
            activation='gelu', batch_first=True, norm_first=True)
        self.context = nn.TransformerEncoder(encoder, layers, norm=nn.LayerNorm(width), enable_nested_tensor=False)
        self.pitch = nn.Linear(width, 129)
        self.boundary = nn.Linear(width, 4)
        self.duration = nn.Linear(width, 192)
        self.activity = nn.Linear(width, 1)
        self.prior_gain = nn.Parameter(torch.tensor(0.))
        self.register_buffer('kernel', pitch_kernel())

    def forward(self, batch):
        acoustic = batch['acoustic']
        h = self.projection(acoustic)
        h = h + self.temporal(h.transpose(1, 2)).transpose(1, 2)
        # Continuous evidence at 10 ms resolution precedes any score grid choice.
        # Separate learned alignment weights for pitch and boundary evidence.
        offsets = torch.arange(-24, 25, device=h.device)
        indices = (batch['query'].round().long()[..., None] + offsets).clamp_min(0)
        indices = torch.minimum(indices, batch['audio_length'][:, None, None] - 1)
        b = torch.arange(len(h), device=h.device)[:, None, None]
        local = h[b, indices] + self.offset
        weights = self.query_score(local).softmax(dim=-2)
        pooled = torch.einsum('btwk,btwd->btkd', weights, local).flatten(-2)
        score = self.fuse(pooled) + self.bar_position(batch['position'] % 48)
        # Absolute-within-chunk encoding provides order beyond repeated bar phase.
        t = torch.arange(score.shape[1], device=h.device).float()[:, None]
        f = torch.exp(torch.arange(0, score.shape[-1], 2, device=h.device) * (-math.log(10000.) / score.shape[-1]))
        pos = torch.zeros_like(score[0]); pos[:, 0::2] = (t*f).sin(); pos[:, 1::2] = (t*f).cos()
        score = self.context(score + pos * .1, src_key_padding_mask=~batch['valid'])
        salience = acoustic[b, indices, :360]
        prior = torch.einsum('btw,btwc->btc', weights[..., 0], salience) @ self.kernel
        prior = (prior + 1e-4) / (prior.sum(-1, keepdim=True) + .0128)
        logits = self.pitch(score)
        logits = torch.cat((logits[..., :1], logits[..., 1:] + F.softplus(self.prior_gain) * prior.log()), -1)
        return {'pitch': logits, 'boundary_state': self.boundary(score),
                'duration': self.duration(score), 'activity': self.activity(score).squeeze(-1)}


def score_loss(outputs, batch, structured_weight=0., batch_index=0):
    valid = batch['valid'] & batch['loss_mask']
    state, state_mask = boundary_targets(batch)
    state_mask &= union_grid(batch['position'])
    pitch = F.cross_entropy(outputs['pitch'][valid], batch['pitch'][valid], reduction='none')
    # Rest and long vowels must not dominate the rare, user-marked note starts.
    weight = torch.where(batch['pitch'][valid] > 0, 1., .35)
    pitch_loss = (pitch * weight).sum() / weight.sum().clamp_min(1)
    onset = valid & (batch['onset'] > .5)
    if onset.any():
        pitch_loss = pitch_loss + .5 * F.cross_entropy(outputs['pitch'][onset], batch['pitch'][onset])
    boundary_loss = F.cross_entropy(outputs['boundary_state'][state_mask], state[state_mask],
        weight=torch.tensor([1., 1., 2., 2.], device=state.device))
    duration_mask = valid & batch['duration_mask']
    duration_loss = (F.cross_entropy(outputs['duration'][duration_mask], batch['duration'][duration_mask])
        if duration_mask.any() else outputs['duration'].sum() * 0)
    activity_loss = F.binary_cross_entropy_with_logits(outputs['activity'][valid], batch['activity'][valid])
    loss = pitch_loss + boundary_loss + .3 * duration_loss + .5 * activity_loss
    if structured_weight:
        scored = dict(outputs)
        scored['note_active_score'] = F.logsigmoid(outputs['activity']) * .5 / 3
        scored['note_rest_score'] = F.logsigmoid(-outputs['activity']) * .5 / 3
        loss = loss + structured_weight * note_event_loss(scored, batch, batch_index=batch_index,
            max_notes=8, semitone_contrasts=True, artifact_contrasts=True,
            allowed_boundary_mask=union_grid(batch['position']))
    return loss


@torch.inference_mode()
def predict_outputs(model, song, device, steps=384):
    model.eval()
    n = len(song['query_frames'])
    last = max(0, n-steps)
    starts = sorted(set([*range(0, last+1, steps//2), last]))
    total, weights = {}, torch.zeros(n)
    for start in starts:
        batch = {k: v.to(device) for k, v in collate([chunk(song, start, steps)]).items()}
        out = model(batch)
        size = out['pitch'].shape[1]
        w = torch.hann_window(size, periodic=False).clamp_min(.05)
        weights[start:start+size] += w
        for key, value in out.items():
            value = value[0].float().cpu()
            if key not in total:
                total[key] = torch.zeros((n, *value.shape[1:]))
            total[key][start:start+size] += value * w.reshape(-1, *([1]*(value.ndim-1)))
    total = {k: v / weights.reshape(-1, *([1]*(v.ndim-1))) for k, v in total.items()}
    total['note_active_score'] = F.logsigmoid(total['activity']) * .5 / 3
    total['note_rest_score'] = F.logsigmoid(-total['activity']) * .5 / 3
    return total


def decode(outputs, song):
    # Both grids are candidates. No target rhythm, note boundary, or pitch is read.
    duration = song['grid']['source']['num_samples'] / song['grid']['source']['sample_rate']
    observed = (song['query_frames'] >= 0) & (song['query_frames'] < duration*100)
    notes = decode_note_events(outputs, None, observed,
        allowed_boundary_mask=union_grid(torch.arange(len(observed))))
    return notes


def predict(model, song, device, steps=384):
    return decode(predict_outputs(model, song, device, steps), song)


def evaluate(model, songs, device):
    result = {}
    for song in songs:
        notes = predict(model, song, device)
        result[song['song_id']] = {**summarize_note_counts(note_counts(notes, song)),
            **semitone_aba_counts(notes, song), **note_artifact_counts(notes, song)}
    keys = ('reference_notes', 'predicted_notes', 'onset_matches', 'pitch_onset_matches', 'full_note_matches')
    return {**summarize_note_counts({k: sum(s[k] for s in result.values()) for k in keys}), 'songs': result}


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def train(args, device):
    torch.manual_seed(2026); np.random.seed(2026); random.seed(2026)
    rows = read_manifest(ROOT / 'dataset/melody_dataset')
    train_ids, dev_ids = song_split([str(r['song_id']) for r in rows], ROOT / 'output/melody_transformer/training_split.json')
    songs = {sid: load_song(args.cache or args.output / 'cache', sid) for sid in train_ids + dev_ids}
    loader = DataLoader(ScoreDataset([songs[s] for s in train_ids]), batch_size=4, shuffle=True,
        num_workers=0, collate_fn=collate)
    model = ScoreModel().to(device)
    if args.init:
        initial = torch.load(args.init, map_location='cpu', weights_only=False)
        if initial['schema'] != SCHEMA or initial['metadata']['development_ids'] != dev_ids:
            raise ValueError('Initializer must be a compatible isolated model with the same split.')
        model.load_state_dict(initial['model_state'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=.02)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=3e-5)
    metadata = {'schema': SCHEMA, 'train_ids': train_ids, 'development_ids': dev_ids,
        'production_sha256_before': file_digest(ROOT / 'output/melody_transformer/final.pt'),
        'label_sha256': {s: songs[s]['midi_sha256'] for s in songs},
        'note': 'Development songs reused historically; song19 is training-fit only. Fixed reviewed tempo/alignment.',
        'rmvpe_sha256': file_digest(RUNTIME / 'rmvpe.pt'), 'source_sha256': file_digest(Path(__file__)),
        'parameters': sum(p.numel() for p in model.parameters()), 'epochs': args.epochs,
        'structured_weight': args.structured_weight, 'learning_rate': args.learning_rate,
        'cache': str((args.cache or args.output / 'cache').resolve()),
        'initializer_sha256': file_digest(args.init) if args.init else None}
    save_json(args.output / 'run.json', metadata)
    history, best = [], -1.
    if args.init:
        initial_dev = evaluate(model, [songs[s] for s in dev_ids], device)
        best = initial_dev['full_note_f1']
        history.append({'epoch': 0, 'development': initial_dev})
        torch.save({'schema': SCHEMA, 'model_config': model.config, 'model_state': model.state_dict(),
            'epoch': 0, 'development': initial_dev, 'metadata': metadata}, args.output / 'best.pt')
    for epoch in range(1, args.epochs+1):
        model.train(); losses = []; start_time = time.monotonic()
        for batch_index, batch in enumerate(loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            loss = score_loss(model(batch), batch, args.structured_weight, batch_index)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite score loss')
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.); optimizer.step()
            losses.append(float(loss.detach()))
        scheduler.step()
        row = {'epoch': epoch, 'train_loss': float(np.mean(losses)), 'seconds': time.monotonic()-start_time}
        if epoch == 1 or epoch % args.eval_every == 0 or epoch == args.epochs:
            row['development'] = evaluate(model, [songs[s] for s in dev_ids], device)
            if row['development']['full_note_f1'] > best:
                best = row['development']['full_note_f1']
                torch.save({'schema': SCHEMA, 'model_config': model.config, 'model_state': model.state_dict(),
                    'epoch': epoch, 'development': row['development'], 'metadata': metadata}, args.output / 'best.pt')
        history.append(row)
        save_json(args.output / 'history.json', history)
        print('[epoch]', json.dumps(row), flush=True)
    torch.save({'schema': SCHEMA, 'model_config': model.config, 'model_state': model.state_dict(),
        'epoch': args.epochs, 'metadata': metadata}, args.output / 'last.pt')
    if file_digest(ROOT / 'output/melody_transformer/final.pt') != metadata['production_sha256_before']:
        raise RuntimeError('Production checkpoint changed during isolated experiment.')


def export(args, device):
    from transmelody.inference.melody_inference import export_prediction
    cp = torch.load(args.output / 'best.pt', map_location='cpu', weights_only=False)
    model = ScoreModel(**cp['model_config']).to(device)
    model.load_state_dict(cp['model_state'])
    report = {'checkpoint_epoch': cp['epoch'], 'songs': {}, 'development': cp.get('development')}
    folder = args.output / 'samples'; folder.mkdir(exist_ok=True)
    for sid in args.ids or ['4', '11', '18', '19']:
        song = load_song(Path(cp['metadata'].get('cache', args.output / 'cache')), sid)
        notes = predict(model, song, device)
        export_prediction(folder / f'{sid}_predicted.mid', song['grid'], notes, None)
        stats = {**summarize_note_counts(note_counts(notes, song)), **semitone_aba_counts(notes, song),
            **note_artifact_counts(notes, song), 'bpm': song['grid']['tempo']['bpm'],
            'tempo_events': song['grid']['timeline']['tempo_events'],
            'audio_start_tick': song['grid']['midi_alignment']['audio_start_tick'],
            'role': 'development' if sid in cp['metadata']['development_ids'] else 'training_fit'}
        report['songs'][sid] = stats
        save_json(folder / f'{sid}_prediction.json', stats)
        print('[export]', sid, json.dumps(stats), flush=True)
    save_json(args.output / 'evaluation.json', report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'train', 'export'])
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--ids', nargs='+')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--eval-every', type=int, default=3)
    parser.add_argument('--cache', type=Path)
    parser.add_argument('--init', type=Path)
    parser.add_argument('--structured-weight', type=float, default=0.)
    parser.add_argument('--learning-rate', type=float, default=3e-4)
    args = parser.parse_args()
    if args.output.resolve().is_relative_to(ROOT / 'output/melody_transformer'):
        parser.error('Experimental output must not be inside the production model directory.')
    if args.epochs < 1 or args.eval_every < 1 or args.structured_weight < 0 or args.learning_rate <= 0:
        parser.error('Epochs/evaluation interval/learning rate must be positive; structured weight must be nonnegative.')
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.command == 'prepare': prepare(args.output, device, args.ids)
    elif args.command == 'train': train(args, device)
    else: export(args, device)


if __name__ == '__main__':
    main()
