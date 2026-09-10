"""Optional note-level pitch refinement. Never changes boundaries or forces a scale.

Tonal profiles are explicit, weak major/natural-minor hypotheses, not annotated
key truth. The HMM allows regional changes; chromatic notes retain nonzero mass.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace

import torch
from torch import nn

from transmelody.models.melody_transformer import STEPS_PER_BAR

PITCH_NAMES = ('C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B')
SCALES = ((0, 2, 4, 5, 7, 9, 11), (0, 2, 3, 5, 7, 8, 10))


def key_name(index):
    return f'{PITCH_NAMES[index % 12]} {"major" if index < 12 else "minor"}'


def tonal_profiles():
    # Deliberately transparent prototype weights, NOT a fitted key classifier.
    profiles = torch.full((24, 12), .25)
    membership = torch.zeros(24, 12)
    for mode, scale in enumerate(SCALES):
        for tonic in range(12):
            row = mode * 12 + tonic
            for degree in scale:
                profiles[row, (tonic + degree) % 12] = 1.
                membership[row, (tonic + degree) % 12] = 1.
            profiles[row, tonic] = 1.6
            profiles[row, (tonic + 7) % 12] = 1.3
    return profiles / profiles.sum(-1, keepdim=True), membership


def pitch_chroma(probability):
    result = probability.new_zeros((*probability.shape[:-1], 12))
    indices = torch.arange(128, device=probability.device) % 12
    result.scatter_add_(-1, indices.expand_as(probability), probability)
    return result


def tonal_context(logits, starts, durations, total_bars=None):
    """Soft per-bar HMM posterior from ACOUSTIC note scores, never reference pitch.

    Each note contributes to every bar it sounds in. Long notes cannot dominate
    a bar beyond one beat. Silent bars have no emission and are marked unknown.
    Profile scores / HMM posteriors are model-dependent, not calibrated certainty.
    """
    logits, starts, durations = logits.float().cpu(), starts.long().cpu(), durations.long().cpu()
    n_bars = total_bars or (int((starts + durations).max() + 47) // 48 if len(starts) else 1)
    chroma = pitch_chroma(logits.softmax(-1))
    histogram, mass = torch.zeros(n_bars, 12), torch.zeros(n_bars)
    for i, (start, duration) in enumerate(zip(starts.tolist(), durations.tolist())):
        end = start + duration
        for bar in range(start // 48, min(n_bars, (end + 47) // 48)):
            weight = min(12, max(0, min(end, (bar + 1) * 48) - max(start, bar * 48))) / 12
            histogram[bar] += weight * chroma[i]
            mass[bar] += weight
    normalized = histogram / mass[:, None].clamp_min(1e-8)
    profiles, membership = tonal_profiles()
    emission = normalized @ profiles.log().T * mass.clamp(max=4)[:, None] * 2
    transition = torch.full((24, 24), .03 / 23)
    transition.fill_diagonal_(.97)
    transition = transition.log()
    forward = torch.empty_like(emission)
    forward[0] = emission[0] - math.log(24)
    forward[0] -= torch.logsumexp(forward[0], 0)
    for bar in range(1, n_bars):
        forward[bar] = emission[bar] + torch.logsumexp(forward[bar-1, :, None] + transition, 0)
        forward[bar] -= torch.logsumexp(forward[bar], 0)
    backward = torch.zeros_like(emission)
    for bar in range(n_bars - 2, -1, -1):
        backward[bar] = torch.logsumexp(transition + emission[bar+1, None] + backward[bar+1, None], 1)
        backward[bar] -= torch.logsumexp(backward[bar], 0)
    posterior = (forward + backward).softmax(-1)
    return posterior, normalized, mass, membership


def note_inputs(outputs, notes):
    pooled = []
    for note in notes:
        start = note.start_step + int(note.end_step - note.start_step >= 3)
        pooled.append(outputs['pitch'][start:note.end_step].float().mean(0)[1:])
    logits = torch.stack(pooled) if pooled else torch.empty(0, 128)
    starts = torch.tensor([n.start_step for n in notes], dtype=torch.long)
    durations = torch.tensor([n.end_step - n.start_step for n in notes], dtype=torch.long)
    return logits, context_features(logits, starts, durations), starts, durations


def context_features(logits, starts, durations):
    posterior, chroma, mass, _ = tonal_context(logits, starts, durations)
    bars = starts // STEPS_PER_BAR
    phase = starts.float() % 48 / 48 * 2 * math.pi
    gaps = torch.zeros(len(logits))
    if len(logits) > 1:
        gaps[1:] = (starts[1:] - starts[:-1] - durations[:-1]).float().clamp_min(0)
    timing = torch.stack([phase.sin(), phase.cos(), torch.log1p(durations.float()) / 5,
        torch.log1p(gaps) / 5, mass[bars].clamp(max=8) / 8], -1)
    features = torch.cat([logits.softmax(-1), chroma[bars], posterior[bars], timing], -1)
    return features


@dataclass
class PitchContextConfig:
    d_model: int = 96
    layers: int = 2
    max_residual: float = 1.5
    dropout: float = .1


class PitchContextRefiner(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or PitchContextConfig()
        c = self.config
        self.projection = nn.Sequential(nn.Linear(169, c.d_model), nn.GELU(), nn.LayerNorm(c.d_model))
        layer = nn.TransformerEncoderLayer(c.d_model, 4, c.d_model * 3, c.dropout,
            activation='gelu', batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, c.layers, norm=nn.LayerNorm(c.d_model),
            enable_nested_tensor=False)
        self.residual = nn.Linear(c.d_model, 128)
        nn.init.zeros_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)

    def forward(self, logits, features, padding_mask=None):
        hidden = self.projection(features)
        positions = torch.arange(hidden.shape[1], device=hidden.device).float()[:, None]
        frequencies = torch.exp(torch.arange(0, self.config.d_model, 2, device=hidden.device)
            * (-math.log(10000.) / self.config.d_model))
        encoding = torch.zeros_like(hidden[0])
        encoding[:, 0::2] = torch.sin(positions * frequencies)
        encoding[:, 1::2] = torch.cos(positions * frequencies)
        hidden = self.encoder(hidden + encoding, src_key_padding_mask=padding_mask)
        # Bounded residual: contextual evidence can resolve uncertainty, but a
        # >2*max_residual acoustic logit margin cannot be overturned. No pitch
        # class, interval, key membership or ABA pattern is prohibited.
        return logits + self.config.max_residual * self.residual(hidden).tanh()


@torch.inference_mode()
def refine_logits(model, logits, features, device, chunk_notes=64):
    if not len(logits):
        return logits.clone()
    if chunk_notes < 2:
        raise ValueError('Need at least two notes of context.')
    starts = list(range(0, max(1, len(logits) - chunk_notes + 1), chunk_notes // 2))
    final = max(0, len(logits) - chunk_notes)
    if starts[-1] != final:
        starts.append(final)
    totals, weights = torch.zeros_like(logits), torch.zeros(len(logits))
    for start in starts:
        end = min(len(logits), start + chunk_notes)
        pred = model(logits[start:end][None].to(device), features[start:end][None].to(device))[0].cpu()
        weight = torch.hann_window(end - start + 2, periodic=False)[1:-1].clamp_min(.05)
        totals[start:end] += pred * weight[:, None]
        weights[start:end] += weight
    return totals / weights[:, None]


def replace_note_pitches(notes, logits):
    if len(notes) != len(logits):
        raise ValueError('Refinement must preserve every original note boundary.')
    return [replace(note, pitch=int(row.argmax())) for note, row in zip(notes, logits)]


def tonality_report(logits, starts, durations, total_bars=None):
    posterior, _, mass, membership = tonal_context(logits, starts, durations, total_bars)
    best = posterior.argmax(-1)
    ranked = posterior.sort(-1, descending=True).values
    # Reporting-only confidence policy, NEVER a pitch-correction rule.
    known = (ranked[:, 0] >= .35) & ((ranked[:, 0] - ranked[:, 1]) >= .10) & (mass > 0)
    regions = []
    for bar, key in enumerate(best.tolist()):
        label = key if bool(known[bar]) else -1
        if regions and regions[-1]['key_index'] == label:
            regions[-1]['end_bar_1based'] = bar + 1
        else:
            regions.append({'start_bar_1based': bar + 1, 'end_bar_1based': bar + 1,
                'key_index': label, 'key_candidate': key_name(label) if label >= 0 else 'uncertain'})
    changes, excursions = [], []
    for i in range(1, len(regions)):
        left, right = regions[i-1], regions[i]
        length = right['end_bar_1based'] - right['start_bar_1based'] + 1
        if left['key_index'] < 0 or right['key_index'] < 0:
            continue
        if length >= 4 and left['end_bar_1based'] - left['start_bar_1based'] + 1 >= 4:
            changes.append({'bar_1based': right['start_bar_1based'],
                'from': left['key_candidate'], 'to': right['key_candidate']})
        elif length < 4 and i + 1 < len(regions) and regions[i+1]['key_index'] == left['key_index']:
            excursions.append(right)
    chromatic = []
    for i, (pitch, start) in enumerate(zip(logits.argmax(-1).tolist(), starts.tolist())):
        bar = start // 48
        if known[bar] and not membership[best[bar], pitch % 12]:
            chromatic.append({'note_index_0based': i, 'bar_1based': bar + 1, 'pitch': pitch,
                'key_candidate': key_name(int(best[bar]))})
    return {'uses_inst_audio': False, 'key_supervision': 'none',
        'time_reference': 'One-based MIDI grid bars, including leading/trailing padding.',
        'scope': 'Melody-only major/natural-minor hypotheses, not verified harmonic analysis. '
        'Other modes, relative-key ambiguity, borrowed tones and minor raised sevenths need review.',
        'confidence_note': 'Profile/HMM scores are not calibrated probabilities. Four-bar persistence '
        'and confidence thresholds label report candidates only; they never force note pitches.',
        'regions': regions, 'modulation_candidates': changes,
        'temporary_tonal_excursion_candidates': excursions,
        'non_diatonic_note_candidates': chromatic,
        'bar_key_scores': posterior.tolist()}


def load_refiner(path, device, base_sha256):
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    if checkpoint['base_sha256'] != base_sha256:
        raise ValueError('Pitch context checkpoint was trained for a different acoustic model.')
    model = PitchContextRefiner(PitchContextConfig(**checkpoint['config'])).to(device)
    model.load_state_dict(checkpoint['model_state'], strict=True)
    model.eval()
    return model, checkpoint
