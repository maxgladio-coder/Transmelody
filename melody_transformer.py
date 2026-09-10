from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import mido
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset

from prepare_dataset import interpolate_with_extrapolation
from project_settings import PROJECT_PPQ
from articulation_encoder import (
    ARTICULATION_FEATURE_SIZE,
    ARTICULATION_HIDDEN_SIZE,
    DEFAULT_ARTICULATION_MODEL,
    extract_articulation_sequence,
    sample_articulation_at_times,
    pronunciation_evidence,
)
from musical_timeline import canonicalize_grid, samples_at_ticks


# A 1/48-note timeline is the smallest shared integer lattice for straight
# sixteenths (120 ticks) and eighth-note triplets (160 ticks) at PPQ 480.
GRID_DIVISION = 12
TICKS_PER_STEP = PROJECT_PPQ // GRID_DIVISION
STEPS_PER_BAR = 4 * GRID_DIVISION
STRAIGHT_TICKS_PER_STEP = PROJECT_PPQ // 4
TRIPLET_TICKS_PER_STEP = PROJECT_PPQ // 3
STRAIGHT_STEP_MULTIPLE = STRAIGHT_TICKS_PER_STEP // TICKS_PER_STEP
TRIPLET_STEP_MULTIPLE = TRIPLET_TICKS_PER_STEP // TICKS_PER_STEP
RHYTHM_STRAIGHT = 0
RHYTHM_TRIPLET = 1
RHYTHM_NAMES = ("straight_16", "triplet_8")
NUM_RHYTHM_CLASSES = len(RHYTHM_NAMES)
NUM_PITCH_CLASSES = 129  # 0=silence, MIDI pitch p is class p+1
MAX_DURATION_STEPS = 192


@dataclass(frozen=True)
class FeatureConfig:
    sample_rate: int = 16000
    n_fft: int = 1024
    hop_length: int = 160
    n_mels: int = 96
    f_min: float = 40.0
    f_max: float = 8000.0
    # Thirteen 10 ms frames let the boundary heads see consonant/vowel
    # transitions around each musical Grid line.
    context_offsets: tuple[int, ...] = tuple(range(-6, 7))
    use_articulation_features: bool = False
    articulation_model_id: str = DEFAULT_ARTICULATION_MODEL

    @property
    def input_dim(self) -> int:
        base = self.n_mels * len(self.context_offsets) + 1
        return base + (
            ARTICULATION_FEATURE_SIZE
            if self.use_articulation_features
            else 0
        )

    @property
    def audio_valid_index(self) -> int:
        return self.n_mels * len(self.context_offsets)

    @property
    def articulation_start_index(self) -> int:
        return self.audio_valid_index + 1

    @property
    def articulation_novelty_index(self) -> int:
        return self.articulation_start_index + ARTICULATION_HIDDEN_SIZE

    @property
    def articulation_token_change_index(self) -> int:
        return self.articulation_novelty_index + 1


@dataclass(frozen=True)
class ModelConfig:
    input_dim: int
    d_model: int = 128
    nhead: int = 4
    num_layers: int = 4
    dim_feedforward: int = 384
    dropout: float = 0.1
    max_duration_steps: int = MAX_DURATION_STEPS
    use_articulation_heads: bool = False
    use_sequence_position: bool = False  # False preserves old checkpoint behavior.
    use_pronunciation_context: bool = False
    articulation_feature_start: int = 0
    pronunciation_radius: int = 24
    use_musical_event_context: bool = False
    musical_event_version: int = 1
    use_note_event_model: bool = False
    note_activity_weight: float = 0.0  # Opt-in soft score, no activity threshold.
    acoustic_bins: int = 96
    acoustic_frames: int = 13


def read_manifest(dataset_root: Path) -> list[dict]:
    manifest_path = dataset_root / "manifest.jsonl"
    return [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _merge_overlapping_same_pitch_notes(notes_payload: dict) -> dict:
    """Collapse accidental same-pitch overlaps without hiding true polyphony."""
    ordered_notes = sorted(
        notes_payload["notes"],
        key=lambda note: (
            int(note["midi_time"]["start_tick"]),
            int(note["midi_time"]["end_tick"]),
            int(note["pitch"]),
        ),
    )
    merged: list[dict] = []
    for note in ordered_notes:
        current = {
            **note,
            "midi_time": dict(note["midi_time"]),
        }
        if merged:
            previous = merged[-1]
            previous_time = previous["midi_time"]
            current_time = current["midi_time"]
            if (
                int(current["pitch"]) == int(previous["pitch"])
                and int(current_time["start_tick"])
                < int(previous_time["end_tick"])
            ):
                previous_time["end_tick"] = max(
                    int(previous_time["end_tick"]),
                    int(current_time["end_tick"]),
                )
                continue
        merged.append(current)
    return {**notes_payload, "notes": merged}


def _labels_from_notes(notes_payload: dict, grid_end_tick: int) -> tuple[torch.Tensor, torch.Tensor]:
    notes_payload = _merge_overlapping_same_pitch_notes(notes_payload)
    num_steps = math.ceil(grid_end_tick / TICKS_PER_STEP)
    pitch = torch.zeros(num_steps, dtype=torch.long)
    onset = torch.zeros(num_steps, dtype=torch.float32)

    for note in notes_payload["notes"]:
        midi_time = note["midi_time"]
        start_tick = int(midi_time["start_tick"])
        end_tick = int(midi_time["end_tick"])
        midi_pitch = int(note["pitch"])
        start_step = round(start_tick / TICKS_PER_STEP)
        end_step = round(end_tick / TICKS_PER_STEP)
        if start_step * TICKS_PER_STEP != start_tick:
            raise ValueError(f"Off-grid note start at tick {start_tick}.")
        if end_step * TICKS_PER_STEP != end_tick:
            raise ValueError(f"Off-grid note end at tick {end_tick}.")
        if end_step <= start_step:
            raise ValueError(f"Non-positive note at tick {start_tick}.")
        occupied = pitch[start_step:end_step]
        if torch.any(occupied != 0):
            raise ValueError(
                f"Polyphonic label at ticks {start_tick}-{end_tick}; "
                "the melody target must be monophonic."
            )
        pitch[start_step:end_step] = midi_pitch + 1
        onset[start_step] = 1.0
    return pitch, onset


def _event_labels_from_notes(
    notes_payload: dict,
    grid_end_tick: int,
    max_duration_steps: int = MAX_DURATION_STEPS,
) -> dict[str, torch.Tensor]:
    notes_payload = _merge_overlapping_same_pitch_notes(notes_payload)
    frame_pitch, onset = _labels_from_notes(notes_payload, grid_end_tick)
    num_steps = len(frame_pitch)
    offset = torch.zeros(num_steps, dtype=torch.float32)
    activity = (frame_pitch != 0).to(torch.float32)
    onset_pitch = torch.zeros(num_steps, dtype=torch.long)
    duration = torch.zeros(num_steps, dtype=torch.long)
    duration_mask = torch.zeros(num_steps, dtype=torch.bool)
    continuation = torch.zeros(num_steps, dtype=torch.float32)
    pitch_change = torch.zeros(num_steps, dtype=torch.float32)
    articulation_boundary = torch.zeros(num_steps, dtype=torch.float32)
    rhythm_mode, rhythm_mask = _rhythm_labels_from_notes(
        notes_payload,
        grid_end_tick,
    )

    ordered_notes = sorted(
        notes_payload["notes"],
        key=lambda note: (
            int(note["midi_time"]["start_tick"]),
            int(note["midi_time"]["end_tick"]),
        ),
    )
    previous_end_step: int | None = None
    previous_pitch: int | None = None
    for note in ordered_notes:
        midi_time = note["midi_time"]
        start_step = int(midi_time["start_tick"]) // TICKS_PER_STEP
        end_step = int(midi_time["end_tick"]) // TICKS_PER_STEP
        duration_steps = end_step - start_step
        onset_pitch[start_step] = int(note["pitch"]) + 1
        duration[start_step] = min(duration_steps, max_duration_steps) - 1
        duration_mask[start_step] = True
        if end_step - start_step > 1:
            continuation[start_step : end_step - 1] = 1.0
        if end_step < num_steps:
            offset[end_step] = 1.0
        note_pitch = int(note["pitch"])
        if (
            previous_end_step == start_step
            and previous_pitch is not None
        ):
            if previous_pitch != note_pitch:
                pitch_change[start_step] = 1.0
            else:
                articulation_boundary[start_step] = 1.0
        previous_end_step = end_step
        previous_pitch = note_pitch

    return {
        "pitch": frame_pitch,
        "onset": onset,
        "offset": offset,
        "activity": activity,
        "onset_pitch": onset_pitch,
        "duration": duration,
        "duration_mask": duration_mask,
        "continuation": continuation,
        "pitch_change": pitch_change,
        "articulation_boundary": articulation_boundary,
        "rhythm_mode": rhythm_mode,
        "rhythm_mask": rhythm_mask,
    }


def _rhythm_labels_from_notes(
    notes_payload: dict,
    grid_end_tick: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Infer confident per-bar straight/triplet labels from exact MIDI edges."""
    num_steps = math.ceil(grid_end_tick / TICKS_PER_STEP)
    num_bars = math.ceil(num_steps / STEPS_PER_BAR)
    bar_mode = torch.zeros(num_bars, dtype=torch.long)
    bar_mask = torch.zeros(num_bars, dtype=torch.bool)
    errors = [[0, 0] for _ in range(num_bars)]
    evidence_count = [0 for _ in range(num_bars)]
    bar_ticks = PROJECT_PPQ * 4

    for note in notes_payload["notes"]:
        midi_time = note["midi_time"]
        for key in ("start_tick", "end_tick"):
            tick = int(midi_time[key])
            if tick <= 0 or tick >= grid_end_tick:
                continue
            local_tick = tick % bar_ticks
            if local_tick == 0:
                continue
            bar_index = min(num_bars - 1, tick // bar_ticks)
            straight_remainder = local_tick % STRAIGHT_TICKS_PER_STEP
            triplet_remainder = local_tick % TRIPLET_TICKS_PER_STEP
            straight_distance = min(
                straight_remainder,
                STRAIGHT_TICKS_PER_STEP - straight_remainder,
            )
            triplet_distance = min(
                triplet_remainder,
                TRIPLET_TICKS_PER_STEP - triplet_remainder,
            )
            errors[bar_index][RHYTHM_STRAIGHT] += straight_distance
            errors[bar_index][RHYTHM_TRIPLET] += triplet_distance
            evidence_count[bar_index] += 1

    for bar_index, (straight_error, triplet_error) in enumerate(errors):
        if (
            evidence_count[bar_index] == 0
            or straight_error == triplet_error
        ):
            continue
        if triplet_error < straight_error:
            bar_mode[bar_index] = RHYTHM_TRIPLET
        else:
            bar_mode[bar_index] = RHYTHM_STRAIGHT
        bar_mask[bar_index] = True

    return (
        bar_mode.repeat_interleave(STEPS_PER_BAR)[:num_steps],
        bar_mask.repeat_interleave(STEPS_PER_BAR)[:num_steps],
    )


def _sample_context(
    mel: torch.Tensor,
    frame_positions: torch.Tensor,
    offsets: tuple[int, ...],
) -> torch.Tensor:
    contexts: list[torch.Tensor] = []
    max_frame = mel.shape[-1] - 1
    for offset in offsets:
        positions = frame_positions + offset
        left = torch.floor(positions).long()
        right = left + 1
        fraction = (positions - left).unsqueeze(0)
        valid = (positions >= 0) & (positions <= max_frame)
        left = left.clamp(0, max_frame)
        right = right.clamp(0, max_frame)
        values = mel[:, left] * (1.0 - fraction) + mel[:, right] * fraction
        values[:, ~valid] = 0.0
        contexts.append(values.transpose(0, 1))
    return torch.cat(contexts, dim=-1)


@torch.inference_mode()
def extract_grid_features(
    audio_path: Path,
    grid: dict,
    feature_config: FeatureConfig,
    device: torch.device,
) -> tuple[torch.Tensor, int, int]:
    audio, source_sr = sf.read(
        audio_path,
        dtype="float32",
        always_2d=True,
    )
    original_num_samples = len(audio)
    mono = torch.from_numpy(np.mean(audio, axis=1)).to(device)
    if source_sr != feature_config.sample_rate:
        mono = torchaudio.functional.resample(
            mono,
            source_sr,
            feature_config.sample_rate,
        )

    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=feature_config.sample_rate,
        n_fft=feature_config.n_fft,
        hop_length=feature_config.hop_length,
        n_mels=feature_config.n_mels,
        f_min=feature_config.f_min,
        f_max=feature_config.f_max,
        power=2.0,
    ).to(device)
    mel = torch.log1p(100.0 * mel_transform(mono))

    grid_end_tick = int(grid["midi_alignment"]["grid_end_tick"])
    num_steps = math.ceil(grid_end_tick / TICKS_PER_STEP)
    step_ticks = np.arange(num_steps, dtype=np.float64) * TICKS_PER_STEP
    beat_ticks = np.asarray([beat["tick"] for beat in grid["beats"]], dtype=np.float64)
    beat_samples = np.asarray([beat["sample"] for beat in grid["beats"]], dtype=np.float64)
    original_samples = interpolate_with_extrapolation(
        step_ticks,
        beat_ticks,
        beat_samples,
    )
    if grid.get("timeline"):
        original_samples = samples_at_ticks(grid, step_ticks, source_sr)
    frame_positions = torch.as_tensor(
        original_samples
        / source_sr
        * feature_config.sample_rate
        / feature_config.hop_length,
        dtype=torch.float32,
        device=device,
    )
    features = _sample_context(
        mel,
        frame_positions,
        feature_config.context_offsets,
    )
    audio_valid = torch.as_tensor(
        (original_samples >= 0) & (original_samples < original_num_samples),
        dtype=torch.float32,
        device=device,
    ).unsqueeze(1)
    features = torch.cat([features, audio_valid], dim=1).cpu().to(torch.float16)
    if feature_config.use_articulation_features:
        articulation_sequence, articulation_times = (
            extract_articulation_sequence(
                mono,
                model_id=feature_config.articulation_model_id,
                device=device,
            )
        )
        query_times = torch.as_tensor(
            original_samples / source_sr,
            dtype=torch.float32,
        )
        articulation = sample_articulation_at_times(
            articulation_sequence,
            articulation_times,
            query_times,
        )
        features = torch.cat([features, articulation], dim=1)
    return features, source_sr, original_num_samples


@torch.inference_mode()
def build_song_cache(
    dataset_root: Path,
    song: dict,
    cache_dir: Path,
    feature_config: FeatureConfig,
    device: torch.device,
) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{song['song_id']}_grid_features.pt"

    audio_path = dataset_root / song["vocal_audio"]
    grid_path = dataset_root / song["grid"]
    labels_path = dataset_root / song["labels"]
    midi_path = dataset_root / song["vocal_midi"]
    source_fingerprint = {
        str(path.resolve()): (path.stat().st_mtime_ns, path.stat().st_size)
        for path in (audio_path, grid_path, labels_path, midi_path)
    }
    from pronunciation_labels import digest, load_reviewed, reviewed_targets
    pronunciation_path = dataset_root / "pronunciation_labels" / f"{song['song_id']}.json"
    reviewed = None
    if pronunciation_path.exists():
        # Content fingerprints catch edits even if an editor preserves mtimes.
        reviewed = load_reviewed(pronunciation_path, audio_path, str(song["song_id"]))
        source_fingerprint[str(pronunciation_path.resolve())] = digest(pronunciation_path)
    if cache_path.exists():
        cached = torch.load(cache_path, map_location="cpu", weights_only=False)
        if (
            cached.get("schema_version") == "grid_feature_cache_v10_reviewed_pronunciation"
            and
            cached.get("source_fingerprint") == source_fingerprint
            and cached.get("feature_config") == asdict(feature_config)
        ):
            return cache_path

    grid = json.loads(grid_path.read_text(encoding="utf-8"))
    from prepare_dataset import midi_metadata, build_note_labels, SongFiles
    metadata = midi_metadata(midi_path)
    grid = canonicalize_grid(grid, [(e["tick"], e["tempo"]) for e in metadata["tempo_events"]],
        audio_start_tick=metadata["audio_start_tick"],
        minimum_end_tick=max((n.original_end_tick for n in metadata["notes"]), default=0),
        source="reviewed_midi")
    # Read current MIDI even when a caller skipped prepare_dataset; stale JSON
    # labels must never win over a newly corrected score.
    info = sf.info(audio_path)
    labels_payload = build_note_labels(song=SongFiles(song["song_id"], dataset_root / song["inst_audio"], audio_path, midi_path),
        grid=grid, midi_data=metadata, sample_rate=info.samplerate, num_samples=info.frames)
    grid_end_tick = int(grid["midi_alignment"]["grid_end_tick"])
    targets = _event_labels_from_notes(labels_payload, grid_end_tick)
    features, _, _ = extract_grid_features(
        audio_path,
        grid,
        feature_config,
        device,
    )
    num_steps = len(features)
    articulation_mask = torch.zeros(num_steps, dtype=torch.bool)
    if feature_config.use_articulation_features:
        articulation_mask = (
            features[:, feature_config.audio_valid_index].float() >= 0.5
        )
    observed = features[:, feature_config.audio_valid_index].float() >= 0.5
    boundary_observed = observed.clone()
    boundary_observed[1:] &= observed[:-1]
    continuation_observed = observed.clone()
    continuation_observed[:-1] &= observed[1:]
    continuation_observed[-1] = False
    targets["loss_mask"] = observed
    targets["boundary_mask"] = boundary_observed
    targets["continuation_mask"] = continuation_observed
    targets["duration_mask"] &= observed
    for note in labels_payload["notes"]:
        if not note["training"]["fully_observable"]:
            index = int(note["midi_time"]["start_tick"]) // TICKS_PER_STEP
            if 0 <= index < len(observed):
                targets["duration_mask"][index] = False
    targets["pronunciation_boundary"] = (pronunciation_evidence(features, feature_config.articulation_novelty_index)
        if feature_config.use_articulation_features else torch.zeros(num_steps))
    targets["pronunciation_weight"] = (articulation_mask & boundary_observed).float() * 0.10
    reviewed_count = 0
    if reviewed is not None:
        times = samples_at_ticks(grid, np.arange(num_steps) * TICKS_PER_STEP, info.samplerate) / info.samplerate
        reviewed_boundary, reviewed_mask = reviewed_targets(reviewed, times)
        reviewed_mask &= articulation_mask & boundary_observed
        targets["pronunciation_boundary"][reviewed_mask] = reviewed_boundary[reviewed_mask]
        targets["pronunciation_weight"][reviewed_mask] = 1.0
        reviewed_count = int(reviewed_mask.sum())
    print(f"[pronunciation] {song['song_id']}: {reviewed_count} reviewed frames", flush=True)

    torch.save(
        {
            "schema_version": "grid_feature_cache_v10_reviewed_pronunciation",
            "song_id": song["song_id"],
            "source_fingerprint": source_fingerprint,
            "feature_config": asdict(feature_config),
            "features": features,
            **targets,
            "articulation_mask": articulation_mask & boundary_observed,
            "grid": grid,
            "source_audio": str(audio_path),
            "reviewed_pronunciation_frames": reviewed_count,
        },
        cache_path,
    )
    return cache_path


def build_feature_caches(
    dataset_root: Path,
    cache_dir: Path,
    feature_config: FeatureConfig,
    device: torch.device,
) -> list[Path]:
    paths = []
    for song in read_manifest(dataset_root):
        print(f"[features] {song['song_id']}", flush=True)
        paths.append(
            build_song_cache(
                dataset_root,
                song,
                cache_dir,
                feature_config,
                device,
            )
        )
    return paths


class GridChunkDataset(Dataset):
    def __init__(
        self,
        cache_paths: list[Path],
        chunk_steps: int,
        stride_steps: int,
    ) -> None:
        self.songs = [
            torch.load(path, map_location="cpu", weights_only=False)
            for path in cache_paths
        ]
        self.chunk_steps = chunk_steps
        self.index: list[tuple[int, int]] = []
        for song_index, song in enumerate(self.songs):
            length = len(song["pitch"])
            starts = list(range(0, max(1, length - chunk_steps + 1), stride_steps))
            final_start = max(0, length - chunk_steps)
            if not starts or starts[-1] != final_start:
                starts.append(final_start)
            self.index.extend((song_index, start) for start in starts)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        song_index, start = self.index[index]
        song = self.songs[song_index]
        end = min(start + self.chunk_steps, len(song["pitch"]))
        length = end - start
        features = torch.zeros(
            self.chunk_steps,
            song["features"].shape[1],
            dtype=torch.float32,
        )
        target_names = (
            "pitch",
            "onset",
            "offset",
            "activity",
            "onset_pitch",
            "duration",
            "duration_mask",
            "continuation",
            "pitch_change",
            "articulation_boundary",
            "articulation_mask",
            "rhythm_mode",
            "rhythm_mask",
            "loss_mask",
            "boundary_mask",
            "continuation_mask",
            "pronunciation_boundary",
            "pronunciation_weight",
        )
        targets = {
            name: torch.zeros(
                self.chunk_steps,
                dtype=song[name].dtype,
            )
            for name in target_names
        }
        valid = torch.zeros(self.chunk_steps, dtype=torch.bool)
        features[:length] = song["features"][start:end].float()
        for name in target_names:
            targets[name][:length] = song[name][start:end]
        valid[:length] = True
        positions = (torch.arange(self.chunk_steps) + start) % STEPS_PER_BAR
        return {
            "features": features,
            **targets,
            "valid": valid,
            "bar_position": positions,
        }


def soft_pronunciation_context(hidden: torch.Tensor, boundary: torch.Tensor,
                               radius: int, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
    """Pool nearby frames with a soft penalty for crossing speech boundaries.

    No discrete word partition is imposed: note/pitch heads can still resolve
    several MIDI notes within one sustained vowel.
    """
    cumulative = torch.cumsum(torch.log1p(-boundary.float().clamp(max=0.999)), dim=1)
    weights = torch.exp(-abs(cumulative[:, :, None] - cumulative[:, None, :]))
    positions = torch.arange(hidden.shape[1], device=hidden.device)
    weights = weights * ((positions[:, None] - positions[None, :]).abs() <= radius)
    if padding_mask is not None:
        weights = weights.masked_fill(padding_mask[:, None, :], 0)
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    return torch.bmm(weights.to(hidden.dtype), hidden)


class MelodyTransformer(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if not math.isfinite(config.note_activity_weight) or config.note_activity_weight < 0:
            raise ValueError('Note activity weight must be finite and nonnegative.')
        self.config = config
        self.input_projection = nn.Sequential(
            nn.LayerNorm(config.input_dim),
            nn.Linear(config.input_dim, config.d_model),
            nn.GELU(),
        )
        self.bar_position = nn.Embedding(STEPS_PER_BAR, config.d_model)
        if config.use_musical_event_context:
            if config.acoustic_bins * config.acoustic_frames >= config.input_dim:
                raise ValueError("Musical-event attention requires a complete local mel patch.")
            self.event_frame_projection = nn.Sequential(nn.LayerNorm(config.acoustic_bins),
                nn.Linear(config.acoustic_bins, config.d_model), nn.GELU())
            self.event_frame_position = nn.Parameter(torch.randn(config.acoustic_frames, config.d_model) * .02)
            self.event_query = nn.Linear(config.d_model, config.d_model)
            self.event_fusion = nn.Sequential(nn.Linear(2 * config.d_model, config.d_model), nn.GELU(),
                nn.LayerNorm(config.d_model))
            if config.musical_event_version >= 2:
                self.event_temporal = nn.Conv1d(config.d_model, config.d_model, kernel_size=5, padding=2)
                event_layer = nn.TransformerEncoderLayer(config.d_model, config.nhead,
                    config.dim_feedforward, config.dropout, activation="gelu", batch_first=True, norm_first=True)
                self.event_encoder = nn.TransformerEncoder(event_layer, num_layers=2,
                    norm=nn.LayerNorm(config.d_model), enable_nested_tensor=False)
                self.event_onset_head = nn.Linear(config.d_model, 1)
                self.event_offset_head = nn.Linear(config.d_model, 1)
                if config.musical_event_version >= 3:
                    # One MIDI-supervised decision at a score boundary. Include
                    # the preceding cell so continuation is aligned to this edge.
                    self.boundary_state_head = nn.Sequential(
                        nn.Linear(3 * config.d_model + 5, config.d_model), nn.GELU(),
                        nn.LayerNorm(config.d_model), nn.Linear(config.d_model, 4))
        if config.use_pronunciation_context:
            if config.input_dim - config.articulation_feature_start != ARTICULATION_FEATURE_SIZE:
                raise ValueError("Pronunciation context requires the frozen Japanese speech features.")
            self.pronunciation_projection = nn.Sequential(
                nn.LayerNorm(ARTICULATION_FEATURE_SIZE),
                nn.Linear(ARTICULATION_FEATURE_SIZE, config.d_model), nn.GELU())
            self.pronunciation_head = nn.Conv1d(config.d_model, 1, kernel_size=3, padding=1)
            self.pronunciation_fusion = nn.Sequential(
                nn.Linear(3 * config.d_model + 1, config.d_model), nn.GELU(),
                nn.LayerNorm(config.d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_layers,
            norm=nn.LayerNorm(config.d_model),
            enable_nested_tensor=False,
        )
        self.pitch_head = nn.Linear(config.d_model, NUM_PITCH_CLASSES)
        self.onset_head = nn.Linear(config.d_model, 1)
        self.offset_head = nn.Linear(config.d_model, 1)
        self.activity_head = nn.Linear(config.d_model, 1)
        self.duration_head = nn.Linear(
            config.d_model,
            config.max_duration_steps,
        )
        self.continuation_head = nn.Linear(config.d_model, 1)
        self.rhythm_head = nn.Linear(config.d_model, NUM_RHYTHM_CLASSES)
        if config.use_articulation_heads:
            self.articulation_head = nn.Linear(config.d_model, 1)
            self.pitch_change_head = nn.Linear(config.d_model, 1)

    def forward(
        self,
        features: torch.Tensor,
        bar_position: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        acoustic_hidden = self.input_projection(features)
        hidden = acoustic_hidden + self.bar_position(bar_position)
        if self.config.use_sequence_position:
            position = torch.arange(hidden.shape[1], device=hidden.device).float()[:, None]
            frequency = torch.exp(torch.arange(0, self.config.d_model, 2, device=hidden.device).float()
                                  * (-math.log(10000.0) / self.config.d_model))
            encoding = torch.zeros(hidden.shape[1], self.config.d_model, device=hidden.device)
            encoding[:, 0::2] = torch.sin(position * frequency)
            encoding[:, 1::2] = torch.cos(position * frequency[:encoding[:, 1::2].shape[1]])
            hidden = hidden + encoding.to(hidden.dtype)
        pronunciation_logits = None
        if self.config.use_pronunciation_context:
            speech = self.pronunciation_projection(features[:, :, self.config.articulation_feature_start:])
            if padding_mask is not None:
                speech = speech.masked_fill(padding_mask[:, :, None], 0)
            pronunciation_logits = self.pronunciation_head(speech.transpose(1, 2)).squeeze(1)
            boundary = torch.sigmoid(pronunciation_logits)
            context = soft_pronunciation_context(hidden, boundary, self.config.pronunciation_radius, padding_mask)
            hidden = hidden + self.pronunciation_fusion(torch.cat([hidden, speech, context, boundary[:, :, None]], dim=-1))
        hidden = self.encoder(hidden, src_key_padding_mask=padding_mask)
        if self.config.use_musical_event_context:
            # The MIDI-trained musical context queries 10 ms acoustic frames
            # around a score coordinate. Attention is evidence selection, NOT
            # a claim that its maximum is the physical/perceptual onset time.
            patch = features[:, :, :self.config.acoustic_bins * self.config.acoustic_frames]
            patch = patch.reshape(*patch.shape[:2], self.config.acoustic_frames, self.config.acoustic_bins)
            frame_hidden = self.event_frame_projection(patch) + self.event_frame_position
            query = self.event_query(hidden)
            weights = torch.softmax((frame_hidden * query[:, :, None]).sum(-1).float()
                / math.sqrt(self.config.d_model), dim=-1).to(frame_hidden.dtype)
            evidence = (frame_hidden * weights[:, :, :, None]).sum(-2)
            hidden = hidden + self.event_fusion(torch.cat([hidden, evidence], dim=-1))
        outputs = {
            "pitch": self.pitch_head(hidden),
            "onset": self.onset_head(hidden).squeeze(-1),
            "offset": self.offset_head(hidden).squeeze(-1),
            "activity": self.activity_head(hidden).squeeze(-1),
            "duration": self.duration_head(hidden),
            "continuation": self.continuation_head(hidden).squeeze(-1),
            "rhythm": self.rhythm_head(hidden),
        }
        if self.config.use_articulation_heads:
            outputs["articulation"] = self.articulation_head(hidden).squeeze(-1)
            outputs["pitch_change"] = self.pitch_change_head(hidden).squeeze(-1)
        if pronunciation_logits is not None:
            outputs["pronunciation"] = pronunciation_logits
        if self.config.use_musical_event_context:
            # Explicit version signal: legacy checkpoints retain their decoder.
            # The onset lane deliberately has NO bar-position embedding or
            # pronunciation-span gate. Otherwise the rare triplet onsets can
            # be suppressed by a learned straight-grid positional shortcut.
            if self.config.musical_event_version >= 2:
                content = acoustic_hidden
                if padding_mask is not None:
                    content = content.masked_fill(padding_mask[:, :, None], 0)
                content = content + self.event_temporal(content.transpose(1, 2)).transpose(1, 2)
                content = self.event_encoder(content, src_key_padding_mask=padding_mask)
                outputs["onset"] = self.event_onset_head(content).squeeze(-1)
                outputs["offset"] = self.event_offset_head(content).squeeze(-1)
                if self.config.musical_event_version >= 3:
                    previous_hidden = torch.cat([torch.zeros_like(hidden[:, :1]), hidden[:, :-1]], dim=1)
                    previous_continue = torch.cat([torch.zeros_like(outputs["continuation"][:, :1]),
                        outputs["continuation"][:, :-1]], dim=1)
                    # Neighboring acoustic evidence can support the exact MIDI
                    # grid edge without moving that edge to a second peak.
                    local_onset = torch.nn.functional.max_pool1d(
                        outputs["onset"][:, None], 3, stride=1, padding=1).squeeze(1)
                    evidence = torch.stack([outputs["onset"], outputs["offset"],
                        outputs["activity"], previous_continue, local_onset], dim=-1)
                    outputs["boundary_state"] = self.boundary_state_head(
                        torch.cat([hidden, content, previous_hidden, evidence], dim=-1))
            outputs["musical_event_onset"] = outputs["onset"]
            if self.config.use_note_event_model:
                outputs["note_event_onset"] = outputs["onset"]
                if self.config.note_activity_weight > 0:
                    outputs['note_active_score'] = F.logsigmoid(outputs['activity']) * self.config.note_activity_weight / 3
                    outputs['note_rest_score'] = F.logsigmoid(-outputs['activity']) * self.config.note_activity_weight / 3
        return outputs


@dataclass(frozen=True)
class PredictedNote:
    start_step: int
    end_step: int
    pitch: int
    onset_probability: float
    duration_steps: int


def decode_rhythm_modes(
    rhythm_logits: torch.Tensor,
    *,
    total_steps: int | None = None,
    triplet_threshold: float = 0.60,
) -> torch.Tensor:
    """Decide each bar independently; confident single-bar triplets survive."""
    logits = rhythm_logits.float()
    if not 0.5 <= triplet_threshold <= 1.0:
        raise ValueError("Triplet confidence must be in [0.5, 1].")
    if logits.ndim != 2 or logits.shape[-1] != NUM_RHYTHM_CLASSES:
        raise ValueError("Rhythm logits must have shape [steps, 2].")
    if total_steps is None:
        total_steps = len(logits)
    num_bars = math.ceil(total_steps / STEPS_PER_BAR)
    modes = []
    for bar_index in range(num_bars):
        start = bar_index * STEPS_PER_BAR
        end = min(len(logits), start + STEPS_PER_BAR)
        if end <= start:
            raise ValueError("Missing rhythm logits for a bar.")
        confidence = torch.softmax(logits[start:end], dim=-1).mean(dim=0)[RHYTHM_TRIPLET]
        modes.append(int(confidence >= triplet_threshold))
    return torch.tensor(modes, dtype=torch.long)


def rhythm_boundary_mask(
    rhythm_modes: torch.Tensor,
    total_steps: int,
) -> torch.Tensor:
    allowed = torch.zeros(total_steps, dtype=torch.bool)
    for step in range(total_steps):
        bar_index = min(len(rhythm_modes) - 1, step // STEPS_PER_BAR)
        local_step = step % STEPS_PER_BAR
        multiple = (
            TRIPLET_STEP_MULTIPLE
            if int(rhythm_modes[bar_index]) == RHYTHM_TRIPLET
            else STRAIGHT_STEP_MULTIPLE
        )
        allowed[step] = local_step % multiple == 0
    return allowed


def decode_segment_events(
    outputs: dict[str, torch.Tensor],
    *,
    onset_threshold: float = 0.55,
    strong_onset_threshold: float = 0.75,
    continuation_threshold: float = 0.50,
    offset_threshold: float = 0.60,
    pitch_change_threshold: float = 0.60,
    articulation_threshold: float = 0.70,
    articulation_onset_floor: float = 0.30,
    activity_threshold: float = 0.35,
    observation_mask: torch.Tensor | None = None,
    rhythm_modes: torch.Tensor | None = None,
) -> list[PredictedNote]:
    """Decode notes from learned boundaries and same-note continuation."""
    onset_probability = torch.sigmoid(outputs["onset"].float())
    offset_probability = torch.sigmoid(outputs["offset"].float())
    activity_probability = torch.sigmoid(outputs["activity"].float())
    continuation_probability = torch.sigmoid(
        outputs["continuation"].float()
    )
    pitch_change_probability = (
        torch.sigmoid(outputs["pitch_change"].float())
        if "pitch_change" in outputs
        else torch.zeros_like(onset_probability)
    )
    articulation_probability = (
        torch.sigmoid(outputs["articulation"].float())
        if "articulation" in outputs
        else torch.zeros_like(onset_probability)
    )
    notes: list[PredictedNote] = []
    step = 0
    total_steps = len(onset_probability)
    if rhythm_modes is None:
        rhythm_modes = (
            decode_rhythm_modes(outputs["rhythm"], total_steps=total_steps)
            if "rhythm" in outputs
            else torch.zeros(
                math.ceil(total_steps / STEPS_PER_BAR),
                dtype=torch.long,
            )
        )
    allowed_boundary = rhythm_boundary_mask(rhythm_modes, total_steps)
    onset_probability = onset_probability.clone()
    onset_probability[~allowed_boundary] = 0.0
    active = activity_probability >= activity_threshold
    if observation_mask is not None:
        active &= observation_mask.cpu().bool()

    while step < total_steps:
        if not bool(allowed_boundary[step]) or not bool(active[step]):
            step += 1
            continue
        onset_value = float(onset_probability[step])
        pitch_turn_value = float(pitch_change_probability[step])
        articulation_value = float(articulation_probability[step])
        # Pitch-change is supervised from score targets. A physical pitch
        # fluctuation cannot override it or veto a learned semitone turn.
        articulation_start = (
            articulation_value >= articulation_threshold
            and onset_value >= articulation_onset_floor
        )
        if (
            onset_value < onset_threshold
            and not (
                pitch_turn_value >= pitch_change_threshold
            )
            and not articulation_start
        ):
            step += 1
            continue

        end_step = step + 1
        first_low_boundary: int | None = None
        while end_step < total_steps:
            if not bool(allowed_boundary[end_step]):
                end_step += 1
                continue
            next_onset = float(onset_probability[end_step])
            pitch_turn = (
                float(pitch_change_probability[end_step])
                >= pitch_change_threshold
                and bool(active[end_step])
            )
            articulation_turn = (
                float(articulation_probability[end_step])
                >= articulation_threshold
                and next_onset >= articulation_onset_floor
                and bool(active[end_step])
            )
            if (
                next_onset >= strong_onset_threshold
                or pitch_turn
                or articulation_turn
            ):
                break

            continues = (
                float(continuation_probability[end_step - 1])
                >= continuation_threshold
            )
            if continues:
                first_low_boundary = None
                end_step += 1
                continue

            if first_low_boundary is None:
                first_low_boundary = end_step
            explicit_end = (
                float(offset_probability[end_step]) >= offset_threshold
                or float(activity_probability[end_step]) < 0.35
            )
            if explicit_end or end_step > first_low_boundary:
                end_step = first_low_boundary
                break
            end_step += 1

        if first_low_boundary is not None and end_step >= total_steps:
            end_step = first_low_boundary
        if end_step <= step:
            end_step = step + 1
        # Ignore consonant/attack-heavy first cell when a stable interior is
        # available. Every frame in the chosen interval was trained with the
        # same MIDI segment pitch.
        pitch_start = step + 1 if end_step - step >= 3 else step
        segment_pitch_logits = outputs["pitch"][
            pitch_start:end_step
        ].float().mean(dim=0)
        pitch_value = int(segment_pitch_logits.argmax())
        if pitch_value == 0:
            pitch_value = int(segment_pitch_logits[1:].argmax()) + 1
        notes.append(
            PredictedNote(
                start_step=step,
                end_step=end_step,
                pitch=pitch_value - 1,
                onset_probability=onset_value,
                duration_steps=end_step - step,
            )
        )
        step = end_step
    return notes


def save_segment_prediction_midi(
    output_path: Path,
    notes: list[PredictedNote],
    bpm: float,
    grid_end_tick: int,
    rhythm_modes: torch.Tensor | None = None,
    tempo_map_events: list[tuple[int, int]] | None = None,
    audio_start_tick: int | None = None,
) -> None:
    events: list[tuple[int, int, mido.Message]] = []
    for note in notes:
        start_tick = note.start_step * TICKS_PER_STEP
        end_tick = min(grid_end_tick, note.end_step * TICKS_PER_STEP)
        if end_tick <= start_tick:
            continue
        events.append(
            (
                start_tick,
                1,
                mido.Message("note_on", note=note.pitch, velocity=100),
            )
        )
        events.append(
            (
                end_tick,
                0,
                mido.Message("note_off", note=note.pitch, velocity=0),
            )
        )

    midi = mido.MidiFile(ticks_per_beat=PROJECT_PPQ)
    meta = mido.MidiTrack()
    meta_events: list[tuple[int, mido.MetaMessage]] = [
        (
            0,
            mido.MetaMessage(
                "time_signature",
                numerator=4,
                denominator=4,
            ),
        )
    ]
    if audio_start_tick is not None:
        if not 0 <= audio_start_tick <= grid_end_tick:
            raise ValueError(
                f"Audio start tick {audio_start_tick} is outside "
                f"the MIDI grid 0-{grid_end_tick}."
            )
        meta_events.append(
            (
                audio_start_tick,
                mido.MetaMessage(
                    "marker",
                    text=(
                        "WAV START (align audio sample 0 here): "
                        f"tick {audio_start_tick}"
                    ),
                ),
            )
        )
    if tempo_map_events:
        meta_events.extend(
            (
                int(tick),
                mido.MetaMessage("set_tempo", tempo=int(tempo)),
            )
            for tick, tempo in tempo_map_events
        )
    else:
        meta_events.append(
            (0, mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(bpm)))
        )
    if rhythm_modes is not None and len(rhythm_modes):
        previous_mode: int | None = None
        for bar_index, mode_value in enumerate(rhythm_modes.tolist()):
            mode = int(mode_value)
            if mode == previous_mode:
                continue
            meta_events.append(
                (
                    bar_index * PROJECT_PPQ * 4,
                    mido.MetaMessage(
                        "marker",
                        text=f"Rhythm Grid: {RHYTHM_NAMES[mode]}",
                    ),
                )
            )
            previous_mode = mode
    previous_tick = 0
    for tick, message in sorted(meta_events, key=lambda item: item[0]):
        tick = min(tick, grid_end_tick)
        meta.append(message.copy(time=tick - previous_tick))
        previous_tick = tick
    meta.append(
        mido.MetaMessage(
            "end_of_track",
            time=max(0, grid_end_tick - previous_tick),
        )
    )
    midi.tracks.append(meta)

    melody = mido.MidiTrack()
    previous_tick = 0
    for tick, _, message in sorted(events, key=lambda item: (item[0], item[1])):
        melody.append(message.copy(time=tick - previous_tick))
        previous_tick = tick
    melody.append(
        mido.MetaMessage(
            "end_of_track",
            time=max(0, grid_end_tick - previous_tick),
        )
    )
    midi.tracks.append(melody)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    midi.save(output_path)


def stabilize_pitch_sequence(
    pitch_logits: torch.Tensor,
    onset_probabilities: torch.Tensor,
    features: torch.Tensor,
    feature_config: FeatureConfig,
    *,
    shape_similarity_threshold: float = 0.96,
    onset_threshold: float = 0.65,
    sustained_steps: int = 2,
    sustained_margin: float = 0.12,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply syllable-aware hysteresis while preserving a 1/16 minimum note."""
    probabilities = torch.softmax(pitch_logits.float(), dim=-1)
    raw_pitch = probabilities.argmax(dim=-1)
    decoded = raw_pitch.clone()

    zero_context = feature_config.context_offsets.index(0)
    start = zero_context * feature_config.n_mels
    spectral_shape = features[:, start : start + feature_config.n_mels].float()
    spectral_shape = spectral_shape - spectral_shape.mean(dim=-1, keepdim=True)
    spectral_norm = torch.linalg.vector_norm(spectral_shape, dim=-1)
    spectral_shape = F.normalize(spectral_shape, dim=-1, eps=1e-6)
    shape_similarity = torch.ones(len(features), dtype=torch.float32)
    if len(features) > 1:
        shape_similarity[1:] = torch.sum(
            spectral_shape[1:] * spectral_shape[:-1],
            dim=-1,
        ).cpu()
        both_flat = (spectral_norm[1:] < 1e-6) & (spectral_norm[:-1] < 1e-6)
        shape_similarity[1:][both_flat.cpu()] = 1.0

    for step in range(1, len(decoded)):
        current = int(decoded[step - 1])
        candidate = int(raw_pitch[step])
        if candidate == current:
            decoded[step] = current
            continue

        # Voice starts and ends remain controlled by the learned silence class.
        if current == 0 or candidate == 0:
            continue

        run_end = step
        while run_end < len(raw_pitch) and int(raw_pitch[run_end]) == candidate:
            run_end += 1
        run_length = run_end - step
        evidence_end = min(run_end, step + max(1, sustained_steps))
        candidate_probability = probabilities[
            step:evidence_end, candidate
        ].mean()
        current_probability = probabilities[step:evidence_end, current].mean()
        margin = float(candidate_probability - current_probability)
        strong_onset = float(onset_probabilities[step]) >= onset_threshold
        same_spectral_shape = (
            float(shape_similarity[step]) >= shape_similarity_threshold
        )
        required_sustained_margin = sustained_margin * (
            1.5 if same_spectral_shape else 1.0
        )
        clear_sustained_turn = (
            run_length >= sustained_steps
            and margin >= required_sustained_margin
        )

        if strong_onset or clear_sustained_turn:
            continue
        # A single 1/16 pitch deviation is never promoted to a note merely
        # because its pitch posterior is confident. True one-cell notes must
        # also carry learned onset evidence; this blocks vibrato/scoop noise.
        decoded[step] = current

    # Remove a one-cell A-B-A spike unless the middle cell has a strong onset.
    for step in range(1, len(decoded) - 1):
        is_single_cell_spike = (
            decoded[step - 1] == decoded[step + 1]
            and decoded[step] != decoded[step - 1]
        )
        if not is_single_cell_spike:
            continue
        if float(onset_probabilities[step]) < onset_threshold:
            decoded[step] = decoded[step - 1]

    return decoded, shape_similarity


def pool_grid_predictions(
    pitch_logits: torch.Tensor,
    onset_probabilities: torch.Tensor,
    features: torch.Tensor,
    factor: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pool fine Grid predictions before coarse quantized decoding."""
    if factor < 1:
        raise ValueError("Pooling factor must be at least one.")
    if factor == 1:
        return pitch_logits, onset_probabilities, features
    if len(pitch_logits) % factor:
        raise ValueError(
            f"Sequence length {len(pitch_logits)} is not divisible by {factor}."
        )
    groups = len(pitch_logits) // factor
    pooled_logits = pitch_logits.reshape(groups, factor, -1).mean(dim=1)
    pooled_onset = onset_probabilities.reshape(groups, factor).amax(dim=1)
    pooled_features = features.reshape(groups, factor, -1).mean(dim=1)
    return pooled_logits, pooled_onset, pooled_features


def quantize_pitch_classes(
    pitch_classes: torch.Tensor,
    pitch_logits: torch.Tensor,
    onset_probabilities: torch.Tensor,
    factor: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Coarsen a decoded sequence without inventing a new pitch class."""
    if factor < 1:
        raise ValueError("Quantization factor must be at least one.")
    if factor == 1:
        return pitch_classes, onset_probabilities
    if len(pitch_classes) % factor:
        raise ValueError(
            f"Sequence length {len(pitch_classes)} is not divisible by {factor}."
        )

    probabilities = torch.softmax(pitch_logits.float(), dim=-1)
    groups = len(pitch_classes) // factor
    quantized = torch.zeros(groups, dtype=pitch_classes.dtype)
    quantized_onset = onset_probabilities.reshape(groups, factor).amax(dim=1)
    previous = 0

    for group_index in range(groups):
        start = group_index * factor
        end = start + factor
        group_classes = pitch_classes[start:end]
        candidates = torch.unique(group_classes)
        best_class = int(candidates[0])
        best_occupancy = -1
        best_confidence = -1.0
        for candidate_tensor in candidates:
            candidate = int(candidate_tensor)
            occupancy = int(torch.sum(group_classes == candidate))
            confidence = float(probabilities[start:end, candidate].mean())
            continuity_bonus = 1e-4 if candidate == previous else 0.0
            if (
                occupancy > best_occupancy
                or (
                    occupancy == best_occupancy
                    and confidence + continuity_bonus > best_confidence
                )
            ):
                best_class = candidate
                best_occupancy = occupancy
                best_confidence = confidence + continuity_bonus
        quantized[group_index] = best_class
        previous = best_class
    return quantized, quantized_onset


def save_prediction_midi(
    output_path: Path,
    pitch_classes: np.ndarray,
    onset_probabilities: np.ndarray,
    bpm: float,
    grid_end_tick: int,
    onset_threshold: float = 0.65,
    ticks_per_step: int = TICKS_PER_STEP,
) -> None:
    events: list[tuple[int, int, mido.Message]] = []
    active_pitch: int | None = None
    active_start = 0

    def close_note(end_step: int) -> None:
        nonlocal active_pitch
        if active_pitch is not None and end_step > active_start:
            tick = end_step * ticks_per_step
            events.append(
                (tick, 0, mido.Message("note_off", note=active_pitch, velocity=0))
            )
        active_pitch = None

    for step, pitch_class in enumerate(pitch_classes):
        midi_pitch = int(pitch_class) - 1 if pitch_class > 0 else None
        new_onset = onset_probabilities[step] >= onset_threshold
        if midi_pitch != active_pitch or (new_onset and active_pitch is not None):
            close_note(step)
            if midi_pitch is not None:
                active_pitch = midi_pitch
                active_start = step
                events.append(
                    (
                        step * ticks_per_step,
                        1,
                        mido.Message("note_on", note=midi_pitch, velocity=100),
                    )
                )
    close_note(len(pitch_classes))

    midi = mido.MidiFile(ticks_per_beat=PROJECT_PPQ)
    meta = mido.MidiTrack()
    meta.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(bpm), time=0))
    meta.append(
        mido.MetaMessage(
            "time_signature",
            numerator=4,
            denominator=4,
            time=0,
        )
    )
    meta.append(mido.MetaMessage("end_of_track", time=grid_end_tick))
    midi.tracks.append(meta)

    melody = mido.MidiTrack()
    previous_tick = 0
    for tick, _, message in sorted(events, key=lambda item: (item[0], item[1])):
        melody.append(message.copy(time=tick - previous_tick))
        previous_tick = tick
    melody.append(
        mido.MetaMessage(
            "end_of_track",
            time=max(0, grid_end_tick - previous_tick),
        )
    )
    midi.tracks.append(melody)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    midi.save(output_path)
