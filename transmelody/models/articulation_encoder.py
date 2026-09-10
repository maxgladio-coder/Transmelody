from __future__ import annotations
from transmelody.paths import PROJECT_ROOT

import gc
import os
from pathlib import Path

import torch
import torch.nn.functional as F


os.environ.setdefault(
    "HF_HOME",
    str(PROJECT_ROOT / ".cache" / "huggingface"),
)

DEFAULT_ARTICULATION_MODEL = (
    "reazon-research/japanese-wav2vec2-base-rs35kh"
)
ARTICULATION_HIDDEN_SIZE = 768
ARTICULATION_AUX_SIZE = 3  # spectral novelty, CTC token change, non-blank confidence
ARTICULATION_FEATURE_SIZE = ARTICULATION_HIDDEN_SIZE + ARTICULATION_AUX_SIZE
_ENCODERS: dict[tuple[str, str], torch.nn.Module] = {}


def _load_encoder(model_id: str, device: torch.device) -> torch.nn.Module:
    key = (model_id, str(device))
    if key in _ENCODERS:
        return _ENCODERS[key]
    try:
        from transformers import AutoModelForCTC
    except ImportError as error:
        raise RuntimeError(
            "Japanese articulation features require transformers. Run: "
            "python -m pip install transformers==4.52.1 safetensors"
        ) from error
    revision = ({'revision': '46afc596052b612293c8db256b3a69447a2f57dc'}
                if model_id == DEFAULT_ARTICULATION_MODEL else {})
    encoder = AutoModelForCTC.from_pretrained(model_id, **revision,
        cache_dir=str(PROJECT_ROOT / ".cache/huggingface/hub")).to(device).eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    _ENCODERS[key] = encoder
    return encoder


@torch.inference_mode()
def extract_articulation_sequence(
    mono_16k: torch.Tensor,
    *,
    model_id: str,
    device: torch.device,
    chunk_seconds: float = 20.0,
    overlap_seconds: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return frozen Japanese speech embeddings and frame-center seconds."""
    encoder = _load_encoder(model_id, device)
    sample_rate = 16000
    chunk_samples = int(round(chunk_seconds * sample_rate))
    hidden_parts: list[torch.Tensor] = []
    time_parts: list[torch.Tensor] = []
    probability_parts: list[torch.Tensor] = []
    waveform = mono_16k.float()

    stride, receptive = 1, 1
    for kernel, conv_stride in zip(encoder.config.conv_kernel, encoder.config.conv_stride):
        receptive += (kernel - 1) * stride
        stride *= conv_stride
    overlap = int(overlap_seconds * sample_rate) // stride * stride
    chunk_samples = max(stride, chunk_samples // stride * stride)

    for start in range(0, len(waveform), chunk_samples):
        left = max(0, start - overlap)
        end = min(len(waveform), start + chunk_samples)
        chunk = waveform[left : min(len(waveform), end + overlap)]
        if len(chunk) < receptive:
            continue
        chunk = (chunk - chunk.mean()) / chunk.std().clamp_min(1e-5)
        base_outputs = encoder.base_model(chunk.unsqueeze(0).to(device))
        hidden = base_outputs.last_hidden_state[0]
        logits = encoder.lm_head(hidden)
        probabilities = torch.softmax(logits.float(), dim=-1)
        frame_count = len(hidden)
        frame_times = (
            left / sample_rate
            + (torch.arange(frame_count, device=device) * stride + (receptive - 1) / 2) / sample_rate
        )
        keep = (frame_times >= start / sample_rate) & (frame_times < end / sample_rate)
        hidden_parts.append(hidden[keep].cpu().to(torch.float16))
        probability_parts.append(probabilities[keep].cpu())
        time_parts.append(frame_times[keep].cpu())

    if not hidden_parts:
        raise ValueError("Audio is too short for articulation feature extraction.")
    hidden = torch.cat(hidden_parts)
    times = torch.cat(time_parts)
    probabilities = torch.cat(probability_parts)
    blank_id = int(encoder.config.pad_token_id)
    top_probability, top_token = probabilities.max(dim=-1)
    token_change = torch.zeros(len(hidden))
    token_change[1:] = ((top_token[1:] != top_token[:-1]) & (top_token[1:] != blank_id)).float() * top_probability[1:]
    novelty = torch.zeros(len(hidden))
    novelty[1:] = 1.0 - F.cosine_similarity(hidden[1:].float(), hidden[:-1].float(), dim=-1)
    auxiliary = torch.stack([novelty.clamp(0, 2), token_change, 1 - probabilities[:, blank_id]], dim=1)
    return torch.cat([hidden, auxiliary.to(torch.float16)], dim=1), times


def pronunciation_evidence(features: torch.Tensor, auxiliary_start: int) -> torch.Tensor:
    """Soft speech-change proxy, NOT a phoneme/character annotation.

    CTC labels are transcription tokens; novelty can also respond to singing
    style. Keep this weak auxiliary separate from MIDI note supervision.
    """
    novelty = features[:, auxiliary_start].float().clamp_min(0)
    token = features[:, auxiliary_start + 1].float().clamp(0, 1)
    confidence = features[:, auxiliary_start + 2].float().clamp(0, 1)
    return (1 - (1 - token) * torch.exp(-novelty)) * confidence


def sample_articulation_at_times(
    sequence: torch.Tensor,
    frame_times: torch.Tensor,
    query_times: torch.Tensor,
) -> torch.Tensor:
    right = torch.searchsorted(frame_times, query_times.float())
    left = right - 1
    valid = (query_times >= frame_times[0]) & (query_times <= frame_times[-1])
    left = left.clamp(0, len(sequence) - 1)
    right = right.clamp(0, len(sequence) - 1)
    left_time = frame_times[left]
    right_time = frame_times[right]
    fraction = (
        (query_times.float() - left_time)
        / (right_time - left_time).clamp_min(1e-6)
    ).unsqueeze(1)
    values = (
        sequence[left].float() * (1.0 - fraction)
        + sequence[right].float() * fraction
    )
    values[~valid] = 0.0
    return values.to(torch.float16)


def clear_articulation_encoder_cache() -> None:
    _ENCODERS.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
