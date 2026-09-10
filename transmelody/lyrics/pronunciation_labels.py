"""Reviewed pronunciation spans in original WAV seconds, independent of MIDI notes."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

SCHEMA = "reviewable_pronunciation_v1"


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validate_labels(data: dict, *, require_reviewed: bool = False) -> None:
    if data.get("schema_version") != SCHEMA or data.get("time_reference") != "original_wav_seconds":
        raise ValueError("Unsupported pronunciation schema/time reference.")
    if data.get("status") not in {"candidate", "reviewed"}:
        raise ValueError("Invalid review status.")
    if require_reviewed and data["status"] != "reviewed":
        raise ValueError("Automatic candidates cannot be used as reviewed training labels.")
    duration = float(data["audio_duration"])
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("Invalid audio duration.")
    if not data.get("audio_sha256") or not str(data.get("song_id", "")).isdigit():
        raise ValueError("Song ID and audio fingerprint are required.")
    coverage = data.get("reviewed_regions", [])
    previous = 0.0
    for region in coverage:
        start, end = map(float, region)
        if not (math.isfinite(start) and math.isfinite(end) and previous <= start < end <= duration):
            raise ValueError("Review regions must be ordered, non-overlapping WAV intervals.")
        previous = end
    if require_reviewed and (not coverage or not data.get("reviewer")):
        raise ValueError("Explicit reviewed regions and reviewer name are required.")
    for tier in ("units", "phones"):
        previous = 0.0
        for span in data.get(tier, []):
            start, end = float(span["start"]), float(span["end"])
            if not (math.isfinite(start) and math.isfinite(end) and previous - 1e-6 <= start < end <= duration + 1e-6):
                raise ValueError(f"Invalid/overlapping {tier} span: {span}")
            if not str(span.get("label", "")).strip():
                raise ValueError(f"Empty {tier} label.")
            previous = end
    if not data.get("units"):
        raise ValueError("At least one pronunciation unit is required.")


def load_reviewed(path: Path, audio_path: Path, song_id: str) -> dict:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    validate_labels(data, require_reviewed=True)
    if str(data["song_id"]) != str(song_id) or data["audio_sha256"] != digest(audio_path):
        raise ValueError(f"Stale/wrong-song pronunciation labels: {path}. Re-align/review against current audio.")
    return data


def reviewed_targets(data: dict, times: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    """Only reviewed coverage is supervised; phone-internal transitions are NOT unit edges.

    Continuous second labels are projected onto the existing feature grid. A
    half-local-step Gaussian avoids losing an edge between two sample centers.
    This projection does not modify the annotation or snap MIDI boundaries.
    """
    times = np.asarray(times, dtype=np.float64)
    if len(times) > 1 and np.any(np.diff(times) <= 0):
        raise ValueError("Query times must increase.")
    target = np.zeros(len(times), dtype=np.float32)
    mask = np.zeros(len(times), dtype=bool)
    if not len(times):
        return torch.from_numpy(target), torch.from_numpy(mask)
    spacing = np.gradient(times) if len(times) > 1 else np.full(1, 0.02)
    sigma = np.maximum(0.020, spacing * 0.5)
    edges = np.array(sorted({float(s[key]) for s in data["units"] for key in ("start", "end")}))
    for start, end in data["reviewed_regions"]:
        # A small margin avoids inventing negative evidence at cropped coverage edges.
        inside = (times >= start + 0.02) & (times < end - 0.02)
        mask |= inside
        local_edges = edges[(edges >= start) & (edges <= end)]
        if len(local_edges):
            distance = np.min(np.abs(times[:, None] - local_edges[None, :]), axis=1)
            target[inside] = np.exp(-0.5 * (distance[inside] / sigma[inside]) ** 2)
    return torch.from_numpy(target), torch.from_numpy(mask)
