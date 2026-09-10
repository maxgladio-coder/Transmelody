"""Lyrics -> Japanese singing phone/unit candidates -> explicit reviewed labels.

No lyrics/audio/MIDI are changed. SOFA is an offline annotation teacher, not a
runtime dependency of melody prediction. See README for the review contract.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np
import soundfile as sf
import torch
import torchaudio

from pronunciation_labels import SCHEMA, digest, validate_labels

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
SOFA_SOURCE = ROOT / ".cache/sofa/source/SOFA-1.0.3"
SOFA_MODEL = ROOT / ".cache/sofa/model"


def write_new(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Never silently replace the user's reviewed work.
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def boundaries_only(data: dict, source_url: str) -> dict:
    """Export timing analysis of a retrieved source without reproducing lyrics.

    In-memory teacher inference may use text, but this artifact contains neither
    full lyrics nor their phonetic transcription. Local user-supplied lyric
    files can use the normal, text-preserving workflow instead.
    """
    import copy
    result = copy.deepcopy(data)
    result.pop("lyrics", None)
    result["lyrics_source_url"] = source_url
    result["text_export"] = "boundary_ids_only"
    for tier in ("units", "phones"):
        for i, span in enumerate(result[tier]):
            span["label"] = f"{tier[:-1]}_{i + 1:04d}"
    result["phrases"] = [{k: v for k, v in phrase.items() if k not in {"text", "reading", "units"}}
        | {"text": f"Phrase {i + 1:02d}"} for i, phrase in enumerate(result["phrases"])]
    return result


def candidate_diagnostics(data: dict) -> dict:
    """Triage only: thresholds here never change boundaries or train labels."""
    units = data["units"]
    durations = np.array([u["end"] - u["start"] for u in units])
    suspect = [{"unit": i + 1, "phrase": u.get("phrase", -1) + 1,
        "start": u["start"], "end": u["end"], "duration_sec": float(durations[i])}
        for i, u in enumerate(units) if durations[i] < .020]
    return {"song_id": data["song_id"], "status": data["status"],
        "purpose": "Review triage only; not measured boundary accuracy.",
        "phrase_count": len(data["phrases"]), "unit_count": len(units), "phone_count": len(data["phones"]),
        "median_unit_duration_sec": float(np.median(durations)),
        "under_20ms_count": len(suspect), "under_20ms_units": suspect,
        "coarse_unknown_phrase_numbers": [i + 1 for i, p in enumerate(data["phrases"]) if p.get("coarse_unknown_tokens", 0)],
        "lowest_sofa_score_phrase_numbers": [i + 1 for i in sorted(range(len(data["phrases"])),
            key=lambda i: data["phrases"][i].get("sofa_confidence", 0))[:8]],
        "first_unit_start_sec": units[0]["start"], "last_unit_end_sec": units[-1]["end"]}


def group_phones(phones: list[str]) -> list[list[str]]:
    """CV/mora candidates, NOT linguistic truth or MIDI notes (melisma allowed)."""
    units, pending = [], []
    for raw in phones:
        phone = raw.lower() if raw in {"A", "I", "U", "E", "O"} else raw
        if phone in {"pau", "sil"}:
            if pending:
                raise ValueError(f"Incomplete pronunciation before silence: {pending}")
            continue
        pending.append(phone)
        if phone in {"a", "i", "u", "e", "o", "N", "cl"}:
            units.append(pending)
            pending = []
    if pending or not units:
        raise ValueError(f"Cannot group pronunciation: {phones}")
    return units


def phrase_units(phrase: dict) -> list[list[str]]:
    # Explicit groups support nonstandard singing readings without guessing.
    if "units" in phrase:
        units = phrase["units"]
        if not isinstance(units, list) or not units or any(not isinstance(u, list) or not u for u in units):
            raise ValueError("units must be nonempty lists of phonemes, e.g. [[\"k\",\"a\"],[\"n\",\"a\"]].")
        return units
    import pyopenjtalk
    return group_phones(pyopenjtalk.g2p(phrase.get("reading") or phrase["text"]).split())


def validate_phrases(phrases: list[dict], duration: float) -> bool:
    if not phrases or any(not p.get("text", "").strip() for p in phrases):
        raise ValueError("Fill lyrics in sung order, including repeated lines.")
    timed = ["start" in p or "end" in p for p in phrases]
    if any(timed) and not all(timed):
        raise ValueError("Either time every phrase or leave all phrases untimed.")
    previous = 0.0
    if all(timed):
        for p in phrases:
            start, end = float(p["start"]), float(p["end"])
            if not (np.isfinite(start) and np.isfinite(end) and previous <= start < end <= duration):
                raise ValueError("Phrase clips must be ordered, non-overlapping, within the original WAV.")
            if end - start > 60:
                raise ValueError("Split timed clips longer than 60 seconds into lyric phrases.")
            previous = end
    return all(timed)


@torch.inference_mode()
def coarse_phrase_times(audio: np.ndarray, sr: int, phrases: list[dict], device: torch.device) -> list[dict]:
    """Known lyric TEXT CTC alignment supplies rough clips, not phoneme labels."""
    from articulation_encoder import DEFAULT_ARTICULATION_MODEL, _load_encoder, clear_articulation_encoder_cache
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_ARTICULATION_MODEL,
        cache_dir=str(ROOT / ".cache/huggingface/hub"), local_files_only=True)
    ids = [tokenizer.encode(p["text"], add_special_tokens=False) for p in phrases]
    unknown_counts = [row.count(tokenizer.unk_token_id) for row in ids]
    if any(not row or n / len(row) > .2 for row, n in zip(ids, unknown_counts)):
        raise ValueError("Too many unsupported ASR tokens for reliable coarse clips. Provide timed phrases.")
    if any(unknown_counts):
        print("[lyrics] Unknown TEXT tokens use coarse wildcard slots in phrases: "
            + str([i + 1 for i, n in enumerate(unknown_counts) if n])
            + "; these clips require timing review, no phonemes are dropped.", flush=True)
    waveform = torchaudio.functional.resample(torch.from_numpy(audio), sr, 16000)
    encoder = _load_encoder(DEFAULT_ARTICULATION_MODEL, device)
    blank = int(encoder.config.pad_token_id)
    # This tokenizer can share UNK and CTC blank ID. Give unknown text a
    # separate virtual class; never put the actual blank inside a CTC target.
    wildcard_id = max([blank] + sum(ids, [])) + 1
    ids = [[wildcard_id if token == tokenizer.unk_token_id else token for token in row] for row in ids]
    if any(blank in row for row in ids):
        raise ValueError("Lyrics encode a blank token; supply timed phrases.")
    # Keep only target vocabulary; renormalization would not change the best path.
    vocabulary = [blank] + sorted(set(sum(ids, [])) - {blank})
    mapping = {token: i for i, token in enumerate(vocabulary)}
    stride, receptive = 1, 1
    for kernel, step in zip(encoder.config.conv_kernel, encoder.config.conv_stride):
        receptive += (kernel - 1) * stride
        stride *= step
    emissions, centers = [], []
    core, overlap = 320000 // stride * stride, 16000 // stride * stride
    for start in range(0, len(waveform), core):
        left, end = max(0, start - overlap), min(len(waveform), start + core)
        chunk = waveform[left:min(len(waveform), end + overlap)]
        if len(chunk) < receptive:
            continue
        chunk = (chunk - chunk.mean()) / chunk.std().clamp_min(1e-5)
        logits = encoder(chunk[None].to(device)).logits[0]
        times = (left + torch.arange(len(logits), device=device) * stride + (receptive - 1) / 2) / 16000
        keep = (times >= start / 16000) & (times < end / 16000)
        full_log_probs = logits.float().log_softmax(-1)[keep]
        selected_log_probs = full_log_probs[:, [blank if token == wildcard_id else token for token in vocabulary]].clone()
        if any(unknown_counts):
            # A wildcard TEXT slot, not a predicted phone. Known surrounding
            # tokens anchor the rough clip; SOFA still receives all G2P phones.
            nonblank = full_log_probs.clone()
            nonblank[:, blank] = -torch.inf
            selected_log_probs[:, mapping[wildcard_id]] = nonblank.max(dim=-1).values
        emissions.append(selected_log_probs.cpu())
        centers.append(times[keep].cpu())
    if not emissions:
        raise ValueError("Audio too short.")
    log_probs, times = torch.cat(emissions), torch.cat(centers)
    target = torch.tensor([[mapping[t] for row in ids for t in row]], dtype=torch.int32)
    path, scores = torchaudio.functional.forced_align(log_probs[None], target, blank=0)
    spans = torchaudio.functional.merge_tokens(path[0], scores[0], blank=0)
    if len(spans) != target.shape[1] or [s.token for s in spans] != target[0].tolist():
        raise ValueError("CTC alignment could not account for every lyric token; use timed phrases.")
    result, index = [], 0
    for phrase, row, unknown in zip(phrases, ids, unknown_counts):
        chosen = spans[index:index + len(row)]
        index += len(row)
        result.append({**phrase, "start": max(0., float(times[chosen[0].start]) - 0.35),
            "end": min(len(audio) / sr, float(times[chosen[-1].end - 1]) + 0.35),
            "coarse_ctc_score": float(np.mean([float(s.score) for s in chosen])),
            "coarse_unknown_tokens": unknown})
    for a, b in zip(result, result[1:]):
        if a["end"] > b["start"]:
            a["end"] = b["start"] = (a["end"] + b["start"]) / 2
    del encoder
    clear_articulation_encoder_cache()
    validate_phrases(result, len(audio) / sr)
    return result


def load_sofa(device: torch.device):
    models = list(SOFA_MODEL.rglob("model.ckpt"))
    if not SOFA_SOURCE.exists() or len(models) != 1:
        raise RuntimeError("SOFA v1.0.3 source / akm_ja_v001 model missing under .cache/sofa.")
    sys.path.insert(0, str(SOFA_SOURCE))
    from modules.task.forced_alignment import LitForcedAlignmentTask
    # Only safe tensor/basic-type deserialization; don't enable arbitrary pickle.
    checkpoint = torch.load(models[0], map_location="cpu", weights_only=True)
    model = LitForcedAlignmentTask(**checkpoint["hyper_parameters"])
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device).eval()
    model.set_inference_mode("force")
    # Direct inference does not run Lightning Trainer lifecycle hooks.
    from modules.utils.get_melspec import MelSpecExtractor
    model.get_melspec = MelSpecExtractor(**model.melspec_config, device=str(device))
    return model, models[0]


@torch.inference_mode()
def align(audio_path: Path, request: dict, device: torch.device) -> dict:
    audio, sr = sf.read(audio_path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if not len(audio) or not np.isfinite(audio).all():
        raise ValueError("Audio is empty or contains non-finite samples.")
    duration = len(audio) / sr
    phrases = request["phrases"]
    timed = validate_phrases(phrases, duration)
    units = [phrase_units(p) for p in phrases]
    if not timed:
        print("[lyrics] Coarse text alignment; these are NOT reviewed phoneme boundaries.", flush=True)
        phrases = coarse_phrase_times(audio, sr, phrases, device)
    model, model_path = load_sofa(device)
    data = {"schema_version": SCHEMA, "status": "candidate", "song_id": str(request["song_id"]),
        "time_reference": "original_wav_seconds", "audio_path": str(audio_path.resolve()),
        "audio_sha256": digest(audio_path), "audio_duration": duration,
        "backend": {"name": "SOFA", "version": "1.0.3", "model": "akm_ja_v001", "model_sha256": digest(model_path)},
        "lyrics": request, "reviewed_regions": [], "units": [], "phones": [], "phrases": [],
        "warnings": ["Automatic candidate only. Check reading, repeated lines, breaths, long vowels and backing vocals.",
            "SOFA confidence is phrase-level, not calibrated phone-boundary probability."]}
    with tempfile.TemporaryDirectory(prefix="sofa-", dir=ROOT / ".cache/sofa") as temp:
        for number, (phrase, groups) in enumerate(zip(phrases, units)):
            print(f"[SOFA] phrase {number + 1}/{len(phrases)}", flush=True)
            left, right = round(phrase["start"] * sr), round(phrase["end"] * sr)
            if right - left < sr // 10 or float(np.std(audio[left:right])) < 1e-7:
                raise ValueError(f"Phrase {number + 1} is too short or silent. Check coarse phrase timing.")
            crop = Path(temp) / "phrase.wav"
            sf.write(crop, audio[left:right], sr, subtype="FLOAT")
            phones, words, indices = ["SP"], [], [-1]
            for i, group in enumerate(groups):
                if any(p not in model.vocab or model.vocab[p] == 0 for p in group):
                    raise ValueError(f"Unsupported/blank phonemes {group}; correct the phrase reading or units.")
                words.append(f"u{i:04d}")
                phones.extend(group + ["SP"])
                indices.extend([i] * len(group) + [-1])
            prediction = model.predict_step((crop, phones, words, indices), 0)
            _, _, confidence, phone_names, phone_intervals, word_names, word_intervals = prediction
            if not np.isfinite(float(confidence)):
                raise ValueError("SOFA returned a non-finite confidence; candidate rejected.")
            offset, limit = left / sr, right / sr
            for names, intervals, tier in ((phone_names, phone_intervals, "phones"), (word_names, word_intervals, "units")):
                for name, (start, end) in zip(names, intervals):
                    if name in {"SP", "AP"}:
                        continue
                    start, end = max(offset, offset + float(start)), min(limit, offset + float(end))
                    if end <= start:
                        raise ValueError(f"Collapsed {tier} alignment in phrase {number + 1}; supply better clip/reading.")
                    label = " ".join(groups[int(name[1:])]) if tier == "units" else str(name)
                    data[tier].append({"start": start, "end": end, "label": label, "phrase": number})
            if len([u for u in data["units"] if u["phrase"] == number]) != len(groups):
                raise ValueError("SOFA dropped pronunciation units; candidate rejected.")
            data["phrases"].append({**phrase, "start": offset, "end": limit, "sofa_confidence": float(confidence)})
    validate_labels(data)
    data["diagnostics"] = candidate_diagnostics(data)
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init")
    init.add_argument("song_id", type=int)
    init.add_argument("output", type=Path)
    run = sub.add_parser("align")
    run.add_argument("audio", type=Path)
    run.add_argument("lyrics", type=Path)
    run.add_argument("output", type=Path)
    run.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    review = sub.add_parser("accept", help="After manual correction, explicitly accept ONLY reviewed_regions.")
    review.add_argument("candidate", type=Path)
    review.add_argument("audio", type=Path)
    review.add_argument("--reviewer", required=True)
    review.add_argument("--dataset", type=Path, default=ROOT / "dataset/melody_dataset")
    args = parser.parse_args()
    if args.command == "init":
        write_new(args.output, {"song_id": str(args.song_id), "phrases": [{"text": "", "reading": ""}]})
    elif args.command == "align":
        if args.output.exists():
            raise FileExistsError(args.output)
        request = json.loads(args.lyrics.read_text(encoding="utf-8-sig"))
        if not str(request.get("song_id", "")).isdigit():
            raise ValueError("Numeric song_id required.")
        data = align(args.audio, request, torch.device(args.device))
        data["lyrics_sha256"] = digest(args.lyrics)
        write_new(args.output, data)
    else:
        data = json.loads(args.candidate.read_text(encoding="utf-8-sig"))
        if data["audio_sha256"] != digest(args.audio):
            raise ValueError("Audio changed since alignment; refusing stale labels.")
        data.update(status="reviewed", reviewer=args.reviewer.strip(),
            review_scope="pronunciation_units",
            reviewed_at=datetime.now(timezone.utc).isoformat(), candidate_sha256=digest(args.candidate))
        validate_labels(data, require_reviewed=True)
        output = args.dataset / "pronunciation_labels" / f"{data['song_id']}.json"
        write_new(output, data)
        print(f"Reviewed labels saved: {output}")


if __name__ == "__main__":
    main()
