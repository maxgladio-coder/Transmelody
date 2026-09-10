from __future__ import annotations

import argparse
import math
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


TARGET_SAMPLE_RATE = 44100
TARGET_SUBTYPE = "PCM_16"
SAFE_PEAK = 32767.0 / 32768.0
BLOCK_FRAMES = 262144


@dataclass(frozen=True)
class StemPair:
    song_id: str
    instrumental: Path
    vocal: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replace paired project stems with 44.1 kHz PCM16 WAV."
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path("."),
        help="Project root containing dataset and test audio.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Perform replacement; without this flag only report planned work.",
    )
    return parser.parse_args()


def _index(folder: Path, suffixes: tuple[str, ...]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for path in folder.glob("*.wav"):
        for suffix in suffixes:
            if path.name.endswith(suffix):
                song_id = path.name[: -len(suffix)]
                if song_id in result:
                    raise ValueError(f"Duplicate stem id {song_id!r} in {folder}.")
                result[song_id] = path
                break
    return result


def _paired(
    inst_dir: Path,
    vocal_dir: Path,
    inst_suffixes: tuple[str, ...],
    vocal_suffixes: tuple[str, ...],
) -> list[StemPair]:
    inst = _index(inst_dir, inst_suffixes)
    vocal = _index(vocal_dir, vocal_suffixes)
    if set(inst) != set(vocal):
        raise ValueError(
            f"Pair mismatch in {inst_dir.parent}: "
            f"inst-only={sorted(set(inst) - set(vocal))}, "
            f"vocal-only={sorted(set(vocal) - set(inst))}"
        )
    return [
        StemPair(song_id, inst[song_id], vocal[song_id])
        for song_id in sorted(inst)
    ]


def discover_pairs(workspace: Path) -> list[StemPair]:
    dataset = workspace / "dataset" / "melody_dataset"
    test = workspace / "test audio"
    return [
        *_paired(
            dataset / "inst_audio",
            dataset / "vocal_audio",
            ("_inst.wav",),
            ("_vocals.wav", "_vocal.wav"),
        ),
        *_paired(
            test / "test_inst",
            test / "test_vocal",
            ("_Instrumental.wav",),
            ("_Vocals.wav",),
        ),
    ]


def _ffmpeg_executable() -> str:
    executable = shutil.which("ffmpeg")
    if executable is None:
        raise RuntimeError(
            "Audio decoding failed and FFmpeg is not available. "
            "Install FFmpeg or add it to PATH."
        )
    return executable


def _ffmpeg_peak_amplitude(path: Path) -> tuple[float, str]:
    command = [
        _ffmpeg_executable(),
        "-hide_banner",
        "-loglevel",
        "error",
        "-err_detect",
        "ignore_err",
        "-i",
        str(path),
        "-map",
        "0:a:0",
        "-vn",
        "-f",
        "f32le",
        "-acodec",
        "pcm_f32le",
        "-",
    ]
    # Drain stderr concurrently. Corrupt files may otherwise fill the pipe
    # while this thread is waiting for more decoded stdout.
    import tempfile
    stderr_file = tempfile.TemporaryFile()
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=stderr_file,
    )
    if process.stdout is None:
        raise RuntimeError("Could not open FFmpeg decoder pipes.")

    peak = 0.0
    sample_count = 0
    carry = b""
    while True:
        chunk = process.stdout.read(BLOCK_FRAMES * 4)
        if not chunk:
            break
        chunk = carry + chunk
        usable = len(chunk) - (len(chunk) % 4)
        carry = chunk[usable:]
        if usable:
            samples = np.frombuffer(chunk[:usable], dtype="<f4")
            sample_count += len(samples)
            peak = max(peak, float(np.max(np.abs(samples))))

    return_code = process.wait()
    process.stdout.close()
    stderr_file.seek(0)
    stderr = stderr_file.read().decode("utf-8", errors="replace").strip()
    stderr_file.close()
    if return_code != 0 or sample_count == 0:
        details = f": {stderr}" if stderr else ""
        raise RuntimeError(f"FFmpeg could not decode {path}{details}")
    return peak, stderr


def peak_amplitude(
    path: Path,
    *,
    warning: Callable[[str], None] | None = None,
) -> float:
    peak = 0.0
    try:
        with sf.SoundFile(path) as source:
            while True:
                block = source.read(
                    BLOCK_FRAMES,
                    dtype="float32",
                    always_2d=True,
                )
                if len(block) == 0:
                    break
                peak = max(peak, float(np.max(np.abs(block))))
        return peak
    except sf.LibsndfileError as exc:
        peak, ffmpeg_message = _ffmpeg_peak_amplitude(path)
        if warning is not None:
            warning(
                f"{path.name}: libsndfile decode failed ({exc}); "
                "recovered with FFmpeg."
            )
            if ffmpeg_message:
                warning(
                    f"{path.name}: FFmpeg skipped damaged audio data."
                )
        return peak


def expected_frames(source_frames: int, source_rate: int) -> int:
    return int(round(source_frames * TARGET_SAMPLE_RATE / source_rate))


def _convert_with_soundfile(
    path: Path,
    temporary: Path,
    source_info: sf._SoundFileInfo,
    target_frames: int,
    gain: float,
) -> None:
    if source_info.samplerate == TARGET_SAMPLE_RATE:
        with sf.SoundFile(path) as source, sf.SoundFile(
            temporary,
            mode="w",
            samplerate=TARGET_SAMPLE_RATE,
            channels=source_info.channels,
            subtype=TARGET_SUBTYPE,
            format="WAV",
        ) as destination:
            while True:
                block = source.read(
                    BLOCK_FRAMES,
                    dtype="float32",
                    always_2d=True,
                )
                if len(block) == 0:
                    break
                destination.write(np.clip(block * gain, -1.0, SAFE_PEAK))
        return

    audio, source_rate = sf.read(
        path,
        dtype="float32",
        always_2d=True,
    )
    divisor = math.gcd(source_rate, TARGET_SAMPLE_RATE)
    audio = resample_poly(
        audio,
        TARGET_SAMPLE_RATE // divisor,
        source_rate // divisor,
        axis=0,
    )
    if len(audio) > target_frames:
        audio = audio[:target_frames]
    elif len(audio) < target_frames:
        audio = np.pad(audio, ((0, target_frames - len(audio)), (0, 0)))
    sf.write(
        temporary,
        np.clip(audio * gain, -1.0, SAFE_PEAK),
        TARGET_SAMPLE_RATE,
        subtype=TARGET_SUBTYPE,
        format="WAV",
    )


def _convert_with_ffmpeg(
    path: Path,
    temporary: Path,
    gain: float,
    target_frames: int,
) -> str:
    command = [
        _ffmpeg_executable(),
        "-hide_banner",
        "-loglevel",
        "warning",
        "-err_detect",
        "ignore_err",
        "-y",
        "-i",
        str(path),
        "-map",
        "0:a:0",
        "-vn",
        "-af",
        (
            f"volume={gain:.12g},"
            f"aresample={TARGET_SAMPLE_RATE},"
            f"apad,atrim=end_sample={target_frames}"
        ),
        "-map_metadata",
        "-1",
        "-c:a",
        "pcm_s16le",
        str(temporary),
    ]
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0 or not temporary.exists():
        details = result.stderr.strip()
        raise RuntimeError(
            f"FFmpeg conversion failed for {path}"
            + (f": {details}" if details else "")
        )
    return result.stderr.strip()


def convert_file(
    path: Path,
    gain: float,
    *,
    target_path: Path | None = None,
    remove_source: bool = False,
    warning: Callable[[str], None] | None = None,
) -> Path:
    target_path = target_path or path
    if target_path != path and target_path.exists():
        raise FileExistsError(f"Target already exists: {target_path}")
    source_info = sf.info(path)
    target_frames = expected_frames(source_info.frames, source_info.samplerate)
    temporary = target_path.with_name(f".{target_path.stem}.pcm16_tmp.wav")
    if temporary.exists():
        temporary.unlink()

    recovered = False
    recovery_message = ""
    try:
        try:
            _convert_with_soundfile(
                path,
                temporary,
                source_info,
                target_frames,
                gain,
            )
        except sf.LibsndfileError as exc:
            recovered = True
            if temporary.exists():
                temporary.unlink()
            ffmpeg_message = _convert_with_ffmpeg(
                path,
                temporary,
                gain,
                target_frames,
            )
            recovery_message = f"libsndfile: {exc}; FFmpeg: {ffmpeg_message}"
            if warning is not None:
                warning(
                    f"{path.name}: libsndfile conversion failed ({exc}); "
                    "used FFmpeg fallback."
                )
                if ffmpeg_message:
                    warning(
                        f"{path.name}: FFmpeg skipped damaged audio data."
                    )
                warning(
                    f"{path.name}: padded any missing tail samples with "
                    "silence to preserve the declared duration."
                )

        converted = sf.info(temporary)
        if converted.samplerate != TARGET_SAMPLE_RATE:
            raise RuntimeError(f"Bad sample rate after conversion: {path}")
        if converted.subtype != TARGET_SUBTYPE:
            raise RuntimeError(f"Bad subtype after conversion: {path}")
        if converted.channels != source_info.channels:
            raise RuntimeError(f"Channel count changed: {path}")
        if converted.frames != target_frames:
            raise RuntimeError(
                f"Frame count mismatch for {path}: "
                f"{converted.frames} != {target_frames}"
            )
        if recovered:
            # Total duration cannot validate internal timing after damaged
            # frames were skipped. Keep both the original and a review copy.
            candidate = target_path.with_name(f"{target_path.stem}.recovered.wav")
            if candidate.exists():
                raise FileExistsError(f"Recovery candidate already exists: {candidate}; source preserved.")
            os.replace(temporary, candidate)
            report_path = candidate.with_suffix(".json")
            report_path.write_text(json.dumps({"source": str(path.resolve()),
                "candidate": str(candidate.resolve()), "status": "needs_alignment_review",
                "original_preserved": True, "internal_alignment_verified": False,
                "decoder_message": recovery_message}, ensure_ascii=False, indent=2), encoding="utf-8")
            raise RuntimeError(f"Recovered audio requires internal alignment review: {candidate}. Original preserved; not replaced.")
        os.replace(temporary, target_path)
        if remove_source and target_path != path:
            path.unlink()
        return target_path
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> None:
    args = parse_args()
    workspace = args.workspace.resolve()
    pairs = discover_pairs(workspace)
    if not pairs:
        raise SystemExit("No paired project stems found.")

    converted_count = 0
    skipped_count = 0
    for pair in pairs:
        inst_info = sf.info(pair.instrumental)
        vocal_info = sf.info(pair.vocal)
        if (
            inst_info.samplerate != vocal_info.samplerate
            or inst_info.frames != vocal_info.frames
        ):
            raise ValueError(f"Unaligned stem pair: {pair.song_id}")

        inst_peak = peak_amplitude(pair.instrumental)
        vocal_peak = peak_amplitude(pair.vocal)
        pair_peak = max(inst_peak, vocal_peak)
        gain = SAFE_PEAK / pair_peak if pair_peak > 1.0 else 1.0
        gain_db = 20.0 * math.log10(gain) if gain > 0 else float("-inf")
        print(
            f"[pair] {pair.song_id}: peak={pair_peak:.6f}, "
            f"shared_gain={gain_db:.2f} dB"
        )

        for path in (pair.instrumental, pair.vocal):
            info = sf.info(path)
            compliant = (
                info.samplerate == TARGET_SAMPLE_RATE
                and info.subtype == TARGET_SUBTYPE
            )
            # A pair-level attenuation must be applied to both stems.
            if compliant and gain == 1.0:
                print(f"  [skip] {path}")
                skipped_count += 1
                continue
            print(f"  [{'write' if args.apply else 'plan'}] {path}")
            if args.apply:
                convert_file(path, gain)
            converted_count += 1

        if args.apply:
            final_inst = sf.info(pair.instrumental)
            final_vocal = sf.info(pair.vocal)
            if final_inst.frames != final_vocal.frames:
                raise RuntimeError(f"Pair length changed: {pair.song_id}")

    action = "Converted" if args.apply else "Would convert"
    print(f"{action} {converted_count} files; skipped {skipped_count}.")


if __name__ == "__main__":
    main()
