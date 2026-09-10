from __future__ import annotations

import argparse
import math
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import soundfile as sf

from normalize_audio_format import (
    SAFE_PEAK,
    TARGET_SAMPLE_RATE,
    TARGET_SUBTYPE,
    convert_file,
    peak_amplitude,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


STEM_SUFFIX = re.compile(
    r"(?i)(?:_instrumental|_vocals|_vocal|_inst)$"
)


@dataclass(frozen=True)
class ConversionResult:
    discovered: int
    converted: int
    skipped: int


def canonical_group(path: Path) -> str:
    """Keep matching instrumental/vocal stems on the same gain."""
    return STEM_SUFFIX.sub("", path.stem).casefold()


def discover_audio_files(folder: Path, recursive: bool = True) -> list[Path]:
    iterator = folder.rglob("*") if recursive else folder.glob("*")
    return sorted(
        (
            path
            for path in iterator
            if path.is_file()
            and path.suffix.casefold() in {".wav", ".flac"}
            and not path.name.endswith(".recovered.wav")
            and not (
                path.name.startswith(".")
                and path.name.endswith(".pcm16_tmp.wav")
            )
        ),
        key=lambda path: str(path).casefold(),
    )


def convert_folder(
    folder: Path,
    *,
    recursive: bool = True,
    apply: bool = False,
    log: Callable[[str], None] = print,
) -> ConversionResult:
    folder = folder.resolve()
    if not folder.is_dir():
        raise ValueError(f"Not a directory: {folder}")
    files = discover_audio_files(folder, recursive=recursive)
    if not files:
        raise ValueError(f"No WAV or FLAC files found under {folder}")

    for path in files:
        if path.suffix.casefold() == ".flac":
            target = path.with_suffix(".wav")
            if target.exists():
                raise FileExistsError(
                    f"Cannot convert {path}: target already exists: {target}"
                )

    groups: dict[str, list[Path]] = defaultdict(list)
    for path in files:
        groups[canonical_group(path)].append(path)

    converted = 0
    skipped = 0
    for group_name, group_paths in sorted(groups.items()):
        infos = [sf.info(path) for path in group_paths]
        if len(group_paths) > 1:
            first = infos[0]
            for path, info in zip(group_paths[1:], infos[1:]):
                if (
                    info.samplerate != first.samplerate
                    or info.frames != first.frames
                ):
                    raise ValueError(
                        f"Paired stems differ in rate/length: "
                        f"{group_paths[0]} vs {path}"
                    )

        peaks = [
            peak_amplitude(
                path,
                warning=lambda message: log(f"  [warn] {message}"),
            )
            for path in group_paths
        ]
        group_peak = max(peaks)
        gain = SAFE_PEAK / group_peak if group_peak > 1.0 else 1.0
        gain_db = 20.0 * math.log10(gain) if gain > 0 else float("-inf")
        log(
            f"[group] {group_name}: {len(group_paths)} file(s), "
            f"peak={group_peak:.6f}, shared gain={gain_db:.2f} dB"
        )

        for path, info in zip(group_paths, infos):
            compliant = (
                path.suffix.casefold() == ".wav"
                and
                info.samplerate == TARGET_SAMPLE_RATE
                and info.subtype == TARGET_SUBTYPE
            )
            if compliant and gain == 1.0:
                log(f"  [skip] {path}")
                skipped += 1
                continue
            target = (
                path.with_suffix(".wav")
                if path.suffix.casefold() == ".flac"
                else path
            )
            arrow = f" -> {target}" if target != path else ""
            log(f"  [{'write' if apply else 'plan'}] {path}{arrow}")
            if apply:
                convert_file(
                    path,
                    gain,
                    target_path=target,
                    remove_source=target != path,
                    warning=lambda message: log(f"  [warn] {message}"),
                )
            converted += 1

        if apply and len(group_paths) > 1:
            final_paths = [
                path.with_suffix(".wav")
                if path.suffix.casefold() == ".flac"
                else path
                for path in group_paths
            ]
            final_infos = [sf.info(path) for path in final_paths]
            if len({info.frames for info in final_infos}) != 1:
                raise RuntimeError(f"Paired stem frame mismatch: {group_name}")

    return ConversionResult(
        discovered=len(files),
        converted=converted,
        skipped=skipped,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replace WAV/FLAC audio in a folder with 44.1 kHz PCM16 WAV."
        )
    )
    parser.add_argument("folder", type=Path)
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Only process WAV/FLAC files directly inside the selected folder.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Perform replacement; omit for a dry-run report.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = convert_folder(
        args.folder,
        recursive=not args.no_recursive,
        apply=args.apply,
    )
    action = "Converted" if args.apply else "Would convert"
    print(
        f"{action} {result.converted}; skipped {result.skipped}; "
        f"discovered {result.discovered} WAV/FLAC files."
    )


if __name__ == "__main__":
    main()
