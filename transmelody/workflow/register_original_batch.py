from __future__ import annotations

import argparse
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path

from transmelody.workflow.batch_song_renamer import (
    DEFAULT_REGISTRY,
    RenameItem,
    apply_plan,
    build_plan,
)


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


@dataclass(frozen=True)
class StemRename:
    source: Path
    destination: Path


STEM_NAMES = (
    ("_Instrumental.wav", "_inst.wav"),
    ("_Vocals.wav", "_vocal.wav"),
)


def build_stem_plan(
    audio_plan: list[RenameItem],
    stem_folder: Path,
) -> list[StemRename]:
    stem_folder = stem_folder.resolve()
    if not stem_folder.is_dir():
        raise ValueError(f"不是有效分轨文件夹：{stem_folder}")

    plan: list[StemRename] = []
    for item in audio_plan:
        for source_suffix, destination_suffix in STEM_NAMES:
            source = stem_folder / f"{item.title}{source_suffix}"
            destination = stem_folder / (
                f"{item.song_id}{destination_suffix}"
            )
            if not source.is_file():
                raise FileNotFoundError(f"缺少对应分轨：{source}")
            if destination.exists() and destination != source:
                raise FileExistsError(f"目标分轨已存在：{destination}")
            plan.append(StemRename(source, destination))

    expected_sources = {item.source.resolve() for item in plan}
    actual_sources = {
        path.resolve()
        for path in stem_folder.glob("*.wav")
        if path.is_file()
    }
    unexpected = sorted(actual_sources - expected_sources)
    if unexpected:
        raise ValueError(
            "分轨文件夹中存在无法对应到原音频的 WAV：\n"
            + "\n".join(str(path) for path in unexpected)
        )
    return plan


def rollback_stems(
    completed: list[StemRename],
    temporary_paths: list[tuple[StemRename, Path]],
) -> None:
    for item in reversed(completed):
        if item.destination.exists() and not item.source.exists():
            item.destination.replace(item.source)
    for item, temporary in reversed(temporary_paths):
        if temporary.exists() and not item.source.exists():
            temporary.replace(item.source)


def apply_batch(
    audio_plan: list[RenameItem],
    stem_plan: list[StemRename],
    registry: Path,
) -> None:
    token = uuid.uuid4().hex
    temporary_paths: list[tuple[StemRename, Path]] = []
    completed: list[StemRename] = []
    try:
        for index, item in enumerate(stem_plan):
            temporary = item.source.with_name(
                f".stem_rename_{token}_{index}{item.source.suffix}"
            )
            item.source.replace(temporary)
            temporary_paths.append((item, temporary))
        for item, temporary in temporary_paths:
            temporary.replace(item.destination)
            completed.append(item)

        apply_plan(
            audio_plan,
            registry,
            status="已分轨",
        )
    except Exception:
        rollback_stems(completed, temporary_paths)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "登记并编号原音频，同时把现有分轨改为 *_inst.wav / "
            "*_vocal.wav。"
        )
    )
    parser.add_argument(
        "--audio-folder",
        type=Path,
        default=Path("original audio"),
    )
    parser.add_argument(
        "--stem-folder",
        type=Path,
        default=Path("original stem"),
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=DEFAULT_REGISTRY,
    )
    parser.add_argument("--apply", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audio_plan = build_plan(args.audio_folder, args.registry)
    stem_plan = build_stem_plan(audio_plan, args.stem_folder)
    print(
        f"识别到 {len(audio_plan)} 首，编号范围 "
        f"{audio_plan[0].song_id}–{audio_plan[-1].song_id}；"
        f"对应 {len(stem_plan)} 个分轨。"
    )
    for audio in audio_plan:
        print(
            f"  {audio.source.name} -> {audio.destination.name}; "
            f"{audio.title}_Instrumental.wav -> "
            f"{audio.song_id}_inst.wav; "
            f"{audio.title}_Vocals.wav -> "
            f"{audio.song_id}_vocal.wav"
        )
    if args.apply:
        apply_batch(audio_plan, stem_plan, args.registry)
        print("完成：原音频、分轨和 registry 已同步更新。")
    else:
        print("当前为预览模式；确认后增加 --apply 执行。")


if __name__ == "__main__":
    main()
