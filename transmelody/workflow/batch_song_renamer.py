from __future__ import annotations
from transmelody.paths import PROJECT_ROOT

import argparse
import re
import sys
import uuid
from dataclasses import dataclass
from datetime import date
from pathlib import Path


DEFAULT_REGISTRY = PROJECT_ROOT / "dataset" / "song_registry.xlsx"
SUPPORTED_EXTENSIONS = {
    ".wav",
    ".flac",
    ".mp3",
    ".m4a",
    ".aac",
    ".ogg",
}
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


@dataclass(frozen=True)
class RenameItem:
    song_id: int
    title: str
    source: Path
    destination: Path


def natural_key(value: str) -> list[object]:
    return [
        int(part) if part.isdigit() else part.casefold()
        for part in re.split(r"(\d+)", value)
    ]


def read_registry(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"记录表不存在：{path}")
    if path.suffix.casefold() != ".xlsx":
        raise ValueError("记录表必须是 .xlsx 文件。")
    from transmelody.workflow.registry_io import read
    rows = read(path)
    ids = [int(row["id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("记录表中存在重复数字编号。")
    if any(song_id < 1 for song_id in ids):
        raise ValueError("记录表编号必须是正整数。")
    return rows


def discover_songs(folder: Path) -> list[Path]:
    supported_files = sorted(
        (
            path
            for path in folder.iterdir()
            if path.is_file()
            and path.suffix.casefold() in SUPPORTED_EXTENSIONS
        ),
        key=lambda path: natural_key(path.name),
    )
    files = [path for path in supported_files if not path.stem.isdigit()]
    if not files:
        raise ValueError(
            f"文件夹中没有需要登记的原名音频：{folder}"
        )
    titles = [path.stem.casefold() for path in files]
    duplicates = sorted({title for title in titles if titles.count(title) > 1})
    if duplicates:
        raise ValueError(
            "发现同名但不同扩展名的音频，请先去重："
            + ", ".join(duplicates)
        )
    return files


def build_plan(folder: Path, registry: Path = DEFAULT_REGISTRY) -> list[RenameItem]:
    folder = folder.resolve()
    if not folder.is_dir():
        raise ValueError(f"不是有效文件夹：{folder}")
    rows = read_registry(registry)
    next_id = max((int(row["id"]) for row in rows), default=0) + 1
    files = discover_songs(folder)
    plan = [
        RenameItem(
            song_id=next_id + index,
            title=source.stem,
            source=source,
            destination=source.with_name(
                f"{next_id + index}{source.suffix.casefold()}"
            ),
        )
        for index, source in enumerate(files)
    ]
    conflicts = [
        str(item.destination)
        for item in plan
        if item.destination.exists() and item.destination != item.source
    ]
    if conflicts:
        raise FileExistsError(
            "目标数字文件已存在：\n" + "\n".join(conflicts)
        )
    return plan


def append_registry_atomic(
    path: Path,
    new_rows: list[dict[str, str]],
) -> None:
    from transmelody.workflow.registry_io import append
    append(path, new_rows)


def update_registry_atomic(
    path: Path,
    updates: list[dict[str, str]],
) -> None:
    from transmelody.workflow.registry_io import update
    if updates:
        update(path, updates)


def apply_plan(
    plan: list[RenameItem],
    registry: Path = DEFAULT_REGISTRY,
    *,
    status: str = "待分轨",
) -> None:
    if not plan:
        raise ValueError("没有需要重命名的歌曲。")
    rows = read_registry(registry)
    expected_next_id = max(
        (int(row["id"]) for row in rows),
        default=0,
    ) + 1
    if plan[0].song_id != expected_next_id:
        raise ValueError(
            "记录表在预览后发生了变化，请重新预览后再执行。"
        )
    token = uuid.uuid4().hex
    temporary_paths: list[tuple[RenameItem, Path]] = []
    completed: list[RenameItem] = []
    try:
        for index, item in enumerate(plan):
            temporary = item.source.with_name(
                f".song_rename_{token}_{index}{item.source.suffix}"
            )
            item.source.replace(temporary)
            temporary_paths.append((item, temporary))
        for item, temporary in temporary_paths:
            temporary.replace(item.destination)
            completed.append(item)

        today = date.today().isoformat()
        source_folder = plan[0].source.parent.name
        new_rows = []
        for item in plan:
            new_rows.append(
                {
                    "id": str(item.song_id),
                    "title": item.title,
                    "original_filename": item.source.name,
                    "numbered_filename": item.destination.name,
                    "source_folder": source_folder,
                    "added_date": today,
                    "status": status,
                    "notes": "",
                }
            )
        append_registry_atomic(registry, new_rows)
    except Exception:
        for item in reversed(completed):
            if item.destination.exists() and not item.source.exists():
                item.destination.replace(item.source)
        for item, temporary in reversed(temporary_paths):
            if temporary.exists() and not item.source.exists():
                temporary.replace(item.source)
        raise


def print_plan(plan: list[RenameItem]) -> None:
    print(
        f"识别到 {len(plan)} 首；编号范围 "
        f"{plan[0].song_id}–{plan[-1].song_id}"
    )
    for item in plan:
        print(f"  {item.source.name}  ->  {item.destination.name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="按记录表的下一编号批量重命名歌曲并追加登记。"
    )
    parser.add_argument("folder", type=Path)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="执行重命名并写入记录表；不加时只预览。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plan = build_plan(args.folder, args.registry)
    print_plan(plan)
    if args.apply:
        apply_plan(plan, args.registry)
        print(f"完成，记录表已更新：{args.registry}")
    else:
        print("当前为预览模式；确认后增加 --apply 执行。")


if __name__ == "__main__":
    main()
