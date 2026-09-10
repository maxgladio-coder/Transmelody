from __future__ import annotations
from transmelody.paths import PROJECT_ROOT

import argparse
import re
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from transmelody.workflow.batch_song_renamer import (
    DEFAULT_REGISTRY,
    read_registry,
    update_registry_atomic,
)


ORIGINAL_STEM = PROJECT_ROOT / "original stem"
TEST_ROOT = PROJECT_ROOT / "test audio"
TEST_INST = TEST_ROOT / "test_inst"
TEST_VOCAL = TEST_ROOT / "test_vocal"
DATASET_ROOT = PROJECT_ROOT / "dataset" / "melody_dataset"
DATASET_INST = DATASET_ROOT / "inst_audio"
DATASET_VOCAL = DATASET_ROOT / "vocal_audio"
DATASET_MIDI = DATASET_ROOT / "vocal_mid"
PREDICTION_OUTPUT = PROJECT_ROOT / "test_output"
CHECKPOINT = PROJECT_ROOT / "output" / "melody_transformer" / "final.pt"
TRAINING_SPLIT = CHECKPOINT.parent / "training_split.json"

STATUS_STAGED = "待预测"
STATUS_REVIEW = "待校对"
STATUS_TRAINING = "已入训练集"

INST_PATTERN = re.compile(r"^(\d+)_inst\.wav$", re.IGNORECASE)
VOCAL_PATTERN = re.compile(r"^(\d+)_vocal\.wav$", re.IGNORECASE)
SOURCE_INST_PATTERNS = (
    INST_PATTERN,
    re.compile(r"^(\d+)_Instrumental\.wav$", re.IGNORECASE),
)
SOURCE_VOCAL_PATTERNS = (
    VOCAL_PATTERN,
    re.compile(r"^(\d+)_Vocals\.wav$", re.IGNORECASE),
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


@dataclass(frozen=True)
class StemPair:
    song_id: int
    inst: Path
    vocal: Path


def index_stems(
    folder: Path,
    pattern: re.Pattern[str] | tuple[re.Pattern[str], ...],
) -> dict[int, Path]:
    result: dict[int, Path] = {}
    patterns = pattern if isinstance(pattern, tuple) else (pattern,)
    if not folder.is_dir():
        return result
    for path in folder.iterdir():
        if not path.is_file():
            continue
        match = next(
            (
                candidate.fullmatch(path.name)
                for candidate in patterns
                if candidate.fullmatch(path.name)
            ),
            None,
        )
        if not match:
            continue
        song_id = int(match.group(1))
        if song_id in result:
            raise ValueError(f"{folder} 中编号 {song_id} 重复。")
        result[song_id] = path.resolve()
    return result


def discover_pairs(inst_dir: Path, vocal_dir: Path) -> list[StemPair]:
    inst = index_stems(inst_dir, INST_PATTERN)
    vocal = index_stems(vocal_dir, VOCAL_PATTERN)
    if set(inst) != set(vocal):
        raise ValueError(
            "分轨配对不完整："
            f"仅 inst={sorted(set(inst) - set(vocal))}，"
            f"仅 vocal={sorted(set(vocal) - set(inst))}"
        )
    return [
        StemPair(song_id, inst[song_id], vocal[song_id])
        for song_id in sorted(inst)
    ]


def discover_source_pairs() -> list[StemPair]:
    inst = index_stems(ORIGINAL_STEM, SOURCE_INST_PATTERNS)
    vocal = index_stems(ORIGINAL_STEM, SOURCE_VOCAL_PATTERNS)
    if set(inst) != set(vocal):
        raise ValueError(
            "original stem 配对不完整："
            f"仅 inst={sorted(set(inst) - set(vocal))}，"
            f"仅 vocal={sorted(set(vocal) - set(inst))}"
        )
    return [
        StemPair(song_id, inst[song_id], vocal[song_id])
        for song_id in sorted(inst)
    ]


def registry_index(registry: Path = DEFAULT_REGISTRY) -> dict[int, dict[str, str]]:
    return {
        int(row["id"]): row
        for row in read_registry(registry)
    }


def describe_status(registry: Path = DEFAULT_REGISTRY) -> dict[str, object]:
    rows = registry_index(registry)
    source_pairs = discover_source_pairs()
    test_pairs = discover_pairs(TEST_INST, TEST_VOCAL)
    review_ids = sorted(
        song_id
        for song_id, row in rows.items()
        if row["status"] == STATUS_REVIEW
    )
    return {
        "source_pairs": len(source_pairs),
        "queued_pairs": len(test_pairs),
        "next_test_id": test_pairs[0].song_id if test_pairs else None,
        "review_ids": review_ids,
        "registry_next_id": max(rows, default=0) + 1,
    }


def print_status(registry: Path = DEFAULT_REGISTRY) -> None:
    status = describe_status(registry)
    print(f"original stem 待入队：{status['source_pairs']} 组")
    print(f"test audio 队列：{status['queued_pairs']} 组")
    print(f"下一组：{status['next_test_id'] or '无'}")
    print(
        "等待校对："
        + (
            ", ".join(map(str, status["review_ids"]))
            if status["review_ids"]
            else "无"
        )
    )
    print(f"Registry 下一编号：{status['registry_next_id']}")


def rollback_moves(
    completed: list[tuple[Path, Path]],
    temporary: list[tuple[Path, Path]],
) -> None:
    for source, destination in reversed(completed):
        if destination.exists() and not source.exists():
            replace_with_windows_retry(destination, source)
    for source, temporary_path in reversed(temporary):
        if temporary_path.exists() and not source.exists():
            replace_with_windows_retry(temporary_path, source)


def replace_with_windows_retry(
    source: Path,
    destination: Path,
    *,
    attempts: int = 20,
    delay_seconds: float = 0.25,
) -> None:
    """Retry atomic moves while Windows releases short-lived audio handles."""
    for attempt in range(attempts):
        try:
            source.replace(destination)
            return
        except PermissionError:
            if attempt + 1 == attempts:
                raise
            time.sleep(delay_seconds)


def move_pairs_transactionally(
    moves: list[tuple[Path, Path]],
    registry: Path,
    updates: list[dict[str, str]],
) -> None:
    token = uuid.uuid4().hex
    temporary: list[tuple[Path, Path]] = []
    completed: list[tuple[Path, Path]] = []
    try:
        for index, (source, _) in enumerate(moves):
            temporary_path = source.with_name(
                f".queue_move_{token}_{index}{source.suffix}"
            )
            replace_with_windows_retry(source, temporary_path)
            temporary.append((source, temporary_path))
        for (source, destination), (_, temporary_path) in zip(
            moves,
            temporary,
        ):
            destination.parent.mkdir(parents=True, exist_ok=True)
            replace_with_windows_retry(temporary_path, destination)
            completed.append((source, destination))
        update_registry_atomic(registry, updates)
    except Exception:
        rollback_moves(completed, temporary)
        raise


def stage_original_stems(
    *,
    registry: Path = DEFAULT_REGISTRY,
    apply: bool = False,
) -> list[StemPair]:
    pairs = discover_source_pairs()
    if not pairs:
        raise ValueError("original stem 中没有可入队的数字分轨。")
    rows = registry_index(registry)
    moves: list[tuple[Path, Path]] = []
    updates: list[dict[str, str]] = []
    for pair in pairs:
        if pair.song_id not in rows:
            raise ValueError(f"Registry 中不存在编号 {pair.song_id}。")
        if rows[pair.song_id]["status"] not in {"待分轨", "已分轨"}:
            raise ValueError(
                f"编号 {pair.song_id} 当前状态为 "
                f"{rows[pair.song_id]['status']!r}，"
                "预期为“待分轨”或“已分轨”。"
            )
        destinations = (
            TEST_INST / f"{pair.song_id}_inst.wav",
            TEST_VOCAL / f"{pair.song_id}_vocal.wav",
        )
        for destination in destinations:
            if destination.exists():
                raise FileExistsError(f"测试队列目标已存在：{destination}")
        moves.extend(
            [
                (pair.inst, destinations[0]),
                (pair.vocal, destinations[1]),
            ]
        )
        updates.append(
            {
                "id": str(pair.song_id),
                "status": STATUS_STAGED,
                "notes": "已进入 test audio 队列",
            }
        )

    print(
        f"将 {len(pairs)} 组分轨移入 test audio，"
        f"编号 {pairs[0].song_id}–{pairs[-1].song_id}。"
    )
    if apply:
        move_pairs_transactionally(moves, registry, updates)
        print("入队完成，Registry 状态已更新为“待预测”。")
    else:
        print("当前为预览；增加 --apply 才会移动文件。")
    return pairs


def run_command(command: list[str]) -> None:
    print("> " + " ".join(command), flush=True)
    subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=True,
    )


def pending_review_rows(
    registry: Path = DEFAULT_REGISTRY,
) -> list[tuple[int, dict[str, str]]]:
    rows = registry_index(registry)
    return sorted(
        (
            (song_id, row)
            for song_id, row in rows.items()
            if row["status"] == STATUS_REVIEW
        ),
        key=lambda item: item[0],
    )


def validate_reviewed_files(song_id: int) -> None:
    required = (
        DATASET_INST / f"{song_id}_inst.wav",
        DATASET_VOCAL / f"{song_id}_vocal.wav",
        DATASET_MIDI / f"{song_id}_vocal.mid",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"编号 {song_id} 仍在等待校对。请把修正后的 MIDI 保存为 "
            f"{DATASET_MIDI / f'{song_id}_vocal.mid'}。\n缺少："
            + "\n".join(missing)
        )


def prepare_and_accept_reviews(
    *,
    registry: Path = DEFAULT_REGISTRY,
) -> list[int]:
    pending = pending_review_rows(registry)
    for song_id, _ in pending:
        validate_reviewed_files(song_id)

    run_command(
        [
            str(sys.executable),
            "-m", "transmelody.training.prepare_dataset",
            str(DATASET_ROOT),
        ]
    )
    if pending:
        updates = [
            {
                "id": str(song_id),
                "status": STATUS_TRAINING,
                "notes": "校对 MIDI 已验证并进入训练集",
            }
            for song_id, _ in pending
        ]
        update_registry_atomic(registry, updates)
        print(
            "已确认校对完成："
            + ", ".join(str(song_id) for song_id, _ in pending)
        )
    return [song_id for song_id, _ in pending]


def move_predicted_pair_to_dataset(
    pair: StemPair,
    *,
    registry: Path = DEFAULT_REGISTRY,
) -> None:
    midi_path = PREDICTION_OUTPUT / f"{pair.song_id}_predicted.mid"
    if not midi_path.is_file():
        raise FileNotFoundError(f"预测 MIDI 未生成：{midi_path}")
    grid_path = (
        PREDICTION_OUTPUT
        / "grid_cache"
        / f"{pair.song_id}_grid.json"
    )
    if not grid_path.is_file():
        raise FileNotFoundError(f"预测 Grid 未生成：{grid_path}")

    destinations = (
        DATASET_INST / f"{pair.song_id}_inst.wav",
        DATASET_VOCAL / f"{pair.song_id}_vocal.wav",
        DATASET_ROOT / "grid_cache" / f"{pair.song_id}_grid.json",
    )
    for destination in destinations:
        if destination.exists():
            raise FileExistsError(f"训练集目标已存在：{destination}")
    moves = [
        (pair.inst, destinations[0]),
        (pair.vocal, destinations[1]),
        (grid_path, destinations[2]),
    ]
    update = [
        {
            "id": str(pair.song_id),
            "status": STATUS_REVIEW,
            "notes": (
                f"预测 MIDI：test_output/{midi_path.name}；"
                f"校对后保存为 vocal_mid/{pair.song_id}_vocal.mid"
            ),
        }
    ]
    move_pairs_transactionally(moves, registry, update)


def workflow_training_command(epochs: int) -> list[str]:
    return [str(sys.executable), "-m", "transmelody.training.train_melody", str(DATASET_ROOT), "--epochs", str(epochs),
        "--output", str(CHECKPOINT.parent), "--split", str(TRAINING_SPLIT),
        "--musical-event-context", "--note-event-training", "--init-checkpoint", str(CHECKPOINT),
        "--learning-rate", "0.00003", "--semitone-guard", "--artifact-training", "--artifact-guard"]


def run_next(
    *,
    registry: Path = DEFAULT_REGISTRY,
    epochs: int = 75,
    skip_train: bool = False,
) -> int | None:
    if epochs < 0:
        raise ValueError("epochs 不能小于 0；0 表示跳过训练。")
    skip_train = skip_train or epochs == 0
    if skip_train:
        print("本轮跳过训练，使用当前模型预测；校对文件检查不会跳过。")

    queue_before = discover_pairs(TEST_INST, TEST_VOCAL)
    pending = pending_review_rows(registry)
    if not queue_before and not pending:
        print("test audio 已耗尽，且没有等待校对的曲目。")
        return None

    if pending or not skip_train:
        prepare_and_accept_reviews(registry=registry)
    if not skip_train:
        run_command(workflow_training_command(epochs))

    queue = discover_pairs(TEST_INST, TEST_VOCAL)
    if not queue:
        print("最后一组校对数据已入库；本轮跳过训练，test audio 队列已耗尽。"
              if skip_train else "最后一组校对数据已纳入训练；test audio 队列已耗尽。")
        return None
    pair = queue[0]
    row = registry_index(registry).get(pair.song_id)
    if row is None:
        raise ValueError(f"Registry 中不存在编号 {pair.song_id}。")
    if row["status"] != STATUS_STAGED:
        raise ValueError(
            f"编号 {pair.song_id} 的 Registry 状态为 "
            f"{row['status']!r}，预期为“{STATUS_STAGED}”。"
        )
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(f"模型 checkpoint 不存在：{CHECKPOINT}")

    run_command(
        [
            str(sys.executable),
            "-m", "transmelody.inference.predict_test_audio",
            "--input",
            str(TEST_ROOT),
            "--output",
            str(PREDICTION_OUTPUT),
            "--checkpoint",
            str(CHECKPOINT),
            "--limit",
            "1",
        ]
    )
    move_predicted_pair_to_dataset(pair, registry=registry)
    print(
        f"编号 {pair.song_id} 已预测并移入训练集音频目录；"
        f"请校对 test_output/{pair.song_id}_predicted.mid，"
        f"另存为 dataset/melody_dataset/vocal_mid/"
        f"{pair.song_id}_vocal.mid。"
    )
    return pair.song_id


def check_next(registry: Path = DEFAULT_REGISTRY) -> None:
    queue = discover_pairs(TEST_INST, TEST_VOCAL)
    pending = pending_review_rows(registry)
    if pending:
        for song_id, _ in pending:
            validate_reviewed_files(song_id)
        print(
            "上一轮校对文件齐全，下一次运行会先纳入训练："
            + ", ".join(str(song_id) for song_id, _ in pending)
        )
    if not queue:
        print("test audio 队列为空。")
        return
    pair = queue[0]
    row = registry_index(registry).get(pair.song_id)
    if row is None or row["status"] != STATUS_STAGED:
        raise ValueError(
            f"编号 {pair.song_id} 的 Registry 状态不是“{STATUS_STAGED}”。"
        )
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(f"模型 checkpoint 不存在：{CHECKPOINT}")
    print(
        f"检查通过：下一次将处理编号 {pair.song_id}；"
        f"inst={pair.inst.name}，vocal={pair.vocal.name}。"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="管理 melody 模型的待预测、待校对和训练集循环。"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("status", help="查看当前队列状态。")
    subparsers.add_parser(
        "check",
        help="验证下一轮所需文件，但不训练、不预测、不移动。",
    )

    stage = subparsers.add_parser(
        "stage",
        help="把 original stem 中的数字分轨移入 test audio。",
    )
    stage.add_argument("--apply", action="store_true")

    next_parser = subparsers.add_parser(
        "next",
        help="训练模型并预测 test audio 中编号最小的一组。",
    )
    next_parser.add_argument("--epochs", type=int, default=75,
        help="继续训练的轮数；设为 0 则跳过训练，仍检查校对文件并预测下一组。")
    next_parser.add_argument(
        "--skip-train",
        action="store_true",
        help="使用现有 checkpoint 直接预测，仅用于明确不需重训时。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "status":
        print_status()
    elif args.command == "check":
        check_next()
    elif args.command == "stage":
        stage_original_stems(apply=args.apply)
    elif args.command == "next":
        run_next(epochs=args.epochs, skip_train=args.skip_train)


if __name__ == "__main__":
    main()
