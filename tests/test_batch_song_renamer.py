from pathlib import Path

import pytest

from batch_song_renamer import build_plan, discover_songs


def test_plan_uses_next_registry_id_and_natural_order(tmp_path: Path, monkeypatch) -> None:
    folder = tmp_path / "songs"
    folder.mkdir()
    (folder / "Song 10.wav").touch()
    (folder / "Song 2.flac").touch()
    registry = tmp_path / "registry.xlsx"
    # Unit-test numbering independently of the external XLSX bridge runtime.
    monkeypatch.setattr("batch_song_renamer.read_registry", lambda path: [{"id": "9", "title": "Existing"}])

    plan = build_plan(folder, registry)

    assert [item.source.name for item in plan] == [
        "Song 2.flac",
        "Song 10.wav",
    ]
    assert [item.destination.name for item in plan] == ["10.flac", "11.wav"]


def test_existing_numeric_audio_is_ignored(tmp_path: Path) -> None:
    (tmp_path / "10.wav").touch()
    (tmp_path / "New song.wav").touch()
    assert [p.name for p in discover_songs(tmp_path)] == ["New song.wav"]


def test_duplicate_title_with_two_extensions_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "Same.wav").touch()
    (tmp_path / "Same.flac").touch()

    with pytest.raises(ValueError, match="去重"):
        discover_songs(tmp_path)
