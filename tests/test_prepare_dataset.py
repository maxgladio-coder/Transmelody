from pathlib import Path

import mido
import numpy as np

from prepare_dataset import (
    discover_dataset,
    interpolate_with_extrapolation,
    midi_metadata,
)


def test_discovers_three_way_song_pair(tmp_path: Path) -> None:
    root = tmp_path / "melody_dataset"
    for folder in ("inst_audio", "vocal_audio", "vocal_mid"):
        (root / folder).mkdir(parents=True)
    (root / "inst_audio" / "song one_inst.wav").touch()
    (root / "vocal_audio" / "song one_vocals.wav").touch()
    (root / "vocal_mid" / "song one_vocal.mid").touch()
    (root / "inst_audio" / "2_inst.wav").touch()
    (root / "vocal_audio" / "2_vocal.wav").touch()
    (root / "vocal_mid" / "2_vocal.mid").touch()

    songs, issues = discover_dataset(root)

    assert not issues
    assert len(songs) == 2
    assert {song.song_id for song in songs} == {"song one", "2"}


def test_tick_mapping_interpolates_and_extrapolates() -> None:
    grid_ticks = np.asarray([480.0, 960.0, 1440.0])
    grid_samples = np.asarray([1000.0, 2000.0, 3000.0])
    values = np.asarray([0.0, 720.0, 1920.0])

    mapped = interpolate_with_extrapolation(
        values,
        grid_ticks,
        grid_samples,
    )

    assert np.allclose(mapped, [0.0, 1500.0, 4000.0])


def test_midi_tempo_map_is_preserved_as_metadata(tmp_path: Path) -> None:
    path = tmp_path / "variable_tempo.mid"
    midi = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    track.append(
        mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(140), time=0)
    )
    track.append(
        mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(195), time=1920)
    )
    track.append(mido.MetaMessage("end_of_track", time=1920))
    midi.tracks.append(track)
    midi.save(path)

    metadata = midi_metadata(path)

    assert metadata["tempo_mode"] == "tempo_map"
    assert [event["tick"] for event in metadata["tempo_events"]] == [0, 1920]
    assert [round(event["bpm"]) for event in metadata["tempo_events"]] == [
        140,
        195,
    ]
