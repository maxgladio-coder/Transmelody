import mido

from transmelody.midi.midi_grid_quantizer import QuantizeConfig, quantize_midi
from transmelody.config import PROJECT_PPQ


def absolute_messages(track: mido.MidiTrack) -> list[tuple[int, mido.Message]]:
    tick = 0
    result = []
    for message in track:
        tick += message.time
        result.append((tick, message))
    return result


def make_test_midi() -> mido.MidiFile:
    midi = mido.MidiFile(type=1, ticks_per_beat=960)
    conductor = mido.MidiTrack()
    conductor.append(mido.MetaMessage("set_tempo", tempo=500000, time=0))
    conductor.append(mido.MetaMessage("end_of_track", time=5000))
    midi.tracks.append(conductor)

    melody = mido.MidiTrack()
    melody.append(mido.Message("program_change", program=0, time=0))
    melody.append(mido.Message("note_on", note=60, velocity=90, time=61))
    melody.append(mido.Message("note_off", note=60, velocity=0, time=410))
    melody.append(mido.Message("note_on", note=62, velocity=90, time=63))
    melody.append(mido.Message("note_off", note=62, velocity=0, time=421))
    melody.append(mido.MetaMessage("end_of_track", time=3000))
    midi.tracks.append(melody)
    return midi


def test_quantizes_notes_and_replaces_tempo() -> None:
    midi = make_test_midi()
    stats = quantize_midi(
        midi,
        QuantizeConfig(
            bpm=177,
            grid="1/16",
            quantize_note_ends=True,
            minimum_length_steps=1,
            repair_monophonic_overlaps=True,
        ),
    )

    melody_events = absolute_messages(midi.tracks[1])
    note_events = [
        (tick, message.type, getattr(message, "note", None))
        for tick, message in melody_events
        if message.type in {"note_on", "note_off"}
    ]
    assert note_events == [
        (0, "note_on", 60),
        (240, "note_off", 60),
        (240, "note_on", 62),
        (480, "note_off", 62),
    ]
    assert stats.notes == 2
    assert stats.moved_onsets == 2

    tempo_events = [
        (tick, message)
        for tick, message in absolute_messages(midi.tracks[0])
        if message.type == "set_tempo"
    ]
    assert len(tempo_events) == 1
    assert tempo_events[0][0] == 0
    assert round(mido.tempo2bpm(tempo_events[0][1].tempo)) == 177
    assert stats.source_ticks_per_beat == 960
    assert stats.ticks_per_beat == PROJECT_PPQ
    assert midi.ticks_per_beat == PROJECT_PPQ
    assert stats.output_end_tick % (4 * midi.ticks_per_beat) == 0
    assert stats.output_end_tick >= stats.original_end_tick


def test_onsets_only_preserves_note_duration() -> None:
    midi = make_test_midi()
    expected_rescaled_duration = 205
    quantize_midi(
        midi,
        QuantizeConfig(
            bpm=177,
                grid="1/16",
                quantize_note_ends=False,
                repair_monophonic_overlaps=False,
                bridge_short_gaps=False,
            ),
    )

    events = absolute_messages(midi.tracks[1])
    note_ticks = [
        tick
        for tick, message in events
        if message.type in {"note_on", "note_off"}
    ]
    assert note_ticks[1] - note_ticks[0] == expected_rescaled_duration


def test_bridges_one_grid_gap_but_preserves_longer_rest() -> None:
    midi = mido.MidiFile(type=1, ticks_per_beat=480)
    track = mido.MidiTrack()
    track.append(mido.MetaMessage("time_signature", numerator=3, denominator=4))
    track.append(mido.Message("note_on", note=60, velocity=90, time=0))
    track.append(mido.Message("note_off", note=60, velocity=0, time=240))
    track.append(mido.Message("note_on", note=62, velocity=90, time=120))
    track.append(mido.Message("note_off", note=62, velocity=0, time=240))
    track.append(mido.Message("note_on", note=64, velocity=90, time=240))
    track.append(mido.Message("note_off", note=64, velocity=0, time=240))
    midi.tracks.append(track)

    stats = quantize_midi(
        midi,
        QuantizeConfig(bpm=182, grid="1/16"),
    )

    events = absolute_messages(midi.tracks[0])
    note_events = [
        (tick, message.type, getattr(message, "note", None))
        for tick, message in events
        if message.type in {"note_on", "note_off"}
    ]
    assert note_events[:4] == [
        (0, "note_on", 60),
        (360, "note_off", 60),
        (360, "note_on", 62),
        (600, "note_off", 62),
    ]
    assert note_events[4][0] == 840
    assert stats.bridged_gaps == 1
    assert stats.time_signature == (3, 4)
    assert stats.output_end_tick % 1440 == 0


def test_deletes_velocity_below_40_and_normalizes_retained_notes() -> None:
    midi = mido.MidiFile(type=1, ticks_per_beat=480)
    track = mido.MidiTrack()
    track.append(mido.Message("note_on", note=60, velocity=39, time=0))
    track.append(mido.Message("note_off", note=60, velocity=0, time=240))
    track.append(mido.Message("note_on", note=62, velocity=40, time=0))
    track.append(mido.Message("note_off", note=62, velocity=0, time=240))
    midi.tracks.append(track)

    stats = quantize_midi(
        midi,
        QuantizeConfig(bpm=120, grid="1/16"),
    )

    note_ons = [
        message
        for _, message in absolute_messages(midi.tracks[0])
        if message.type == "note_on" and message.velocity > 0
    ]
    assert [(message.note, message.velocity) for message in note_ons] == [
        (62, 100)
    ]
    assert stats.input_notes == 2
    assert stats.notes == 1
    assert stats.deleted_low_velocity_notes == 1


def test_preserves_original_tempo_by_default() -> None:
    midi = make_test_midi()
    stats = quantize_midi(midi, QuantizeConfig(grid="1/16"))

    tempo_events = [
        message.tempo
        for _, message in absolute_messages(midi.tracks[0])
        if message.type == "set_tempo"
    ]
    assert tempo_events == [500000]
    assert stats.input_bpm == 120.0
    assert stats.output_bpm == 120.0
    assert stats.tempo_replaced is False
