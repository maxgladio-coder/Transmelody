import json
import os
from pathlib import Path

import mido
import numpy as np
import pytest
import soundfile as sf

from audio_audition import WavePlayer, midi_notes_in_wav_time, render_audition


@pytest.mark.parametrize("ppq", [480, 960])
def test_midi_tempo_changes_and_marker_use_actual_event_time(tmp_path, ppq):
    midi = mido.MidiFile(ticks_per_beat=ppq)
    midi.tracks.append(mido.MidiTrack([
        mido.MetaMessage("set_tempo", tempo=500000),
        mido.MetaMessage("marker", text="WAV START (text tick deliberately wrong: 999)", time=ppq),
        mido.Message("note_on", note=60, velocity=100),
        mido.MetaMessage("set_tempo", tempo=1000000, time=ppq),
        mido.Message("note_off", note=60, time=ppq),
    ]))
    path = tmp_path / "tempo.mid"
    midi.save(path)
    assert midi_notes_in_wav_time(path) == [(0., 1.5, 60, 100)]


def test_missing_marker_never_silently_aligns_to_zero(tmp_path):
    midi = mido.MidiFile()
    midi.tracks.append(mido.MidiTrack())
    path = tmp_path / "unknown.mid"
    midi.save(path)
    with pytest.raises(ValueError, match="WAV START"):
        midi_notes_in_wav_time(path)


def test_click_onsets_only_with_absolute_crop_origin():
    wave = np.zeros(44100 * 3)
    result = render_audition(wave, 44100, 1., 2., vocal_gain=0., clicks=True,
        units=[{"start": 1.2, "end": 1.6}, {"start": .5, "end": .8}])
    assert len(result) == 44100
    assert not result[:8820].any()
    assert result[8820:9600].any()
    assert not result[10000:].any()  # no second click at the unit end


def test_resample_pitch_and_note_crossing_clip_boundaries():
    result = render_audition(np.zeros(48000 * 2), 48000, .5, 1., vocal_gain=0.,
        notes=[(0., 1.5, 69, 127)], cue_gain=.5)
    assert len(result) == 22050 and np.isfinite(result).all()
    peak_frequency = np.fft.rfftfreq(len(result), 1 / 44100)[np.argmax(abs(np.fft.rfft(result)))]
    assert peak_frequency == 440
    assert np.max(np.abs(result)) <= .98


def test_mix_protects_from_clipping_and_does_not_mutate_wave():
    wave = np.ones(44100, dtype=np.float32)
    result = render_audition(wave, 44100, 0., 1., vocal_gain=1., cue_gain=1., notes=[(0., 1., 69, 127)])
    assert np.max(abs(result)) <= .980001
    assert (wave == 1).all()
    with pytest.raises(ValueError):
        render_audition(wave, 44100, float("nan"), 1.)


@pytest.mark.skipif(os.environ.get("RUN_AUDITION_SMOKE") != "1", reason="Optional Windows playback smoke")
def test_native_player_on_silent_pcm(tmp_path):
    path = tmp_path / "静音 播放测试.wav"
    sf.write(path, np.zeros(44100), 44100, subtype="PCM_16")
    player = WavePlayer()
    try:
        player.open(path)
        assert player.mode() == "stopped"
        player.play(.2)
        player.pause()
        assert player.position() >= .199
        player.play(.4)
        player.stop()
    finally:
        player.close()
    assert not player.opened


@pytest.mark.skipif(os.environ.get("RUN_AUDITION_SMOKE") != "1", reason="Optional hidden Tk audition smoke")
def test_ui_reads_actual_18_and_renders_without_playing_sound():
    import tkinter as tk
    from melody_audition_ui import AuditionWindow, ROOT
    class SilentPlayer:
        opened = False
        current = 0.
        def open(self, path):
            self.opened = True
            assert sf.info(path).subtype == "PCM_16"
        def play(self, seconds=0.):
            self.current = seconds
        def position(self):
            return self.current
        def mode(self):
            return "playing"
        def pause(self):
            pass
        def stop(self):
            pass
        def close(self):
            self.opened = False
    root = tk.Tk()
    root.withdraw()
    window = AuditionWindow(root, ROOT / "output/lyric_alignment/18_candidate.json", player=SilentPlayer())
    try:
        assert {"旧预测 MIDI（不是本轮结果）", "手工校对 MIDI"} <= set(window.midi_options)
        window.play()
        assert window.playing
        window.position.set(.5)
        window.seek()
        assert window.player.current == .5
        window.pause()
        window.mode.set("人声 + 手工校对 MIDI")
        window.change_mode()
        window.play()
        assert window.loaded_key[2] == "人声 + 手工校对 MIDI"
        if (ROOT / "output/musical_event_experiment/sample_joint/18_predicted.mid").exists():
            assert "联合边界实验 MIDI" in window.midi_options
            window.pause()
            window.mode.set("人声 + 联合边界实验 MIDI")
            window.change_mode()
            window.play()
            assert window.loaded_key[2] == "人声 + 联合边界实验 MIDI"
        if (ROOT / "output/pitch_context_experiment/sample/18_predicted.mid").exists():
            assert "音高上下文实验 MIDI" in window.midi_options
            window.pause()
            window.mode.set("人声 + 音高上下文实验 MIDI")
            window.change_mode()
            window.play()
            assert window.loaded_key[2] == "人声 + 音高上下文实验 MIDI"
        if (ROOT / "output/joint_pitch_experiment/sample/18_predicted.mid").exists():
            assert "音高与边界联合 MIDI" in window.midi_options
            window.pause()
            window.mode.set("人声 + 音高与边界联合 MIDI")
            window.change_mode()
            window.play()
            assert window.loaded_key[2] == "人声 + 音高与边界联合 MIDI"
        if (ROOT / "output/note_event_experiment/sample/18_predicted.mid").exists():
            assert "完整音符工作流 MIDI" in window.midi_options
            window.pause()
            window.mode.set("人声 + 完整音符工作流 MIDI")
            window.change_mode()
            window.play()
            assert window.loaded_key[2] == "人声 + 完整音符工作流 MIDI"
        temporary = window.wav_path
    finally:
        window.close()
    assert not temporary.exists()
