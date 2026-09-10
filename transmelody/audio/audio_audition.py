"""Read-only audition rendering. Boundary clicks are NOT inferred melody notes."""
from __future__ import annotations

from collections import defaultdict, deque
import ctypes
import math
from pathlib import Path
import uuid

import mido
import numpy as np
from scipy.signal import resample_poly

PLAYBACK_RATE = 44100


def midi_notes_in_wav_time(path: Path, *, audio_start_seconds: float | None = None):
    """Honor all MIDI tempo events and the actual tick of the WAV START marker."""
    midi = mido.MidiFile(path)
    if midi.type == 2 or midi.ticks_per_beat <= 0:
        raise ValueError("试听仅支持同步 PPQ MIDI（type 0/1）。")
    tempo, time = 500000, 0.0
    active, notes, markers = defaultdict(deque), [], []
    for message in mido.merge_tracks(midi.tracks):
        time += mido.tick2second(message.time, midi.ticks_per_beat, tempo)
        if message.type == "set_tempo":
            tempo = message.tempo
        elif message.type == "marker" and message.text.startswith("WAV START"):
            markers.append(time)
        elif message.type == "note_on" and message.velocity and message.channel != 9:
            active[(message.channel, message.note)].append((time, message.velocity))
        elif message.type == "note_off" or (message.type == "note_on" and not message.velocity):
            queue = active[(message.channel, message.note)]
            if queue:
                start, velocity = queue.popleft()
                if time > start:
                    notes.append((start, time, message.note, velocity))
    if any(active.values()):
        raise ValueError("MIDI 有未关闭的音符，不能可靠试听。")
    if markers:
        if max(markers) - min(markers) > 1e-6:
            raise ValueError("MIDI 中存在冲突的 WAV START 标记。")
        origin = markers[0]
    elif audio_start_seconds is not None and math.isfinite(audio_start_seconds):
        origin = audio_start_seconds
    else:
        raise ValueError("此 MIDI 缺少 WAV START 标记，不能猜测音频偏移。")
    return [(start - origin, end - origin, pitch, velocity) for start, end, pitch, velocity in sorted(notes)]


def render_audition(wave, sample_rate, start, end, *, units=(), notes=(),
                    vocal_gain=.65, cue_gain=.22, clicks=False, rate=PLAYBACK_RATE):
    """Render one selection; no source files/labels/MIDI are modified.

    Clicks mark unit STARTS only (not both edges, avoiding double cues). MIDI
    notes use a simple harmonic tone, not an external instrument or soundfont.
    """
    wave = np.asarray(wave)
    if wave.ndim != 1 or not len(wave) or sample_rate <= 0 or rate <= 0:
        raise ValueError("需要有效的单声道音频与采样率。")
    if not (np.isfinite([start, end, vocal_gain, cue_gain]).all()
            and 0 <= start < end <= len(wave) / sample_rate + 1e-6
            and 0 <= vocal_gain <= 1 and 0 <= cue_gain <= 1):
        raise ValueError("试听范围或音量无效。")
    length = round((end - start) * rate)
    if length < 1:
        raise ValueError("试听范围太短。")
    selected = wave[round(start * sample_rate):round(end * sample_rate)].astype(np.float32)
    if not np.isfinite(selected).all():
        raise ValueError("音频中有非有限数值。")
    divisor = math.gcd(int(sample_rate), rate)
    if sample_rate != rate:
        selected = resample_poly(selected, rate // divisor, int(sample_rate) // divisor)
    mix = np.zeros(length, dtype=np.float32)
    count = min(length, len(selected))
    mix[:count] = selected[:count] * vocal_gain
    if clicks:
        t = np.arange(round(.018 * rate)) / rate
        cue = (np.sin(2 * np.pi * 1500 * t) * np.exp(-t * 180) * np.minimum(t / .001, 1)).astype(np.float32)
        # Dedup at the output-sample precision only, without changing labels.
        for index in sorted({round((u["start"] - start) * rate) for u in units if start <= u["start"] < end}):
            count = min(len(cue), length - index)
            if count > 0:
                mix[index:index + count] += cue[:count] * cue_gain
    for onset, offset, pitch, velocity in notes:
        left, right = max(start, onset), min(end, offset)
        if right <= left:
            continue
        a, b = max(0, round((left - start) * rate)), min(length, round((right - start) * rate))
        t = start + np.arange(a, b) / rate - onset
        frequency = 440 * 2 ** ((pitch - 69) / 12)
        tone = np.sin(2 * np.pi * frequency * t)
        if 2 * frequency < rate / 2:
            tone += .18 * np.sin(4 * np.pi * frequency * t)
        duration = offset - onset
        envelope = np.minimum(np.maximum(t, 0) / .006, 1) * np.minimum(np.maximum(duration - t, 0) / .012, 1)
        mix[a:b] += (tone * envelope * cue_gain * velocity / 127).astype(np.float32)
    peak = float(np.max(np.abs(mix)))
    if peak > .98:
        mix *= .98 / peak
    return mix


class WavePlayer:
    """Small Windows PCM player with seek/pause; all commands target our own alias."""
    def __init__(self):
        self.api = ctypes.WinDLL("winmm")
        self.api.mciSendStringW.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_uint, ctypes.c_void_p]
        self.api.mciSendStringW.restype = ctypes.c_uint
        self.api.mciGetErrorStringW.argtypes = [ctypes.c_uint, ctypes.c_wchar_p, ctypes.c_uint]
        self.alias = "audition_" + uuid.uuid4().hex
        self.opened = False

    def command(self, text):
        buffer = ctypes.create_unicode_buffer(1024)
        error = self.api.mciSendStringW(text, buffer, len(buffer), None)
        if error:
            self.api.mciGetErrorStringW(error, buffer, len(buffer))
            raise RuntimeError(f"音频播放错误 {error}: {buffer.value}")
        return buffer.value

    def open(self, path):
        self.close()
        path = str(Path(path).resolve())
        if '"' in path:
            raise ValueError("不支持带引号的文件路径。")
        self.command(f'open "{path}" type waveaudio alias {self.alias}')
        self.opened = True
        self.command(f"set {self.alias} time format milliseconds")

    def play(self, seconds=0.):
        self.command(f"play {self.alias} from {max(0, round(seconds * 1000))}")

    def pause(self):
        self.command(f"pause {self.alias}")

    def stop(self):
        if self.opened:
            self.command(f"stop {self.alias}")

    def position(self):
        return float(self.command(f"status {self.alias} position")) / 1000 if self.opened else 0.

    def mode(self):
        return self.command(f"status {self.alias} mode") if self.opened else "stopped"

    def close(self):
        if self.opened:
            self.command(f"close {self.alias}")
            self.opened = False
