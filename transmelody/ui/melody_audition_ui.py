"""Simple local player for vocal, pronunciation cues and explicitly labeled old MIDI."""
from __future__ import annotations
from transmelody.paths import PROJECT_ROOT

import argparse
import json
from pathlib import Path
import tempfile
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import soundfile as sf

from transmelody.audio.audio_audition import PLAYBACK_RATE, WavePlayer, midi_notes_in_wav_time, render_audition
from transmelody.lyrics.pronunciation_labels import digest, validate_labels

ROOT = PROJECT_ROOT


class AuditionWindow:
    def __init__(self, root, candidate=None, *, player=None):
        self.root = root
        self.player = player or WavePlayer()
        cache = ROOT / ".cache/audition"
        cache.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="listen-", dir=cache)
        self.wav_path = Path(self.temporary.name) / "playback.wav"
        self.loaded_key = None
        self.data = None
        self.playing = False
        self.dragging = False
        self.midi_options = {}
        self.phrase = tk.StringVar()
        self.mode = tk.StringVar(value="人声 + 发音起点提示")
        self.position = tk.DoubleVar(value=0.)
        self.vocal_gain = tk.DoubleVar(value=.65)
        self.cue_gain = tk.DoubleVar(value=.22)
        self.loop = tk.BooleanVar(value=False)
        self.status = tk.StringVar(value="请选择一个 *_candidate.json")
        self.clock = tk.StringVar(value="原 WAV 00:00.000")
        root.title("旋律 / 发音边界试听")
        root.geometry("860x410")
        top = ttk.Frame(root, padding=12)
        top.pack(fill="both", expand=True)
        ttk.Button(top, text="打开边界候选 JSON", command=self.guard(self.browse)).pack(anchor="w")
        ttk.Button(top, text="载入其他 / 实验 MIDI", command=self.guard(self.browse_midi)).pack(anchor="w")
        ttk.Label(top, text="发音提示音只表示起点，不代表音高；旋律试听请选择已载入的 MIDI。", wraplength=810).pack(anchor="w", pady=10)
        row = ttk.Frame(top)
        row.pack(fill="x", pady=5)
        ttk.Label(row, text="范围：").pack(side="left")
        self.phrase_box = ttk.Combobox(row, textvariable=self.phrase, state="readonly", width=25)
        self.phrase_box.pack(side="left", padx=5)
        self.phrase_box.bind("<<ComboboxSelected>>", lambda _: self.change_range())
        ttk.Button(row, text="上一句", command=lambda: self.step(-1)).pack(side="left")
        ttk.Button(row, text="下一句", command=lambda: self.step(1)).pack(side="left", padx=5)
        row = ttk.Frame(top)
        row.pack(fill="x", pady=5)
        ttk.Label(row, text="试听：").pack(side="left")
        self.mode_box = ttk.Combobox(row, textvariable=self.mode, state="readonly", width=43)
        self.mode_box.pack(side="left", padx=5)
        self.mode_box.bind("<<ComboboxSelected>>", lambda _: self.change_mode())
        ttk.Checkbutton(row, text="循环选中范围", variable=self.loop).pack(side="left", padx=10)
        self.slider = ttk.Scale(top, from_=0, to=1, variable=self.position)
        self.slider.pack(fill="x", pady=8)
        self.slider.bind("<ButtonPress-1>", lambda _: setattr(self, "dragging", True))
        self.slider.bind("<ButtonRelease-1>", self.guard(self.seek))
        ttk.Label(top, textvariable=self.clock).pack(anchor="w")
        row = ttk.Frame(top)
        row.pack(fill="x", pady=8)
        for title, action in (("播放 / 继续", self.play), ("暂停", self.pause), ("停止", self.stop)):
            ttk.Button(row, text=title, command=self.guard(action)).pack(side="left", padx=3)
        for title, variable in (("人声音量", self.vocal_gain), ("提示 / MIDI 音量", self.cue_gain)):
            ttk.Label(row, text=title).pack(side="left", padx=5)
            gain = ttk.Scale(row, from_=0, to=1, variable=variable, length=105)
            gain.pack(side="left")
            gain.bind("<ButtonRelease-1>", lambda _: self.change_mode())
        ttk.Label(top, textvariable=self.status, wraplength=810).pack(fill="x", pady=8)
        root.protocol("WM_DELETE_WINDOW", self.close)
        self.timer = root.after(100, self.tick)
        if candidate:
            self.load(candidate)

    def guard(self, action):
        def run(*args):
            try:
                action(*args)
            except Exception as error:
                self.playing = False
                messagebox.showerror("试听工具", str(error), parent=self.root)
        return run

    def browse(self):
        path = filedialog.askopenfilename(initialdir=ROOT / "output/lyric_alignment", filetypes=[("边界候选", "*.json")])
        if path:
            self.load(Path(path))

    def load(self, path):
        data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        validate_labels(data)
        source = Path(data["audio_path"])
        if digest(source) != data["audio_sha256"]:
            raise ValueError("原音频发生变化，边界可能已错位，请重新对齐。")
        wave, sr = sf.read(source, dtype="float32", always_2d=True)
        self.stop()
        self.player.close()
        self.data, self.wave, self.sr = data, wave.mean(axis=1), sr
        self.midi_options = {}
        sid = data["song_id"]
        warnings = []
        for label, midi in (("旧预测 MIDI（不是本轮结果）", ROOT / f"test_output/{sid}_predicted.mid"),
                            ("手工校对 MIDI", ROOT / f"dataset/melody_dataset/vocal_mid/{sid}_vocal.mid"),
                            ("音乐起点实验 MIDI", ROOT / f"output/musical_event_experiment/sample/{sid}_predicted.mid"),
                            ("联合边界实验 MIDI", ROOT / f"output/musical_event_experiment/sample_joint/{sid}_predicted.mid"),
                            ("音高上下文实验 MIDI", ROOT / f"output/pitch_context_experiment/sample/{sid}_predicted.mid"),
                            ("音高与边界联合 MIDI", ROOT / f"output/joint_pitch_experiment/sample/{sid}_predicted.mid"),
                            ("完整音符工作流 MIDI", ROOT / f"output/note_event_experiment/sample/{sid}_predicted.mid")):
            if midi.exists():
                try:
                    self.midi_options[label] = midi_notes_in_wav_time(midi)
                except ValueError as error:
                    warnings.append(f"{label}不可用：{error}")
        modes = ["原人声", "人声 + 发音起点提示", "仅发音起点提示"]
        for label in self.midi_options:
            modes.extend([label, "人声 + " + label])
        self.mode_box.configure(values=modes)
        self.mode.set("人声 + 发音起点提示")
        self.phrase_box.configure(values=["全曲"] + [f"第 {i + 1:02d} 句" for i in range(len(data["phrases"]))])
        self.phrase_box.current(1)
        self.change_range()
        self.status.set(f"曲目 {sid}：{len(data['units'])} 个发音单位；点击在单位起点响。MIDI 使用简易合成音色，按自己的 tempo map + WAV START 对齐。" + " ".join(warnings))

    def browse_midi(self):
        if not self.data:
            raise ValueError("请先选择歌曲的边界候选，以确定对应的原人声。")
        path = filedialog.askopenfilename(initialdir=ROOT / "output", filetypes=[("MIDI", "*.mid *.midi")])
        if path:
            notes = midi_notes_in_wav_time(Path(path))
            label = f"载入 MIDI：{Path(path).name}"
            self.midi_options[label] = notes
            modes = ["原人声", "人声 + 发音起点提示", "仅发音起点提示"]
            for name in self.midi_options:
                modes.extend([name, "人声 + " + name])
            self.mode_box.configure(values=modes)
            self.mode.set("人声 + " + label)
            self.change_mode()
            self.status.set(f"已载入 {path}。请确认 MIDI 与当前人声属于同一首歌；使用 MIDI 内的 WAV START 对齐。")

    def bounds(self):
        index = self.phrase_box.current()
        if index <= 0:
            return 0., len(self.wave) / self.sr
        phrase = self.data["phrases"][index - 1]
        return max(0., phrase["start"] - .15), min(len(self.wave) / self.sr, phrase["end"] + .15)

    def change_range(self):
        self.stop()
        self.loaded_key = None
        if self.data:
            start, end = self.bounds()
            self.slider.configure(to=end - start)
            self.clock.set(f"原 WAV {start:.3f} 秒 — {end:.3f} 秒")

    def step(self, direction):
        if self.data:
            self.phrase_box.current(min(len(self.data["phrases"]), max(0, self.phrase_box.current() + direction)))
            self.change_range()

    def change_mode(self):
        was_playing = self.playing
        if self.player.opened and self.playing:
            self.position.set(self.player.position())
        self.player.stop()
        self.playing = False
        self.loaded_key = None
        if was_playing:
            self.guard(self.play)()

    def play(self):
        if not self.data:
            raise ValueError("请先打开边界候选文件。")
        start, end = self.bounds()
        mode = self.mode.get()
        key = (start, end, mode, self.vocal_gain.get(), self.cue_gain.get())
        if self.loaded_key != key:
            self.player.close()
            self.loaded_key = None
            self.root.update_idletasks()
            notes = self.midi_options.get(mode.removeprefix("人声 + "), ())
            rendered = render_audition(self.wave, self.sr, start, end, units=self.data["units"], notes=notes,
                vocal_gain=self.vocal_gain.get() if mode == "原人声" or mode.startswith("人声 + ") else 0,
                cue_gain=self.cue_gain.get(), clicks="发音起点提示" in mode)
            sf.write(self.wav_path, rendered, PLAYBACK_RATE, subtype="PCM_16")
            self.player.open(self.wav_path)
            self.loaded_key = key
        position = self.position.get()
        if position >= end - start - .002:
            position = 0.
            self.position.set(0.)
        self.player.play(position)
        self.playing = True

    def pause(self):
        if self.playing:
            self.player.pause()
            self.position.set(self.player.position())
            self.playing = False

    def stop(self):
        self.player.stop()
        self.playing = False
        self.position.set(0.)

    def seek(self, _=None):
        self.dragging = False
        if self.playing:
            self.play()

    def tick(self):
        try:
            if self.data:
                start, end = self.bounds()
                if self.playing and not self.dragging:
                    self.position.set(min(self.player.position(), end - start))
                    if self.player.mode() == "stopped":
                        self.playing = False
                        if self.loop.get():
                            self.position.set(0.)
                            self.play()
                self.clock.set(f"原 WAV {start + self.position.get():.3f} 秒  |  当前范围 {start:.3f} — {end:.3f} 秒")
        except Exception as error:
            self.playing = False
            self.status.set(str(error))
        self.timer = self.root.after(100, self.tick)

    def close(self):
        self.root.after_cancel(self.timer)
        try:
            self.player.close()
        finally:
            self.temporary.cleanup()
            self.root.destroy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", nargs="?", type=Path)
    args = parser.parse_args()
    candidate = args.candidate
    if candidate is None:
        files = sorted((ROOT / "output/lyric_alignment").glob("*_candidate.json"), key=lambda p: p.stat().st_mtime_ns, reverse=True)
        candidate = files[0] if files else None
    root = tk.Tk()
    try:
        AuditionWindow(root, candidate)
    except Exception as error:
        messagebox.showerror("无法启动试听", str(error), parent=root)
        root.destroy()
        return
    root.mainloop()


if __name__ == "__main__":
    main()
