"""Local waveform/spectrogram review of pronunciation UNIT candidates."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import threading
import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np
import soundfile as sf
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".cache/matplotlib"))
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from lyric_alignment import ROOT, write_new
from pronunciation_labels import digest, validate_labels


def merge_regions(regions: list[list[float]]) -> list[list[float]]:
    result = []
    for start, end in sorted(regions):
        if result and start <= result[-1][1]:
            result[-1][1] = max(result[-1][1], end)
        else:
            result.append([start, end])
    return result


class ReviewWindow:
    def __init__(self, root: tk.Tk, path: Path, dataset: Path):
        self.root, self.path, self.dataset = root, path, dataset
        self.data = json.loads(path.read_text(encoding="utf-8-sig"))
        validate_labels(self.data)
        if self.data["status"] != "candidate":
            raise ValueError("请打开候选/校对草稿，不要直接编辑已进入训练集的正式标签。")
        audio = Path(self.data["audio_path"])
        if digest(audio) != self.data["audio_sha256"]:
            raise ValueError("音频已更改，请重新对齐；不能在错位音频上校对。")
        self.wave, self.sr = sf.read(audio, dtype="float32", always_2d=True)
        self.wave = self.wave.mean(axis=1)
        self.phrase = tk.IntVar(value=0)
        self.start, self.end, self.label, self.reviewer = (tk.StringVar() for _ in range(4))
        self.cursor = None
        self.checked = {i for i, p in enumerate(self.data["phrases"])
            if any(a <= p["start"] and b >= p["end"] for a, b in self.data["reviewed_regions"])}
        root.title(f"发音边界校对 — {self.data['song_id']}（不是 MIDI 切音）")
        root.geometry("1250x860")
        top = ttk.Frame(root, padding=8)
        top.pack(fill="x")
        ttk.Label(top, text="歌词句编号（从 0 开始）").pack(side="left")
        picker = ttk.Combobox(top, textvariable=self.phrase, values=list(range(len(self.data["phrases"]))), width=6, state="readonly")
        picker.pack(side="left")
        picker.bind("<<ComboboxSelected>>", lambda _: self.refresh())
        for title, action in (("试听本句", self.play_phrase), ("试听选中发音", self.play_unit),
                              ("确认本句已校对", self.confirm_phrase), ("保存校对草稿", self.save),
                              ("导出已确认标签", self.accept)):
            ttk.Button(top, text=title, command=self.guard(action)).pack(side="left", padx=3)
        info = ttk.Frame(root, padding=8)
        info.pack(fill="x")
        ttk.Label(info, text="校对者：").pack(side="left")
        ttk.Entry(info, textvariable=self.reviewer, width=16).pack(side="left")
        ttk.Label(info, text="只确认发音单位边界；phones 字段仍是模型候选。一个字内可包含多个 MIDI 音符。").pack(side="left", padx=10)
        self.description = ttk.Label(root, wraplength=1180)
        self.description.pack(fill="x", padx=8)
        self.figure = Figure(figsize=(11, 4.1), dpi=100)
        self.axes = self.figure.subplots(2, 1, sharex=True, gridspec_kw={"height_ratios": [1, 3]})
        self.canvas = FigureCanvasTkAgg(self.figure, master=root)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.canvas.mpl_connect("button_press_event", self.clicked)
        self.table = ttk.Treeview(root, columns=("start", "end", "label"), show="headings", height=7, selectmode="browse")
        for key, title in (("start", "起点 / WAV 秒"), ("end", "终点 / WAV 秒"), ("label", "发音（可修正）")):
            self.table.heading(key, text=title)
        self.table.pack(fill="x", padx=8)
        self.table.tag_configure("short", foreground="#b34700")
        self.table.bind("<<TreeviewSelect>>", self.select)
        edit = ttk.Frame(root, padding=8)
        edit.pack(fill="x")
        for title, variable in (("起点", self.start), ("终点", self.end), ("发音", self.label)):
            ttk.Label(edit, text=title).pack(side="left")
            ttk.Entry(edit, textvariable=variable, width=14).pack(side="left", padx=3)
        for title, action in (("光标→起点", lambda: self.set_cursor(self.start)),
                              ("光标→终点", lambda: self.set_cursor(self.end)),
                              ("应用修改", self.update_unit), ("与下一单位合并", self.merge_unit),
                              ("在光标处分开", self.split_unit)):
            ttk.Button(edit, text=title, command=self.guard(action)).pack(side="left", padx=3)
        self.status = ttk.Label(root, text="点击声谱定位；修改时间后应用。修改本句会撤销该句的确认状态。")
        self.status.pack(fill="x", padx=8, pady=5)
        self.refresh()

    def guard(self, action):
        def run():
            try:
                action()
            except Exception as error:
                messagebox.showerror("边界校对", str(error), parent=self.root)
        return run

    def selected(self):
        selection = self.table.selection()
        if not selection:
            raise ValueError("请先选一个发音单位。")
        return int(selection[0])

    def refresh(self):
        p = self.data["phrases"][self.phrase.get()]
        self.description.configure(text=f"{p['text']}   |   {'已确认' if self.phrase.get() in self.checked else '待校对'}   |   WAV {p['start']:.3f}–{p['end']:.3f} 秒")
        self.table.delete(*self.table.get_children())
        for i, unit in enumerate(self.data["units"]):
            if unit["phrase"] == self.phrase.get():
                self.table.insert("", "end", iid=str(i), values=(f"{unit['start']:.4f}", f"{unit['end']:.4f}", unit["label"]),
                    tags=("short",) if unit["end"] - unit["start"] < .020 else ())
        self.draw()

    def draw(self):
        p = self.data["phrases"][self.phrase.get()]
        left, right = round(p["start"] * self.sr), round(p["end"] * self.sr)
        wave = self.wave[left:right]
        for ax in self.axes:
            ax.clear()
        stride = max(1, len(wave) // 8000)
        self.axes[0].plot((left + np.arange(0, len(wave), stride)) / self.sr, wave[::stride], linewidth=.6)
        if len(wave) >= 1024:
            self.axes[1].specgram(wave, NFFT=1024, Fs=self.sr, noverlap=768, cmap="magma",
                xextent=(left / self.sr, right / self.sr), vmin=-110, vmax=-20)
        self.axes[1].set_ylim(0, min(6000, self.sr / 2))
        self.axes[1].set_ylabel("Hz")
        self.axes[1].set_xlabel("Original WAV seconds")
        selected = self.table.selection()
        for i, u in enumerate(self.data["units"]):
            if u["phrase"] != self.phrase.get():
                continue
            self.axes[1].axvline(u["start"], color="cyan", linewidth=.8)
            self.axes[1].axvline(u["end"], color="cyan", linewidth=.5, linestyle=":")
            self.axes[0].text(u["start"], .7, u["label"], transform=self.axes[0].get_xaxis_transform(), fontsize=7)
            if str(i) in selected:
                self.axes[1].axvspan(u["start"], u["end"], color="cyan", alpha=.18)
        if self.cursor is not None:
            self.axes[1].axvline(self.cursor, color="lime")
        self.axes[1].set_xlim(p["start"], p["end"])
        self.figure.tight_layout()
        self.canvas.draw_idle()

    def select(self, _=None):
        if not self.table.selection():
            return
        u = self.data["units"][self.selected()]
        self.start.set(f"{u['start']:.6f}")
        self.end.set(f"{u['end']:.6f}")
        self.label.set(u["label"])
        self.draw()

    def clicked(self, event):
        if event.xdata is not None:
            self.cursor = float(event.xdata)
            self.status.configure(text=f"光标 WAV {self.cursor:.4f} 秒")
            self.draw()

    def set_cursor(self, variable):
        if self.cursor is None:
            raise ValueError("先点击声谱选择一个时间。")
        variable.set(f"{self.cursor:.6f}")

    def changed(self, previous):
        try:
            validate_labels(self.data)
            p = self.data["phrases"][self.phrase.get()]
            if any(u["start"] < p["start"] or u["end"] > p["end"] for u in self.data["units"] if u["phrase"] == self.phrase.get()):
                raise ValueError("发音边界不能超出本句范围；粗对齐错误请修改歌词句时间并重新运行对齐。")
        except Exception:
            self.data["units"] = previous
            raise
        self.checked.discard(self.phrase.get())
        self.sync_review()
        self.refresh()

    def update_unit(self):
        import copy
        previous = copy.deepcopy(self.data["units"])
        self.data["units"][self.selected()].update(start=float(self.start.get()), end=float(self.end.get()), label=self.label.get().strip())
        self.changed(previous)

    def merge_unit(self):
        import copy
        i = self.selected()
        units = self.data["units"]
        if i + 1 >= len(units) or units[i + 1]["phrase"] != units[i]["phrase"]:
            raise ValueError("本句没有下一个发音单位。")
        previous = copy.deepcopy(units)
        units[i]["end"] = units[i + 1]["end"]
        units[i]["label"] += " / " + units[i + 1]["label"]
        del units[i + 1]
        self.changed(previous)

    def split_unit(self):
        import copy
        i = self.selected()
        u = self.data["units"][i]
        if self.cursor is None or not u["start"] < self.cursor < u["end"]:
            raise ValueError("光标须位于选中单位内部。")
        previous = copy.deepcopy(self.data["units"])
        self.data["units"].insert(i + 1, {**u, "start": self.cursor})
        u["end"] = self.cursor
        self.changed(previous)

    def play(self, start, end):
        import winsound
        buffer = io.BytesIO()
        sf.write(buffer, self.wave[round(start * self.sr):round(end * self.sr)], self.sr, format="WAV", subtype="PCM_16")
        threading.Thread(target=winsound.PlaySound, args=(buffer.getvalue(), winsound.SND_MEMORY), daemon=True).start()

    def play_phrase(self):
        p = self.data["phrases"][self.phrase.get()]
        self.play(p["start"], p["end"])

    def play_unit(self):
        u = self.data["units"][self.selected()]
        self.play(max(0, u["start"] - .12), min(self.data["audio_duration"], u["end"] + .12))

    def sync_review(self):
        self.data["reviewed_regions"] = merge_regions([[p["start"], p["end"]] for i, p in enumerate(self.data["phrases"]) if i in self.checked])
        self.data["status"] = "candidate"

    def confirm_phrase(self):
        validate_labels(self.data)
        self.checked.add(self.phrase.get())
        self.sync_review()
        self.refresh()

    def save(self):
        self.sync_review()
        validate_labels(self.data)
        # Explicit save updates this draft only, using a sibling atomic replace.
        temporary = self.path.with_suffix(".saving.json")
        temporary.write_text(json.dumps(self.data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(self.path)
        self.status.configure(text=f"已保存草稿；已确认 {len(self.checked)} / {len(self.data['phrases'])} 句")

    def accept(self):
        if digest(Path(self.data["audio_path"])) != self.data["audio_sha256"]:
            raise ValueError("音频已更改，请重新对齐。")
        self.save()
        data = {**self.data, "status": "reviewed", "review_scope": "pronunciation_units",
            "reviewer": self.reviewer.get().strip(), "reviewed_at": datetime.now(timezone.utc).isoformat(),
            "candidate_sha256": digest(self.path)}
        validate_labels(data, require_reviewed=True)
        output = self.dataset / "pronunciation_labels" / f"{data['song_id']}.json"
        if output.exists():
            raise FileExistsError(f"已有正式标签 {output}。请先保留旧版本再移开它；不会静默覆盖。")
        write_new(output, data)
        messagebox.showinfo("完成", f"已导出确认过的 {len(self.checked)} 句；其他区域仍使用弱监督。\n{output}", parent=self.root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--dataset", type=Path, default=ROOT / "dataset/melody_dataset")
    args = parser.parse_args()
    root = tk.Tk()
    ReviewWindow(root, args.candidate, args.dataset)
    root.mainloop()


if __name__ == "__main__":
    main()
