from __future__ import annotations

import traceback
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from midi_grid_quantizer import GRID_STEPS, QuantizeConfig, quantize_midi_file


class MidiQuantizerApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Melody MIDI Grid Quantizer")
        self.geometry("760x560")
        self.minsize(700, 520)

        self.input_path = tk.StringVar()
        self.output_path = tk.StringVar()
        self.bpm = tk.StringVar(value="177")
        self.replace_tempo = tk.BooleanVar(value=False)
        self.grid = tk.StringVar(value="1/16")
        self.quantize_ends = tk.BooleanVar(value=True)
        self.repair_overlaps = tk.BooleanVar(value=True)
        self.bridge_short_gaps = tk.BooleanVar(value=True)
        self.maximum_gap_steps = tk.IntVar(value=1)
        self.complete_final_bar = tk.BooleanVar(value=True)
        self.minimum_length_steps = tk.IntVar(value=1)
        self.status = tk.StringVar(value="请选择 Melodyne 导出的 MIDI 文件。")

        self._build_ui()

    def _build_ui(self) -> None:
        root = ttk.Frame(self, padding=18)
        root.pack(fill="both", expand=True)
        root.columnconfigure(1, weight=1)

        ttk.Label(
            root,
            text="Melody MIDI Grid Quantizer",
            font=("Segoe UI", 18, "bold"),
        ).grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 18))

        ttk.Label(root, text="输入 MIDI").grid(row=1, column=0, sticky="w")
        ttk.Entry(root, textvariable=self.input_path).grid(
            row=1, column=1, sticky="ew", padx=10
        )
        ttk.Button(root, text="选择…", command=self.choose_input).grid(
            row=1, column=2
        )

        ttk.Label(root, text="输出 MIDI").grid(
            row=2, column=0, sticky="w", pady=(10, 0)
        )
        ttk.Entry(root, textvariable=self.output_path).grid(
            row=2, column=1, sticky="ew", padx=10, pady=(10, 0)
        )
        ttk.Button(root, text="选择…", command=self.choose_output).grid(
            row=2, column=2, pady=(10, 0)
        )

        settings = ttk.LabelFrame(root, text="量化设置", padding=14)
        settings.grid(
            row=3,
            column=0,
            columnspan=3,
            sticky="ew",
            pady=18,
        )

        ttk.Checkbutton(
            settings,
            text="替换 Tempo",
            variable=self.replace_tempo,
        ).grid(row=0, column=0, sticky="w")
        ttk.Entry(settings, textvariable=self.bpm, width=12).grid(
            row=0, column=1, sticky="w", padx=(8, 28)
        )

        ttk.Label(settings, text="吸附网格").grid(row=0, column=2, sticky="w")
        ttk.Combobox(
            settings,
            textvariable=self.grid,
            values=list(GRID_STEPS),
            width=10,
            state="readonly",
        ).grid(row=0, column=3, sticky="w", padx=8)

        ttk.Label(settings, text="最短音符").grid(
            row=1, column=0, sticky="w", pady=(14, 0)
        )
        ttk.Spinbox(
            settings,
            from_=1,
            to=16,
            textvariable=self.minimum_length_steps,
            width=10,
        ).grid(row=1, column=1, sticky="w", padx=(8, 28), pady=(14, 0))
        ttk.Label(settings, text="个网格").grid(
            row=1, column=2, sticky="w", pady=(14, 0)
        )
        ttk.Label(
            settings,
            text="力度规则：低于 40 删除，其余统一为 100",
        ).grid(row=1, column=3, sticky="w", pady=(14, 0))

        ttk.Checkbutton(
            settings,
            text="起音和结束点都吸附（取消时保持原音长）",
            variable=self.quantize_ends,
        ).grid(row=2, column=0, columnspan=4, sticky="w", pady=(14, 0))
        ttk.Checkbutton(
            settings,
            text="修复同一旋律通道中的音符重叠",
            variable=self.repair_overlaps,
        ).grid(row=3, column=0, columnspan=4, sticky="w", pady=(8, 0))
        ttk.Checkbutton(
            settings,
            text="桥接短间隙",
            variable=self.bridge_short_gaps,
        ).grid(row=4, column=0, sticky="w", pady=(8, 0))
        ttk.Spinbox(
            settings,
            from_=0,
            to=8,
            textvariable=self.maximum_gap_steps,
            width=10,
        ).grid(row=4, column=1, sticky="w", padx=(8, 8), pady=(8, 0))
        ttk.Label(settings, text="个网格以内").grid(
            row=4, column=2, columnspan=2, sticky="w", pady=(8, 0)
        )
        ttk.Checkbutton(
            settings,
            text="将 MIDI 终点补齐到完整小节（沿用原拍号）",
            variable=self.complete_final_bar,
        ).grid(row=5, column=0, columnspan=4, sticky="w", pady=(8, 0))

        ttk.Button(
            root,
            text="生成量化 MIDI",
            command=self.run_quantize,
        ).grid(row=4, column=0, columnspan=3, sticky="ew", ipady=8)

        ttk.Label(root, textvariable=self.status).grid(
            row=5,
            column=0,
            columnspan=3,
            sticky="w",
            pady=(14, 6),
        )

        self.report = tk.Text(
            root,
            height=10,
            wrap="word",
            state="disabled",
            font=("Consolas", 10),
        )
        self.report.grid(row=6, column=0, columnspan=3, sticky="nsew")
        root.rowconfigure(6, weight=1)

    def choose_input(self) -> None:
        selected = filedialog.askopenfilename(
            title="选择 MIDI",
            filetypes=[("MIDI files", "*.mid *.midi"), ("All files", "*.*")],
        )
        if not selected:
            return
        input_path = Path(selected)
        self.input_path.set(str(input_path))
        self.output_path.set(
            str(input_path.with_name(f"{input_path.stem}_quantized.mid"))
        )

    def choose_output(self) -> None:
        initial = Path(self.output_path.get()) if self.output_path.get() else None
        selected = filedialog.asksaveasfilename(
            title="保存量化 MIDI",
            defaultextension=".mid",
            initialdir=str(initial.parent) if initial else None,
            initialfile=initial.name if initial else "melody_quantized.mid",
            filetypes=[("MIDI files", "*.mid"), ("All files", "*.*")],
        )
        if selected:
            self.output_path.set(selected)

    def set_report(self, text: str) -> None:
        self.report.configure(state="normal")
        self.report.delete("1.0", "end")
        self.report.insert("1.0", text)
        self.report.configure(state="disabled")

    def run_quantize(self) -> None:
        try:
            input_text = self.input_path.get().strip()
            output_text = self.output_path.get().strip()
            if not input_text:
                raise ValueError("请选择输入 MIDI 文件。")
            if not output_text:
                raise ValueError("请选择输出文件。")

            input_path = Path(input_text)
            output_path = Path(output_text)
            if not input_path.is_file():
                raise ValueError("请选择有效的输入 MIDI 文件。")

            config = QuantizeConfig(
                bpm=(
                    float(self.bpm.get())
                    if self.replace_tempo.get()
                    else None
                ),
                grid=self.grid.get(),
                quantize_note_ends=self.quantize_ends.get(),
                minimum_length_steps=int(self.minimum_length_steps.get()),
                repair_monophonic_overlaps=self.repair_overlaps.get(),
                bridge_short_gaps=self.bridge_short_gaps.get(),
                maximum_gap_steps=int(self.maximum_gap_steps.get()),
                complete_final_bar=self.complete_final_bar.get(),
            )
            self.status.set("正在量化…")
            self.update_idletasks()
            stats = quantize_midi_file(input_path, output_path, config)
            report = (
                f"输出：{output_path}\n\n"
                f"音符数量：{stats.notes}\n"
                f"输入音符：{stats.input_notes}\n"
                f"删除低力度音符：{stats.deleted_low_velocity_notes}\n"
                f"移动起音：{stats.moved_onsets}\n"
                f"移动结束点：{stats.moved_ends}\n"
                f"修复重叠：{stats.repaired_overlaps}\n"
                f"桥接短间隙：{stats.bridged_gaps}\n"
                f"未配对 note_on / note_off："
                f"{stats.unmatched_note_ons} / {stats.unmatched_note_offs}\n"
                f"PPQ：{stats.source_ticks_per_beat} → "
                f"{stats.ticks_per_beat}\n"
                f"网格宽度：{stats.grid_step_ticks:g} ticks\n"
                f"拍号：{stats.time_signature[0]}/"
                f"{stats.time_signature[1]}\n"
                f"Tempo：{stats.input_bpm:.4f} → "
                f"{stats.output_bpm:.4f} BPM"
                f"{'（已替换）' if stats.tempo_replaced else '（已保留）'}\n"
                f"最大修正：{stats.max_adjustment_ticks} ticks "
                f"({stats.max_adjustment_ms:.2f} ms)\n"
                f"MIDI 终点：{stats.original_end_tick} → "
                f"{stats.output_end_tick} ticks\n"
                f"尾部补齐：{stats.trailing_padding_ticks} ticks"
            )
            self.set_report(report)
            self.status.set("完成。原始 MIDI 未被修改。")
            messagebox.showinfo("完成", f"已生成：\n{output_path}")
        except Exception as error:
            self.status.set("处理失败。")
            self.set_report(f"{error}\n\n{traceback.format_exc()}")
            messagebox.showerror("处理失败", str(error))


if __name__ == "__main__":
    MidiQuantizerApp().mainloop()
