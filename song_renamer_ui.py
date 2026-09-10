from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from batch_song_renamer import DEFAULT_REGISTRY, apply_plan, build_plan


class SongRenamerApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("歌曲批量编号与记录")
        self.root.geometry("760x520")
        self.folder = tk.StringVar()
        self.plan = []

        frame = ttk.Frame(root, padding=14)
        frame.pack(fill="both", expand=True)
        ttk.Label(
            frame,
            text="按记录表下一个编号，批量重命名文件夹中的歌曲",
        ).pack(anchor="w")
        ttk.Label(
            frame,
            text="记录表：dataset/song_registry.xlsx（自动同步）",
        ).pack(anchor="w", pady=(3, 10))

        row = ttk.Frame(frame)
        row.pack(fill="x")
        ttk.Entry(row, textvariable=self.folder).pack(
            side="left",
            fill="x",
            expand=True,
        )
        ttk.Button(row, text="选择文件夹", command=self.choose_folder).pack(
            side="left",
            padx=(8, 0),
        )

        buttons = ttk.Frame(frame)
        buttons.pack(fill="x", pady=10)
        ttk.Button(buttons, text="生成预览", command=self.preview).pack(
            side="left"
        )
        self.apply_button = ttk.Button(
            buttons,
            text="确认重命名并写入记录表",
            command=self.apply,
            state="disabled",
        )
        self.apply_button.pack(side="left", padx=8)

        self.text = tk.Text(frame, wrap="none", state="disabled")
        self.text.pack(fill="both", expand=True)

    def set_text(self, value: str) -> None:
        self.text.configure(state="normal")
        self.text.delete("1.0", "end")
        self.text.insert("end", value)
        self.text.configure(state="disabled")

    def choose_folder(self) -> None:
        selected = filedialog.askdirectory()
        if selected:
            self.folder.set(selected)
            self.preview()

    def preview(self) -> None:
        try:
            self.plan = build_plan(Path(self.folder.get()), DEFAULT_REGISTRY)
            lines = [
                f"识别到 {len(self.plan)} 首",
                f"编号范围：{self.plan[0].song_id}–{self.plan[-1].song_id}",
                "",
            ]
            lines.extend(
                f"{item.source.name}  →  {item.destination.name}"
                for item in self.plan
            )
            self.set_text("\n".join(lines))
            self.apply_button.configure(state="normal")
        except Exception as exc:
            self.plan = []
            self.apply_button.configure(state="disabled")
            self.set_text(f"错误：{exc}")

    def apply(self) -> None:
        if not self.plan:
            return
        if not messagebox.askyesno(
            "确认批量重命名",
            f"将重命名 {len(self.plan)} 首歌曲并追加记录表。\n"
            "操作前请确认预览顺序正确。继续吗？",
        ):
            return
        try:
            apply_plan(self.plan, DEFAULT_REGISTRY)
            self.set_text(self.text.get("1.0", "end") + "\n完成。")
            self.apply_button.configure(state="disabled")
            messagebox.showinfo("完成", "歌曲已编号，记录表已更新。")
        except Exception as exc:
            messagebox.showerror("失败", str(exc))


def main() -> None:
    root = tk.Tk()
    SongRenamerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
