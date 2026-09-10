from __future__ import annotations

import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from transmelody.audio.audio_folder_converter import convert_folder, discover_audio_files


class AudioConverterApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("WAV 44.1 kHz / 16-bit 原位转换")
        self.root.geometry("780x520")
        self.folder = tk.StringVar()
        self.recursive = tk.BooleanVar(value=True)
        self.messages: queue.Queue[tuple[str, object]] = queue.Queue()

        frame = ttk.Frame(root, padding=14)
        frame.pack(fill="both", expand=True)
        ttk.Label(
            frame,
            text="选择文件夹，将 WAV/FLAC 转换为 44.1 kHz / PCM16 WAV",
        ).pack(anchor="w")

        row = ttk.Frame(frame)
        row.pack(fill="x", pady=(12, 6))
        ttk.Entry(row, textvariable=self.folder).pack(
            side="left",
            fill="x",
            expand=True,
        )
        ttk.Button(row, text="选择文件夹", command=self.choose_folder).pack(
            side="left",
            padx=(8, 0),
        )

        ttk.Checkbutton(
            frame,
            text="递归处理子文件夹（匹配的 Instrumental/Vocals 使用共同增益）",
            variable=self.recursive,
        ).pack(anchor="w", pady=4)

        buttons = ttk.Frame(frame)
        buttons.pack(fill="x", pady=8)
        ttk.Button(buttons, text="扫描", command=self.scan).pack(side="left")
        self.convert_button = ttk.Button(
            buttons,
            text="一键原位转换",
            command=self.start_conversion,
        )
        self.convert_button.pack(side="left", padx=8)

        self.log = tk.Text(frame, wrap="word", state="disabled")
        self.log.pack(fill="both", expand=True, pady=(8, 0))
        self.root.after(100, self.poll_messages)

    def choose_folder(self) -> None:
        selected = filedialog.askdirectory()
        if selected:
            self.folder.set(selected)
            self.scan()

    def append(self, text: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", text + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def selected_folder(self) -> Path | None:
        path = Path(self.folder.get().strip())
        if not path.is_dir():
            messagebox.showerror("无效目录", "请先选择包含 WAV 的文件夹。")
            return None
        return path

    def scan(self) -> None:
        folder = self.selected_folder()
        if folder is None:
            return
        files = discover_audio_files(folder, recursive=self.recursive.get())
        self.append(f"扫描到 {len(files)} 个 WAV/FLAC：{folder}")
        for path in files:
            self.append(f"  {path}")

    def start_conversion(self) -> None:
        folder = self.selected_folder()
        if folder is None:
            return
        files = discover_audio_files(folder, recursive=self.recursive.get())
        if not files:
            messagebox.showwarning(
                "没有文件",
                "目录中没有找到 WAV 或 FLAC 文件。",
            )
            return
        confirmed = messagebox.askyesno(
            "确认原位替换",
            f"将处理 {len(files)} 个 WAV/FLAC。\n"
            "目标格式：44.1 kHz / PCM16。\n\n继续吗？",
        )
        if not confirmed:
            return
        self.convert_button.configure(state="disabled")
        threading.Thread(
            target=self.convert_worker,
            args=(folder, self.recursive.get()),
            daemon=True,
        ).start()

    def convert_worker(self, folder: Path, recursive: bool) -> None:
        try:
            result = convert_folder(
                folder,
                recursive=recursive,
                apply=True,
                log=lambda text: self.messages.put(("log", text)),
            )
            self.messages.put(("done", result))
        except Exception as exc:
            self.messages.put(("error", exc))

    def poll_messages(self) -> None:
        try:
            while True:
                kind, payload = self.messages.get_nowait()
                if kind == "log":
                    self.append(str(payload))
                elif kind == "done":
                    self.convert_button.configure(state="normal")
                    self.append(
                        f"完成：转换 {payload.converted}，"
                        f"跳过 {payload.skipped}。"
                    )
                    messagebox.showinfo("转换完成", "所有文件已验证并原位替换。")
                elif kind == "error":
                    self.convert_button.configure(state="normal")
                    self.append(f"错误：{payload}")
                    messagebox.showerror("转换失败", str(payload))
        except queue.Empty:
            pass
        self.root.after(100, self.poll_messages)


def main() -> None:
    root = tk.Tk()
    AudioConverterApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
