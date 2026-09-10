from __future__ import annotations
from transmelody.paths import PROJECT_ROOT

import queue
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk


WORKFLOW = 'transmelody.workflow.melody_queue_workflow'


class MelodyQueueApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Melody 数据循环工作流")
        self.root.geometry("820x560")
        self.messages: queue.Queue[tuple[str, object]] = queue.Queue()
        self.epochs = tk.IntVar(value=75)
        self.skip_train = tk.BooleanVar(value=False)

        frame = ttk.Frame(root, padding=14)
        frame.pack(fill="both", expand=True)
        ttk.Label(
            frame,
            text=(
                "original stem → test audio 队列 → 预测 → "
                "校对 MIDI → 训练集"
            ),
        ).pack(anchor="w")

        controls = ttk.Frame(frame)
        controls.pack(fill="x", pady=(12, 8))
        self.status_button = ttk.Button(
            controls,
            text="刷新状态",
            command=self.refresh_status,
        )
        self.status_button.pack(side="left")
        self.stage_button = ttk.Button(
            controls,
            text="导入 original stem",
            command=self.stage,
        )
        self.stage_button.pack(side="left", padx=8)
        self.next_button = ttk.Button(
            controls,
            text="运行下一轮",
            command=self.run_next,
        )
        self.next_button.pack(side="left")

        ttk.Label(controls, text="训练轮数").pack(side="left", padx=(20, 4))
        ttk.Spinbox(
            controls,
            from_=0,
            to=500,
            textvariable=self.epochs,
            width=6,
        ).pack(side="left")
        ttk.Checkbutton(
            controls,
            text="跳过训练，直接预测",
            variable=self.skip_train,
        ).pack(side="left", padx=12)

        ttk.Label(frame, text="训练轮数设为 0：跳过训练，检查校对文件后直接预测下一首。").pack(
            anchor="w", pady=(0, 8))

        self.log = tk.Text(frame, wrap="word", state="disabled")
        self.log.pack(fill="both", expand=True)
        self.root.after(100, self.poll_messages)
        self.refresh_status()

    def append(self, text: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", text + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def set_busy(self, busy: bool) -> None:
        state = "disabled" if busy else "normal"
        self.status_button.configure(state=state)
        self.stage_button.configure(state=state)
        self.next_button.configure(state=state)

    def start_command(self, args: list[str]) -> None:
        self.set_busy(True)
        self.append("")
        self.append("> " + " ".join(args))
        threading.Thread(
            target=self.command_worker,
            args=(args,),
            daemon=True,
        ).start()

    def command_worker(self, args: list[str]) -> None:
        try:
            process = subprocess.Popen(
                [sys.executable, "-B", "-m", WORKFLOW, *args],
                cwd=PROJECT_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            assert process.stdout is not None
            for line in process.stdout:
                self.messages.put(("log", line.rstrip()))
            return_code = process.wait()
            if return_code:
                raise RuntimeError(f"工作流退出码：{return_code}")
            self.messages.put(("done", args[0]))
        except Exception as exc:
            self.messages.put(("error", exc))

    def refresh_status(self) -> None:
        self.start_command(["status"])

    def stage(self) -> None:
        if not messagebox.askyesno(
            "确认导入队列",
            "把 original stem 中所有完整数字分轨移入 test audio，"
            "并把 Registry 状态改为“待预测”？",
        ):
            return
        self.start_command(["stage", "--apply"])

    def run_next(self) -> None:
        try:
            epochs = self.epochs.get()
            if epochs < 0:
                raise ValueError
        except (tk.TclError, ValueError):
            messagebox.showerror("训练轮数无效", "请输入不小于 0 的整数；0 表示跳过训练。")
            return
        args = ["next", "--epochs", str(epochs)]
        if self.skip_train.get():
            args.append("--skip-train")
        description = (
            "使用现有 checkpoint 直接预测"
            if self.skip_train.get() or epochs == 0
            else f"先训练 {epochs} 轮，再预测"
        )
        if not messagebox.askyesno(
            "确认运行下一轮",
            f"{description} test audio 中编号最小的一组。\n"
            "上一组校对 MIDI 未入库时，工作流会自动停止。\n\n继续吗？",
        ):
            return
        self.start_command(args)

    def poll_messages(self) -> None:
        try:
            while True:
                kind, payload = self.messages.get_nowait()
                if kind == "log":
                    self.append(str(payload))
                elif kind == "done":
                    self.set_busy(False)
                    self.append("完成。")
                elif kind == "error":
                    self.set_busy(False)
                    self.append(f"错误：{payload}")
                    messagebox.showerror("工作流失败", str(payload))
        except queue.Empty:
            pass
        self.root.after(100, self.poll_messages)


def main() -> None:
    root = tk.Tk()
    MelodyQueueApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
