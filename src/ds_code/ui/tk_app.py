import tkinter as tk
from tkinter import ttk

from ds_code.runtime.api import RuntimeApi
from ds_code.runtime.models import PromptRequest


class DsCodeWindow:
    def __init__(self, runtime: RuntimeApi) -> None:
        self._runtime = runtime
        self._root = tk.Tk()
        self._root.title("DS Code")
        self._root.geometry("800x500")
        self._build_menu()

        header = ttk.Label(self._root, text="DS Code skeleton is running.")
        header.pack(anchor="w", padx=12, pady=(12, 6))

        prompt_frame = ttk.Frame(self._root)
        prompt_frame.pack(fill="x", padx=12)
        self._prompt = ttk.Entry(prompt_frame)
        self._prompt.pack(side="left", fill="x", expand=True)
        send_btn = ttk.Button(prompt_frame, text="Send", command=self._on_send)
        send_btn.pack(side="left", padx=(8, 0))

        controls = ttk.Frame(self._root)
        controls.pack(fill="x", padx=12, pady=(8, 0))
        ttk.Button(controls, text="Create Thread", command=self._on_create_thread).pack(
            side="left"
        )
        ttk.Button(controls, text="List Threads", command=self._on_list_threads).pack(
            side="left", padx=(8, 0)
        )

        self._log = tk.Text(self._root, height=18, state="disabled")
        self._log.pack(fill="both", expand=True, padx=12, pady=12)

        self._status = ttk.Label(self._root, text="Ready", anchor="w")
        self._status.pack(fill="x", padx=12, pady=(0, 8))

        self._prompt.bind("<Return>", self._on_send_event)
        self._root.bind_all("<Command-q>", lambda _e: self._root.quit())
        self._root.bind_all("<Command-n>", lambda _e: self._on_create_thread())
        self._root.bind_all("<Command-l>", lambda _e: self._on_list_threads())

    def _build_menu(self) -> None:
        menubar = tk.Menu(self._root)

        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Exit", command=self._root.quit, accelerator="Cmd+Q")
        menubar.add_cascade(label="File", menu=file_menu)

        thread_menu = tk.Menu(menubar, tearoff=0)
        thread_menu.add_command(
            label="Create", command=self._on_create_thread, accelerator="Cmd+N"
        )
        thread_menu.add_command(
            label="List", command=self._on_list_threads, accelerator="Cmd+L"
        )
        menubar.add_cascade(label="Threads", menu=thread_menu)

        self._root.config(menu=menubar)

    def run(self) -> None:
        self._root.mainloop()

    def _append_log(self, message: str) -> None:
        self._log.configure(state="normal")
        self._log.insert("end", message + "\n")
        self._log.see("end")
        self._log.configure(state="disabled")
        self._status.configure(text=message)

    def _on_send_event(self, _event: object) -> None:
        self._on_send()

    def _on_send(self) -> None:
        prompt = self._prompt.get().strip()
        if not prompt:
            return
        self._prompt.delete(0, "end")
        response = self._runtime.handle_prompt(PromptRequest(prompt=prompt))
        self._append_log(f"prompt: {prompt}")
        self._append_log(f"response: {response.output}")

    def _on_create_thread(self) -> None:
        response = self._runtime.create_thread()
        self._append_log(f"thread created: {response.thread_id}")

    def _on_list_threads(self) -> None:
        threads = self._runtime.list_threads()
        if not threads:
            self._append_log("no threads")
            return
        for thread in threads:
            name = thread.name or "(unnamed)"
            self._append_log(f"thread {thread.id} {name} {thread.status}")
