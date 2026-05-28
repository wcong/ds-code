from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Footer, Header, Input, Static

from ds_code.runtime.api import RuntimeApi
from ds_code.runtime.models import PromptRequest


class DsCodeApp(App):
    CSS = """
    #status {
        margin: 1 2;
        text-style: bold;
    }

    #prompt {
        margin: 0 2;
    }

    #controls {
        margin: 1 2;
        height: 3;
    }

    #log {
        margin: 1 2;
        height: 12;
        border: solid;
        padding: 1;
    }
    """

    def __init__(self, runtime: RuntimeApi) -> None:
        super().__init__()
        self._runtime = runtime
        self._log: list[str] = []

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical():
            yield Static("DS Code skeleton is running.", id="status")
            yield Input(placeholder="Enter prompt and press Enter", id="prompt")
            with Horizontal(id="controls"):
                yield Button("Create Thread", id="create-thread")
                yield Button("List Threads", id="list-threads")
            yield Static("", id="log")
        yield Footer()

    def _append_log(self, message: str) -> None:
        self._log.append(message)
        log = self.query_one("#log", Static)
        log.update("\n".join(self._log[-12:]))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "create-thread":
            response = self._runtime.create_thread()
            self._append_log(f"thread created: {response.thread_id}")
        elif event.button.id == "list-threads":
            threads = self._runtime.list_threads()
            if not threads:
                self._append_log("no threads")
            else:
                for thread in threads:
                    name = thread.name or "(unnamed)"
                    self._append_log(f"thread {thread.id} {name} {thread.status}")

    def on_input_submitted(self, event: Input.Submitted) -> None:
        prompt = event.value.strip()
        if not prompt:
            return
        event.input.value = ""
        response = self._runtime.handle_prompt(PromptRequest(prompt=prompt))
        self._append_log(f"prompt: {prompt}")
        self._append_log(f"response: {response.output}")
