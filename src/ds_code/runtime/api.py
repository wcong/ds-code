import uuid
from typing import Optional

from ds_code.core.engine import CoreEngine
from ds_code.llm.models import ProviderKind
from ds_code.execpolicy import AskForApproval
from ds_code.runtime.models import PromptRequest, PromptResponse, RuntimeConfig, ThreadResponse
from ds_code.runtime.prompt_router import parse_prompt
from ds_code.runtime.approvals import ApprovalStore
from ds_code.tools.execution import ToolCall
from ds_code.tasks.manager import TaskManager


class RuntimeApi:
    def __init__(
        self,
        core: CoreEngine,
        tasks: TaskManager,
        config: Optional[RuntimeConfig] = None,
    ) -> None:
        self._core = core
        self._tasks = tasks
        self._config = config or RuntimeConfig()
        self._approvals = ApprovalStore()

    def handle_prompt(self, req: PromptRequest) -> PromptResponse:
        route = parse_prompt(req.prompt)
        if route.call is not None:
            result = self.invoke_tool(route.call, AskForApproval.UnlessTrusted, ".")
            return PromptResponse(
                output=str(result),
                model=self._config.model,
                events=result.get("events", []),
            )
        requested_model = req.model or self._config.model
        provider = req.provider or self._config.provider
        selection = self._core.resolve_model(requested_model, provider)
        resolved_model = selection.resolved.id
        response_id = f"resp-{uuid.uuid4()}"

        payload = {
            "provider": provider.as_str()
            if isinstance(provider, ProviderKind)
            else str(provider),
            "model": resolved_model,
            "prompt": req.prompt,
            "telemetry": self._config.telemetry,
            "base_url": self._config.base_url,
            "has_api_key": bool(self._config.api_key and self._config.api_key.strip()),
            "approval_policy": self._config.approval_policy,
            "sandbox_mode": self._config.sandbox_mode,
        }
        if req.thread_id:
            self._core.touch_thread_message(req.thread_id, req.prompt)
        return PromptResponse(
            output=str(payload),
            model=resolved_model,
            events=[
                {"type": "response_start", "response_id": response_id},
                {
                    "type": "response_delta",
                    "response_id": response_id,
                    "delta": "model-selected",
                    "channel": "text",
                },
                {"type": "response_end", "response_id": response_id},
            ],
        )

    def submit_task(self, name: str, payload: str):
        return self._tasks.submit(name, payload)

    def create_thread(self) -> ThreadResponse:
        new = self._core.create_thread(self._config.provider.as_str(), "auto")
        return ThreadResponse(
            status="created",
            thread_id=new.thread.id,
            model=new.model,
            model_provider=new.model_provider,
        )

    def resume_thread(self, thread_id: str) -> ThreadResponse:
        resumed = self._core.resume_thread(thread_id, self._config.provider.as_str(), "auto")
        if resumed is None:
            return ThreadResponse(status="missing", thread_id=thread_id, data={"error": "thread not found"})
        return ThreadResponse(
            status="resumed",
            thread_id=resumed.thread.id,
            model=resumed.model,
            model_provider=resumed.model_provider,
        )

    def list_threads(self) -> list:
        return self._core.list_threads()

    def invoke_tool(
        self,
        call: ToolCall,
        approval_mode: AskForApproval = AskForApproval.UnlessTrusted,
        cwd: str = ".",
    ) -> dict:
        result = self._core.invoke_tool(call, approval_mode, cwd)
        approval_id = result.get("approval_id")
        if approval_id:
            self._approvals.record(approval_id, result.get("status", "unknown"), result.get("reason"))
        return result

    def review_approval(self, approval_id: str, decision: str, remember: bool = False) -> dict:
        record = self._approvals.get(approval_id)
        if record is None:
            return {"ok": False, "error": "approval not found"}
        record.status = decision
        if remember:
            self._core._exec_policy.remember_session_approval(approval_id)
        return {"ok": True, "approval_id": approval_id, "decision": decision}
