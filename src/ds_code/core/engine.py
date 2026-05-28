from typing import Optional
import uuid

from ds_code.core.jobs import JobManager, JobRecord, JobHistoryEntry
from ds_code.core.threads import NewThread, Thread, ThreadManager
from ds_code.execpolicy import AskForApproval, ExecPolicyContext, ExecPolicyEngine
from ds_code.llm.client import LlmClient
from ds_code.llm.models import ModelResolution, ProviderKind
from ds_code.tools.registry import ToolRegistry
from ds_code.tools.execution import ToolCall
from ds_code.runtime.events import (
    exec_approval_request,
    error_frame,
    tool_call_result,
    tool_call_start,
)


class CoreEngine:
    def __init__(
        self,
        llm: LlmClient,
        tools: ToolRegistry,
        jobs: Optional[JobManager] = None,
        threads: Optional[ThreadManager] = None,
        exec_policy: Optional[ExecPolicyEngine] = None,
    ) -> None:
        self._llm = llm
        self._tools = tools
        self._jobs = jobs or JobManager()
        self._threads = threads or ThreadManager()
        self._exec_policy = exec_policy or ExecPolicyEngine()

    def run(self, prompt: str) -> str:
        _ = self._tools.list_tools()
        return self._llm.generate(prompt)

    def resolve_model(
        self, requested: Optional[str], provider_hint: Optional[ProviderKind]
    ) -> ModelResolution:
        return self._llm.resolve_model(requested, provider_hint)

    def generate_with_model(
        self,
        prompt: str,
        requested_model: Optional[str] = None,
        provider_hint: Optional[ProviderKind] = None,
    ) -> str:
        return self._llm.generate(prompt, requested_model, provider_hint)

    def enqueue_job(self, name: str) -> JobRecord:
        return self._jobs.enqueue(name)

    def set_job_running(self, job_id: str) -> None:
        self._jobs.set_running(job_id)

    def update_job_progress(self, job_id: str, progress: int, detail: Optional[str]) -> None:
        self._jobs.update_progress(job_id, progress, detail)

    def complete_job(self, job_id: str) -> None:
        self._jobs.complete(job_id)

    def fail_job(self, job_id: str, detail: str) -> None:
        self._jobs.fail(job_id, detail)

    def cancel_job(self, job_id: str) -> None:
        self._jobs.cancel(job_id)

    def pause_job(self, job_id: str, detail: Optional[str]) -> None:
        self._jobs.pause(job_id, detail)

    def resume_job(self, job_id: str, detail: Optional[str]) -> None:
        self._jobs.resume(job_id, detail)

    def job_history(self, job_id: str) -> list[JobHistoryEntry]:
        return self._jobs.history(job_id)

    def create_thread(self, model_provider: str, model: str = "auto") -> NewThread:
        return self._threads.create(model_provider, model)

    def resume_thread(
        self, thread_id: str, model_provider: str, model: str = "auto"
    ) -> Optional[NewThread]:
        return self._threads.resume(thread_id, model_provider, model)

    def list_threads(self) -> list[Thread]:
        return self._threads.list_threads()

    def read_thread(self, thread_id: str) -> Optional[Thread]:
        return self._threads.read_thread(thread_id)

    def set_thread_name(self, thread_id: str, name: str) -> Optional[Thread]:
        return self._threads.set_thread_name(thread_id, name)

    def archive_thread(self, thread_id: str) -> None:
        self._threads.archive(thread_id)

    def unarchive_thread(self, thread_id: str) -> None:
        self._threads.unarchive(thread_id)

    def touch_thread_message(self, thread_id: str, message: str) -> None:
        self._threads.touch_message(thread_id, message)

    def invoke_tool(
        self,
        call: ToolCall,
        approval_mode: AskForApproval,
        cwd: str,
    ) -> dict:
        command, policy_cwd, execution_kind = call.execution_subject(cwd)
        response_id = f"tool-{uuid.uuid4()}"
        call_id = call.raw_tool_call_id or f"tool-call-{uuid.uuid4()}"
        decision = self._exec_policy.check(
            ExecPolicyContext(
                command=command,
                cwd=policy_cwd,
                ask_for_approval=approval_mode,
                sandbox_mode=None,
            )
        )

        if not decision.allow:
            frame = error_frame(response_id, decision.requirement.reason)
            return {
                "ok": False,
                "status": "denied",
                "execution_kind": execution_kind,
                "error": decision.requirement.reason,
                "events": [frame],
            }

        if decision.requires_approval:
            approval_id = f"approval-{uuid.uuid4()}"
            frame = exec_approval_request(
                call_id=call_id,
                approval_id=approval_id,
                response_id=response_id,
                command=command,
                cwd=policy_cwd,
                reason=decision.requirement.reason,
                proposed_execpolicy_amendment=(
                    decision.requirement.proposed_execpolicy_amendment.prefixes
                    if decision.requirement.proposed_execpolicy_amendment
                    else []
                ),
                proposed_network_policy_amendments=(
                    decision.requirement.proposed_network_policy_amendments or []
                ),
            )
            return {
                "ok": False,
                "status": "approval_required",
                "execution_kind": execution_kind,
                "reason": decision.requirement.reason,
                "approval_id": approval_id,
                "events": [frame],
                "proposed_execpolicy_amendment": (
                    decision.requirement.proposed_execpolicy_amendment.prefixes
                    if decision.requirement.proposed_execpolicy_amendment
                    else []
                ),
            }

        result = self._tools.dispatch(call)
        start_frame = tool_call_start(response_id, call.name, {"payload": call.payload.__dict__})
        if result.ok:
            end_frame = tool_call_result(response_id, call.name, {"content": result.content})
        else:
            end_frame = error_frame(response_id, result.content)
        return {
            "ok": result.ok,
            "status": "completed" if result.ok else "failed",
            "execution_kind": execution_kind,
            "output": result.content,
            "events": [start_frame, end_frame],
        }
