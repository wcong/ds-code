import uuid
from typing import List, Optional

from ds_code.execpolicy import NetworkPolicyAmendment


def response_start() -> dict:
    response_id = f"resp-{uuid.uuid4()}"
    return {"event": "response_start", "response_id": response_id}


def response_delta(response_id: str, delta: str, channel: str = "text") -> dict:
    return {
        "event": "response_delta",
        "response_id": response_id,
        "delta": delta,
        "channel": channel,
    }


def response_end(response_id: str) -> dict:
    return {"event": "response_end", "response_id": response_id}


def tool_call_start(response_id: str, tool_name: str, arguments: dict) -> dict:
    return {
        "event": "tool_call_start",
        "response_id": response_id,
        "tool_name": tool_name,
        "arguments": arguments,
    }


def tool_call_result(response_id: str, tool_name: str, output: dict) -> dict:
    return {
        "event": "tool_call_result",
        "response_id": response_id,
        "tool_name": tool_name,
        "output": output,
    }


def error_frame(response_id: str, message: str) -> dict:
    return {"event": "error", "response_id": response_id, "message": message}


def exec_approval_request(
    call_id: str,
    approval_id: str,
    response_id: str,
    command: str,
    cwd: str,
    reason: str,
    proposed_execpolicy_amendment: List[str],
    proposed_network_policy_amendments: List[NetworkPolicyAmendment],
    additional_permissions: Optional[List[str]] = None,
) -> dict:
    return {
        "event": "exec_approval_request",
        "request": {
            "call_id": call_id,
            "approval_id": approval_id,
            "turn_id": response_id,
            "command": command,
            "cwd": cwd,
            "reason": reason,
            "proposed_execpolicy_amendment": proposed_execpolicy_amendment,
            "proposed_network_policy_amendments": [
                {"host": item.host, "action": item.action.value}
                for item in proposed_network_policy_amendments
            ],
            "additional_permissions": additional_permissions or [],
        },
    }
