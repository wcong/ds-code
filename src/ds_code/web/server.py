from __future__ import annotations

from typing import List
import difflib
import os

import json

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
import uvicorn

from ds_code.core.engine import CoreEngine
from ds_code.llm.client import LlmClient
from ds_code.llm.models import ModelRegistry
from ds_code.runtime.api import RuntimeApi
from ds_code.runtime.models import PromptRequest, RuntimeConfig
from ds_code.config import Config
from ds_code.tasks.manager import TaskManager
from ds_code.tools.registry import ToolRegistry
from ds_code.tools.specs import ToolSpec
from ds_code.tools.mcp import InMemoryMcpClient, McpManager, McpServerConfig, ToolFilter
from ds_code.tools.execution import ToolCall, ToolPayload
from ds_code.execpolicy import AskForApproval


def build_runtime(config: Config) -> RuntimeApi:
    llm = LlmClient()
    mcp = McpManager()
    mcp.load_state()
    mcp.register_server(
        McpServerConfig(name="local", command="in-memory"),
        ToolFilter(),
        InMemoryMcpClient().with_tool("health", {"status": "ok"}),
    )
    tools = ToolRegistry(mcp=mcp)
    core = CoreEngine(llm=llm, tools=tools)
    tasks = TaskManager()
    config = RuntimeConfig.from_config(config)
    return RuntimeApi(core=core, tasks=tasks, config=config)


def _repo_root() -> str:
    return os.path.abspath(os.getcwd())


def _resolve_repo_path(raw_path: str) -> tuple[str, str]:
    if not raw_path or not raw_path.strip():
      raise ValueError("path required")
    root = _repo_root()
    candidate = raw_path.strip()
    if os.path.isabs(candidate):
      abs_path = os.path.abspath(candidate)
    else:
      abs_path = os.path.abspath(os.path.join(root, candidate))
    if abs_path != root and not abs_path.startswith(root + os.sep):
      raise ValueError("path must be within repo")
    rel_path = os.path.relpath(abs_path, root)
    return abs_path, rel_path


def _read_text(path: str) -> str:
    if not os.path.exists(path):
      return ""
    with open(path, "r", encoding="utf-8") as handle:
      return handle.read()


def _diff_text(before: str, after: str, rel_path: str) -> str:
    lines = difflib.unified_diff(
      before.splitlines(),
      after.splitlines(),
      fromfile=f"a/{rel_path}",
      tofile=f"b/{rel_path}",
      lineterm="",
    )
    return "\n".join(lines)


def create_app() -> FastAPI:
    app = FastAPI()
    config = Config.load()
    runtime = build_runtime(config)

    @app.get("/")
    def index() -> HTMLResponse:
        return HTMLResponse(_INDEX_HTML)

    @app.get("/api/health")
    def health() -> JSONResponse:
        return JSONResponse({"ok": True})

    @app.post("/api/prompt")
    def prompt(payload: dict) -> JSONResponse:
        text = str(payload.get("prompt", "")).strip()
        if not text:
            return JSONResponse({"ok": False, "error": "prompt required"}, status_code=400)
        response = runtime.handle_prompt(PromptRequest(prompt=text))
        return JSONResponse(
            {
                "ok": True,
                "output": response.output,
                "model": response.model,
                "events": response.events,
            }
        )

    @app.get("/api/prompt/stream")
    def prompt_stream(prompt: str) -> StreamingResponse:
        response = runtime.handle_prompt(PromptRequest(prompt=prompt))

        def event_stream():
            for event in response.events:
                yield f"event: message\ndata: {json.dumps(event)}\n\n"
            payload = {"output": response.output, "model": response.model}
            yield f"event: done\ndata: {json.dumps(payload)}\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    @app.post("/api/threads/create")
    def create_thread() -> JSONResponse:
        response = runtime.create_thread()
        return JSONResponse(
            {
                "ok": True,
                "thread_id": response.thread_id,
                "model": response.model,
                "model_provider": response.model_provider,
            }
        )

    @app.get("/api/threads/list")
    def list_threads() -> JSONResponse:
        threads = runtime.list_threads()
        items: List[dict] = []
        for thread in threads:
            items.append(
                {
                    "id": thread.id,
                    "name": thread.name,
                    "status": thread.status,
                    "updated_at": thread.updated_at,
                }
            )
        return JSONResponse({"ok": True, "threads": items})

    @app.get("/api/threads/{thread_id}/messages")
    def list_thread_messages(thread_id: str) -> JSONResponse:
        messages = runtime._core._threads.list_messages(thread_id, 100)
        payload = [
          {
            "id": item.id,
            "role": item.role,
            "content": item.content,
            "created_at": item.created_at,
          }
          for item in messages
        ]
        return JSONResponse({"ok": True, "messages": payload})

    @app.get("/api/jobs")
    def list_jobs() -> JSONResponse:
        jobs = runtime._tasks.list_jobs()
        payload = [
          {
            "id": job.id,
            "name": job.name,
            "status": job.status.value,
            "progress": job.progress,
            "detail": job.detail,
            "updated_at": job.updated_at,
          }
          for job in jobs
        ]
        return JSONResponse({"ok": True, "jobs": payload})

    @app.post("/api/jobs/enqueue")
    def enqueue_job(payload: dict) -> JSONResponse:
        name = str(payload.get("name", "")).strip() or "job"
        job = runtime.submit_task(name, payload.get("payload", ""))
        return JSONResponse({"ok": True, "job_id": job.id})

    @app.post("/api/jobs/update")
    def update_job(payload: dict) -> JSONResponse:
        job_id = str(payload.get("job_id", "")).strip()
        status = str(payload.get("status", "")).strip()
        progress = payload.get("progress")
        detail = payload.get("detail")
        if not job_id or not status:
          return JSONResponse({"ok": False, "error": "job_id and status required"}, status_code=400)
        if status == "running":
          runtime._tasks.set_running(job_id)
        elif status == "completed":
          runtime._tasks.complete(job_id)
        elif status == "failed":
          runtime._tasks.fail(job_id, str(detail or "failed"))
        elif status == "cancelled":
          runtime._tasks.cancel(job_id)
        elif status == "paused":
          runtime._tasks.pause(job_id, str(detail or "paused"))
        elif status == "progress":
          try:
            runtime._tasks.update_progress(job_id, int(progress or 0), str(detail or ""))
          except ValueError:
            return JSONResponse({"ok": False, "error": "invalid progress"}, status_code=400)
        else:
          return JSONResponse({"ok": False, "error": "unsupported status"}, status_code=400)
        return JSONResponse({"ok": True})

    @app.get("/api/models")
    def list_models() -> JSONResponse:
        models = ModelRegistry.default().list()
        payload = [
          {
            "id": model.id,
            "provider": model.provider.as_str(),
            "aliases": model.aliases,
            "supports_tools": model.supports_tools,
            "supports_reasoning": model.supports_reasoning,
          }
          for model in models
        ]
        return JSONResponse({"ok": True, "models": payload})

    @app.get("/api/tools/specs")
    def list_tool_specs() -> JSONResponse:
        specs = runtime._core._tools.list_specs()
        payload = [
          {
            "name": item.spec.name,
            "input_schema": item.spec.input_schema,
            "output_schema": item.spec.output_schema,
            "supports_parallel_tool_calls": item.supports_parallel_tool_calls,
            "timeout_ms": item.spec.timeout_ms,
          }
          for item in specs
        ]
        return JSONResponse({"ok": True, "specs": payload})

    @app.post("/api/tools/register")
    def register_tool(payload: dict) -> JSONResponse:
        name = str(payload.get("name", "")).strip()
        if not name:
          return JSONResponse({"ok": False, "error": "name required"}, status_code=400)
        runtime._core._tools.register(name, lambda args: f"{name} -> {args}")
        return JSONResponse({"ok": True})

    @app.post("/api/tools/specs/register")
    def register_tool_spec(payload: dict) -> JSONResponse:
        name = str(payload.get("name", "")).strip()
        if not name:
          return JSONResponse({"ok": False, "error": "name required"}, status_code=400)
        spec = {
          "name": name,
          "input_schema": payload.get("input_schema", {"type": "object"}),
          "output_schema": payload.get("output_schema", {"type": "object"}),
          "supports_parallel_tool_calls": bool(payload.get("supports_parallel_tool_calls", False)),
          "timeout_ms": payload.get("timeout_ms"),
        }
        runtime._core._tools.register_spec(
          ToolSpec(
            name=spec["name"],
            input_schema=spec["input_schema"],
            output_schema=spec["output_schema"],
            supports_parallel_tool_calls=spec["supports_parallel_tool_calls"],
            timeout_ms=spec["timeout_ms"],
          )
        )
        return JSONResponse({"ok": True})

    @app.get("/api/mcp/tools")
    def list_mcp_tools() -> JSONResponse:
        tools = runtime._core._tools.list_mcp_tools()
        payload = [tool.__dict__ for tool in tools]
        return JSONResponse({"ok": True, "tools": payload})

    @app.get("/api/mcp/resources")
    def list_mcp_resources() -> JSONResponse:
        mcp = runtime._core._tools._mcp
        if mcp is None:
          return JSONResponse({"ok": True, "resources": []})
        resources = mcp.list_resources()
        payload = [resource.__dict__ for resource in resources]
        return JSONResponse({"ok": True, "resources": payload})

    @app.get("/api/mcp/servers")
    def list_mcp_servers() -> JSONResponse:
        mcp = runtime._core._tools._mcp
        if mcp is None:
          return JSONResponse({"ok": True, "servers": []})
        servers = []
        for name, (config, _tool_filter) in mcp._configs.items():
          servers.append(
            {
              "name": name,
              "enabled": config.enabled,
              "command": config.command,
              "running": name in mcp._clients,
            }
          )
        return JSONResponse({"ok": True, "servers": servers})

    @app.post("/api/mcp/servers/register")
    def register_mcp_server(payload: dict) -> JSONResponse:
        name = str(payload.get("name", "")).strip()
        if not name:
          return JSONResponse({"ok": False, "error": "name required"}, status_code=400)
        mcp = runtime._core._tools._mcp
        if mcp is None:
          return JSONResponse({"ok": False, "error": "mcp not configured"}, status_code=400)
        config = McpServerConfig(name=name, command=str(payload.get("command", "mcp")))
        allow = payload.get("allow") or []
        deny = payload.get("deny") or []
        tool_filter = ToolFilter(allow=list(allow), deny=list(deny))
        mcp.register_server(config, tool_filter, InMemoryMcpClient())
        return JSONResponse({"ok": True})

    @app.post("/api/mcp/servers/unregister")
    def unregister_mcp_server(payload: dict) -> JSONResponse:
        name = str(payload.get("name", "")).strip()
        if not name:
          return JSONResponse({"ok": False, "error": "name required"}, status_code=400)
        mcp = runtime._core._tools._mcp
        if mcp is None:
          return JSONResponse({"ok": False, "error": "mcp not configured"}, status_code=400)
        try:
          mcp.unregister_server(name)
        except KeyError as exc:
          return JSONResponse({"ok": False, "error": str(exc)}, status_code=404)
        return JSONResponse({"ok": True})

    @app.post("/api/mcp/servers/start")
    def start_mcp_server(payload: dict) -> JSONResponse:
        name = str(payload.get("name", "")).strip()
        if not name:
          return JSONResponse({"ok": False, "error": "name required"}, status_code=400)
        mcp = runtime._core._tools._mcp
        if mcp is None:
          return JSONResponse({"ok": False, "error": "mcp not configured"}, status_code=400)
        try:
          mcp.start_server(name, InMemoryMcpClient())
        except KeyError as exc:
          return JSONResponse({"ok": False, "error": str(exc)}, status_code=404)
        return JSONResponse({"ok": True})

    @app.post("/api/mcp/servers/stop")
    def stop_mcp_server(payload: dict) -> JSONResponse:
        name = str(payload.get("name", "")).strip()
        if not name:
          return JSONResponse({"ok": False, "error": "name required"}, status_code=400)
        mcp = runtime._core._tools._mcp
        if mcp is None:
          return JSONResponse({"ok": False, "error": "mcp not configured"}, status_code=400)
        try:
          mcp.stop_server(name)
        except KeyError as exc:
          return JSONResponse({"ok": False, "error": str(exc)}, status_code=404)
        return JSONResponse({"ok": True})

    @app.post("/api/tools/invoke")
    def invoke_tool(payload: dict) -> JSONResponse:
        name = str(payload.get("name", "")).strip()
        tool_type = str(payload.get("type", "function")).strip().lower()
        arguments = payload.get("arguments")
        command = payload.get("command")
        cwd = payload.get("cwd")
        timeout_ms = payload.get("timeout_ms")
        if not name:
          return JSONResponse({"ok": False, "error": "tool name required"}, status_code=400)
        tool_payload = ToolPayload(
          type=tool_type,
          arguments=str(arguments or ""),
          command=str(command or ""),
          cwd=str(cwd) if cwd else None,
          timeout_ms=int(timeout_ms) if timeout_ms else None,
          raw_arguments=payload.get("raw_arguments"),
          server=payload.get("server"),
          tool=payload.get("tool"),
        )
        call = ToolCall(name=name, payload=tool_payload)
        result = runtime.invoke_tool(call, AskForApproval.UnlessTrusted, ".")
        return JSONResponse(result)

    @app.post("/api/approvals/review")
    def review_approval(payload: dict) -> JSONResponse:
        approval_id = str(payload.get("approval_id", "")).strip()
        decision = str(payload.get("decision", "")).strip()
        remember = bool(payload.get("remember", False))
        if not approval_id or not decision:
          return JSONResponse(
            {"ok": False, "error": "approval_id and decision required"},
            status_code=400,
          )
        result = runtime.review_approval(approval_id, decision, remember)
        return JSONResponse(result)

    @app.get("/api/config")
    def list_config() -> JSONResponse:
        return JSONResponse({"ok": True, "values": config.list_values()})

    @app.get("/api/config/{key}")
    def get_config(key: str) -> JSONResponse:
        value = config.get_value(key)
        if value is None:
          return JSONResponse({"ok": False, "error": "key not found"}, status_code=404)
        return JSONResponse({"ok": True, "value": value})

    @app.post("/api/config/{key}")
    def set_config(key: str, payload: dict) -> JSONResponse:
        value = str(payload.get("value", ""))
        try:
          config.set_value(key, value)
          config.save()
          runtime._config = RuntimeConfig.from_config(config)
        except ValueError as exc:
          return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)
        return JSONResponse({"ok": True})

    @app.post("/api/composer/preview")
    def composer_preview(payload: dict) -> JSONResponse:
      raw_path = str(payload.get("path", ""))
      content = str(payload.get("content", ""))
      try:
        abs_path, rel_path = _resolve_repo_path(raw_path)
      except ValueError as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)
      before = _read_text(abs_path)
      diff = _diff_text(before, content, rel_path)
      return JSONResponse({"ok": True, "diff": diff, "changed": before != content})

    @app.post("/api/composer/apply")
    def composer_apply(payload: dict) -> JSONResponse:
      raw_path = str(payload.get("path", ""))
      content = str(payload.get("content", ""))
      try:
        abs_path, rel_path = _resolve_repo_path(raw_path)
      except ValueError as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)
      os.makedirs(os.path.dirname(abs_path), exist_ok=True)
      with open(abs_path, "w", encoding="utf-8") as handle:
        handle.write(content)
      return JSONResponse({"ok": True, "path": rel_path})

    return app


def run_server() -> None:
    uvicorn.run(create_app(), host="127.0.0.1", port=8000, log_level="info")


_INDEX_HTML = """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>DS Code</title>
    <style>
      :root {
        --bg: #f6f4ef;
        --ink: #1b1a17;
        --muted: #6c6860;
        --accent: #2a6f6f;
        --accent-2: #c7502a;
        --panel: #ffffff;
      }
      body {
        margin: 0;
        font-family: "Space Grotesk", "IBM Plex Sans", "Helvetica Neue", Arial, sans-serif;
        background: radial-gradient(circle at top, #fef3e1 0%, var(--bg) 55%, #efe9df 100%);
        color: var(--ink);
      }
      header {
        padding: 24px 28px 8px;
        font-size: 22px;
        font-weight: 600;
      }
      main {
        display: grid;
        grid-template-columns: minmax(280px, 1fr) minmax(320px, 1fr);
        gap: 16px;
        padding: 0 28px 28px;
      }
      section {
        background: var(--panel);
        border-radius: 16px;
        padding: 16px;
        box-shadow: 0 16px 24px rgba(28, 25, 20, 0.08);
      }
      label {
        display: block;
        font-size: 13px;
        color: var(--muted);
        margin-bottom: 6px;
      }
      textarea {
        width: 100%;
        min-height: 120px;
        border-radius: 10px;
        border: 1px solid #ddd2c1;
        padding: 10px;
        font-size: 14px;
        resize: vertical;
      }
      button {
        margin-top: 10px;
        border: none;
        border-radius: 10px;
        padding: 10px 14px;
        font-weight: 600;
        background: var(--accent);
        color: white;
        cursor: pointer;
      }
      button.accent {
        background: var(--accent-2);
      }
      button.secondary {
        background: transparent;
        border: 1px solid var(--accent);
        color: var(--accent);
        margin-left: 8px;
      }
      select,
      input[type="text"] {
        width: 100%;
        border-radius: 10px;
        border: 1px solid #ddd2c1;
        padding: 8px 10px;
        font-size: 14px;
      }
      .log {
        min-height: 240px;
        white-space: pre-wrap;
        font-size: 13px;
        color: var(--muted);
      }
      .diff {
        min-height: 160px;
        white-space: pre-wrap;
        font-size: 12px;
        color: #4f4a42;
        background: #fbf7f0;
        border: 1px solid #e4dccf;
        border-radius: 10px;
        padding: 10px;
        font-family: "IBM Plex Mono", "SFMono-Regular", Menlo, monospace;
      }
      .timeline {
        border-left: 2px solid #e4dccf;
        padding-left: 12px;
        font-size: 12px;
        color: var(--muted);
      }
      .timeline-item {
        margin-bottom: 8px;
      }
      .progress {
        height: 6px;
        border-radius: 999px;
        background: #e6e0d7;
        overflow: hidden;
        margin-top: 8px;
      }
      .progress span {
        display: block;
        height: 100%;
        width: 0%;
        background: var(--accent);
        transition: width 0.2s ease;
      }
      .config-row {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 10px;
        margin-top: 10px;
      }
      @media (max-width: 900px) {
        main {
          grid-template-columns: 1fr;
        }
      }
    </style>
  </head>
  <body>
    <header>DS Code Web</header>
    <main>
      <section>
        <label for="prompt">Prompt</label>
        <textarea id="prompt" placeholder="Type a prompt..."></textarea>
        <div>
          <button id="send">Send</button>
          <button id="send-stream" class="secondary">Send (Stream)</button>
          <button id="create-thread" class="secondary">Create Thread</button>
          <button id="list-threads" class="secondary">List Threads</button>
          <button id="list-tools" class="secondary">List Tool Specs</button>
          <button id="list-mcp-tools" class="secondary">List MCP Tools</button>
          <button id="list-mcp-resources" class="secondary">List MCP Resources</button>
        </div>
      </section>
      <section>
        <label>Composer (Code Gen)</label>
        <label for="composer-prompt">Composer Prompt</label>
        <textarea id="composer-prompt" placeholder="Describe what to generate..."></textarea>
        <div class="config-row">
          <div>
            <label for="composer-path">Target Path</label>
            <input id="composer-path" type="text" placeholder="src/ds_code/..." />
          </div>
          <div>
            <label for="composer-actions">Actions</label>
            <div>
              <button id="composer-generate">Generate</button>
              <button id="composer-preview" class="secondary">Preview Diff</button>
              <button id="composer-apply" class="accent">Apply</button>
            </div>
          </div>
        </div>
        <label for="composer-output">Generated / Edited Content</label>
        <textarea id="composer-output" placeholder="Generated code appears here..."></textarea>
        <label>Diff Preview</label>
        <div id="composer-diff" class="diff"></div>
      </section>
      <section>
        <label>Activity</label>
        <div id="log" class="log"></div>
        <div class="progress"><span id="stream-progress"></span></div>
      </section>
      <section>
        <label>Config</label>
        <div class="config-row">
          <div>
            <label for="config-key">Key</label>
            <select id="config-key">
              <option value="provider">provider</option>
              <option value="model">model</option>
              <option value="base_url">base_url</option>
              <option value="default_text_model">default_text_model</option>
              <option value="approval_policy">approval_policy</option>
              <option value="sandbox_mode">sandbox_mode</option>
            </select>
          </div>
          <div>
            <label for="config-value">Value</label>
            <input id="config-value" type="text" placeholder="Enter value" />
            <label id="config-hint" style="margin-top:6px;color:var(--muted);font-size:12px;"></label>
          </div>
        </div>
        <div>
          <button id="config-get" class="secondary">Get</button>
          <button id="config-set" class="accent">Set</button>
          <button id="config-list" class="secondary">List</button>
          <button id="config-save" class="secondary">Save</button>
        </div>
      </section>
      <section>
        <label>Thread History</label>
        <div class="config-row">
          <div>
            <label for="thread-id">Thread ID</label>
            <input id="thread-id" type="text" placeholder="thread-..." />
          </div>
          <div>
            <label for="thread-select">Recent Threads</label>
            <select id="thread-select"></select>
          </div>
        </div>
        <div>
          <button id="load-thread" class="secondary">Load Messages</button>
        </div>
      </section>
      <section>
        <label>Event Timeline</label>
        <div id="timeline" class="timeline"></div>
      </section>
      <section>
        <label>Tool Invocation</label>
        <div class="config-row">
          <div>
            <label for="tool-name">Tool Name</label>
            <input id="tool-name" type="text" placeholder="tool name or mcp__..." />
          </div>
          <div>
            <label for="tool-type">Type</label>
            <select id="tool-type">
              <option value="function">function</option>
              <option value="local_shell">local_shell</option>
              <option value="mcp">mcp</option>
            </select>
          </div>
        </div>
        <div class="config-row">
          <div>
            <label for="tool-args">Arguments</label>
            <input id="tool-args" type="text" placeholder="args or input" />
          </div>
          <div>
            <label for="tool-cwd">CWD</label>
            <input id="tool-cwd" type="text" placeholder="." />
          </div>
        </div>
        <div class="config-row">
          <div>
            <label for="mcp-server">MCP Server</label>
            <input id="mcp-server" type="text" placeholder="local" />
          </div>
          <div>
            <label for="mcp-tool">MCP Tool</label>
            <input id="mcp-tool" type="text" placeholder="health" />
          </div>
        </div>
        <div class="config-row">
          <div>
            <label for="mcp-tool-select">Available MCP Tools</label>
            <select id="mcp-tool-select"></select>
          </div>
          <div>
            <label>&nbsp;</label>
            <button id="refresh-mcp" class="secondary">Refresh MCP Tools</button>
          </div>
        </div>
        <div class="config-row">
          <div>
            <label for="mcp-resource-select">MCP Resources</label>
            <select id="mcp-resource-select"></select>
          </div>
          <div>
            <label>&nbsp;</label>
            <button id="refresh-mcp-resources" class="secondary">Refresh MCP Resources</button>
          </div>
        </div>
        <div class="config-row">
          <div>
            <label for="tool-timeout">Timeout (ms)</label>
            <input id="tool-timeout" type="text" placeholder="2000" />
          </div>
          <div>
            <label>&nbsp;</label>
            <button id="invoke-tool" class="accent">Invoke Tool</button>
          </div>
        </div>
      </section>
      <section>
        <label>Model Selection</label>
        <div class="config-row">
          <div>
            <label for="provider-select">Provider</label>
            <select id="provider-select"></select>
          </div>
          <div>
            <label for="model-select">Model</label>
            <select id="model-select"></select>
          </div>
        </div>
        <div>
          <button id="apply-model" class="accent">Apply</button>
          <button id="refresh-models" class="secondary">Refresh</button>
        </div>
      </section>
      <section>
        <label>Job Queue</label>
        <div class="config-row">
          <div>
            <label for="job-name">Job Name</label>
            <input id="job-name" type="text" placeholder="compile" />
          </div>
          <div>
            <label for="job-payload">Payload</label>
            <input id="job-payload" type="text" placeholder="optional details" />
          </div>
        </div>
        <div>
          <button id="enqueue-job" class="accent">Enqueue</button>
          <button id="list-jobs" class="secondary">List Jobs</button>
        </div>
        <div class="config-row">
          <div>
            <label for="job-id">Job ID</label>
            <input id="job-id" type="text" placeholder="job-..." />
          </div>
          <div>
            <label for="job-status">Status</label>
            <select id="job-status">
              <option value="running">running</option>
              <option value="progress">progress</option>
              <option value="paused">paused</option>
              <option value="completed">completed</option>
              <option value="failed">failed</option>
              <option value="cancelled">cancelled</option>
            </select>
          </div>
        </div>
        <div class="config-row">
          <div>
            <label for="job-progress">Progress (%)</label>
            <input id="job-progress" type="text" placeholder="40" />
          </div>
          <div>
            <label for="job-detail">Detail</label>
            <input id="job-detail" type="text" placeholder="optional status message" />
          </div>
        </div>
        <div>
          <button id="update-job" class="secondary">Update Job</button>
          <button id="simulate-job" class="secondary">Simulate Progress</button>
        </div>
      </section>
      <section>
        <label>Approval Review</label>
        <div class="config-row">
          <div>
            <label for="approval-id">Approval ID</label>
            <input id="approval-id" type="text" placeholder="approval-..." />
          </div>
          <div>
            <label for="approval-decision">Decision</label>
            <select id="approval-decision">
              <option value="approved">approved</option>
              <option value="approved_for_session">approved_for_session</option>
              <option value="denied">denied</option>
              <option value="abort">abort</option>
            </select>
          </div>
        </div>
        <div>
          <button id="submit-approval" class="accent">Submit Decision</button>
        </div>
      </section>
      <section>
        <label>Tool Registry</label>
        <div class="config-row">
          <div>
            <label for="registry-tool-name">Tool Name</label>
            <input id="registry-tool-name" type="text" placeholder="my_tool" />
          </div>
          <div>
            <label for="registry-timeout">Timeout (ms)</label>
            <input id="registry-timeout" type="text" placeholder="1000" />
          </div>
        </div>
        <div class="config-row">
          <div>
            <label for="registry-input">Input Schema (JSON)</label>
            <input id="registry-input" type="text" placeholder='{"type":"object"}' />
          </div>
          <div>
            <label for="registry-output">Output Schema (JSON)</label>
            <input id="registry-output" type="text" placeholder='{"type":"object"}' />
          </div>
        </div>
        <div>
          <button id="register-tool" class="accent">Register Tool</button>
          <button id="register-spec" class="secondary">Register Spec</button>
        </div>
      </section>
      <section>
        <label>MCP Servers</label>
        <div style="margin-bottom:8px;">
          <span>Running servers:</span>
          <span id="mcp-running" style="margin-left:6px;padding:2px 8px;border-radius:999px;background:#d8ede5;color:#1b6b55;font-size:12px;">0</span>
        </div>
        <div class="config-row">
          <div>
            <label for="mcp-server-name">Server Name</label>
            <input id="mcp-server-name" type="text" placeholder="local" />
          </div>
          <div>
            <label for="mcp-server-command">Command</label>
            <input id="mcp-server-command" type="text" placeholder="mcp" />
          </div>
        </div>
        <div class="config-row">
          <div>
            <label for="mcp-allow">Allow (comma)</label>
            <input id="mcp-allow" type="text" placeholder="health" />
          </div>
          <div>
            <label for="mcp-deny">Deny (comma)</label>
            <input id="mcp-deny" type="text" placeholder="" />
          </div>
        </div>
        <div>
          <button id="register-mcp-server" class="accent">Register</button>
          <button id="list-mcp-servers" class="secondary">List</button>
          <button id="start-mcp-server" class="secondary">Start</button>
          <button id="stop-mcp-server" class="secondary">Stop</button>
          <button id="unregister-mcp-server" class="secondary">Unregister</button>
        </div>
      </section>
    </main>
    <script>
      const log = document.getElementById("log");
      const promptBox = document.getElementById("prompt");
      const composerPrompt = document.getElementById("composer-prompt");
      const composerPath = document.getElementById("composer-path");
      const composerOutput = document.getElementById("composer-output");
      const composerDiff = document.getElementById("composer-diff");
      const configKey = document.getElementById("config-key");
      const configValue = document.getElementById("config-value");
      const configHint = document.getElementById("config-hint");
      const threadIdInput = document.getElementById("thread-id");
      const threadSelect = document.getElementById("thread-select");
      const providerSelect = document.getElementById("provider-select");
      const modelSelect = document.getElementById("model-select");
      const timeline = document.getElementById("timeline");
        <div class="config-row">
          <div>
            <label>Quick Prefixes</label>
            <button id="insert-tool" class="secondary">tool:</button>
            <button id="insert-shell" class="secondary">shell:</button>
            <button id="insert-mcp" class="secondary">mcp:</button>
          </div>
          <div>
            <label>Examples</label>
            <div class="log">tool:my_tool arg1 arg2\nshell:ls -la\nmcp:local::health {"key":"value"}</div>
          </div>
        </div>
      const streamProgress = document.getElementById("stream-progress");
      const toolName = document.getElementById("tool-name");
      const toolType = document.getElementById("tool-type");
      const toolArgs = document.getElementById("tool-args");
      const toolCwd = document.getElementById("tool-cwd");
      const toolTimeout = document.getElementById("tool-timeout");
      const mcpServer = document.getElementById("mcp-server");
      const mcpTool = document.getElementById("mcp-tool");
      const mcpToolSelect = document.getElementById("mcp-tool-select");
      const mcpResourceSelect = document.getElementById("mcp-resource-select");
      const jobName = document.getElementById("job-name");
      const jobPayload = document.getElementById("job-payload");
      const jobId = document.getElementById("job-id");
      const jobStatus = document.getElementById("job-status");
      const jobProgress = document.getElementById("job-progress");
      const jobDetail = document.getElementById("job-detail");
      const approvalId = document.getElementById("approval-id");
      const approvalDecision = document.getElementById("approval-decision");
      const registryToolName = document.getElementById("registry-tool-name");
      const registryTimeout = document.getElementById("registry-timeout");
      const registryInput = document.getElementById("registry-input");
      const registryOutput = document.getElementById("registry-output");
      const mcpServerName = document.getElementById("mcp-server-name");
      const mcpServerCommand = document.getElementById("mcp-server-command");
      const mcpAllow = document.getElementById("mcp-allow");
      const mcpDeny = document.getElementById("mcp-deny");
      const mcpRunning = document.getElementById("mcp-running");
      let modelCache = [];
      const hints = {
        provider: "deepseek | nvidia-nim | openai | atlascloud | wanjie-ark | openrouter | novita | fireworks | sglang | vllm | ollama",
        approval_policy: "unless_trusted | on_failure | on_request | never | reject",
        sandbox_mode: "workspace-write | read-only | none",
      };
      function updateHint() {
        const key = configKey.value;
        configHint.textContent = hints[key] || "";
      }
      async function loadConfigValue() {
        const key = configKey.value;
        const res = await fetch(`/api/config/${key}`);
        const data = await res.json();
        if (data.ok) {
          configValue.value = data.value || "";
        }
      }
      configKey.addEventListener("change", async () => {
        updateHint();
        await loadConfigValue();
      });
      updateHint();
      loadConfigValue();
      function append(message) {
        log.textContent += message + "\n";
        log.scrollTop = log.scrollHeight;
      }
      function pushTimeline(label, data) {
        const item = document.createElement("div");
        item.className = "timeline-item";
        item.textContent = `${label}: ${data}`;
        timeline.appendChild(item);
        timeline.scrollTop = timeline.scrollHeight;
      }
      async function composerGenerate() {
        const prompt = composerPrompt.value.trim();
        if (!prompt) return;
        const res = await fetch("/api/prompt", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ prompt })
        });
        const data = await res.json();
        if (!data.ok) {
          append("composer error: " + data.error);
          return;
        }
        composerOutput.value = data.output || "";
        append("composer generated output");
      }
      async function composerPreview() {
        const path = composerPath.value.trim();
        const content = composerOutput.value;
        if (!path) return;
        const res = await fetch("/api/composer/preview", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ path, content })
        });
        const data = await res.json();
        if (!data.ok) {
          append("composer preview error: " + data.error);
          composerDiff.textContent = "";
          return;
        }
        composerDiff.textContent = data.diff || "(no changes)";
        append("composer diff ready");
      }
      async function composerApply() {
        const path = composerPath.value.trim();
        const content = composerOutput.value;
        if (!path) return;
        const res = await fetch("/api/composer/apply", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ path, content })
        });
        const data = await res.json();
        if (!data.ok) {
          append("composer apply error: " + data.error);
          return;
        }
        append("composer applied: " + data.path);
      }
      document.getElementById("send").addEventListener("click", async () => {
        const prompt = promptBox.value.trim();
        if (!prompt) return;
        promptBox.value = "";
        append("prompt: " + prompt);
        const res = await fetch("/api/prompt", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ prompt })
        });
        const data = await res.json();
        if (!data.ok) {
          append("error: " + data.error);
          return;
        }
        append("response: " + data.output);
        if (data.events && data.events.length) {
          data.events.forEach((event) => {
            append("event: " + JSON.stringify(event));
            pushTimeline(event.type || event.event || "event", JSON.stringify(event));
          });
        }
      });
      document.getElementById("insert-tool").addEventListener("click", () => {
        promptBox.value = `tool:${promptBox.value}`;
        promptBox.focus();
      });
      document.getElementById("insert-shell").addEventListener("click", () => {
        promptBox.value = `shell:${promptBox.value}`;
        promptBox.focus();
      });
      document.getElementById("insert-mcp").addEventListener("click", () => {
        promptBox.value = `mcp:${promptBox.value}`;
        promptBox.focus();
      });
      document.getElementById("send-stream").addEventListener("click", () => {
        const prompt = promptBox.value.trim();
        if (!prompt) return;
        promptBox.value = "";
        append("prompt(stream): " + prompt);
        streamProgress.style.width = "20%";
        const source = new EventSource(`/api/prompt/stream?prompt=${encodeURIComponent(prompt)}`);
        source.addEventListener("message", (event) => {
          append("event: " + event.data);
          pushTimeline("stream", event.data);
          streamProgress.style.width = "60%";
        });
        source.addEventListener("done", (event) => {
          append("response: " + event.data);
          pushTimeline("done", event.data);
          streamProgress.style.width = "100%";
          source.close();
          setTimeout(() => {
            streamProgress.style.width = "0%";
          }, 600);
        });
        source.addEventListener("error", () => {
          append("stream error");
          pushTimeline("error", "stream error");
          streamProgress.style.width = "0%";
          source.close();
        });
      });
      document.getElementById("create-thread").addEventListener("click", async () => {
        const res = await fetch("/api/threads/create", { method: "POST" });
        const data = await res.json();
        if (data.ok) {
          append("thread created: " + data.thread_id);
        }
      });
      document.getElementById("list-threads").addEventListener("click", async () => {
        const res = await fetch("/api/threads/list");
        const data = await res.json();
        if (!data.ok) return;
        if (!data.threads.length) {
          append("no threads");
          return;
        }
        data.threads.forEach((t) => {
          append(`thread ${t.id} ${t.name || "(unnamed)"} ${t.status}`);
        });
      });
      document.getElementById("list-tools").addEventListener("click", async () => {
        const res = await fetch("/api/tools/specs");
        const data = await res.json();
        if (!data.ok) return;
        if (!data.specs.length) {
          append("no tool specs registered");
          return;
        }
        data.specs.forEach((spec) => {
          append(`tool ${spec.name} parallel=${spec.supports_parallel_tool_calls}`);
        });
      });
      document.getElementById("list-mcp-tools").addEventListener("click", async () => {
        const res = await fetch("/api/mcp/tools");
        const data = await res.json();
        if (!data.ok) return;
        if (!data.tools.length) {
          append("no mcp tools registered");
          return;
        }
        data.tools.forEach((tool) => {
          append(`mcp ${tool.server_name}::${tool.tool_name}`);
        });
      });
      document.getElementById("list-mcp-resources").addEventListener("click", async () => {
        const res = await fetch("/api/mcp/resources");
        const data = await res.json();
        if (!data.ok) return;
        if (!data.resources.length) {
          append("no mcp resources");
          return;
        }
        data.resources.forEach((resource) => {
          append(`mcp resource ${resource.server_name} ${resource.uri}`);
        });
      });
      document.getElementById("composer-generate").addEventListener("click", composerGenerate);
      document.getElementById("composer-preview").addEventListener("click", composerPreview);
      document.getElementById("composer-apply").addEventListener("click", composerApply);
      document.getElementById("config-get").addEventListener("click", async () => {
        const key = configKey.value;
        const res = await fetch(`/api/config/${key}`);
        const data = await res.json();
        if (!data.ok) {
          append("config error: " + data.error);
          return;
        }
        configValue.value = data.value || "";
        append(`config ${key} = ${data.value}`);
      });
      document.getElementById("config-set").addEventListener("click", async () => {
        const key = configKey.value;
        const value = configValue.value.trim();
        const res = await fetch(`/api/config/${key}` ,{
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ value })
        });
        const data = await res.json();
        if (!data.ok) {
          append("config error: " + data.error);
          return;
        }
        append(`config ${key} updated`);
      });
      document.getElementById("config-list").addEventListener("click", async () => {
        const res = await fetch("/api/config");
        const data = await res.json();
        if (!data.ok) return;
        Object.entries(data.values).forEach(([key, value]) => {
          append(`config ${key} = ${value}`);
        });
      });
      document.getElementById("config-save").addEventListener("click", () => {
        append("config saved");
      });
      document.getElementById("load-thread").addEventListener("click", async () => {
        const threadId = threadIdInput.value.trim();
        if (!threadId) return;
        const res = await fetch(`/api/threads/${threadId}/messages`);
        const data = await res.json();
        if (!data.ok) return;
        if (!data.messages.length) {
          append("no messages for thread");
          return;
        }
        data.messages.forEach((msg) => {
          append(`[${msg.role}] ${msg.content}`);
        });
      });
      async function refreshThreads() {
        const res = await fetch("/api/threads/list");
        const data = await res.json();
        threadSelect.innerHTML = "";
        if (!data.ok || !data.threads.length) {
          const option = document.createElement("option");
          option.value = "";
          option.textContent = "No threads";
          threadSelect.appendChild(option);
          return;
        }
        data.threads.forEach((thread) => {
          const option = document.createElement("option");
          option.value = thread.id;
          option.textContent = `${thread.id} (${thread.status})`;
          threadSelect.appendChild(option);
        });
        threadIdInput.value = data.threads[0].id;
      }
      threadSelect.addEventListener("change", () => {
        threadIdInput.value = threadSelect.value;
      });
      refreshThreads();
      async function refreshModels() {
        const res = await fetch("/api/models");
        const data = await res.json();
        if (!data.ok) return;
        modelCache = data.models;
        const providers = [...new Set(modelCache.map((m) => m.provider))];
        providerSelect.innerHTML = "";
        providers.forEach((provider) => {
          const option = document.createElement("option");
          option.value = provider;
          option.textContent = provider;
          providerSelect.appendChild(option);
        });
        updateModelOptions();
      }
      function updateModelOptions() {
        const provider = providerSelect.value;
        const models = modelCache.filter((m) => m.provider === provider);
        modelSelect.innerHTML = "";
        models.forEach((model) => {
          const option = document.createElement("option");
          option.value = model.id;
          option.textContent = model.id;
          modelSelect.appendChild(option);
        });
      }
      providerSelect.addEventListener("change", updateModelOptions);
      document.getElementById("refresh-models").addEventListener("click", refreshModels);
      document.getElementById("apply-model").addEventListener("click", async () => {
        const provider = providerSelect.value;
        const model = modelSelect.value;
        await fetch(`/api/config/provider`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ value: provider })
        });
        await fetch(`/api/config/model`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ value: model })
        });
        append(`model set: ${provider} / ${model}`);
      });
      refreshModels();
      document.getElementById("enqueue-job").addEventListener("click", async () => {
        const name = jobName.value.trim() || "job";
        const payload = jobPayload.value.trim();
        const res = await fetch("/api/jobs/enqueue", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ name, payload })
        });
        const data = await res.json();
        if (!data.ok) {
          append("job error: " + data.error);
          return;
        }
        append("job queued: " + data.job_id);
      });
      document.getElementById("list-jobs").addEventListener("click", async () => {
        const res = await fetch("/api/jobs");
        const data = await res.json();
        if (!data.ok) return;
        if (!data.jobs.length) {
          append("no jobs");
          return;
        }
        data.jobs.forEach((job) => {
          append(`job ${job.id} ${job.name} ${job.status} ${job.progress ?? 0}%`);
        });
      });
      document.getElementById("update-job").addEventListener("click", async () => {
        const payload = {
          job_id: jobId.value.trim(),
          status: jobStatus.value,
          progress: jobProgress.value.trim(),
          detail: jobDetail.value.trim(),
        };
        const res = await fetch("/api/jobs/update", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload)
        });
        const data = await res.json();
        if (!data.ok) {
          append("job update error: " + data.error);
          return;
        }
        append(`job ${payload.job_id} updated: ${payload.status}`);
      });
      document.getElementById("simulate-job").addEventListener("click", async () => {
        const job_id = jobId.value.trim();
        if (!job_id) return;
        for (const step of [10, 30, 60, 90, 100]) {
          const status = step === 100 ? "completed" : "progress";
          await fetch("/api/jobs/update", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              job_id,
              status,
              progress: step,
              detail: `step ${step}%`
            })
          });
          append(`job ${job_id} progress ${step}%`);
          await new Promise((resolve) => setTimeout(resolve, 300));
        }
      });
      document.getElementById("submit-approval").addEventListener("click", async () => {
        const payload = {
          approval_id: approvalId.value.trim(),
          decision: approvalDecision.value,
          remember: approvalDecision.value === "approved_for_session"
        };
        if (!payload.approval_id) return;
        const res = await fetch("/api/approvals/review", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload)
        });
        const data = await res.json();
        if (!data.ok) {
          append("approval error: " + data.error);
          return;
        }
        append(`approval ${payload.approval_id} -> ${payload.decision}`);
      });
      document.getElementById("register-tool").addEventListener("click", async () => {
        const name = registryToolName.value.trim();
        if (!name) return;
        const res = await fetch("/api/tools/register", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ name })
        });
        const data = await res.json();
        if (!data.ok) {
          append("tool register error: " + data.error);
          return;
        }
        append(`tool registered: ${name}`);
      });
      document.getElementById("register-spec").addEventListener("click", async () => {
        const name = registryToolName.value.trim();
        if (!name) return;
        const inputSchema = _safeJson(registryInput.value.trim());
        const outputSchema = _safeJson(registryOutput.value.trim());
        const timeout = registryTimeout.value.trim();
        const res = await fetch("/api/tools/specs/register", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            name,
            input_schema: inputSchema,
            output_schema: outputSchema,
            timeout_ms: timeout ? Number(timeout) : undefined
          })
        });
        const data = await res.json();
        if (!data.ok) {
          append("spec register error: " + data.error);
          return;
        }
        append(`spec registered: ${name}`);
      });
      document.getElementById("register-mcp-server").addEventListener("click", async () => {
        const name = mcpServerName.value.trim();
        if (!name) return;
        const allow = mcpAllow.value.split(",").map((item) => item.trim()).filter(Boolean);
        const deny = mcpDeny.value.split(",").map((item) => item.trim()).filter(Boolean);
        const res = await fetch("/api/mcp/servers/register", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ name, command: mcpServerCommand.value.trim(), allow, deny })
        });
        const data = await res.json();
        if (!data.ok) {
          append("mcp register error: " + data.error);
          return;
        }
        append(`mcp server registered: ${name}`);
      });
      document.getElementById("unregister-mcp-server").addEventListener("click", async () => {
        const name = mcpServerName.value.trim();
        if (!name) return;
        const res = await fetch("/api/mcp/servers/unregister", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ name })
        });
        const data = await res.json();
        if (!data.ok) {
          append("mcp unregister error: " + data.error);
          return;
        }
        append(`mcp server unregistered: ${name}`);
      });
      document.getElementById("list-mcp-servers").addEventListener("click", async () => {
        const res = await fetch("/api/mcp/servers");
        const data = await res.json();
        if (!data.ok) return;
        if (!data.servers.length) {
          append("no mcp servers");
          return;
        }
        data.servers.forEach((server) => {
          append(`mcp server ${server.name} running=${server.running}`);
        });
        const running = data.servers.filter((item) => item.running).length;
        mcpRunning.textContent = String(running);
      });
      document.getElementById("start-mcp-server").addEventListener("click", async () => {
        const name = mcpServerName.value.trim();
        if (!name) return;
        const res = await fetch("/api/mcp/servers/start", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ name })
        });
        const data = await res.json();
        if (!data.ok) {
          append("mcp start error: " + data.error);
          return;
        }
        append(`mcp server started: ${name}`);
      });
      document.getElementById("stop-mcp-server").addEventListener("click", async () => {
        const name = mcpServerName.value.trim();
        if (!name) return;
        const res = await fetch("/api/mcp/servers/stop", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ name })
        });
        const data = await res.json();
        if (!data.ok) {
          append("mcp stop error: " + data.error);
          return;
        }
        append(`mcp server stopped: ${name}`);
      });
      document.getElementById("invoke-tool").addEventListener("click", async () => {
        const name = toolName.value.trim();
        const type = toolType.value;
        const argumentsValue = toolArgs.value.trim();
        if (!name) return;
        const payload = {
          name,
          type,
          arguments: argumentsValue,
          command: type === "local_shell" ? argumentsValue : undefined,
          cwd: toolCwd.value.trim() || undefined,
          timeout_ms: toolTimeout.value ? Number(toolTimeout.value) : undefined,
          raw_arguments: type === "mcp" ? _safeJson(argumentsValue) : undefined,
          server: type === "mcp" ? mcpServer.value.trim() : undefined,
          tool: type === "mcp" ? mcpTool.value.trim() : undefined,
        };
        const res = await fetch("/api/tools/invoke", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload)
        });
        const data = await res.json();
        append("tool result: " + JSON.stringify(data));
        if (data.events) {
          data.events.forEach((event) => pushTimeline("tool", JSON.stringify(event)));
        }
      });
      async function refreshMcpTools() {
        const res = await fetch("/api/mcp/tools");
        const data = await res.json();
        mcpToolSelect.innerHTML = "";
        if (!data.ok || !data.tools.length) {
          const option = document.createElement("option");
          option.value = "";
          option.textContent = "No MCP tools";
          mcpToolSelect.appendChild(option);
          return;
        }
        data.tools.forEach((tool) => {
          const option = document.createElement("option");
          option.value = tool.qualified_name;
          option.textContent = `${tool.server_name}::${tool.tool_name}`;
          mcpToolSelect.appendChild(option);
        });
        mcpTool.value = data.tools[0].tool_name;
        mcpServer.value = data.tools[0].server_name;
        toolName.value = data.tools[0].qualified_name;
      }
      mcpToolSelect.addEventListener("change", () => {
        const selected = mcpToolSelect.value;
        const parts = selected.split("::");
        if (parts.length === 2) {
          mcpServer.value = parts[0];
          mcpTool.value = parts[1];
        }
        toolName.value = selected;
      });
      toolType.addEventListener("change", () => {
        if (toolType.value === "mcp") {
          toolName.value = mcpToolSelect.value || "";
        }
      });
      document.getElementById("refresh-mcp").addEventListener("click", refreshMcpTools);
      refreshMcpTools();

      async function refreshMcpResources() {
        const res = await fetch("/api/mcp/resources");
        const data = await res.json();
        mcpResourceSelect.innerHTML = "";
        if (!data.ok || !data.resources.length) {
          const option = document.createElement("option");
          option.value = "";
          option.textContent = "No MCP resources";
          mcpResourceSelect.appendChild(option);
          return;
        }
        data.resources.forEach((resource) => {
          const option = document.createElement("option");
          option.value = resource.uri;
          option.textContent = `${resource.server_name} ${resource.uri}`;
          mcpResourceSelect.appendChild(option);
        });
      }
      document.getElementById("refresh-mcp-resources").addEventListener("click", refreshMcpResources);
      refreshMcpResources();

      function _safeJson(value) {
        if (!value) return {};
        try {
          return JSON.parse(value);
        } catch {
          return { input: value };
        }
      }
    </script>
  </body>
</html>
"""
