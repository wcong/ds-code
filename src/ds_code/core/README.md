# Core

## Overview
The core layer owns threads, jobs, state persistence, and tool execution decisions. It is the main orchestration engine used by the runtime API.

## Files
- [engine.py](engine.py): CoreEngine that wires LLM, tools, jobs, threads, and exec policy.
- [jobs.py](jobs.py): JobManager and job history with retry metadata.
- [threads.py](threads.py): ThreadManager for creating/resuming threads and storing messages.
- [state.py](state.py): StateStore that persists threads and messages to ~/.ds_code/state.json.
- [__init__.py](__init__.py): Re-exports core symbols.

## How It Fits Together
- [engine.py](engine.py) receives tool calls from [runtime/api.py](../runtime/api.py) and uses [tools/registry.py](../tools/registry.py) to dispatch them.
- Thread changes are persisted via [state.py](state.py), which is used by [threads.py](threads.py).
- Job state is tracked by [jobs.py](jobs.py) and surfaced through [tasks/manager.py](../tasks/manager.py) and the web API.
