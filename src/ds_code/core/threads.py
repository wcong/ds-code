from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional
import time
import uuid

from ds_code.core.state import MessageRecord, StateStore, ThreadRecord


class ThreadStatus(str, Enum):
    Running = "running"
    Idle = "idle"
    Completed = "completed"
    Failed = "failed"
    Paused = "paused"
    Archived = "archived"


@dataclass
class Thread:
    id: str
    preview: str
    created_at: int
    updated_at: int
    status: ThreadStatus
    name: Optional[str] = None


@dataclass
class NewThread:
    thread: Thread
    model: str
    model_provider: str


class ThreadManager:
    def __init__(self, store: Optional[StateStore] = None) -> None:
        self._threads: Dict[str, Thread] = {}
        self._store = store or StateStore()
        self._load_from_store()

    @staticmethod
    def _now_ts() -> int:
        return int(time.time())

    @staticmethod
    def _truncate_preview(value: str) -> str:
        return value[:120]

    def create(self, model_provider: str, model: str = "auto") -> NewThread:
        thread_id = f"thread-{uuid.uuid4()}"
        now = self._now_ts()
        thread = Thread(
            id=thread_id,
            preview="New conversation",
            created_at=now,
            updated_at=now,
            status=ThreadStatus.Running,
            name=None,
        )
        self._threads[thread_id] = thread
        self._store.upsert_thread(
            ThreadRecord(
                id=thread.id,
                preview=thread.preview,
                created_at=thread.created_at,
                updated_at=thread.updated_at,
                status=thread.status.value,
                name=thread.name,
            )
        )
        return NewThread(thread=thread, model=model, model_provider=model_provider)

    def resume(self, thread_id: str, model_provider: str, model: str = "auto") -> Optional[NewThread]:
        thread = self._threads.get(thread_id)
        if thread is None:
            return None
        thread.status = ThreadStatus.Running
        thread.updated_at = self._now_ts()
        self._store.upsert_thread(
            ThreadRecord(
                id=thread.id,
                preview=thread.preview,
                created_at=thread.created_at,
                updated_at=thread.updated_at,
                status=thread.status.value,
                name=thread.name,
            )
        )
        return NewThread(thread=thread, model=model, model_provider=model_provider)

    def list_threads(self) -> List[Thread]:
        return sorted(self._threads.values(), key=lambda t: t.updated_at, reverse=True)

    def read_thread(self, thread_id: str) -> Optional[Thread]:
        return self._threads.get(thread_id)

    def set_thread_name(self, thread_id: str, name: str) -> Optional[Thread]:
        thread = self._threads.get(thread_id)
        if thread is None:
            return None
        thread.name = name
        thread.updated_at = self._now_ts()
        self._store.upsert_thread(
            ThreadRecord(
                id=thread.id,
                preview=thread.preview,
                created_at=thread.created_at,
                updated_at=thread.updated_at,
                status=thread.status.value,
                name=thread.name,
            )
        )
        return thread

    def touch_message(self, thread_id: str, message: str) -> None:
        thread = self._threads.get(thread_id)
        if thread is None:
            return
        thread.preview = self._truncate_preview(message)
        thread.updated_at = self._now_ts()
        thread.status = ThreadStatus.Running
        self._store.upsert_thread(
            ThreadRecord(
                id=thread.id,
                preview=thread.preview,
                created_at=thread.created_at,
                updated_at=thread.updated_at,
                status=thread.status.value,
                name=thread.name,
            )
        )
        self._store.append_message(thread_id, "user", message)

    def archive(self, thread_id: str) -> None:
        thread = self._threads.get(thread_id)
        if thread is None:
            return
        thread.status = ThreadStatus.Archived
        thread.updated_at = self._now_ts()
        self._store.upsert_thread(
            ThreadRecord(
                id=thread.id,
                preview=thread.preview,
                created_at=thread.created_at,
                updated_at=thread.updated_at,
                status=thread.status.value,
                name=thread.name,
            )
        )

    def unarchive(self, thread_id: str) -> None:
        thread = self._threads.get(thread_id)
        if thread is None:
            return
        thread.status = ThreadStatus.Running
        thread.updated_at = self._now_ts()
        self._store.upsert_thread(
            ThreadRecord(
                id=thread.id,
                preview=thread.preview,
                created_at=thread.created_at,
                updated_at=thread.updated_at,
                status=thread.status.value,
                name=thread.name,
            )
        )

    def list_messages(self, thread_id: str, limit: Optional[int] = None) -> List[MessageRecord]:
        return self._store.list_messages(thread_id, limit)

    def _load_from_store(self) -> None:
        for record in self._store.list_threads():
            self._threads[record.id] = Thread(
                id=record.id,
                preview=record.preview,
                created_at=record.created_at,
                updated_at=record.updated_at,
                status=ThreadStatus(record.status),
                name=record.name,
            )
