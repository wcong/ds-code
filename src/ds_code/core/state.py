from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional
import json
import os
import time


@dataclass
class ThreadRecord:
    id: str
    preview: str
    created_at: int
    updated_at: int
    status: str
    name: Optional[str] = None


@dataclass
class MessageRecord:
    id: str
    thread_id: str
    role: str
    content: str
    created_at: int


class StateStore:
    def __init__(self, path: Optional[str] = None) -> None:
        self._path = path or os.path.join(os.path.expanduser("~"), ".ds_code", "state.json")
        self._threads: Dict[str, ThreadRecord] = {}
        self._messages: List[MessageRecord] = []
        self._load()

    def _load(self) -> None:
        if not os.path.exists(self._path):
            return
        try:
            with open(self._path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, json.JSONDecodeError):
            return
        for item in data.get("threads", []):
            self._threads[item["id"]] = ThreadRecord(**item)
        for item in data.get("messages", []):
            self._messages.append(MessageRecord(**item))

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self._path), exist_ok=True)
        data = {
            "threads": [asdict(item) for item in self._threads.values()],
            "messages": [asdict(item) for item in self._messages],
        }
        with open(self._path, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2)

    def list_threads(self) -> List[ThreadRecord]:
        return list(self._threads.values())

    def get_thread(self, thread_id: str) -> Optional[ThreadRecord]:
        return self._threads.get(thread_id)

    def upsert_thread(self, record: ThreadRecord) -> None:
        self._threads[record.id] = record
        self._save()

    def append_message(self, thread_id: str, role: str, content: str) -> MessageRecord:
        message = MessageRecord(
            id=f"msg-{int(time.time() * 1000)}",
            thread_id=thread_id,
            role=role,
            content=content,
            created_at=int(time.time()),
        )
        self._messages.append(message)
        self._save()
        return message

    def list_messages(self, thread_id: str, limit: Optional[int] = None) -> List[MessageRecord]:
        items = [msg for msg in self._messages if msg.thread_id == thread_id]
        if limit is not None:
            items = items[-limit:]
        return items
