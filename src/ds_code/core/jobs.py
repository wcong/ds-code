from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional
import time


class JobStatus(str, Enum):
    Queued = "queued"
    Running = "running"
    Paused = "paused"
    Completed = "completed"
    Failed = "failed"
    Cancelled = "cancelled"


JOB_DETAIL_SCHEMA_VERSION = 1
DEFAULT_JOB_MAX_ATTEMPTS = 3
DEFAULT_JOB_BACKOFF_BASE_MS = 500
MAX_JOB_HISTORY_ENTRIES = 64


@dataclass
class JobRetryMetadata:
    attempt: int = 0
    max_attempts: int = DEFAULT_JOB_MAX_ATTEMPTS
    backoff_base_ms: int = DEFAULT_JOB_BACKOFF_BASE_MS
    next_backoff_ms: int = 0
    next_retry_at: Optional[int] = None


@dataclass
class JobHistoryEntry:
    at: int
    phase: str
    status: JobStatus
    progress: Optional[int]
    detail: Optional[str]
    retry: JobRetryMetadata


@dataclass
class JobRecord:
    id: str
    name: str
    status: JobStatus
    progress: Optional[int]
    detail: Optional[str]
    retry: JobRetryMetadata
    history: List[JobHistoryEntry] = field(default_factory=list)
    created_at: int = 0
    updated_at: int = 0


class JobManager:
    def __init__(self) -> None:
        self._jobs: Dict[str, JobRecord] = {}

    @staticmethod
    def _now_ts() -> int:
        return int(time.time())

    @staticmethod
    def _deterministic_backoff_ms(retry: JobRetryMetadata) -> int:
        if retry.attempt == 0:
            return 0
        exponent = min(max(retry.attempt - 1, 0), 20)
        multiplier = 1 << exponent
        return retry.backoff_base_ms * multiplier

    @staticmethod
    def _clear_retry_schedule(retry: JobRetryMetadata) -> None:
        retry.next_backoff_ms = 0
        retry.next_retry_at = None

    @staticmethod
    def _push_history(job: JobRecord, phase: str) -> None:
        job.history.append(
            JobHistoryEntry(
                at=job.updated_at,
                phase=phase,
                status=job.status,
                progress=job.progress,
                detail=job.detail,
                retry=JobRetryMetadata(
                    attempt=job.retry.attempt,
                    max_attempts=job.retry.max_attempts,
                    backoff_base_ms=job.retry.backoff_base_ms,
                    next_backoff_ms=job.retry.next_backoff_ms,
                    next_retry_at=job.retry.next_retry_at,
                ),
            )
        )
        if len(job.history) > MAX_JOB_HISTORY_ENTRIES:
            job.history = job.history[-MAX_JOB_HISTORY_ENTRIES:]

    def enqueue(self, name: str) -> JobRecord:
        now = self._now_ts()
        job_id = f"job-{int(time.time() * 1000)}"
        job = JobRecord(
            id=job_id,
            name=name,
            status=JobStatus.Queued,
            progress=0,
            detail=None,
            retry=JobRetryMetadata(),
            history=[],
            created_at=now,
            updated_at=now,
        )
        self._push_history(job, "created")
        self._jobs[job_id] = job
        return job

    def set_running(self, job_id: str) -> None:
        job = self._jobs.get(job_id)
        if job is None:
            return
        job.status = JobStatus.Running
        self._clear_retry_schedule(job.retry)
        job.updated_at = self._now_ts()
        self._push_history(job, "running")

    def update_progress(self, job_id: str, progress: int, detail: Optional[str]) -> None:
        job = self._jobs.get(job_id)
        if job is None:
            return
        job.progress = min(progress, 100)
        job.detail = detail
        job.updated_at = self._now_ts()
        self._push_history(job, "progress_updated")

    def complete(self, job_id: str) -> None:
        job = self._jobs.get(job_id)
        if job is None:
            return
        job.status = JobStatus.Completed
        job.progress = 100
        self._clear_retry_schedule(job.retry)
        job.updated_at = self._now_ts()
        self._push_history(job, "completed")

    def fail(self, job_id: str, detail: str) -> None:
        job = self._jobs.get(job_id)
        if job is None:
            return
        now = self._now_ts()
        job.status = JobStatus.Failed
        job.detail = detail
        if job.retry.attempt < job.retry.max_attempts:
            job.retry.attempt += 1
            job.retry.next_backoff_ms = self._deterministic_backoff_ms(job.retry)
            delay_secs = min((job.retry.next_backoff_ms + 999) // 1000, (2**63 - 1))
            job.retry.next_retry_at = now + int(delay_secs)
        else:
            self._clear_retry_schedule(job.retry)
        job.updated_at = now
        self._push_history(job, "failed")

    def cancel(self, job_id: str) -> None:
        job = self._jobs.get(job_id)
        if job is None:
            return
        job.status = JobStatus.Cancelled
        self._clear_retry_schedule(job.retry)
        job.updated_at = self._now_ts()
        self._push_history(job, "cancelled")

    def pause(self, job_id: str, detail: Optional[str]) -> None:
        job = self._jobs.get(job_id)
        if job is None:
            return
        job.status = JobStatus.Paused
        if detail is not None:
            job.detail = detail
        job.updated_at = self._now_ts()
        self._push_history(job, "paused")

    def resume(self, job_id: str, detail: Optional[str]) -> None:
        job = self._jobs.get(job_id)
        if job is None:
            return
        job.status = JobStatus.Running
        if detail is not None:
            job.detail = detail
        self._clear_retry_schedule(job.retry)
        job.updated_at = self._now_ts()
        self._push_history(job, "resumed")

    def list(self) -> List[JobRecord]:
        return sorted(self._jobs.values(), key=lambda job: job.updated_at, reverse=True)

    def history(self, job_id: str) -> List[JobHistoryEntry]:
        job = self._jobs.get(job_id)
        return list(job.history) if job else []

    def resume_pending(self) -> List[JobRecord]:
        resumed: List[JobRecord] = []
        for job in self._jobs.values():
            if job.status in (JobStatus.Queued, JobStatus.Running):
                job.status = JobStatus.Queued
                job.updated_at = self._now_ts()
                self._push_history(job, "queued_after_resume")
                resumed.append(job)
        return resumed
