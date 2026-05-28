from typing import Optional

from ds_code.core.jobs import JobHistoryEntry, JobManager, JobRecord


class TaskManager:
    def __init__(self, jobs: Optional[JobManager] = None) -> None:
        self._jobs = jobs or JobManager()

    def submit(self, name: str, payload: str) -> JobRecord:
        _ = payload
        return self._jobs.enqueue(name)

    def list_jobs(self) -> list[JobRecord]:
        return self._jobs.list()

    def set_running(self, job_id: str) -> None:
        self._jobs.set_running(job_id)

    def update_progress(self, job_id: str, progress: int, detail: Optional[str]) -> None:
        self._jobs.update_progress(job_id, progress, detail)

    def complete(self, job_id: str) -> None:
        self._jobs.complete(job_id)

    def fail(self, job_id: str, detail: str) -> None:
        self._jobs.fail(job_id, detail)

    def cancel(self, job_id: str) -> None:
        self._jobs.cancel(job_id)

    def pause(self, job_id: str, detail: Optional[str]) -> None:
        self._jobs.pause(job_id, detail)

    def resume(self, job_id: str, detail: Optional[str]) -> None:
        self._jobs.resume(job_id, detail)

    def job_history(self, job_id: str) -> list[JobHistoryEntry]:
        return self._jobs.history(job_id)
