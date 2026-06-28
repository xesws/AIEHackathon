"""Single-writer background consolidation queue for online serving."""
from __future__ import annotations

import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

from memory import consolidate
from serving import shadow_editing


@dataclass
class Job:
    id: str
    trigger: str = "manual"
    ids: Optional[list[str]] = None
    status: str = "queued"
    n_written: Optional[int] = None
    error: Optional[str] = None
    submitted_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None


_q: "queue.Queue[str | None]" = queue.Queue()
_jobs: dict[str, Job] = {}
_lock = threading.Lock()
_worker: threading.Thread | None = None
_stop = threading.Event()


def _snapshot(job: Job) -> dict:
    return {
        "id": job.id,
        "trigger": job.trigger,
        "ids": list(job.ids) if job.ids is not None else None,
        "status": job.status,
        "n_written": job.n_written,
        "error": job.error,
        "submitted_at": job.submitted_at,
        "started_at": job.started_at,
        "finished_at": job.finished_at,
    }


def _worker_loop() -> None:
    while not _stop.is_set():
        job_id = _q.get()
        try:
            if job_id is None:
                return
            with _lock:
                job = _jobs[job_id]
                job.status = "running"
                job.started_at = time.time()
            try:
                n_written = consolidate.run_pass(
                    job.trigger,
                    ids=job.ids,
                    editing_module=shadow_editing,
                )
                with _lock:
                    job.status = "succeeded"
                    job.n_written = n_written
                    job.finished_at = time.time()
            except Exception as exc:
                with _lock:
                    job.status = "failed"
                    job.error = repr(exc)
                    job.finished_at = time.time()
        finally:
            _q.task_done()


def start() -> None:
    """Start the single background editor worker if it is not already running."""
    global _worker
    with _lock:
        if _worker is not None and _worker.is_alive():
            return
        _stop.clear()
        _worker = threading.Thread(target=_worker_loop, name="engram-async-editor", daemon=True)
        _worker.start()


def stop(timeout: float = 5.0) -> None:
    """Ask the worker to stop; a long-running edit may outlive the timeout."""
    global _worker
    worker = _worker
    if worker is None:
        return
    _stop.set()
    _q.put(None)
    worker.join(timeout=timeout)
    if not worker.is_alive():
        with _lock:
            _worker = None


def submit(*, trigger: str = "manual", ids: Optional[list[str]] = None) -> dict:
    """Queue one consolidation pass and return its job snapshot immediately."""
    start()
    job = Job(id=uuid.uuid4().hex, trigger=trigger, ids=list(ids) if ids is not None else None)
    with _lock:
        _jobs[job.id] = job
        snapshot = _snapshot(job)
    _q.put(job.id)
    return snapshot


def get(job_id: str) -> dict | None:
    with _lock:
        job = _jobs.get(job_id)
        return _snapshot(job) if job is not None else None


def status() -> dict:
    with _lock:
        jobs = [_snapshot(j) for j in _jobs.values()]
        running = [j for j in jobs if j["status"] == "running"]
        queued = [j for j in jobs if j["status"] == "queued"]
        failed = [j for j in jobs if j["status"] == "failed"]
        succeeded = [j for j in jobs if j["status"] == "succeeded"]
        worker_alive = _worker is not None and _worker.is_alive()
    return {
        "worker_alive": worker_alive,
        "queue_depth": len(queued),
        "running": running[0] if running else None,
        "counts": {
            "queued": len(queued),
            "running": len(running),
            "succeeded": len(succeeded),
            "failed": len(failed),
        },
        "latest": jobs[-10:],
    }
