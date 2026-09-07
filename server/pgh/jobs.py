"""Single-worker job queue.

One worker, deliberately. Both COLMAP's dense stereo and OpenMVS's densify will
happily consume all 24 GB of VRAM and most of 64 GB of RAM; running two stages
concurrently is the fastest route to an out-of-memory failure halfway through an
hour of compute. Queueing is the feature.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Any

from .fingerprint import compute_fingerprints
from .manifest import RunManifest, StageId, StageRecord, StageState
from .stages.base import CancelledError, CancelToken, Progress, StageContext
from .stages.registry import StageRegistry, registry as default_registry
from .store import RunHandle

logger = logging.getLogger("pgh.jobs")

#: Progress events are emitted at most this often, so a fast inner loop cannot
#: flood the SSE stream. Log lines are not throttled -- they are the useful detail.
PROGRESS_INTERVAL_S = 0.2


class JobState(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Job:
    job_id: str
    run_id: str
    stage_id: StageId
    state: JobState = JobState.QUEUED
    queued_at: float = field(default_factory=time.monotonic)
    started_at: float | None = None
    finished_at: float | None = None
    error: str | None = None
    cancel: CancelToken = field(default_factory=CancelToken)
    #: Set when cancellation must also kill a child process tree.
    proc_cancel: threading.Event = field(default_factory=threading.Event)

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "run_id": self.run_id,
            "stage_id": self.stage_id.value,
            "state": self.state.value,
            "error": self.error,
            "duration_s": (
                round((self.finished_at or time.monotonic()) - self.started_at, 2)
                if self.started_at
                else None
            ),
        }


class JobRunner:
    """Owns the worker thread and the job queue."""

    def __init__(self, stage_registry: StageRegistry | None = None) -> None:
        self.registry = stage_registry or default_registry
        self._queue: queue.Queue[Job] = queue.Queue()
        self._jobs: dict[str, Job] = {}
        self._runs: dict[str, RunHandle] = {}
        self._current: Job | None = None
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stopping = threading.Event()
        self._counter = 0

    # -- lifecycle ----------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._work, daemon=True, name="pgh-worker")
        self._thread.start()

    def stop(self) -> None:
        self._stopping.set()
        with self._lock:
            if self._current is not None:
                self._current.cancel.cancel()
                self._current.proc_cancel.set()

    # -- submission ---------------------------------------------------------

    def submit(self, run: RunHandle, stage_id: StageId) -> Job:
        stage = self.registry.get(stage_id)
        if stage is None:
            raise ValueError(f"stage {stage_id} is not implemented yet")

        with self._lock:
            for job in self._jobs.values():
                if (
                    job.run_id == run.run_id
                    and job.stage_id == stage_id
                    and job.state in (JobState.QUEUED, JobState.RUNNING)
                ):
                    raise ValueError(f"{stage_id} is already {job.state} for this run")
            self._counter += 1
            job = Job(job_id=f"job{self._counter:05d}", run_id=run.run_id, stage_id=stage_id)
            self._jobs[job.job_id] = job
            self._runs[run.run_id] = run

        run.update(lambda m: _mark_queued(m, stage_id))
        run.bus.publish("stage.queued", stage=stage_id.value, job=job.to_dict())
        self._queue.put(job)
        self.start()
        return job

    def cancel(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.state not in (JobState.QUEUED, JobState.RUNNING):
                return False
            job.cancel.cancel()
            job.proc_cancel.set()
            if job.state is JobState.QUEUED:
                job.state = JobState.CANCELLED
            return True

    def cancel_stage(self, run_id: str, stage_id: StageId) -> bool:
        with self._lock:
            job = next(
                (
                    j
                    for j in self._jobs.values()
                    if j.run_id == run_id
                    and j.stage_id == stage_id
                    and j.state in (JobState.QUEUED, JobState.RUNNING)
                ),
                None,
            )
        return self.cancel(job.job_id) if job else False

    def active(self) -> list[dict[str, Any]]:
        with self._lock:
            return [
                j.to_dict()
                for j in self._jobs.values()
                if j.state in (JobState.QUEUED, JobState.RUNNING)
            ]

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    # -- the worker ---------------------------------------------------------

    def _work(self) -> None:
        while not self._stopping.is_set():
            try:
                job = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if job.state is JobState.CANCELLED:
                self._finalise_cancelled(job)
                continue

            with self._lock:
                self._current = job
            try:
                self._execute(job)
            except Exception:  # pragma: no cover - defensive
                logger.exception("worker crashed on %s", job.job_id)
            finally:
                with self._lock:
                    self._current = None
                self._queue.task_done()

    def _execute(self, job: Job) -> None:
        run = self._runs[job.run_id]
        stage = self.registry.get(job.stage_id)
        assert stage is not None

        job.state = JobState.RUNNING
        job.started_at = time.monotonic()

        log_path = run.logs_dir / f"{job.stage_id.value}__{_stamp()}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        scratch = run.stage_scratch(job.stage_id.value)

        run.update(lambda m: _mark_running(m, job.stage_id, log_path, run.dir))
        run.bus.publish("stage.started", stage=job.stage_id.value, job=job.to_dict())

        last_progress = 0.0

        with log_path.open("w", encoding="utf-8", buffering=1) as log_file:
            stage_logger = _make_logger(job, log_file, run)

            def report(progress: Progress) -> None:
                nonlocal last_progress
                now = time.monotonic()
                if now - last_progress < PROGRESS_INTERVAL_S and progress.fraction != 1.0:
                    return
                last_progress = now
                run.bus.publish(
                    "stage.progress",
                    stage=job.stage_id.value,
                    fraction=progress.fraction,
                    message=progress.message,
                    current=progress.current,
                    total=progress.total,
                )

            manifest = run.load()
            params = stage.params_model.model_validate(
                manifest.stages[job.stage_id].params or {}
            )

            ctx = StageContext(
                run_dir=run.dir,
                manifest=manifest,
                params=params,
                logger=stage_logger,
                cancel=job.cancel,
                report=report,
                scratch=scratch,
                proc_cancel=job.proc_cancel,
            )

            try:
                _reset_scratch(scratch)
                result = stage.run(ctx)
            except CancelledError:
                job.state = JobState.CANCELLED
                job.finished_at = time.monotonic()
                stage_logger.warning("cancelled by user")
                _reset_scratch(scratch, remove=True)
                run.update(lambda m: _mark_terminal(m, job.stage_id, StageState.CANCELLED))
                run.bus.publish("stage.cancelled", stage=job.stage_id.value, job=job.to_dict())
                return
            except Exception as exc:
                job.state = JobState.FAILED
                job.error = f"{type(exc).__name__}: {exc}"
                job.finished_at = time.monotonic()
                stage_logger.error("failed: %s", job.error)
                stage_logger.debug(traceback.format_exc())
                _reset_scratch(scratch, remove=True)
                run.update(
                    lambda m: _mark_terminal(m, job.stage_id, StageState.FAILED, job.error)
                )
                run.bus.publish(
                    "stage.failed", stage=job.stage_id.value, error=job.error, job=job.to_dict()
                )
                return

            job.state = JobState.SUCCEEDED
            job.finished_at = time.monotonic()
            duration = round(job.finished_at - (job.started_at or 0), 2)
            stage_logger.info("completed in %ss", duration)
            _reset_scratch(scratch, remove=True)

            def commit(m: RunManifest) -> None:
                record = m.stages[job.stage_id]
                record.state = StageState.DONE
                record.finished_at = _iso()
                record.duration_s = duration
                record.artifacts = result.artifacts
                record.metrics = result.metrics
                record.warnings = result.warnings
                record.tool_versions = result.tool_versions
                record.error = None
                record.stale_reason = None
                # Stamp the fingerprint only after the artifacts are committed, so a
                # crash between the two leaves the stage stale rather than falsely fresh.
                record.fingerprint = compute_fingerprints(m, self.registry)[job.stage_id]

            run.update(commit)
            run.bus.publish(
                "stage.finished",
                stage=job.stage_id.value,
                duration_s=duration,
                metrics=result.metrics,
                warnings=result.warnings,
                job=job.to_dict(),
            )

    def _finalise_cancelled(self, job: Job) -> None:
        run = self._runs.get(job.run_id)
        if run is None:
            return
        run.update(lambda m: _mark_terminal(m, job.stage_id, StageState.CANCELLED))
        run.bus.publish("stage.cancelled", stage=job.stage_id.value, job=job.to_dict())


# -- helpers -----------------------------------------------------------------


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _reset_scratch(scratch: Path, *, remove: bool = False) -> None:
    """Stages write into scratch and commit by rename, so it always starts clean."""
    import shutil

    shutil.rmtree(scratch, ignore_errors=True)
    if not remove:
        scratch.mkdir(parents=True, exist_ok=True)


def _mark_queued(manifest: RunManifest, stage_id: StageId) -> None:
    record = manifest.stages.setdefault(stage_id, StageRecord())
    record.error = None


def _mark_running(
    manifest: RunManifest, stage_id: StageId, log_path: Path, run_dir: Path
) -> None:
    record = manifest.stages.setdefault(stage_id, StageRecord())
    record.state = StageState.RUNNING
    record.started_at = _iso()
    record.finished_at = None
    record.error = None
    record.log_path = str(log_path.relative_to(run_dir)).replace("\\", "/")


def _mark_terminal(
    manifest: RunManifest, stage_id: StageId, state: StageState, error: str | None = None
) -> None:
    record = manifest.stages.setdefault(stage_id, StageRecord())
    record.state = state
    record.finished_at = _iso()
    record.error = error


class _BusLogHandler(logging.Handler):
    """Tees stage log lines onto the event bus so the UI sees them live."""

    def __init__(self, run: RunHandle, stage_id: StageId) -> None:
        super().__init__()
        self.run = run
        self.stage_id = stage_id

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.run.bus.publish(
                "stage.log",
                stage=self.stage_id.value,
                level=record.levelname,
                line=self.format(record),
            )
        except Exception:  # pragma: no cover
            pass


def _make_logger(job: Job, log_file: Any, run: RunHandle) -> logging.Logger:
    stage_logger = logging.getLogger(f"pgh.stage.{job.run_id}.{job.stage_id.value}")
    stage_logger.setLevel(logging.DEBUG)
    stage_logger.propagate = False
    for handler in list(stage_logger.handlers):
        stage_logger.removeHandler(handler)
        handler.close()

    file_handler = logging.StreamHandler(log_file)
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(message)s"))
    stage_logger.addHandler(file_handler)

    bus_handler = _BusLogHandler(run, job.stage_id)
    bus_handler.setFormatter(logging.Formatter("%(message)s"))
    stage_logger.addHandler(bus_handler)
    return stage_logger


runner = JobRunner()
