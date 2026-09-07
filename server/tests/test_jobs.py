"""Job runner, subprocess streaming, and cancellation.

The load-bearing assertion here is that cancelling a stage leaves **no orphaned
child process**. COLMAP and OpenMVS both spawn workers; a cancel that kills only
the process we launched would leave those holding the GPU and the run directory,
and the next stage would fail for reasons that look nothing like the cause.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import psutil
import pytest

from pgh import proc
from pgh.jobs import JobRunner, JobState
from pgh.manifest import StageId, StageState
from pgh.stages.base import Stage, StageParams, StageResult
from pgh.stages.registry import StageRegistry
from pgh.store import RunStore


# -- fake stages -------------------------------------------------------------


class SleepParams(StageParams):
    seconds: int = 30


class ShellSleepStage(Stage):
    """Shells out to a long-running child, the way every real stage does."""

    id = StageId.EXTRACT
    label = "Fake shell stage"
    params_model = SleepParams
    depends_on: list[StageId] = []

    def __init__(self) -> None:
        self.pid: int | None = None
        self.started = threading.Event()

    def run(self, ctx):
        # `ping -n N 127.0.0.1` is the classic Windows sleep: no extra deps, and
        # it emits a line per second so streaming is exercised too.
        seconds = ctx.params.seconds

        def on_line(line: str) -> None:
            if line.strip():
                ctx.logger.info(line)
            ctx.progress(line[:60])

        def capture_pid(run: proc.ProcessRun) -> None:
            self.pid = run.pid
            self.started.set()

        code = proc.stream(
            ["ping", "-n", str(seconds), "127.0.0.1"],
            cwd=ctx.scratch,
            on_line=on_line,
            on_start=capture_pid,
        )
        return StageResult(metrics={"exit_code": code})


class QuickStage(Stage):
    id = StageId.EXTRACT
    label = "Quick"
    depends_on: list[StageId] = []

    def run(self, ctx):
        ctx.progress("working", fraction=0.5)
        ctx.logger.info("hello from the stage")
        (ctx.scratch / "out.txt").write_text("done", encoding="utf-8")
        return StageResult(artifacts={"out": "out.txt"}, metrics={"n": 1})


class BoomStage(Stage):
    id = StageId.EXTRACT
    label = "Boom"
    depends_on: list[StageId] = []

    def run(self, ctx):
        raise RuntimeError("deliberate failure")


# -- fixtures ----------------------------------------------------------------


@pytest.fixture
def run_store(tmp_path: Path) -> RunStore:
    return RunStore(root=tmp_path / "runs")


def make_runner(stage: Stage) -> JobRunner:
    registry = StageRegistry()
    registry.register(stage)
    runner = JobRunner(registry)
    runner.start()
    return runner


def wait_for(predicate, timeout: float = 30.0, interval: float = 0.05) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


# -- tests -------------------------------------------------------------------


def test_successful_stage_commits_artifacts_and_fingerprint(run_store):
    run = run_store.create("quick")
    runner = make_runner(QuickStage())
    try:
        job = runner.submit(run, StageId.EXTRACT)
        assert wait_for(lambda: job.state is JobState.SUCCEEDED)
    finally:
        runner.stop()

    record = run_store.get(run.run_id).io.load().stages[StageId.EXTRACT]
    assert record.state is StageState.DONE
    assert record.artifacts == {"out": "out.txt"}
    assert record.metrics == {"n": 1}
    assert record.fingerprint, "a completed stage must store its fingerprint"
    assert record.log_path


def test_failure_is_recorded_not_raised(run_store):
    run = run_store.create("boom")
    runner = make_runner(BoomStage())
    try:
        job = runner.submit(run, StageId.EXTRACT)
        assert wait_for(lambda: job.state is JobState.FAILED)
    finally:
        runner.stop()

    record = run_store.get(run.run_id).io.load().stages[StageId.EXTRACT]
    assert record.state is StageState.FAILED
    assert "deliberate failure" in record.error
    assert record.fingerprint is None, "a failed stage must not look fresh"


def test_events_are_published_and_persisted(run_store):
    run = run_store.create("events")
    runner = make_runner(QuickStage())
    try:
        job = runner.submit(run, StageId.EXTRACT)
        assert wait_for(lambda: job.state is JobState.SUCCEEDED)
    finally:
        runner.stop()

    types = [e.type for e in run.bus.replay(0)]
    assert "stage.started" in types
    assert "stage.finished" in types
    # Log lines are broadcast live but deliberately kept out of events.ndjson.
    assert "stage.log" not in types


def test_duplicate_submission_is_rejected(run_store):
    run = run_store.create("dupe")
    stage = ShellSleepStage()
    runner = make_runner(stage)
    try:
        runner.submit(run, StageId.EXTRACT)
        assert stage.started.wait(timeout=20)
        with pytest.raises(ValueError, match="already"):
            runner.submit(run, StageId.EXTRACT)
    finally:
        runner.cancel_stage(run.run_id, StageId.EXTRACT)
        runner.stop()


def test_cancel_kills_the_child_process_tree(run_store):
    """The one that matters: no orphan survives a cancel."""
    run = run_store.create("cancel")
    stage = ShellSleepStage()
    runner = make_runner(stage)
    try:
        job = runner.submit(run, StageId.EXTRACT)
        assert stage.started.wait(timeout=20), "child process never started"
        child_pid = stage.pid
        assert child_pid is not None
        assert psutil.pid_exists(child_pid)

        assert runner.cancel(job.job_id)

        assert wait_for(
            lambda: job.state in (JobState.CANCELLED, JobState.FAILED, JobState.SUCCEEDED),
            timeout=30,
        ), "job never reached a terminal state after cancel"

        # The child must be gone, or a zombie awaiting reap -- never alive.
        def child_is_dead() -> bool:
            if not psutil.pid_exists(child_pid):
                return True
            try:
                return psutil.Process(child_pid).status() == psutil.STATUS_ZOMBIE
            except psutil.NoSuchProcess:
                return True

        assert wait_for(child_is_dead, timeout=20), (
            f"pid {child_pid} survived cancellation -- the process tree leaked"
        )
    finally:
        runner.stop()


def test_streaming_forwards_output_lines(tmp_path):
    lines: list[str] = []
    code = proc.stream(
        ["ping", "-n", "2", "127.0.0.1"],
        cwd=tmp_path,
        on_line=lines.append,
    )
    assert code == 0
    assert any("127.0.0.1" in line for line in lines), lines


def test_stream_with_cancel_terminates_early(tmp_path):
    cancel = threading.Event()
    started = threading.Event()

    def on_line(_line: str) -> None:
        started.set()

    def trigger() -> None:
        started.wait(timeout=20)
        cancel.set()

    threading.Thread(target=trigger, daemon=True).start()

    begin = time.monotonic()
    proc.stream_with_cancel(
        ["ping", "-n", "60", "127.0.0.1"],
        cwd=tmp_path,
        on_line=on_line,
        cancel=cancel,
    )
    elapsed = time.monotonic() - begin
    assert elapsed < 30, f"cancel did not shorten a 60s command (took {elapsed:.1f}s)"
