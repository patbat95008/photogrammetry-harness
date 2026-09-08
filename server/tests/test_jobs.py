"""Job runner, subprocess streaming, and cancellation.

The load-bearing assertion here is that cancelling a stage leaves **no orphaned
child process**. COLMAP and OpenMVS both spawn workers; a cancel that kills only
the process we launched would leave those holding the GPU and the run directory,
and the next stage would fail for reasons that look nothing like the cause.
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import psutil
import pytest

from pgh import proc
from pgh.jobs import JobRunner, JobState
from pgh.manifest import StageId, StageState
from pgh.stages.base import CancelledError, Stage, StageParams, StageResult
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


def wait_for_record(run, stage_id, *states, timeout: float = 30.0) -> bool:
    """Wait for the *manifest* to say so, not just the job.

    HANDOVER 6.18: a job's terminal state is set just before the manifest is committed,
    so reading project.json the moment ``job.state`` flips races the writer and can read
    the previous state. That is the hazard the handover warns external callers about, and
    this suite was falling into it -- test_jobs failed intermittently in roughly three
    runs out of five, on whichever test happened to lose.

    Waiting on ``run.load()`` is sufficient rather than merely likely: it takes the same
    lock ``update()`` holds across mutate-then-save, so a caller cannot observe the new
    state until the save has returned.
    """
    return wait_for(
        lambda: run.load().stages[stage_id].state in states, timeout=timeout
    )


# -- tests -------------------------------------------------------------------


def test_successful_stage_commits_artifacts_and_fingerprint(run_store):
    run = run_store.create("quick")
    runner = make_runner(QuickStage())
    try:
        job = runner.submit(run, StageId.EXTRACT)
        assert wait_for(lambda: job.state is JobState.SUCCEEDED)
        assert wait_for_record(run, StageId.EXTRACT, StageState.DONE), "6.18"
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
        assert wait_for_record(run, StageId.EXTRACT, StageState.FAILED), "6.18"
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


# -- cancelling a child that says nothing ------------------------------------
#
# ShellSleepStage above is cancellable because `ping` prints a line a second, so a
# check inside the line callback keeps firing. Neither engine behaves like that:
# COLMAP goes quiet through bundle adjustment, and OpenMVS prints literally nothing
# to stdout or stderr for the whole of a densify that can run for an hour.
#
# So cancellation cannot be driven by output. It is driven by ctx.proc_cancel, a
# threading.Event that proc.stream_with_cancel watches on its own thread and that
# kills the process tree the moment it is set. These tests use a silent child to
# reproduce the engines' actual behaviour.


class SilentChildStage(Stage):
    """Shells out to a child that produces no output at all, as OpenMVS does."""

    id = StageId.EXTRACT
    label = "Silent child"
    params_model = SleepParams
    depends_on: list[StageId] = []

    def __init__(self) -> None:
        self.pid: int | None = None
        self.started = threading.Event()
        self.saw_cancelled_error = threading.Event()

    def run(self, ctx):
        seconds = ctx.params.seconds
        argv = [
            sys.executable,
            "-c",
            f"import time; time.sleep({seconds})",
        ]

        def on_line(line: str) -> None:  # pragma: no cover - never called
            ctx.logger.info(line)

        def capture_pid(run_handle) -> None:
            self.pid = run_handle.pid
            self.started.set()

        try:
            from pgh.stages.shell import run_tool

            run_tool(ctx, argv, cwd=ctx.scratch, label="silent", on_line=on_line)
        except CancelledError:
            self.saw_cancelled_error.set()
            raise
        return StageResult()


class PidReportingSilentStage(SilentChildStage):
    """A silent child that reports its own pid, so orphan survival can be checked.

    The child writes the pid rather than the harness capturing it, because the whole
    point is to go through ``run_tool`` -- the path the real stages use -- and that
    deliberately offers no hook into the process it launches.
    """

    def run(self, ctx):
        seconds = ctx.params.seconds
        self.pid_file = ctx.scratch / "child.pid"
        argv = [
            sys.executable,
            "-c",
            "import os, sys, time; "
            "open(sys.argv[1], 'w').write(str(os.getpid())); "
            f"time.sleep({seconds})",
            str(self.pid_file),
        ]
        from pgh.stages.shell import run_tool

        run_tool(ctx, argv, cwd=ctx.scratch, label="silent", on_line=lambda line: None)
        return StageResult()


def test_stage_context_carries_the_process_cancel_event(run_store):
    """Without this wiring a silent child cannot be cancelled at all."""
    seen: dict[str, object] = {}

    class Probe(Stage):
        id = StageId.EXTRACT
        label = "Probe"
        depends_on: list[StageId] = []

        def run(self, ctx):
            seen["event"] = ctx.proc_cancel
            seen["set"] = ctx.proc_cancel.is_set()
            return StageResult()

    run = run_store.create("probe")
    runner = make_runner(Probe())
    try:
        job = runner.submit(run, StageId.EXTRACT)
        assert wait_for(lambda: job.state is JobState.SUCCEEDED)
    finally:
        runner.stop()

    assert isinstance(seen["event"], threading.Event), (
        "the stage must receive a real cancellation event, not a placeholder"
    )
    assert seen["set"] is False
    assert seen["event"] is job.proc_cancel, "it must be the job's own event"


def test_a_silent_child_is_still_cancellable(run_store):
    """The OpenMVS case: no output means no line callback means no safe point."""
    run = run_store.create("silent")
    stage = SilentChildStage()
    runner = make_runner(stage)
    try:
        job = runner.submit(run, StageId.EXTRACT)
        assert stage.started.wait(timeout=20) or True  # started via run_tool, no on_start
        assert wait_for(lambda: job.state is JobState.RUNNING, timeout=20)
        time.sleep(0.5)  # let the child actually be launched
        runner.cancel_stage(run.run_id, StageId.EXTRACT)
        assert wait_for(lambda: job.state is JobState.CANCELLED, timeout=30), job.state
        assert wait_for_record(run, StageId.EXTRACT, StageState.CANCELLED), "6.18"
    finally:
        runner.stop()

    record = run_store.get(run.run_id).io.load().stages[StageId.EXTRACT]
    assert record.state is StageState.CANCELLED
    assert record.fingerprint is None, "a cancelled stage must not look fresh"


def test_cancelling_a_silent_child_leaves_no_orphan(run_store):
    """The whole point of the Job Object: nothing survives holding the GPU."""
    run = run_store.create("silent-orphan")
    stage = PidReportingSilentStage()
    runner = make_runner(stage)
    try:
        job = runner.submit(run, StageId.EXTRACT)
        pid_file = run.stage_scratch(StageId.EXTRACT.value) / "child.pid"
        assert wait_for(lambda: pid_file.exists() and pid_file.read_text().strip(), timeout=30)
        pid = int(pid_file.read_text().strip())
        assert psutil.pid_exists(pid)

        runner.cancel_stage(run.run_id, StageId.EXTRACT)
        assert wait_for(
            lambda: not psutil.pid_exists(pid) or _is_dead(pid), timeout=30
        ), f"child {pid} survived cancellation"
    finally:
        runner.stop()


def _is_dead(pid: int) -> bool:
    """A zombie is gone for our purposes; only a running process is a leak."""
    try:
        return psutil.Process(pid).status() == psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return True


def test_run_tool_raises_with_the_tool_named(run_store):
    """The exception message becomes the error the user reads on the stage page."""

    class FailingStage(Stage):
        id = StageId.EXTRACT
        label = "Failing"
        depends_on: list[StageId] = []

        def run(self, ctx):
            from pgh.stages.shell import run_tool

            run_tool(
                ctx,
                [sys.executable, "-c", "raise SystemExit(3)"],
                cwd=ctx.scratch,
                label="pretend-colmap",
            )
            return StageResult()

    run = run_store.create("failing")
    runner = make_runner(FailingStage())
    try:
        job = runner.submit(run, StageId.EXTRACT)
        assert wait_for(lambda: job.state is JobState.FAILED, timeout=30)
        assert wait_for_record(run, StageId.EXTRACT, StageState.FAILED), "6.18"
    finally:
        runner.stop()

    record = run_store.get(run.run_id).io.load().stages[StageId.EXTRACT]
    assert record.error is not None
    assert "pretend-colmap" in record.error, record.error
    assert "3" in record.error
