"""Running an external tool from inside a stage.

``extract.py`` shells out to ffmpeg by checking ``ctx.cancel`` inside its line
callback, which works only because ffmpeg narrates constantly. COLMAP goes quiet
for minutes during bundle adjustment and OpenMVS prints nothing whatsoever -- not
one byte, ever, not even for ``--help`` (see HANDOVER 6.4). A cancellation check
driven by output would therefore never fire for the two engines that actually need
it, and the user would be left with a wedged 40-minute densify holding the GPU.

So everything here drives cancellation from ``ctx.proc_cancel``, a
``threading.Event`` watched by a separate thread inside ``proc.stream_with_cancel``,
which kills the whole process tree the moment it is set.
"""

from __future__ import annotations

import re
import threading
import time
from collections.abc import Callable, Iterable
from pathlib import Path

import psutil

from .. import proc
from ..vendor import openmvs
from .base import Progress, StageContext

#: COLMAP reports progress as bracketed counters, never percentages (HANDOVER 6.5).
COUNTER_RE = re.compile(r"\[(\d+)\s*/\s*(\d+)\]")

LineCallback = Callable[[str], None]


def run_tool(
    ctx: StageContext,
    argv: Iterable[str | Path],
    *,
    cwd: Path,
    label: str,
    on_line: LineCallback | None = None,
) -> None:
    """Stream a child process, logging every line, and raise if it fails.

    ``label`` appears both in the log prefix and in the exception message, and that
    message becomes the error string the user sees on the stage page -- so name the
    tool, not the function.
    """
    argv = list(argv)
    cwd.mkdir(parents=True, exist_ok=True)
    ctx.logger.info("%s %s", label, " ".join(str(a) for a in argv[1:]))

    def handle(line: str) -> None:
        if line.strip():
            ctx.logger.info("[%s] %s", label, line)
        if on_line is not None:
            on_line(line)

    code = proc.stream_with_cancel(
        argv, cwd=cwd, on_line=handle, cancel=ctx.proc_cancel
    )

    # A killed process exits non-zero. Check cancellation first so the user sees
    # "cancelled", not a spurious failure they did not cause.
    ctx.cancel.raise_if_cancelled()
    if code != 0:
        raise RuntimeError(f"{label} exited {code} -- see the stage log for its output")


def counter_progress(
    ctx: StageContext, message: str, *, scale: tuple[float, float] | None = None
) -> LineCallback:
    """A line callback that turns COLMAP's ``[42/412]`` counters into progress.

    ``scale`` maps the counter onto a sub-range of the stage's overall progress, so
    several tools can share one bar: ``(0.0, 0.4)`` for the first of them, and so on.
    """

    def handle(line: str) -> None:
        match = COUNTER_RE.search(line)
        if match is None:
            return
        current, total = int(match.group(1)), int(match.group(2))
        if total <= 0:
            return
        if scale is None:
            ctx.progress(message, current=current, total=total)
        else:
            low, high = scale
            ctx.progress(message, fraction=low + (high - low) * (current / total))

    return handle


class OpenMVSLog:
    """Tails the log file an OpenMVS tool drops in its working directory.

    OpenMVS writes ``<Tool>-<timestamp>.log`` into the *current* directory and sends
    nothing to stdout or stderr, so this is the only way to see what it is doing or
    whether it is still alive. The file does not exist yet when the process starts,
    hence the polling for it rather than opening it up front.
    """

    def __init__(self, cwd: Path, tool: str) -> None:
        self.cwd = cwd
        self.tool = tool
        self._before: set[Path] = set()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self.path: Path | None = None

    def __enter__(self) -> OpenMVSLog:
        self.cwd.mkdir(parents=True, exist_ok=True)
        # Snapshot first: a previous run of the same tool leaves its own log behind,
        # and tailing that would report stale progress for a job that never started.
        self._before = set(self.cwd.glob(f"{self.tool}-*.log"))
        return self

    def start(self, on_line: LineCallback) -> None:
        self._thread = threading.Thread(
            target=self._tail, args=(on_line,), daemon=True, name=f"mvs-log-{self.tool}"
        )
        self._thread.start()

    def _find(self) -> Path | None:
        fresh = set(self.cwd.glob(f"{self.tool}-*.log")) - self._before
        if not fresh:
            return None
        return max(fresh, key=lambda p: p.stat().st_mtime)

    def _tail(self, on_line: LineCallback) -> None:
        # Poll for the log first: the tool has started but has not necessarily
        # created it yet, and on a large scene that gap runs to seconds.
        while self.path is None:
            self.path = self._find()
            if self.path is not None:
                break
            if self._stop.wait(0.2):
                return

        # Binary, not text. A text-mode seek only accepts offsets that tell()
        # produced; seeking to a byte count raises ValueError, which would kill this
        # thread silently and freeze the progress bar for the rest of the run.
        offset = 0
        while True:
            try:
                with self.path.open("rb") as fh:
                    fh.seek(offset)
                    while True:
                        raw = fh.readline()
                        # A line still being written is held back and re-read next
                        # pass, so it is never delivered twice nor truncated.
                        if not raw.endswith(b"\n"):
                            break
                        offset += len(raw)
                        on_line(raw.decode("utf-8", errors="replace").rstrip("\r\n"))
            except OSError:
                pass
            if self._stop.wait(0.2):
                return

    def __exit__(self, *exc: object) -> None:
        # Let the tail catch the final lines the tool wrote as it exited.
        time.sleep(0.4)
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3)


#: How often a silent OpenMVS tool is reported as still alive. TextureMesh spent 4m2s
#: on one phase without writing a single log line during the cup run, so without a
#: heartbeat the message would freeze at whatever it last said.
HEARTBEAT_S = 5.0


def _elapsed(seconds: float) -> str:
    minutes, secs = divmod(int(seconds), 60)
    return f"{minutes}m{secs:02d}s" if minutes else f"{secs}s"


def _percent_only(line: str) -> tuple[float, str] | None:
    fraction = openmvs.percent_progress(line)
    return None if fraction is None else (fraction, "")


def run_openmvs(
    ctx: StageContext,
    argv: Iterable[str | Path],
    *,
    cwd: Path,
    tool: str,
    label: str,
    span: tuple[float, float],
    progress_from: Callable[[str], tuple[float, str] | None] | None = None,
    on_line: LineCallback | None = None,
) -> None:
    """Run an OpenMVS tool, taking progress from the log file it drops.

    Nothing arrives on stdout or stderr, so without the log tail the UI would show a
    motionless bar for however long the tool takes.

    ``progress_from`` maps a log line to a fraction within ``span`` and a short phase
    name, defaulting to reading a percentage out of the line. Percentages are a poor
    signal here -- v2.4.0 mostly emits statistics rather than progress -- so callers
    with a table of phase markers should pass their own.

    Two things keep the bar honest. The reported fraction never decreases, so a stray
    statistic cannot drag it backwards. And a heartbeat re-reports the current position
    with a running elapsed time, so a tool that says nothing for four minutes reads as
    working rather than wedged.
    """
    low, high = span
    mapper = progress_from or _percent_only
    started = time.monotonic()
    state = {"fraction": low, "phase": ""}
    lock = threading.Lock()

    def publish() -> None:
        parts = [label]
        if state["phase"]:
            parts.append(state["phase"])
        parts.append(_elapsed(time.monotonic() - started))
        # ctx.progress() checks the cancel token, which would raise on the tailing and
        # heartbeat threads where nothing can catch it. Report directly.
        ctx.report(Progress(fraction=state["fraction"], message=" · ".join(parts)))

    def handle_log_line(line: str) -> None:
        if line.strip():
            ctx.logger.info("[%s] %s", label, line)
        if on_line is not None:
            on_line(line)

        found = mapper(line)
        if found is None:
            return
        fraction, phase = found
        with lock:
            # Monotonic: a percentage in an OpenMVS log is far more often a statistic
            # than progress, and a bar that jumps backwards is worse than a still one.
            state["fraction"] = max(state["fraction"], low + (high - low) * fraction)
            if phase:
                state["phase"] = phase
        publish()

    stop = threading.Event()

    def heartbeat() -> None:
        while not stop.wait(HEARTBEAT_S):
            publish()

    beat = threading.Thread(target=heartbeat, daemon=True, name=f"mvs-beat-{tool}")
    beat.start()
    try:
        with OpenMVSLog(cwd, tool) as log:
            log.start(handle_log_line)
            run_tool(ctx, argv, cwd=cwd, label=label)
    finally:
        stop.set()
        beat.join(timeout=2)


class PeakMemory:
    """Samples system memory while a stage runs, so the report can name the peak.

    The dense and mesh stages both exhaust system RAM long before they trouble the GPU,
    and "peaked at 61 of 64 GB" is the number that tells you to raise a resolution level
    before attempting anything larger.
    """

    def __init__(self, interval: float = 2.0) -> None:
        self.interval = interval
        self.peak_gb = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        def sample() -> None:
            while not self._stop.wait(self.interval):
                used = psutil.virtual_memory().used / 1e9
                self.peak_gb = max(self.peak_gb, used)

        self._thread = threading.Thread(target=sample, daemon=True, name="stage-mem")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3)
