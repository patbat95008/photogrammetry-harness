"""The only module in this codebase permitted to import ``subprocess``.

Three rules, all load-bearing on this machine:

1. **argv is always a list, shell is always False.** The project path contains
   spaces and a ``' - '`` sequence; a single f-string command line produces a
   quoting bug that reproduces nowhere else.
2. **cwd is always explicit.** Every OpenMVS tool writes a timestamped ``.log``
   into its *current working directory* on every invocation -- and prints nothing
   to stdout or stderr at all. Left to chance those logs scatter across the repo.
3. **Children die with their job.** COLMAP and OpenMVS spawn worker processes.
   Terminating only the process we launched orphans them, holding the GPU and the
   run directory. On Windows the reliable fix is a Job Object.
"""

from __future__ import annotations

import subprocess  # noqa: S404 -- see module docstring; banned everywhere else
import sys
import threading
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)

LineCallback = Callable[[str], None]


@dataclass(slots=True)
class CaptureResult:
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool = False

    @property
    def combined(self) -> str:
        return f"{self.stdout}\n{self.stderr}".strip()


def capture(
    argv: list[str | Path],
    *,
    cwd: Path,
    timeout: float = 30.0,
) -> CaptureResult:
    """Run a short-lived command to completion and capture its output.

    For probes only. Anything that can run for minutes uses :func:`stream`.
    """
    args = [str(a) for a in argv]
    try:
        completed = subprocess.run(  # noqa: S603 -- list argv, shell=False
            args,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            shell=False,
            creationflags=_NO_WINDOW,
        )
    except subprocess.TimeoutExpired as exc:
        return CaptureResult(
            returncode=-1,
            stdout=exc.stdout if isinstance(exc.stdout, str) else "",
            stderr=exc.stderr if isinstance(exc.stderr, str) else "",
            timed_out=True,
        )
    except (OSError, ValueError) as exc:
        return CaptureResult(returncode=-1, stdout="", stderr=str(exc))

    return CaptureResult(
        returncode=completed.returncode,
        stdout=completed.stdout or "",
        stderr=completed.stderr or "",
    )


def capture_bytes(
    argv: list[str | Path],
    *,
    cwd: Path,
    timeout: float = 300.0,
) -> tuple[int, bytes, str]:
    """Run a command and capture stdout as raw bytes. Returns (code, stdout, stderr).

    Needed wherever the payload is binary -- decoded PCM, raw frames. The text-mode
    :func:`capture` decodes with ``errors="replace"``, which silently corrupts any
    byte sequence that is not valid UTF-8, and that damage cannot be undone.
    """
    args = [str(a) for a in argv]
    try:
        completed = subprocess.run(  # noqa: S603 -- list argv, shell=False
            args,
            cwd=str(cwd),
            capture_output=True,
            timeout=timeout,
            shell=False,
            creationflags=_NO_WINDOW,
        )
    except subprocess.TimeoutExpired:
        return -1, b"", "timed out"
    except (OSError, ValueError) as exc:
        return -1, b"", str(exc)

    stderr = (completed.stderr or b"").decode("utf-8", errors="replace")
    return completed.returncode, completed.stdout or b"", stderr


class _JobObject:
    """A Windows Job Object that kills the whole process tree when terminated.

    Falls back to a psutil walk if pywin32 is unavailable, which is less reliable
    against grandchildren that re-parent but is better than leaking processes.
    """

    def __init__(self) -> None:
        self._handle = None
        if sys.platform != "win32":
            return
        try:
            import win32job

            handle = win32job.CreateJobObject(None, "")
            info = win32job.QueryInformationJobObject(
                handle, win32job.JobObjectExtendedLimitInformation
            )
            # KILL_ON_JOB_CLOSE means that even if we crash, closing our handle
            # takes the children with it.
            info["BasicLimitInformation"]["LimitFlags"] |= (
                win32job.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            )
            win32job.SetInformationJobObject(
                handle, win32job.JobObjectExtendedLimitInformation, info
            )
            self._handle = handle
        except Exception:  # pragma: no cover - pywin32 missing or policy-restricted
            self._handle = None

    def assign(self, pid: int) -> None:
        if self._handle is None:
            return
        try:
            import win32api
            import win32con
            import win32job

            process = win32api.OpenProcess(
                win32con.PROCESS_SET_QUOTA | win32con.PROCESS_TERMINATE, False, pid
            )
            try:
                win32job.AssignProcessToJobObject(self._handle, process)
            finally:
                win32api.CloseHandle(process)
        except Exception:  # pragma: no cover
            pass

    def terminate(self) -> bool:
        if self._handle is None:
            return False
        try:
            import win32job

            win32job.TerminateJobObject(self._handle, 1)
            return True
        except Exception:  # pragma: no cover
            return False

    def close(self) -> None:
        if self._handle is None:
            return
        try:
            import win32api

            win32api.CloseHandle(self._handle)
        except Exception:  # pragma: no cover
            pass
        finally:
            self._handle = None


def _kill_tree_fallback(pid: int) -> None:
    try:
        import psutil

        parent = psutil.Process(pid)
        for child in parent.children(recursive=True):
            child.kill()
        parent.kill()
    except Exception:  # pragma: no cover
        pass


class ProcessRun:
    """A running child process whose output is pumped line by line."""

    def __init__(self, popen: subprocess.Popen[str], job: _JobObject) -> None:
        self._popen = popen
        self._job = job

    @property
    def pid(self) -> int:
        return self._popen.pid

    def kill(self) -> None:
        """Terminate the process and every descendant it spawned."""
        if not self._job.terminate():
            _kill_tree_fallback(self._popen.pid)


def stream(
    argv: Iterable[str | Path],
    *,
    cwd: Path,
    on_line: LineCallback,
    on_start: Callable[[ProcessRun], None] | None = None,
    env: dict[str, str] | None = None,
) -> int:
    """Run a command, forwarding every output line to ``on_line``. Returns the exit code.

    stdout and stderr are merged deliberately: COLMAP logs progress to *stderr* via
    glog, ffmpeg writes ``-progress`` output to stdout, and a caller that watched
    only one of them would see a silent job.
    """
    args = [str(a) for a in argv]
    cwd.mkdir(parents=True, exist_ok=True)

    job = _JobObject()
    popen = subprocess.Popen(  # noqa: S603 -- list argv, shell=False
        args,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        shell=False,
        env=env,
        creationflags=_NO_WINDOW,
    )
    job.assign(popen.pid)

    run = ProcessRun(popen, job)
    if on_start is not None:
        on_start(run)

    try:
        assert popen.stdout is not None
        for line in popen.stdout:
            on_line(line.rstrip("\r\n"))
        return popen.wait()
    finally:
        if popen.stdout is not None:
            popen.stdout.close()
        if popen.poll() is None:
            run.kill()
            popen.wait(timeout=10)
        job.close()


def stream_with_cancel(
    argv: Iterable[str | Path],
    *,
    cwd: Path,
    on_line: LineCallback,
    cancel: threading.Event,
    env: dict[str, str] | None = None,
) -> int:
    """As :func:`stream`, but kills the process tree as soon as ``cancel`` is set."""
    holder: dict[str, ProcessRun] = {}
    stop_watch = threading.Event()

    def watch() -> None:
        while not stop_watch.wait(0.2):
            if cancel.is_set():
                run = holder.get("run")
                if run is not None:
                    run.kill()
                return

    watcher = threading.Thread(target=watch, daemon=True, name="proc-cancel-watch")
    watcher.start()
    try:
        return stream(
            argv,
            cwd=cwd,
            on_line=on_line,
            on_start=lambda run: holder.__setitem__("run", run),
            env=env,
        )
    finally:
        stop_watch.set()
        watcher.join(timeout=2)
