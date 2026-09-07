"""Reading progress out of OpenMVS, which never says anything on stdout.

Not one byte reaches stdout or stderr from any OpenMVS tool -- not for a densify, not
for ``--help``. Everything goes to ``<Tool>-<timestamp>.log`` in the current working
directory. Without tailing that file the UI shows a motionless progress bar for the
length of a run that can take an hour, and there is no way to tell a working job from
a wedged one.

Two details the tail has to get right, both of which are silent when wrong:

* the log does not exist yet when the process starts, so it has to be waited for;
* a previous run of the same tool leaves its own log in that directory, and tailing
  that one would report stale progress for a job that has not produced any.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from pgh.stages.shell import COUNTER_RE, OpenMVSLog
from pgh.vendor import openmvs


def wait_for(predicate, timeout: float = 5.0, interval: float = 0.05) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


def collector() -> tuple[list[str], threading.Lock, callable]:
    lines: list[str] = []
    lock = threading.Lock()

    def on_line(line: str) -> None:
        with lock:
            lines.append(line)

    return lines, lock, on_line


def snapshot(lines: list[str], lock: threading.Lock) -> list[str]:
    with lock:
        return list(lines)


# -- percentage parsing ------------------------------------------------------


@pytest.mark.parametrize(
    "line,expected",
    [
        ("14:31:49 [PtchMtch] Estimated depth-map 42%", 0.42),
        ("Fused depth-maps 100%", 1.0),
        ("starting 0%", 0.0),
        ("no percentage here", None),
        ("image 518x1144 (107ms)", None),
    ],
)
def test_percent_progress(line: str, expected: float | None) -> None:
    assert openmvs.percent_progress(line) == expected


def test_impossible_percentages_are_rejected() -> None:
    """A frame size or a timing should never be mistaken for progress."""
    assert openmvs.percent_progress("999%") is None


def test_colmap_counters_are_recognised() -> None:
    """COLMAP reports [42/412] counters rather than percentages."""
    match = COUNTER_RE.search("Processed file [42/412]")
    assert match is not None
    assert match.group(1) == "42" and match.group(2) == "412"


# -- log tailing -------------------------------------------------------------


def test_tails_a_log_created_after_the_tool_starts(tmp_path: Path) -> None:
    """The file does not exist at launch; the tail has to wait for it."""
    lines, lock, on_line = collector()

    with OpenMVSLog(tmp_path, "DensifyPointCloud") as log:
        log.start(on_line)
        # Nothing yet, exactly as when the process has started but written nothing.
        assert snapshot(lines, lock) == []

        path = tmp_path / "DensifyPointCloud-20260907.log"
        path.write_text("first line\nsecond line\n", encoding="utf-8")
        assert wait_for(lambda: len(snapshot(lines, lock)) >= 2), snapshot(lines, lock)

        with path.open("a", encoding="utf-8") as fh:
            fh.write("third line\n")
        assert wait_for(lambda: len(snapshot(lines, lock)) >= 3)

    assert snapshot(lines, lock)[:3] == ["first line", "second line", "third line"]


def test_ignores_a_log_left_by_a_previous_run(tmp_path: Path) -> None:
    """Otherwise a re-run replays the last run's progress and never shows its own."""
    stale = tmp_path / "DensifyPointCloud-20260101.log"
    stale.write_text("progress from the last run 99%\n", encoding="utf-8")

    lines, lock, on_line = collector()
    with OpenMVSLog(tmp_path, "DensifyPointCloud") as log:
        log.start(on_line)
        fresh = tmp_path / "DensifyPointCloud-20260907.log"
        fresh.write_text("this run 1%\n", encoding="utf-8")
        assert wait_for(lambda: snapshot(lines, lock) != [])

    assert snapshot(lines, lock) == ["this run 1%"]


def test_ignores_logs_from_a_different_tool(tmp_path: Path) -> None:
    """Several tools share one working directory during the dense stage."""
    lines, lock, on_line = collector()
    with OpenMVSLog(tmp_path, "DensifyPointCloud") as log:
        log.start(on_line)
        (tmp_path / "InterfaceCOLMAP-20260907.log").write_text("other tool\n", encoding="utf-8")
        (tmp_path / "DensifyPointCloud-20260907.log").write_text("mine\n", encoding="utf-8")
        assert wait_for(lambda: snapshot(lines, lock) != [])

    assert "other tool" not in snapshot(lines, lock)


def test_a_partial_line_is_not_emitted_twice(tmp_path: Path) -> None:
    """A line still being written must be held back, not delivered then repeated."""
    lines, lock, on_line = collector()
    path = tmp_path / "DensifyPointCloud-20260907.log"

    with OpenMVSLog(tmp_path, "DensifyPointCloud") as log:
        log.start(on_line)
        path.write_text("complete line\nhalf a li", encoding="utf-8")
        assert wait_for(lambda: snapshot(lines, lock) == ["complete line"])
        with path.open("a", encoding="utf-8") as fh:
            fh.write("ne\n")
        assert wait_for(lambda: len(snapshot(lines, lock)) == 2)

    assert snapshot(lines, lock) == ["complete line", "half a line"]


def test_a_tool_that_writes_no_log_is_survivable(tmp_path: Path) -> None:
    """Failing to find a log must not hang or raise; the tool's exit code still counts."""
    lines, lock, on_line = collector()
    with OpenMVSLog(tmp_path, "DensifyPointCloud") as log:
        log.start(on_line)
    assert snapshot(lines, lock) == []


def test_exit_is_prompt_even_with_a_live_log(tmp_path: Path) -> None:
    """The context manager must not outlive the stage by more than a moment."""
    _, _, on_line = collector()
    started = time.time()
    with OpenMVSLog(tmp_path, "DensifyPointCloud") as log:
        log.start(on_line)
        (tmp_path / "DensifyPointCloud-1.log").write_text("x\n", encoding="utf-8")
    assert time.time() - started < 4.0
