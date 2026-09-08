"""Blender, driven headlessly.

The only engine here that is optional: ``config.py`` marks it ``required=False`` because
nothing before the export stage needs it. So the export stage has to check for it in
preflight rather than assuming the doctor did.

``--factory-startup`` on every invocation. Blender otherwise loads the user's
preferences, add-ons and startup file, any of which can change units, the active render
engine, or import defaults -- an export that quietly depends on whose machine it ran on.

The script talks back through a single sentinel-prefixed JSON line on stdout. Blender is
extremely chatty there, and its version, its add-on registrations and its render progress
all arrive on the same stream, so a prefix is the only reliable way to find the one line
that is a result rather than noise.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

from ..config import PROJECT_ROOT, TOOL_SPECS, resolve_tool
from ..proc import capture

#: Where the headless scripts live, beside the package rather than in it -- they run in
#: Blender's interpreter, not the harness's, and can import neither pgh nor numpy.
SCRIPTS = PROJECT_ROOT / "server" / "pgh" / "blender"

#: What a result line looks like. Everything else Blender says is ignored.
RESULT_PREFIX = "PGH_RESULT "


def blender_path() -> Path:
    spec = next(s for s in TOOL_SPECS if s.key == "blender")
    path = resolve_tool(spec)
    if path is None:
        raise RuntimeError("Blender not found; check the doctor page")
    return path


def available() -> bool:
    """Whether Blender can be found at all, without raising. Used by preflight."""
    spec = next(s for s in TOOL_SPECS if s.key == "blender")
    return resolve_tool(spec) is not None


@lru_cache(maxsize=1)
def version() -> str:
    """Cached, like the other engines: fingerprints ask for it on every page load."""
    result = capture([blender_path(), "--version"], cwd=Path.cwd(), timeout=60)
    match = re.search(r"Blender\s+([0-9]+\.[0-9]+(?:\.[0-9]+)?)", result.combined)
    return match.group(1) if match else "unknown"


def background_script(script: str | Path, job_file: Path) -> list[str | Path]:
    """Run one of the scripts in ``blender/`` headlessly against a JSON job file.

    Everything after ``--`` is hidden from Blender's own argument parser and reaches the
    script, which is how a job file is handed over without inventing an environment
    variable for it.
    """
    path = Path(script)
    if not path.is_absolute():
        path = SCRIPTS / path
    return [
        blender_path(),
        "-b",
        "--factory-startup",
        "-P",
        path,
        "--",
        job_file,
    ]


def parse_result(lines: list[str]) -> dict | None:
    """The last sentinel-prefixed JSON line, or None if the script never got there.

    The last rather than the first: a script that reported progress and then a result
    would otherwise be read as having finished at its first step.
    """
    found = None
    for line in lines:
        stripped = line.strip()
        if not stripped.startswith(RESULT_PREFIX):
            continue
        try:
            found = json.loads(stripped[len(RESULT_PREFIX) :])
        except json.JSONDecodeError:
            continue
    return found
