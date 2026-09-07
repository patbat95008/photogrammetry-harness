"""OpenMVS: argv builders for the v2.4.0 prebuilt binaries.

**The flags below were read off the binaries in ``tools/``, not off the docs.** The
``openMVS - Source/`` checkout in this repo tracks ``develop`` and is months ahead of
the v2.4.0 drop actually being run, so its options do not all exist here. Each tool
was invoked with ``--help`` and the resulting log read back, because -- see below --
that is the only way to see anything an OpenMVS tool has to say.

Captured from ``OpenMVS x64 v2.4.0``, build date Jan 20 2026:

``InterfaceCOLMAP``
    ``-i/--input-file``, ``-o/--output-file``, ``--image-folder`` (default
    ``images/``), ``-w/--working-folder``, ``--common-intrinsics``, ``--binary``.
    Its help states that only the PINHOLE camera model is supported for import,
    which is exactly what ``colmap image_undistorter`` produces.

``DensifyPointCloud``
    ``-i/--input-file``, ``-o/--output-file``, ``-w/--working-folder``,
    ``--resolution-level`` (=1), ``--max-resolution`` (=2560), ``--min-resolution``
    (=640), ``--number-views`` (=8), ``--number-views-fuse`` (=2),
    ``--sub-resolution-levels`` (=2), ``--estimate-colors`` (=2),
    ``--estimate-normals`` (=2), ``--remove-dmaps`` (=0), ``--fusion-mode`` (=0),
    ``-m/--mask-path`` (expects ``.mask.png``, matching HANDOVER 6.7),
    ``--cuda-device`` (=-1, best GPU), ``--max-threads``, ``-v/--verbosity``.

**These tools print nothing.** Not to stdout, not to stderr, not for ``--help``, not
ever. Everything goes to ``<Tool>-<timestamp>.log`` in the *current working
directory*. Two consequences run through the whole codebase: every invocation gets an
explicit ``cwd`` so those logs land somewhere known, and progress has to be read by
tailing that file (``stages/shell.py:OpenMVSLog``) rather than from a pipe.
"""

from __future__ import annotations

import re
import tempfile
from pathlib import Path

from ..config import TOOL_SPECS, resolve_tool
from ..proc import capture

#: OpenMVS reports work as a percentage inside its log lines.
PERCENT_RE = re.compile(r"(\d{1,3})%")

#: Stem of the log file each tool writes, used to find it in the working directory.
LOG_STEMS = {
    "openmvs_interface_colmap": "InterfaceCOLMAP",
    "openmvs_densify": "DensifyPointCloud",
    "openmvs_reconstruct": "ReconstructMesh",
    "openmvs_refine": "RefineMesh",
    "openmvs_texture": "TextureMesh",
}


def _tool(key: str) -> Path:
    spec = next(s for s in TOOL_SPECS if s.key == key)
    path = resolve_tool(spec)
    if path is None:
        raise RuntimeError(f"{spec.label} not found; check the doctor page")
    return path


def version(key: str = "openmvs_densify") -> str:
    """Read the version banner out of the log, since nothing reaches stdout."""
    with tempfile.TemporaryDirectory(prefix="pgh-mvs-") as tmp:
        tmpdir = Path(tmp)
        capture([_tool(key), "--help"], cwd=tmpdir, timeout=30)
        logs = sorted(tmpdir.glob("*.log"), key=lambda p: p.stat().st_mtime)
        text = logs[-1].read_text(encoding="utf-8", errors="replace") if logs else ""
    match = re.search(r"OpenMVS\s+\S*\s*v?([0-9]+\.[0-9]+\.[0-9]+)", text)
    return match.group(1) if match else "unknown"


def interface_colmap(
    *,
    input_dir: Path,
    output_file: Path,
    image_folder: str = "images",
    working_folder: Path | None = None,
) -> list[str | Path]:
    """Convert an undistorted COLMAP workspace into a ``.mvs`` scene."""
    argv: list[str | Path] = [
        _tool("openmvs_interface_colmap"),
        "-i", input_dir,
        "-o", output_file,
        "--image-folder", image_folder,
    ]
    if working_folder is not None:
        argv += ["-w", working_folder]
    return argv


def densify_point_cloud(
    *,
    input_file: Path,
    output_file: Path,
    resolution_level: int = 1,
    max_resolution: int = 2560,
    min_resolution: int = 640,
    number_views: int = 8,
    number_views_fuse: int = 2,
    sub_resolution_levels: int = 2,
    estimate_colors: int = 2,
    estimate_normals: int = 2,
    remove_dmaps: bool = False,
    max_threads: int = 0,
    working_folder: Path | None = None,
) -> list[str | Path]:
    """Estimate per-view depth maps and fuse them into a dense point cloud.

    ``resolution_level`` is the knob that matters: it halves the working image size
    per level, and this stage exhausts system RAM long before it troubles VRAM.

    ``remove_dmaps`` is left off by default and the depth maps are deleted by the
    stage instead, so the count and size can be reported before they go.
    """
    argv: list[str | Path] = [
        _tool("openmvs_densify"),
        "-i", input_file,
        "-o", output_file,
        "--resolution-level", str(resolution_level),
        "--max-resolution", str(max_resolution),
        "--min-resolution", str(min_resolution),
        "--number-views", str(number_views),
        "--number-views-fuse", str(number_views_fuse),
        "--sub-resolution-levels", str(sub_resolution_levels),
        "--estimate-colors", str(estimate_colors),
        "--estimate-normals", str(estimate_normals),
        "--remove-dmaps", "1" if remove_dmaps else "0",
        "--max-threads", str(max_threads),
    ]
    if working_folder is not None:
        argv += ["-w", working_folder]
    return argv


def percent_progress(line: str) -> float | None:
    """Extract a 0-1 fraction from an OpenMVS log line, if it carries one."""
    match = PERCENT_RE.search(line)
    if match is None:
        return None
    value = int(match.group(1))
    return value / 100.0 if 0 <= value <= 100 else None
