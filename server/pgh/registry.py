"""Tool discovery, version probing, and the environment health report.

Everything the harness needs from the outside world is interrogated here, so the
UI can show one honest page of what is and is not working rather than failing
three stages later with a cryptic subprocess error.
"""

from __future__ import annotations

import re
import shutil
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from . import config
from .proc import capture


@dataclass(slots=True)
class ToolStatus:
    key: str
    label: str
    found: bool
    path: str | None = None
    version: str | None = None
    required: bool = True
    cuda: bool | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.found or not self.required


def _probe_dir() -> Path:
    """A throwaway cwd for version probes.

    Every OpenMVS tool writes ``<Tool>-<timestamp>.log`` into its *current working
    directory* on every invocation, including ``--help``. Probing from a temp dir
    keeps that litter out of the project tree.
    """
    d = Path(tempfile.gettempdir()) / "pgh-probe"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _capture_openmvs(path: Path) -> str:
    """Run an OpenMVS tool and return what it wrote to its log file.

    OpenMVS tools print *nothing* to stdout or stderr -- not even for ``--help``.
    The banner (version, build date) and the full option list go only into
    ``<Tool>-<timestamp>.log`` in the working directory. So the probe runs inside a
    disposable directory and reads the log back out.
    """
    with tempfile.TemporaryDirectory(prefix="pgh-mvs-") as tmp:
        tmpdir = Path(tmp)
        capture([path, "--help"], cwd=tmpdir, timeout=30)
        logs = sorted(tmpdir.glob("*.log"), key=lambda p: p.stat().st_mtime)
        if not logs:
            return ""
        return logs[-1].read_text(encoding="utf-8", errors="replace")


# OpenMVS tools that actually have a CUDA code path worth reporting on.
_OPENMVS_CUDA_TOOLS = {"openmvs_densify", "openmvs_refine"}

_VERSION_PATTERNS: dict[str, re.Pattern[str]] = {
    "ffmpeg": re.compile(r"ffmpeg version (\S+)"),
    "ffprobe": re.compile(r"ffprobe version (\S+)"),
    "colmap": re.compile(r"COLMAP\s+([0-9]+\.[0-9]+\.[0-9]+[^\n)]*)"),
    "openmvs": re.compile(r"OpenMVS\s+\S*\s*v?([0-9]+\.[0-9]+\.[0-9]+)"),
    "blender": re.compile(r"Blender\s+([0-9]+\.[0-9]+\.[0-9]+(?:\s+LTS)?)"),
}


def _extract_version(key: str, text: str) -> str | None:
    pattern = (
        _VERSION_PATTERNS["openmvs"]
        if key.startswith("openmvs")
        else _VERSION_PATTERNS.get(key)
    )
    if pattern is None:
        return None
    match = pattern.search(text)
    return match.group(1).strip() if match else None


def probe_tool(spec: config.ToolSpec) -> ToolStatus:
    path = config.resolve_tool(spec)
    if path is None:
        return ToolStatus(
            key=spec.key,
            label=spec.label,
            found=False,
            required=spec.required,
            notes=["not found in tools/, Program Files, or PATH"],
        )

    status = ToolStatus(
        key=spec.key,
        label=spec.label,
        found=True,
        path=str(path),
        required=spec.required,
    )

    if spec.key.startswith("openmvs"):
        text = _capture_openmvs(path)
        returncode = 0 if text else -1
    else:
        result = capture([path, *spec.version_args], cwd=_probe_dir())
        text = result.combined
        returncode = result.returncode

    status.version = _extract_version(spec.key, text)

    if spec.key == "colmap":
        status.cuda = "with CUDA" in text
        if not status.cuda:
            status.notes.append("built WITHOUT CUDA - dense stereo will be very slow")
    elif spec.key.startswith("openmvs"):
        # Only the GPU-accelerated tools expose --cuda-device; InterfaceCOLMAP and
        # the mesh tools have no CUDA path at all, so absence there means nothing.
        if spec.key in _OPENMVS_CUDA_TOOLS:
            status.cuda = "--cuda-device" in text
        if not text:
            status.notes.append("ran but produced no log - could not read version")
    elif spec.key == "ffmpeg":
        if "--enable-libzimg" not in text:
            status.notes.append(
                "libzimg missing - HDR/HLG phone footage cannot be tone-mapped"
            )

    if status.version is None and returncode not in (0, 1):
        status.notes.append(f"version probe exited {returncode}")

    return status


@lru_cache(maxsize=1)
def colmap_dialect() -> dict[str, Any]:
    """Record which option namespaces this COLMAP binary accepts.

    COLMAP 4.2 splits options across two prefixes that BOTH exist: pipeline and
    GPU options moved to ``FeatureMatching.*`` / ``FeatureExtraction.*`` while
    algorithm tuning stayed on ``SiftMatching.*`` / ``SiftExtraction.*``. A blanket
    rename in either direction breaks. The argv builders consult this rather than guess.
    """
    spec = next(s for s in config.TOOL_SPECS if s.key == "colmap")
    path = config.resolve_tool(spec)
    if path is None:
        return {"available": False}

    matcher = capture(
        [path, "exhaustive_matcher", "-h"], cwd=_probe_dir(), timeout=20
    ).combined
    extractor = capture(
        [path, "feature_extractor", "-h"], cwd=_probe_dir(), timeout=20
    ).combined

    return {
        "available": True,
        "feature_matching_ns": "FeatureMatching.use_gpu" in matcher,
        "sift_matching_ns": "SiftMatching." in matcher,
        "feature_extraction_ns": "FeatureExtraction.use_gpu" in extractor,
        "sift_extraction_ns": "SiftExtraction." in extractor,
        "single_camera_per_folder": "single_camera_per_folder" in extractor,
        "mask_path": "ImageReader.mask_path" in extractor,
        "filter_stationary_matches": "filter_stationary_matches" in matcher,
    }


def probe_sam2() -> dict[str, Any]:
    """Report which SAM 2.1 checkpoints are present, paired with their configs."""
    sam2_dir = config.get_settings().sam2_dir
    models: list[dict[str, Any]] = []

    for key, spec in config.SAM2_MODELS.items():
        ckpt = sam2_dir / spec["checkpoint"]
        cfg = sam2_dir / "sam2" / spec["config"]
        models.append(
            {
                "key": key,
                "label": spec["label"],
                "checkpoint": str(ckpt),
                "checkpoint_present": ckpt.exists(),
                "checkpoint_mb": (
                    round(ckpt.stat().st_size / 1024 / 1024, 1) if ckpt.exists() else None
                ),
                "config": spec["config"],
                "config_present": cfg.exists(),
                "is_default": key == config.DEFAULT_SAM2_MODEL,
            }
        )

    importable = False
    import_error: str | None = None
    try:
        import sam2  # noqa: F401

        importable = True
    except Exception as exc:  # pragma: no cover - environment dependent
        import_error = f"{type(exc).__name__}: {exc}"

    return {
        "package_importable": importable,
        "import_error": import_error,
        "default_model": config.DEFAULT_SAM2_MODEL,
        "models": models,
    }


def probe_torch() -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:
        return {"installed": False, "error": f"{type(exc).__name__}: {exc}"}

    info: dict[str, Any] = {
        "installed": True,
        "version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": getattr(torch.version, "cuda", None),
    }
    if info["cuda_available"]:
        info["device_name"] = torch.cuda.get_device_name(0)
        info["vram_total_gb"] = round(
            torch.cuda.get_device_properties(0).total_memory / 1024**3, 1
        )
    else:
        info["warning"] = (
            "torch cannot see the GPU - SAM 2 masking would run on CPU, "
            "which is minutes per frame instead of a fraction of a second"
        )
    return info


def probe_gpu() -> dict[str, Any]:
    smi = shutil.which("nvidia-smi")
    if smi is None:
        return {"available": False}

    result = capture(
        [
            smi,
            "--query-gpu=name,memory.total,memory.used,memory.free,driver_version",
            "--format=csv,noheader,nounits",
        ],
        cwd=_probe_dir(),
        timeout=15,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return {"available": False}

    try:
        name, total, used, free, driver = (
            p.strip() for p in result.stdout.strip().splitlines()[0].split(",")
        )
    except ValueError:
        return {"available": False}

    return {
        "available": True,
        "name": name,
        "vram_total_mb": int(total),
        "vram_used_mb": int(used),
        "vram_free_mb": int(free),
        "driver_version": driver,
    }


def probe_storage() -> dict[str, Any]:
    runs_root = config.get_settings().runs_root

    # Report on the drive even when the runs directory has not been created yet.
    target = runs_root if runs_root.exists() else Path(runs_root.anchor)
    try:
        usage = shutil.disk_usage(target)
    except OSError as exc:
        return {"runs_root": str(runs_root), "error": str(exc)}

    free_gb = round(usage.free / 1024**3, 1)
    notes: list[str] = []
    if free_gb < 100:
        notes.append(f"only {free_gb} GB free - a single dense run can consume 15-40 GB")

    return {
        "runs_root": str(runs_root),
        "runs_root_exists": runs_root.exists(),
        "free_gb": free_gb,
        "total_gb": round(usage.total / 1024**3, 1),
        "long_paths_enabled": _long_paths_enabled(),
        "notes": notes,
    }


def _long_paths_enabled() -> bool | None:
    if sys.platform != "win32":
        return None
    try:
        import winreg

        with winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            r"SYSTEM\CurrentControlSet\Control\FileSystem",
        ) as key:
            value, _ = winreg.QueryValueEx(key, "LongPathsEnabled")
            return bool(value)
    except OSError:
        return False


def doctor_report() -> dict[str, Any]:
    """The full environment health report backing GET /api/doctor."""
    tools = [probe_tool(spec) for spec in config.TOOL_SPECS]
    sam2 = probe_sam2()
    torch_info = probe_torch()

    blocking: list[str] = [f"{t.label} not found" for t in tools if not t.ok]
    if not torch_info.get("installed"):
        blocking.append("torch not installed")
    elif not torch_info.get("cuda_available"):
        blocking.append("torch has no CUDA support")
    if not any(m["checkpoint_present"] for m in sam2["models"]):
        blocking.append("no SAM 2 checkpoints present")

    return {
        "project_root": str(config.PROJECT_ROOT),
        "python": sys.version.split()[0],
        "tools": [asdict(t) for t in tools],
        "colmap_dialect": colmap_dialect(),
        "sam2": sam2,
        "torch": torch_info,
        "gpu": probe_gpu(),
        "storage": probe_storage(),
        "blocking": blocking,
        "healthy": not blocking,
    }
