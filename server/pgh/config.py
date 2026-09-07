"""Settings and filesystem layout.

Tool paths are resolved once, here, from three sources in priority order:
``data/tools.local.toml`` overrides, then the vendored ``tools/`` tree, then PATH.
Nothing else in the codebase should hardcode a path to an external binary.
"""

from __future__ import annotations

import os
import shutil
import tomllib
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel

# server/pgh/config.py -> server/pgh -> server -> <project root>
PROJECT_ROOT = Path(__file__).resolve().parents[2]

TOOLS_DIR = PROJECT_ROOT / "tools"
SAM2_DIR = PROJECT_ROOT / "sam2"
DATA_DIR = PROJECT_ROOT / "data"
DOCS_DIR = PROJECT_ROOT / "docs"

# Run data lives off the project tree and off the system drive: short root keeps
# the deep dense-stage paths (dense/undistorted/stereo/depth_maps/<cam>/<slot>.jpg.geometric.bin)
# clear of MAX_PATH, which COLMAP and OpenMVS may not honour even with LongPathsEnabled.
DEFAULT_RUNS_ROOT = Path(r"D:\pgh-runs")

# Ingest is restricted to these roots; the /api/fs/browse endpoint refuses to
# escape them, so the UI cannot be talked into reading arbitrary disk.
DEFAULT_INGEST_ROOTS = [
    Path(r"D:\\"),
    Path(r"E:\\"),
    Path.home() / "Videos",
    Path.home() / "Downloads",
]


class ToolSpec(BaseModel):
    """Where a binary is expected, and how to ask it its version."""

    key: str
    label: str
    # Candidate paths relative to PROJECT_ROOT, tried in order.
    candidates: list[str] = []
    # Absolute candidates (e.g. Program Files), tried after the relative ones.
    absolute_candidates: list[str] = []
    # Fall back to this name on PATH.
    path_name: str | None = None
    version_args: list[str] = ["--version"]
    required: bool = True


TOOL_SPECS: list[ToolSpec] = [
    ToolSpec(
        key="ffmpeg",
        label="FFmpeg",
        path_name="ffmpeg",
        version_args=["-version"],
    ),
    ToolSpec(
        key="ffprobe",
        label="ffprobe",
        path_name="ffprobe",
        version_args=["-version"],
    ),
    ToolSpec(
        key="colmap",
        label="COLMAP",
        candidates=["tools/colmap-x64-windows-cuda/bin/colmap.exe"],
        path_name="colmap",
        version_args=["-h"],  # COLMAP prints its version banner in help
    ),
    ToolSpec(
        key="openmvs_densify",
        label="OpenMVS DensifyPointCloud",
        candidates=["tools/OpenMVS_Windows_x64_CUDA/DensifyPointCloud.exe"],
        version_args=["--help"],
    ),
    ToolSpec(
        key="openmvs_interface_colmap",
        label="OpenMVS InterfaceCOLMAP",
        candidates=["tools/OpenMVS_Windows_x64_CUDA/InterfaceCOLMAP.exe"],
        version_args=["--help"],
    ),
    ToolSpec(
        key="openmvs_reconstruct",
        label="OpenMVS ReconstructMesh",
        candidates=["tools/OpenMVS_Windows_x64_CUDA/ReconstructMesh.exe"],
        version_args=["--help"],
    ),
    ToolSpec(
        key="openmvs_refine",
        label="OpenMVS RefineMesh",
        candidates=["tools/OpenMVS_Windows_x64_CUDA/RefineMesh.exe"],
        version_args=["--help"],
    ),
    ToolSpec(
        key="openmvs_texture",
        label="OpenMVS TextureMesh",
        candidates=["tools/OpenMVS_Windows_x64_CUDA/TextureMesh.exe"],
        version_args=["--help"],
    ),
    ToolSpec(
        key="blender",
        label="Blender",
        absolute_candidates=[
            r"C:\Program Files\Blender Foundation\Blender 5.2\blender.exe",
            r"C:\Program Files\Blender Foundation\Blender 5.0\blender.exe",
            r"C:\Program Files\Blender Foundation\Blender 4.4\blender.exe",
            r"C:\Program Files\Blender Foundation\Blender 4.2\blender.exe",
        ],
        path_name="blender",
        version_args=["--version"],
        required=False,  # only needed at the export stage
    ),
]


# SAM 2 checkpoints are paired with their Hydra config here, once. A mismatched
# pair loads WITHOUT raising and silently produces poor masks, so this mapping is
# the single source of truth -- never derive the config from the filename at a call site.
SAM2_MODELS: dict[str, dict[str, str]] = {
    "tiny": {
        "checkpoint": "checkpoints/sam2.1_hiera_tiny.pt",
        "config": "configs/sam2.1/sam2.1_hiera_t.yaml",
        "label": "SAM 2.1 Hiera Tiny",
    },
    "small": {
        "checkpoint": "checkpoints/sam2.1_hiera_small.pt",
        "config": "configs/sam2.1/sam2.1_hiera_s.yaml",
        "label": "SAM 2.1 Hiera Small",
    },
    "base_plus": {
        "checkpoint": "checkpoints/sam2.1_hiera_base_plus.pt",
        "config": "configs/sam2.1/sam2.1_hiera_b+.yaml",
        "label": "SAM 2.1 Hiera Base+",
    },
    "large": {
        "checkpoint": "checkpoints/sam2.1_hiera_large.pt",
        "config": "configs/sam2.1/sam2.1_hiera_l.yaml",
        "label": "SAM 2.1 Hiera Large",
    },
}

DEFAULT_SAM2_MODEL = "large"


class Settings(BaseModel):
    project_root: Path = PROJECT_ROOT
    runs_root: Path = DEFAULT_RUNS_ROOT
    ingest_roots: list[Path] = DEFAULT_INGEST_ROOTS
    sam2_dir: Path = SAM2_DIR
    tool_overrides: dict[str, Path] = {}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load settings, applying data/tools.local.toml overrides if present."""
    settings = Settings()

    override_file = DATA_DIR / "tools.local.toml"
    if override_file.exists():
        with override_file.open("rb") as fh:
            raw = tomllib.load(fh)
        if runs_root := raw.get("runs_root"):
            settings.runs_root = Path(runs_root)
        if roots := raw.get("ingest_roots"):
            settings.ingest_roots = [Path(p) for p in roots]
        settings.tool_overrides = {
            k: Path(v) for k, v in (raw.get("tools") or {}).items()
        }

    if env_runs := os.environ.get("PGH_RUNS_ROOT"):
        settings.runs_root = Path(env_runs)

    return settings


def resolve_tool(spec: ToolSpec) -> Path | None:
    """Find a binary: explicit override, then vendored tree, then PATH."""
    settings = get_settings()

    if override := settings.tool_overrides.get(spec.key):
        return override if override.exists() else None

    for rel in spec.candidates:
        candidate = PROJECT_ROOT / rel
        if candidate.exists():
            return candidate

    for absolute in spec.absolute_candidates:
        candidate = Path(absolute)
        if candidate.exists():
            return candidate

    if spec.path_name and (found := shutil.which(spec.path_name)):
        return Path(found)

    return None
