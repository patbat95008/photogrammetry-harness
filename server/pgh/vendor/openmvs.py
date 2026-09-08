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

``ReconstructMesh``
    ``-i/--input-file``, ``-p/--pointcloud-file``, ``-o/--output-file``,
    ``-d/--min-point-distance`` (=1.5), ``--integrate-only-roi`` (=0),
    ``--constant-weight`` (=1), ``-f/--free-space-support`` (=0),
    ``--thickness-factor`` (=1), ``--quality-factor`` (=1), and a Clean pass:
    ``--decimate`` (=1), ``--target-face-num`` (=0), ``--remove-spurious`` (=20),
    ``--remove-spikes`` (=1), ``--close-holes`` (=30), ``--smooth`` (=2),
    ``--edge-length`` (=0), ``--roi-border`` (=0), ``--crop-to-roi`` (=1).
    ``--export-type`` (=ply) accepts **ply or obj only**. ``--cuda-device`` (=-1).

``RefineMesh``
    ``-i/--input-file``, ``-m/--mesh-file``, ``-o/--output-file``,
    ``--resolution-level`` (=0), ``--min-resolution`` (=640), ``--max-views`` (=8),
    ``--decimate`` (=0, auto), ``--close-holes`` (=30), ``--ensure-edge-size`` (=1),
    ``--max-face-area`` (=32), ``--scales`` (=2), ``--scale-step`` (=0.5),
    ``--alternate-pair`` (=0), ``--regularity-weight`` (=0.2),
    ``--rigidity-elasticity-ratio`` (=0.9), ``--gradient-step`` (=45.05),
    ``--planar-vertex-ratio`` (=0), ``--reduce-memory`` (=1).
    ``--export-type`` (=ply) accepts **ply or obj only**.
    ``--cuda-device`` (=**-2**, i.e. the CPU).

``TextureMesh``
    ``-i/--input-file``, ``-m/--mesh-file``, ``-o/--output-file``,
    ``--decimate`` (=1), ``--close-holes`` (=30), ``--resolution-level`` (=0),
    ``--min-resolution`` (=640), ``--outlier-threshold`` (=0.06),
    ``--cost-smoothness-ratio`` (=0.1), ``--virtual-face-images`` (=0),
    ``--global-seam-leveling`` (=1), ``--local-seam-leveling`` (=1),
    ``--texture-size-multiple`` (=0), ``--patch-packing-heuristic`` (=3),
    ``--empty-color`` (=16744231), ``--sharpness-weight`` (=0.5),
    ``--orthographic-image-resolution`` (=0), ``--ignore-mask-label`` (=-1),
    ``--max-texture-size`` (=8192). ``--export-type`` (=ply) accepts
    **ply, obj, glb or gltf** -- the only one of the three that does.
    ``--cuda-device`` (=-1).

**These tools print nothing.** Not to stdout, not to stderr, not for ``--help``, not
ever. Everything goes to ``<Tool>-<timestamp>.log`` in the *current working
directory*. Two consequences run through the whole codebase: every invocation gets an
explicit ``cwd`` so those logs land somewhere known, and progress has to be read by
tailing that file (``stages/shell.py:OpenMVSLog``) rather than from a pipe.

**Five things the help text does not say, all confirmed by running the binaries
against the cup-1 dense scene rather than read off the ``develop`` source.**

*Export types are not uniform.* Only ``TextureMesh`` accepts ``glb``/``gltf``; the
other two take ``ply``/``obj`` and silently coerce anything else to PLY. Ask
``ReconstructMesh`` for ``glb`` and you get a file called ``mesh.ply`` while the caller
looks for ``mesh.glb`` -- which is why the builders below raise instead. The whole
chain therefore runs in PLY and only the last step chooses a format.

*``RefineMesh`` defaults to the CPU.* Its ``--cuda-device`` default is ``-2`` where the
other two default to ``-1`` (best GPU). That is the engine's own judgement about which
path is safe, so the harness keeps it and makes the GPU something you ask for.

*Image paths inside a ``.mvs`` are relative to the working folder,* not to the scene
file. ``scene_dense.mvs`` names its photographs ``undistorted/images/cup/000019.jpg``
and OpenMVS resolves that against ``-w``. Get it wrong and ``TextureMesh`` reports a
missing image for a file that is plainly there.

*The mesh is written to ``<-o with its extension stripped> + <export type>``, and the
``.mvs`` named by ``-o`` may never be written at all* -- it is skipped when the archive
type is the default and the scene was loaded in interface format, which is exactly this
pipeline. Confirmed: ``-o mesh.ply`` produced ``mesh.ply`` and no ``mesh.mvs``. So
nothing downstream may chain on ``-o``; pass the original dense scene plus ``-m``.

*A ``.glb`` is not self-contained.* ``TextureMesh`` writes the atlas beside it as
``<stem>_0.png`` and references it from the glTF by relative URI, so the mesh is two
files rather than one. It is still fewer files and far fewer bytes than the OBJ trio,
and it carries UVs and a ``KHR_materials_unlit`` material -- which is the right material
for a photographic texture that already contains the lighting from the shoot.
"""

from __future__ import annotations

import re
import tempfile
from functools import lru_cache
from pathlib import Path

from ..config import TOOL_SPECS, resolve_tool
from ..proc import capture

#: A percentage inside an OpenMVS log line. Anchored on the left and tolerant of a
#: decimal part, because ``(\d{1,3})%`` reads "points inside ROI (71.46%)" as 46% --
#: it matches the two digits before the sign rather than the number they belong to.
PERCENT_RE = re.compile(r"(?<![\d.])(\d{1,3})(?:\.\d+)?\s*%")

#: Mesh sizes, as the tools report them: "Mesh reconstruction completed: 249632
#: vertices, 498780 faces (21s6ms)". Several lines per run match, including the scene
#: load's "50189 points, 0 vertices, 0 faces", so callers keep the LAST match.
MESH_COUNT_RE = re.compile(r"(\d+)\s+vertices,\s+(\d+)\s+faces")

#: Export types each tool accepts. Only TextureMesh understands glb/gltf; the other two
#: silently write PLY for anything they do not recognise, so asking is worse than
#: refusing -- the file lands under a name the caller is not looking for.
MESH_EXPORT_TYPES = ("ply", "obj")
TEXTURE_EXPORT_TYPES = ("ply", "obj", "glb", "gltf")

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


@lru_cache(maxsize=4)
def version(key: str = "openmvs_densify") -> str:
    """Read the version banner out of the log, since nothing reaches stdout.

    Cached, following ``registry.colmap_dialect()``. Every stage that names a tool
    version in its fingerprint calls this, and fingerprints are recomputed on every
    stage page load and once per run when the runs list is drawn -- so uncached, simply
    listing runs spawns a process per stage per run and waits on each one. A binary
    swapped underneath a running server keeps its old version until restart, which is
    the same trade the dialect probe already makes.
    """
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
    ignore_mask_label: int | None = None,
    mask_path: Path | None = None,
) -> list[str | Path]:
    """Estimate per-view depth maps and fuse them into a dense point cloud.

    ``resolution_level`` is the knob that matters: it halves the working image size
    per level, and this stage exhausts system RAM long before it troubles VRAM.

    ``remove_dmaps`` is left off by default and the depth maps are deleted by the
    stage instead, so the count and size can be reported before they go.

    **Masks arrive as siblings, never through** ``-m``. The tool's own help says
    ``-m`` is "path to folder containing mask images with '.mask.png' extension" --
    one flat folder -- and it builds each name with ``Util::getFileName``, which
    strips the directory *and* the extension. Two camera groups sharing the slot
    clock (HANDOVER 4.1) therefore both map ``<group>/000042.jpg`` onto
    ``000042.mask.png``, and one camera's mask is silently applied to the other. It
    looks fine on a single-camera run and is wrong on every rig. So this builder
    refuses ``mask_path`` outright, and masks are written beside each undistorted
    image instead, which ``ignore_mask_label`` reads per-image.

    ``ignore_mask_label`` must be >= 0 or no mask is read at all: -1 is the default
    and means "estimate a lens-distortion mask", not "use the files on disk".
    """
    _refuse_mask_path(mask_path)
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
    if ignore_mask_label is not None:
        argv += ["--ignore-mask-label", str(ignore_mask_label)]
    if working_folder is not None:
        argv += ["-w", working_folder]
    return argv


def _refuse_mask_path(mask_path: Path | None) -> None:
    if mask_path is None:
        return
    raise ValueError(
        "DensifyPointCloud's -m/--mask-path takes one flat folder and names each "
        "mask with Util::getFileName, which strips the directory and the extension. "
        "Two camera groups share the slot clock, so cam_high/000042.jpg and "
        "cam_eye/000042.jpg both resolve to 000042.mask.png and one camera's mask is "
        "applied to the other -- silently, and only on a rig, so a single-camera test "
        "run looks perfect. Write <stem>.mask.png beside each undistorted image and "
        "pass ignore_mask_label instead."
    )


def _check_export_type(value: str, allowed: tuple[str, ...], tool: str) -> None:
    if value not in allowed:
        raise ValueError(
            f"{tool} cannot export {value!r}; it accepts {' or '.join(allowed)}. It "
            "would not refuse this itself -- it writes PLY for anything it does not "
            "recognise, so the file would land under a name nobody is looking for."
        )


def reconstruct_mesh(
    *,
    input_file: Path,
    output_file: Path,
    point_cloud_file: Path | None = None,
    min_point_distance: float = 1.5,
    free_space_support: bool = False,
    constant_weight: bool = True,
    decimate: float = 1.0,
    target_face_num: int = 0,
    remove_spurious: float = 20.0,
    remove_spikes: bool = True,
    close_holes: int = 30,
    smooth: int = 2,
    crop_to_roi: bool = True,
    export_type: str = "ply",
    cuda_device: int = -1,
    max_threads: int = 0,
    working_folder: Path | None = None,
) -> list[str | Path]:
    """Build a surface from a dense point cloud by Delaunay triangulation and graph cut.

    ``point_cloud_file`` is not the optional extra it looks like. The ``.mvs`` the dense
    stage writes carries only the *sparse* cloud -- its own save line reads "50189
    points, 0 vertices, 0 faces" -- while the dense points live beside it in
    ``scene_dense.ply``. Without ``-p`` this reconstructs the sparse cloud and returns a
    mesh built from a fifteenth of the data, with nothing in the log calling it an error.

    ``target_face_num`` and ``decimate`` both bound the size, and the engine folds them
    into a single target with ``target_face_num`` winning whenever it is not 0.
    """
    _check_export_type(export_type, MESH_EXPORT_TYPES, "ReconstructMesh")
    argv: list[str | Path] = [
        _tool("openmvs_reconstruct"),
        "-i", input_file,
        "-o", output_file,
    ]
    if point_cloud_file is not None:
        argv += ["-p", point_cloud_file]
    argv += [
        "--min-point-distance", str(min_point_distance),
        "--free-space-support", "1" if free_space_support else "0",
        "--constant-weight", "1" if constant_weight else "0",
        "--decimate", str(decimate),
        "--target-face-num", str(target_face_num),
        "--remove-spurious", str(remove_spurious),
        "--remove-spikes", "1" if remove_spikes else "0",
        "--close-holes", str(close_holes),
        "--smooth", str(smooth),
        "--crop-to-roi", "1" if crop_to_roi else "0",
        "--export-type", export_type,
        "--cuda-device", str(cuda_device),
        "--max-threads", str(max_threads),
    ]
    if working_folder is not None:
        argv += ["-w", working_folder]
    return argv


def refine_mesh(
    *,
    input_file: Path,
    mesh_file: Path,
    output_file: Path,
    resolution_level: int = 0,
    min_resolution: int = 640,
    max_views: int = 8,
    decimate: float = 0.0,
    close_holes: int = 30,
    scales: int = 2,
    scale_step: float = 0.5,
    max_face_area: int = 32,
    regularity_weight: float = 0.2,
    reduce_memory: bool = True,
    export_type: str = "ply",
    cuda_device: int = -2,
    max_threads: int = 0,
    working_folder: Path | None = None,
) -> list[str | Path]:
    """Push a reconstructed surface back against the photographs.

    ``cuda_device`` defaults to ``-2`` -- the CPU -- because that is what the binary
    itself defaults to, alone among the three mesh tools. Mirroring the engine rather
    than "helpfully" defaulting to the GPU keeps this layer a faithful record of what
    the tool does; the stage above turns a plain switch into it.
    """
    _check_export_type(export_type, MESH_EXPORT_TYPES, "RefineMesh")
    argv: list[str | Path] = [
        _tool("openmvs_refine"),
        "-i", input_file,
        "-m", mesh_file,
        "-o", output_file,
        "--resolution-level", str(resolution_level),
        "--min-resolution", str(min_resolution),
        "--max-views", str(max_views),
        "--decimate", str(decimate),
        "--close-holes", str(close_holes),
        "--scales", str(scales),
        "--scale-step", str(scale_step),
        "--max-face-area", str(max_face_area),
        "--regularity-weight", str(regularity_weight),
        "--reduce-memory", "1" if reduce_memory else "0",
        "--export-type", export_type,
        "--cuda-device", str(cuda_device),
        "--max-threads", str(max_threads),
    ]
    if working_folder is not None:
        argv += ["-w", working_folder]
    return argv


def texture_mesh(
    *,
    input_file: Path,
    mesh_file: Path,
    output_file: Path,
    resolution_level: int = 0,
    min_resolution: int = 640,
    decimate: float = 1.0,
    close_holes: int = 30,
    outlier_threshold: float = 0.06,
    cost_smoothness_ratio: float = 0.1,
    global_seam_leveling: bool = True,
    local_seam_leveling: bool = True,
    sharpness_weight: float = 0.5,
    max_texture_size: int = 8192,
    export_type: str = "ply",
    cuda_device: int = -1,
    max_threads: int = 0,
    working_folder: Path | None = None,
) -> list[str | Path]:
    """Paint a mesh with the photographs it was reconstructed from.

    The only one of the three that can write ``glb``, and the only one that reads the
    image pixels -- so it is where a wrong ``working_folder`` shows up, because the
    paths inside the ``.mvs`` are resolved against it rather than against the scene file.
    """
    _check_export_type(export_type, TEXTURE_EXPORT_TYPES, "TextureMesh")
    argv: list[str | Path] = [
        _tool("openmvs_texture"),
        "-i", input_file,
        "-m", mesh_file,
        "-o", output_file,
        "--resolution-level", str(resolution_level),
        "--min-resolution", str(min_resolution),
        "--decimate", str(decimate),
        "--close-holes", str(close_holes),
        "--outlier-threshold", str(outlier_threshold),
        "--cost-smoothness-ratio", str(cost_smoothness_ratio),
        "--global-seam-leveling", "1" if global_seam_leveling else "0",
        "--local-seam-leveling", "1" if local_seam_leveling else "0",
        "--sharpness-weight", str(sharpness_weight),
        "--max-texture-size", str(max_texture_size),
        "--export-type", export_type,
        "--cuda-device", str(cuda_device),
        "--max-threads", str(max_threads),
    ]
    if working_folder is not None:
        argv += ["-w", working_folder]
    return argv


def mesh_counts(line: str) -> tuple[int, int] | None:
    """Vertices and faces from a log line that reports them, if it does.

    Keep the last match across a tool's output rather than the first: the scene load
    reports "50189 points, 0 vertices, 0 faces" before any mesh exists.
    """
    match = MESH_COUNT_RE.search(line)
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


def percent_progress(line: str) -> float | None:
    """Extract a 0-1 fraction from an OpenMVS log line, if it carries one.

    Treat the result as a hint rather than as progress. v2.4.0 emits no marching
    percentage at all: of the 1,144 lines a densify wrote for the cup run, the six
    carrying a percent sign were every one of them a statistic. A bar driven straight
    from this jumps to an arbitrary place and stops, so pair it with phase markers and
    never let the reported fraction go backwards.
    """
    match = PERCENT_RE.search(line)
    if match is None:
        return None
    value = int(match.group(1))
    return value / 100.0 if 0 <= value <= 100 else None
