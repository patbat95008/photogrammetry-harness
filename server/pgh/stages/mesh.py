"""Stage 6: turn a dense cloud into a surface, and paint it with the photographs.

Three tools in sequence, of which the middle one is optional:

1. ``ReconstructMesh`` triangulates the dense points and carves a surface out of the
   result with a graph cut.
2. ``RefineMesh`` pushes that surface back against the images. Slow, memory-hungry,
   and off by default.
3. ``TextureMesh`` unwraps the mesh and bakes an atlas from the photographs.

**Refinement is off by default and that is not timidity.** OpenMVS ships ``RefineMesh``
with ``--cuda-device -2`` -- the CPU -- alone among the three, which is the engine's own
judgement about how routine that path is. With it off the chain is a few minutes and
produces something worth looking at; arm it once the cheap pass has succeeded.

**The working folder is the scratch, and it has to be.** Image paths inside a ``.mvs``
are stored relative to the working folder, so ``scene_dense.mvs`` names its photographs
``undistorted/images/cup/000019.jpg`` and OpenMVS resolves that against ``-w``. Pointing
``-w`` at the committed ``dense/`` directory would work, but each tool also drops its
log into the working folder, so a failed mesh run would litter a finished stage's
output. Hardlinking the images into the scratch instead keeps ``-w`` and ``cwd`` the
same directory and leaves ``dense/`` untouched. The links cost nothing and are removed
before the commit, so the disk report does not count those bytes twice.

**The mesh is discovered, not assumed.** OpenMVS builds its output name from the stem
of ``-o`` plus ``--export-type``, so the extension is decided by a different flag from
the one that names the file -- and the ``.mvs`` that ``-o`` appears to ask for is never
written at all for a scene loaded in interface format. Nothing here may chain on it.

**Progress cannot come from percentages.** These tools emit statistics with percent
signs in them and no marching percentage at all, and TextureMesh spent four minutes on
a single phase of the cup run without writing one line. So progress is a table of phase
markers with a monotonic floor, and ``run_openmvs`` beats a heartbeat underneath it so
a silent tool reads as working rather than wedged.
"""

from __future__ import annotations

import os
import shutil
import time
from pathlib import Path
from typing import Any, Callable, Literal

import psutil
from pydantic import Field

from .. import ply
from ..manifest import RunManifest, StageId
from ..vendor import openmvs
from .base import Stage, StageContext, StageParams, StageResult
from .shell import PeakMemory, run_openmvs

#: Below this the graph cut did not find a surface, it found nothing.
EMPTY_FACES_WARN = 1000

#: Above this nothing downstream benefits: the browser viewer crawls and the export
#: stage decimates it anyway. The cup came out at 498k, for scale.
BUSY_FACES_WARN = 2_000_000

#: A textured mesh larger than this is not loaded into the viewer without being asked.
LARGE_MESH_WARN_BYTES = 40_000_000

#: Refining on the processor for longer than this is worth naming the GPU switch over.
SLOW_REFINE_WARN_S = 600

#: Phase markers, read off a real ReconstructMesh run over the cup-1 dense cloud. Each
#: entry is (substring, fraction of this tool's span reached, what it starts next).
#: The fractions are its observed timings, not guesses: tetrahedralization finished at
#: 4s of 21s, weighting at 11s, the graph cut at 21s.
RECONSTRUCT_PHASES: list[tuple[str, float, str]] = [
    ("Delaunay tetrahedralization completed", 0.20, "weighting"),
    ("weighting completed", 0.52, "graph cut"),
    ("graph-cut completed", 0.86, "building the mesh"),
    ("Mesh reconstruction completed", 0.90, "cleaning"),
    ("Cleaned mesh", 0.97, "saving"),
    ("saved", 1.00, ""),
]

#: TextureMesh, from the same run. Assigning views took 4m2s of the 4m17s total, so it
#: is nearly the whole span -- a bar that treated it as half would sit at 50% for four
#: minutes and then leap to done.
TEXTURE_PHASES: list[tuple[str, float, str]] = [
    ("Assigning the best view to each face completed", 0.94, "building the atlas"),
    ("Generating texture atlas and image completed", 0.99, "saving"),
    ("Mesh texturing completed", 1.00, ""),
]

#: RefineMesh, which has NOT yet been run end to end here -- unlike the two above,
#: these markers are read from its output format rather than from a completed run. A
#: marker that never matches costs nothing: the bar holds at its floor and the
#: heartbeat keeps reporting elapsed time, which is the honest thing to show anyway.
REFINE_PHASES: list[tuple[str, float, str]] = [
    ("Mesh subdivided", 0.30, "optimising"),
    ("Mesh refinement completed", 0.95, "saving"),
    ("saved", 1.00, ""),
]

#: Extensions a mesh may come back as, most specific first.
MESH_SUFFIXES = (".glb", ".gltf", ".obj", ".ply")


class MeshParams(StageParams):
    refine: bool = Field(
        False,
        description=(
            "Push the surface back against the photographs after it is built. This is "
            "the slow step and the one that runs out of memory, and OpenMVS ships it "
            "defaulting to the processor rather than the graphics card. Leave it off "
            "until the cheap pass has produced a mesh worth refining."
        ),
    )
    refine_resolution_level: int = Field(
        1,
        ge=0,
        le=4,
        description=(
            "How many times to halve the images while refining. 0 is full resolution "
            "and the most detail; each step up cuts memory and time to roughly a "
            "quarter. Raise this first if refinement exhausts memory."
        ),
    )
    refine_scales: int = Field(
        2,
        ge=1,
        le=4,
        description=(
            "How many times to subdivide and re-solve while refining. Each scale "
            "roughly doubles the faces and the time, and buys detail only where the "
            "photographs actually have it."
        ),
    )
    refine_on_gpu: bool = Field(
        False,
        description=(
            "Refine on the graphics card. OpenMVS ships this step defaulting to the "
            "processor, so the default here is its default; try the card once the "
            "stage has succeeded once, and note it is the step most likely to run the "
            "card out of memory."
        ),
    )
    min_point_distance: float = Field(
        1.5,
        ge=0.0,
        le=10.0,
        description=(
            "How close two dense points may project before one is dropped while "
            "triangulating. Lower keeps more of the cloud and gives a finer, larger, "
            "noisier mesh; higher smooths and shrinks it."
        ),
    )
    free_space_support: bool = Field(
        False,
        description=(
            "Let the space the cameras saw through carve the surface. It recovers thin "
            "structures that would otherwise be filled in solid, and it will also carve "
            "into a genuine surface the cameras only ever saw at a glancing angle."
        ),
    )
    target_face_num: int = Field(
        0,
        ge=0,
        le=20_000_000,
        description=(
            "Cap the mesh at this many triangles, or 0 for no cap. Prefer this to the "
            "decimation fraction below: it is an absolute size you can reason about, "
            "and it overrides that fraction whenever it is not 0."
        ),
    )
    decimate: float = Field(
        1.0,
        gt=0.0,
        le=1.0,
        description=(
            "Keep this fraction of the triangles. 1 disables it. A fraction of an "
            "unknown number is not a size, so prefer the cap above; this is here for "
            "trimming a mesh you have already measured."
        ),
    )
    remove_spurious: float = Field(
        20.0,
        ge=0.0,
        le=200.0,
        description=(
            "How aggressively to delete small disconnected fragments. Raise it when a "
            "reflective or transparent surface has left a haze of floating shells "
            "around the subject; too high and it takes thin real parts with them."
        ),
    )
    close_holes: int = Field(
        30,
        ge=0,
        le=1000,
        description=(
            "Fill holes bounded by up to this many edges. Anything filled here is "
            "invented rather than measured -- the crown of a head shot from a single "
            "elevation is the case to watch."
        ),
    )
    smooth: int = Field(
        2,
        ge=0,
        le=10,
        description=(
            "Smoothing passes over the finished surface. Each one removes noise, and a "
            "little real detail with it."
        ),
    )
    crop_to_roi: bool = Field(
        True,
        description=(
            "Discard everything outside the region the cameras concentrated on. This is "
            "what keeps the room out of the model; turn it off if the subject itself is "
            "coming back clipped."
        ),
    )
    export_type: Literal["glb", "obj", "ply"] = Field(
        "glb",
        description=(
            "What to write. GLB is one binary file plus its texture image, which is "
            "what the viewer here and Blender both want. OBJ writes three files and is "
            "easier to hand-edit. PLY carries no texture at all and will display "
            "untextured."
        ),
    )
    texture_resolution_level: int = Field(
        0,
        ge=0,
        le=4,
        description=(
            "How many times to halve the photographs before sampling colour from them. "
            "0 is full resolution."
        ),
    )
    max_texture_size: int = Field(
        8192,
        ge=1024,
        le=16384,
        description=(
            "Largest texture page to produce. This is most of the file's size: halving "
            "it quarters both the bytes and the video memory the viewer needs."
        ),
    )


class MeshStage(Stage):
    id = StageId.MESH
    label = "Mesh"
    description = "Build a surface from the dense cloud and paint it with the photographs."
    params_model = MeshParams

    def external_inputs(self, manifest: RunManifest) -> dict[str, Any]:
        # Three keys, and one version probe rather than one per tool: every key here
        # costs a subprocess with a timeout on every stage page load, and all five
        # OpenMVS binaries ship from a single build carrying a single version.
        #
        # capture.flip_x is deliberately absent. It is a display and export choice, and
        # HANDOVER 8 assigns it to the export stage's fingerprint -- added there at
        # creation, so it can never rehash work that already exists.
        dense = manifest.stages[StageId.DENSE]
        return {
            "dense_fingerprint": dense.fingerprint,
            "num_points": dense.metrics.get("num_points"),
            "openmvs": _version(openmvs.version),
        }

    def preflight(self, manifest: RunManifest) -> list[str]:
        problems: list[str] = []
        dense = manifest.stages[StageId.DENSE]

        if not dense.artifacts.get("scene"):
            problems.append("no dense scene: run the dense stage first")
        if not dense.artifacts.get("cloud"):
            problems.append(
                "the dense stage recorded no point cloud. The surface is built from "
                "scene_dense.ply, not from the points inside the .mvs -- that file "
                "carries only the sparse cloud the alignment produced, so without it "
                "this would quietly reconstruct a fifteenth of the data."
            )
        if not dense.artifacts.get("undistorted"):
            problems.append(
                "the dense stage recorded no undistorted images. Texturing reads the "
                "photographs, and the paths inside the scene file are relative to that "
                "folder, so without it the mesh can be built but never painted."
            )
        if not dense.metrics.get("num_points"):
            problems.append("the dense cloud is empty; there is no surface to find in it")

        # Masks are not checked here. The dense stage already refuses to run when they
        # are present (HANDOVER 6.8) and it is a hard dependency of this one, so a
        # masked run cannot reach this point without 6.8 having been implemented first.
        # Repeating the block would add a message nothing can exercise.
        return problems

    def run(self, ctx: StageContext) -> StageResult:
        params: MeshParams = ctx.params  # type: ignore[assignment]
        run_dir = ctx.run_dir
        scratch = ctx.scratch / "mesh"
        scratch.mkdir(parents=True, exist_ok=True)

        dense = ctx.manifest.stages[StageId.DENSE]
        scene = run_dir / str(dense.artifacts["scene"])
        cloud = run_dir / str(dense.artifacts["cloud"])
        undistorted = run_dir / str(dense.artifacts["undistorted"])
        _check_inputs(scene, cloud, undistorted)

        # Spans weighted by what the tools actually cost. On the cup run reconstruction
        # took 21s and texturing 4m17s, so an even split would leave the bar parked at
        # half for the whole of the slow half.
        if params.refine:
            spans = {"reconstruct": (0.02, 0.10), "refine": (0.10, 0.62), "texture": (0.62, 0.95)}
        else:
            spans = {"reconstruct": (0.02, 0.14), "texture": (0.14, 0.95)}

        timings: dict[str, float] = {}
        counts: dict[str, tuple[int, int]] = {}

        peak = PeakMemory()
        peak.start()
        try:
            ctx.progress("linking the undistorted images", fraction=0.01)
            linked, link_mode = _link_images(undistorted, scratch)
            ctx.logger.info("staged %d undistorted files by %s", linked, link_mode)

            ctx.progress("reconstructing the surface", fraction=0.02)
            with _timed(timings, "reconstruct"):
                run_openmvs(
                    ctx,
                    openmvs.reconstruct_mesh(
                        input_file=scene,
                        output_file=scratch / "mesh.ply",
                        point_cloud_file=cloud,
                        min_point_distance=params.min_point_distance,
                        free_space_support=params.free_space_support,
                        decimate=params.decimate,
                        target_face_num=params.target_face_num,
                        remove_spurious=params.remove_spurious,
                        close_holes=params.close_holes,
                        smooth=params.smooth,
                        crop_to_roi=params.crop_to_roi,
                        export_type="ply",
                        working_folder=scratch,
                    ),
                    cwd=scratch,
                    tool="ReconstructMesh",
                    label="ReconstructMesh",
                    span=spans["reconstruct"],
                    progress_from=_phase_mapper(RECONSTRUCT_PHASES, "tetrahedralization"),
                    on_line=_count_recorder(counts, "reconstruct"),
                )
            raw = _find_mesh(scratch, "mesh")
            if raw is None:
                raise RuntimeError(
                    "ReconstructMesh produced no mesh. Its log is in the stage log "
                    "above -- it writes nothing to the console, so that is the only "
                    "place its complaint appears."
                )

            surface = raw
            if params.refine:
                ctx.progress("refining against the photographs", fraction=spans["refine"][0])
                with _timed(timings, "refine"):
                    run_openmvs(
                        ctx,
                        openmvs.refine_mesh(
                            input_file=scene,
                            mesh_file=raw,
                            output_file=scratch / "mesh_refined.ply",
                            resolution_level=params.refine_resolution_level,
                            scales=params.refine_scales,
                            export_type="ply",
                            cuda_device=-1 if params.refine_on_gpu else -2,
                            working_folder=scratch,
                        ),
                        cwd=scratch,
                        tool="RefineMesh",
                        label="RefineMesh",
                        span=spans["refine"],
                        progress_from=_phase_mapper(REFINE_PHASES, "subdividing"),
                        on_line=_count_recorder(counts, "refine"),
                    )
                refined = _find_mesh(scratch, "mesh_refined")
                if refined is None:
                    raise RuntimeError(
                        "RefineMesh produced no mesh. It is the step most likely to "
                        "exhaust memory: raise refine_resolution_level, or lower "
                        "refine_scales, and try again."
                    )
                surface = refined

            ctx.progress("texturing", fraction=spans["texture"][0])
            with _timed(timings, "texture"):
                run_openmvs(
                    ctx,
                    openmvs.texture_mesh(
                        input_file=scene,
                        mesh_file=surface,
                        output_file=scratch / f"mesh_textured.{params.export_type}",
                        resolution_level=params.texture_resolution_level,
                        max_texture_size=params.max_texture_size,
                        export_type=params.export_type,
                        working_folder=scratch,
                    ),
                    cwd=scratch,
                    tool="TextureMesh",
                    label="TextureMesh",
                    span=spans["texture"],
                    progress_from=_phase_mapper(TEXTURE_PHASES, "assigning views"),
                    on_line=_count_recorder(counts, "texture"),
                )
        finally:
            peak.stop()

        textured = _find_mesh(scratch, "mesh_textured")
        if textured is None:
            raise RuntimeError(
                "TextureMesh produced no mesh. Check the stage log for a missing image: "
                "the paths inside the scene file are relative to the working folder, so "
                "that is what a complaint about an image that plainly exists means."
            )

        ctx.progress("measuring", fraction=0.96)
        # Log the header before parsing. Scratch is discarded when a stage fails, so if
        # the read goes wrong this is the only surviving evidence of what was written.
        for path in (raw, textured):
            if path.suffix == ".ply":
                try:
                    ctx.logger.info(
                        "%s header: %s",
                        path.name,
                        ply.read_header_text(path).replace("\n", " | "),
                    )
                except ply.PlyError as exc:
                    ctx.logger.warning("could not read the header of %s: %s", path.name, exc)

        measured, count_warnings = _measure(raw, surface, textured, counts, params)
        textures = _find_textures(scratch, "mesh_textured", textured)
        # Sizes now, while these still exist: the commit renames the scratch out from
        # under every path held here.
        sizes = {
            "mesh_bytes": textured.stat().st_size,
            "texture_bytes": sum(t.stat().st_size for t in textures),
        }

        _unlink_images(scratch)
        _commit(run_dir / "mesh", scratch)

        artifacts = {
            "mesh": f"mesh/{textured.name}",
            "mesh_untextured": f"mesh/{raw.name}",
        }
        if surface is not raw:
            artifacts["mesh_refined"] = f"mesh/{surface.name}"
        if textures:
            artifacts["texture"] = f"mesh/{textures[0].name}"
        material = next((t for t in textures if t.suffix == ".mtl"), None)
        if material is not None:
            artifacts["material"] = f"mesh/{material.name}"

        metrics, warnings = _summarise(
            ctx.manifest, params, measured, sizes, len(textures), timings, peak.peak_gb
        )
        return StageResult(
            artifacts=artifacts,
            metrics=metrics,
            warnings=count_warnings + warnings,
            tool_versions={"openmvs": _version(openmvs.version)},
        )


# -- helpers -----------------------------------------------------------------


def _version(fn) -> str:
    try:
        return fn()
    except RuntimeError:
        return "missing"


class _timed:
    """Record how long a block took, so the report can name the slow tool."""

    def __init__(self, into: dict[str, float], key: str) -> None:
        self.into = into
        self.key = key

    def __enter__(self) -> _timed:
        self.started = time.monotonic()
        return self

    def __exit__(self, *exc: object) -> None:
        self.into[self.key] = round(time.monotonic() - self.started, 1)


def _check_inputs(scene: Path, cloud: Path, undistorted: Path) -> None:
    """Fail early and specifically. preflight cannot do this -- it has no run dir."""
    if not scene.is_file():
        raise RuntimeError(f"the dense scene is missing from {scene}")
    if not cloud.is_file():
        raise RuntimeError(
            f"the dense point cloud is missing from {cloud}. The .mvs beside it holds "
            "only the sparse cloud, so there is nothing to reconstruct without it."
        )
    if not (undistorted / "images").is_dir():
        raise RuntimeError(f"the undistorted images are missing from {undistorted}")


def _phase_mapper(
    phases: list[tuple[str, float, str]], first: str
) -> Callable[[str], tuple[float, str] | None]:
    """Map a log line onto a fraction of a tool's span, by the phase it just finished.

    Percentages are not used: these tools print statistics with percent signs in them
    and no marching progress, so a marker table is the only honest signal available.
    A line matching nothing returns None and the bar holds where it was.
    """
    started = {"done": False}

    def mapper(line: str) -> tuple[float, str] | None:
        if not started["done"]:
            started["done"] = True
            return 0.0, first
        for marker, fraction, next_phase in phases:
            if marker in line:
                return fraction, next_phase
        return None

    return mapper


def _count_recorder(into: dict[str, tuple[int, int]], key: str) -> Callable[[str], None]:
    """Capture vertex and face counts from a tool's log as it runs.

    Live rather than by re-reading the log afterwards, because a failed stage's scratch
    is wiped and the log goes with it. The last match wins: the scene load reports
    "50189 points, 0 vertices, 0 faces" before any mesh exists.
    """

    def record(line: str) -> None:
        found = openmvs.mesh_counts(line)
        if found is not None and found != (0, 0):
            into[key] = found

    return record


def _link_images(undistorted: Path, scratch: Path) -> tuple[int, str]:
    """Stage the undistorted workspace inside the scratch so it can be ``-w``.

    Hardlinks, because the images are already on disk and this is only about giving
    OpenMVS a working folder it can resolve relative paths against without writing into
    the committed dense directory. Falls back to copying if the runs root turns out to
    be on a different volume from the scratch, which is the one case ``os.link`` refuses.
    """
    target = scratch / "undistorted"
    count = 0
    mode = "hardlink"
    for source in undistorted.rglob("*"):
        destination = target / source.relative_to(undistorted)
        if source.is_dir():
            destination.mkdir(parents=True, exist_ok=True)
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(source, destination)
        except OSError:
            shutil.copy2(source, destination)
            mode = "copy"
        count += 1
    return count, mode


def _unlink_images(scratch: Path) -> None:
    """Drop the staged images before committing, so disk usage does not count them twice."""
    shutil.rmtree(scratch / "undistorted", ignore_errors=True)


def _find_mesh(scratch: Path, stem: str) -> Path | None:
    """The file the tool actually wrote, which is not always the one it was told to.

    OpenMVS builds the output name from the stem of ``-o`` plus the export type, so the
    extension is chosen by a different flag from the one that names the file. Matching
    on the exact stem also keeps the texture atlas out of it: that is written as
    ``<stem>_0.png``, which is a different stem rather than a different extension.
    """
    for suffix in MESH_SUFFIXES:
        candidate = scratch / f"{stem}{suffix}"
        if candidate.is_file():
            return candidate
    return None


def _find_textures(scratch: Path, stem: str, mesh: Path) -> list[Path]:
    """The files the mesh references but does not contain.

    A .glb is not self-contained: TextureMesh writes the atlas beside it as
    ``<stem>_0.png`` and points at it by relative URI. An .obj needs its .mtl and the
    .jpg that names. Both resolve because everything lands in the same directory.
    """
    found = [
        path
        for path in sorted(scratch.glob(f"{stem}*"))
        if path.is_file()
        and path != mesh
        and path.suffix.lower() in (".png", ".jpg", ".jpeg", ".mtl")
    ]
    return found


def _measure(
    raw: Path,
    surface: Path,
    textured: Path,
    counts: dict[str, tuple[int, int]],
    params: MeshParams,
) -> tuple[dict[str, Any], list[str]]:
    """Vertex and face counts for each stage of the chain.

    The intermediates are always PLY, so their counts are read exactly from the header.
    The textured output may be GLB or OBJ, which this cannot parse, so those counts come
    from the tool's own completion line -- and if that was not seen, from the surface it
    was given, with a warning saying so rather than a number presented as certain.
    """
    warnings: list[str] = []
    measured: dict[str, Any] = {}

    measured["raw_vertices"], measured["raw_faces"] = ply.count_elements(raw)
    if surface != raw:
        measured["refined_vertices"], measured["refined_faces"] = ply.count_elements(surface)

    if textured.suffix == ".ply":
        measured["vertices"], measured["faces"] = ply.count_elements(textured)
    elif "texture" in counts:
        measured["vertices"], measured["faces"] = counts["texture"]
    else:
        source = "refined" if surface != raw else "raw"
        measured["vertices"] = measured[f"{source}_vertices"]
        measured["faces"] = measured[f"{source}_faces"]
        warnings.append(
            f"the counts for {textured.name} could not be read from it or from the "
            "texturing log, so they are the surface texturing was given. Texturing does "
            "not change topology unless it decimates, so they are very probably right -- "
            "but they were not measured on the file that was written."
        )

    return measured, warnings


def _commit(target: Path, scratch: Path) -> None:
    if target.exists():
        shutil.rmtree(target, ignore_errors=True)
    scratch.replace(target)


def _summarise(
    manifest: RunManifest,
    params: MeshParams,
    measured: dict[str, Any],
    sizes: dict[str, int],
    textures: int,
    timings: dict[str, float],
    peak_gb: float,
) -> tuple[dict[str, Any], list[str]]:
    points = manifest.stages[StageId.DENSE].metrics.get("num_points") or 0
    faces = int(measured.get("faces") or 0)
    total_bytes = sizes["mesh_bytes"] + sizes["texture_bytes"]

    metrics: dict[str, Any] = dict(measured)
    metrics.update(
        {
            "refined": params.refine,
            "export_type": params.export_type,
            "faces_per_thousand_points": round(faces / (points / 1000), 1) if points else 0.0,
            "mesh_bytes": sizes["mesh_bytes"],
            "texture_bytes": sizes["texture_bytes"],
            "textures": textures,
            "peak_system_memory_gb": round(peak_gb, 1),
        }
    )
    if params.refine:
        metrics["refined_on_gpu"] = params.refine_on_gpu
    for key, seconds in timings.items():
        metrics[f"{key}_s"] = seconds

    warnings: list[str] = []

    if faces < EMPTY_FACES_WARN:
        warnings.append(
            f"the surface came out with only {faces} triangles from {points} dense "
            "points. That is not a coarse mesh, it is an empty one: the usual causes "
            "are a cloud that is mostly noise from a reflective or transparent surface, "
            "or crop_to_roi having trimmed the subject away along with the background."
        )
    elif faces > BUSY_FACES_WARN:
        warnings.append(
            f"the surface came out with {faces:,} triangles. Nothing downstream needs "
            "that many -- the viewer will crawl and the export stage will decimate it "
            "anyway. Set target_face_num to bound it."
        )

    if total_bytes > LARGE_MESH_WARN_BYTES:
        warnings.append(
            f"the mesh and its texture come to {total_bytes / 1e6:.0f} MB, so the "
            "viewer will ask before loading them rather than doing it on arrival. "
            "max_texture_size is most of that -- halving it quarters the bytes -- and "
            "target_face_num bounds the rest."
        )

    if not params.refine:
        warnings.append(
            "this surface is the graph cut's answer straight from the point cloud, with "
            "no pass against the photographs. Expect it to be faithful where the cloud "
            "was dense and lumpy where it was not; turn refinement on now that the "
            "stage has succeeded once."
        )
    elif not params.refine_on_gpu and timings.get("refine", 0) > SLOW_REFINE_WARN_S:
        warnings.append(
            f"refinement took {timings['refine'] / 60:.0f} minutes on the processor. "
            "OpenMVS defaults this step to the processor because the graphics card path "
            "is the less safe one, but refine_on_gpu is there if you want to try it."
        )

    total_ram = psutil.virtual_memory().total / 1e9
    if peak_gb > total_ram * 0.9:
        warnings.append(
            f"memory peaked at {peak_gb:.0f} GB of {total_ram:.0f} GB. Raise "
            "refine_resolution_level before attempting anything larger."
        )

    return metrics, warnings
