"""Stage 7: clean up, set real-world scale, and write files you can use.

The reconstruction is finished by the time this runs; what is left is everything that
makes the result *usable*, and all three parts of it fail quietly rather than loudly.

**Orientation.** COLMAP's world frame is arbitrary, so the model comes out on its side.
The camera centres lie in a plane whose normal is the up axis, and ``orient.py`` recovers
it -- but the *sign* of a plane normal is not recoverable, so ``capture.flip_x`` supplies
it. That is the toggle in the cloud viewer: somebody looked at the model and said which
way up it goes. This is the first stage to read it, and therefore the first to put it in
a fingerprint -- added here at creation, exactly as HANDOVER 8 asks, so it can never
rehash work that already exists.

**Scale.** Photogrammetry recovers shape but never size. A fixed two-camera rig has one
measurable length in it -- the tape-measured baseline between the lenses -- and paired
slots are the same instant, so the distance between those two camera centres is that
baseline. A single orbiting camera has no such pair, and there is nothing to recover
scale from at all: the honest options are a ruler in frame or a measured dimension typed
in. So ``scale_mode`` defaults to ``auto``, which uses the baseline when there is one and
**exports in model units when there is not, saying so in a warning**. An STL carries no
units, so a confidently mis-scaled export is indistinguishable from a correct one until
something is printed at the wrong size.

**Cleanup** happens inside Blender, and the trap there is documented at HANDOVER 6.23:
a textured GLB splits its vertices at every atlas seam, so "keep the largest connected
component" run naively keeps half a percent of the model. The script merges by distance
first, which heals the seams exactly and leaves UVs intact.

The geometry lives in ``orient.py`` as pure functions over numpy, tested against
synthetic rings. The Blender script is a dumb executor handed a matrix and a scale.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from .. import orient
from ..manifest import RunManifest, StageId
from ..vendor import blender
from .base import Stage, StageContext, StageParams, StageResult
from .shell import run_tool

#: Losing more than this fraction of the mesh to component filtering is worth naming.
#: Some floating debris is normal; a third of the model is not.
DROPPED_FRACTION_WARN = 0.15

#: Below this the decimation has gone past where detail survives.
THIN_FACES_WARN = 5000


class ExportParams(StageParams):
    scale_mode: Literal["auto", "manual", "none"] = Field(
        "auto",
        description=(
            "How to give the model a real-world size. 'auto' measures it from the "
            "camera baseline when this capture had a fixed rig, and leaves the model in "
            "arbitrary units when it did not. 'manual' scales from a dimension you have "
            "measured. 'none' exports in model units. An unscaled export is the quiet "
            "failure here: an STL carries no units, so it looks correct until something "
            "is printed at the wrong size."
        ),
    )
    known_dimension_mm: float | None = Field(
        None,
        gt=0,
        description=(
            "With 'manual': how big the model really is, in millimetres, along the axis "
            "below. For a handheld orbit with no ruler in frame this is the only route "
            "to real size, so measure the subject before it moves."
        ),
    )
    known_dimension_axis: Literal["height", "longest", "width", "depth"] = Field(
        "height",
        description=(
            "Which dimension the measurement above refers to. Height is along the "
            "recovered up axis; longest is whichever of the three is biggest."
        ),
    )
    largest_component_only: bool = Field(
        True,
        description=(
            "Keep the biggest connected shell and drop the rest. Reflective and "
            "transparent surfaces leave a haze of small floating fragments around a "
            "reconstruction, and this is what removes them. Turn it off if the subject "
            "is genuinely in several pieces."
        ),
    )
    target_faces: int = Field(
        0,
        ge=0,
        le=20_000_000,
        description=(
            "Decimate to this many triangles, or 0 to leave the mesh alone. Slicers and "
            "game engines rarely want more than a few hundred thousand."
        ),
    )
    orient_model: bool = Field(
        True,
        description=(
            "Stand the model up, using the plane the cameras lie in as the ground and "
            "the Flip toggle from the cloud viewer for which way up. Off exports "
            "COLMAP's raw frame, which is arbitrary and usually on its side."
        ),
    )
    centre: bool = Field(
        True,
        description=(
            "Move the model to the origin and sit it on the ground plane, which is "
            "where a slicer and a viewer both expect to find it."
        ),
    )
    export_glb: bool = Field(
        True, description="Write a GLB: one file, keeps the texture, good for sharing and viewing."
    )
    export_obj: bool = Field(
        False, description="Write an OBJ with its material and texture, for older tools."
    )
    export_stl: bool = Field(
        True,
        description=(
            "Write an STL, which is what a slicer wants. Geometry only -- no colour, "
            "and no units, so it is only as right as the scale above."
        ),
    )
    turntable_frames: int = Field(
        36,
        ge=0,
        le=180,
        description=(
            "Render this many frames orbiting the finished model, as a record of what "
            "was exported. 0 skips it."
        ),
    )
    turntable_resolution: int = Field(
        720,
        ge=256,
        le=2048,
        description="Size of each turntable frame.",
        json_schema_extra={"affects_fingerprint": False},
    )


class ExportStage(Stage):
    id = StageId.EXPORT
    label = "Export"
    description = "Clean up, set real-world scale, and write files you can use."
    params_model = ExportParams

    def external_inputs(self, manifest: RunManifest) -> dict[str, Any]:
        mesh = manifest.stages[StageId.MESH]
        sparse = manifest.stages[StageId.SPARSE]
        return {
            "mesh_fingerprint": mesh.fingerprint,
            # poses.json is a sparse artifact, and the up axis is computed from it.
            "sparse_fingerprint": sparse.fingerprint,
            # The first and only fingerprint flip_x appears in. Added at this stage's
            # creation so it invalidates nothing that already exists (HANDOVER 8).
            "flip_x": manifest.capture.flip_x,
            "baseline_mm": manifest.capture.baseline_mm,
            "blender": _version(),
        }

    def preflight(self, manifest: RunManifest) -> list[str]:
        problems: list[str] = []
        # Saved params can predate a field being added, so fall back to defaults rather
        # than letting preflight raise where the UI expects a list of reasons.
        try:
            params = self.params_model(**manifest.stages[StageId.EXPORT].params)
        except Exception:
            params = self.params_model()

        if not manifest.stages[StageId.MESH].artifacts.get("mesh"):
            problems.append("no mesh: run the mesh stage first")
        if not manifest.stages[StageId.SPARSE].artifacts.get("poses"):
            problems.append(
                "no camera poses: the up axis is recovered from the plane the cameras "
                "lie in, and sparse/poses.json is where they are recorded"
            )
        if not blender.available():
            problems.append(
                "Blender was not found. It is the only optional engine in the doctor "
                "because nothing before this stage needs it; this stage does. Install "
                "it, or point at it from data/tools.local.toml."
            )
        if params.scale_mode == "manual" and not params.known_dimension_mm:
            problems.append(
                "scale_mode is 'manual' but no measurement was given. Set "
                "known_dimension_mm to the real size of the subject along the axis you "
                "chose, or switch to 'none' to export in model units deliberately."
            )
        return problems

    def run(self, ctx: StageContext) -> StageResult:
        params: ExportParams = ctx.params  # type: ignore[assignment]
        run_dir = ctx.run_dir
        scratch = ctx.scratch / "export"
        scratch.mkdir(parents=True, exist_ok=True)

        mesh_rel = str(ctx.manifest.stages[StageId.MESH].artifacts["mesh"])
        mesh = run_dir / mesh_rel
        poses_path = run_dir / str(ctx.manifest.stages[StageId.SPARSE].artifacts["poses"])
        if not mesh.is_file():
            raise RuntimeError(f"the mesh is missing from {mesh}")
        if not poses_path.is_file():
            raise RuntimeError(f"the camera poses are missing from {poses_path}")

        poses = json.loads(poses_path.read_text(encoding="utf-8")).get("images", [])

        ctx.progress("working out which way is up", fraction=0.03)
        axis, planarity = _orientation(ctx, params, poses)
        scale, scale_source = _scale(ctx, params, poses)

        job = scratch / "job.json"
        job.write_text(
            json.dumps(
                {
                    "mesh": str(mesh),
                    "out_dir": str(scratch),
                    "stem": "model",
                    # The axis, not a rotation: the rotation onto +Z is built inside
                    # Blender, where the importer's own axis convention is known.
                    "up_axis": axis,
                    "scale": scale,
                    # A measured dimension only becomes a factor once the model is
                    # standing up, so Blender resolves it and reports what it used.
                    "fit": (
                        {
                            "mm": params.known_dimension_mm,
                            "axis": params.known_dimension_axis,
                        }
                        if params.scale_mode == "manual"
                        else None
                    ),
                    "centre": params.centre,
                    "largest_component_only": params.largest_component_only,
                    "target_faces": params.target_faces,
                    "formats": {
                        "glb": params.export_glb,
                        "obj": params.export_obj,
                        "stl": params.export_stl,
                    },
                    "turntable_frames": params.turntable_frames,
                    "turntable_resolution": params.turntable_resolution,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        ctx.progress("cleaning and exporting in Blender", fraction=0.1)
        lines: list[str] = []
        run_tool(
            ctx,
            blender.background_script("export_mesh.py", job),
            cwd=scratch,
            label="blender",
            on_line=lines.append,
        )

        result = blender.parse_result(lines)
        if result is None:
            raise RuntimeError(
                "the Blender script did not report a result. It prints one line "
                "beginning 'PGH_RESULT'; its absence means the script raised before "
                "finishing, and the traceback is in the stage log above."
            )

        ctx.progress("committing", fraction=0.95)
        _commit(run_dir / "export", scratch)

        artifacts: dict[str, str] = {"report": "export/report.json"}
        for kind, name in (result.get("files") or {}).items():
            artifacts[f"model_{kind}"] = f"export/{name}"
        if result.get("turntable_frames"):
            artifacts["turntable"] = "export/turntable"

        # Blender reports the factor it actually applied, which for a measured dimension
        # is the only place it could have been worked out.
        applied = result.get("scale") or scale
        metrics, warnings = _summarise(params, result, applied, scale_source, planarity, axis)
        (run_dir / "export" / "report.json").write_text(
            json.dumps({"metrics": metrics, "warnings": warnings}, indent=2), encoding="utf-8"
        )

        return StageResult(
            artifacts=artifacts,
            metrics=metrics,
            warnings=warnings,
            tool_versions={"blender": _version()},
        )


# -- helpers -----------------------------------------------------------------


def _version() -> str:
    try:
        return blender.version()
    except RuntimeError:
        return "missing"


def _orientation(
    ctx: StageContext, params: ExportParams, poses: list[dict]
) -> tuple[list[float] | None, float | None]:
    """The up axis in reconstruction coordinates, and how planar the camera fit was.

    None means "leave the frame alone", which is what ``orient_model`` off asks for.
    """
    if not params.orient_model:
        return None, None

    try:
        axis, planarity = orient.up_axis(poses, flip=ctx.manifest.capture.flip_x)
    except ValueError as exc:
        raise RuntimeError(
            f"could not recover an up axis: {exc} Turn orient_model off to export "
            "COLMAP's raw frame instead."
        ) from exc

    ctx.logger.info(
        "up axis %s from %d cameras, planarity %.4f (flip_x=%s)",
        [round(float(v), 4) for v in axis],
        len(poses),
        planarity,
        ctx.manifest.capture.flip_x,
    )
    return [float(v) for v in axis], planarity


def _scale(
    ctx: StageContext, params: ExportParams, poses: list[dict]
) -> tuple[float | None, str]:
    """Millimetres per model unit, and where that number came from.

    ``manual`` cannot be resolved here: it needs the model's dimensions, which do not
    exist until Blender has oriented and cleaned it. The measurement is passed through
    instead, and Blender turns it into a factor and reports what it used.
    """
    if params.scale_mode == "none":
        return None, "none"

    if params.scale_mode == "auto":
        factor = orient.baseline_scale(poses, ctx.manifest.capture.baseline_mm)
        if factor is None:
            return None, "none"
        ctx.logger.info("scale %.4f mm per model unit, from the measured baseline", factor)
        return factor, "baseline"

    return None, "manual"


def _commit(target: Path, scratch: Path) -> None:
    if target.exists():
        shutil.rmtree(target, ignore_errors=True)
    scratch.replace(target)


def _summarise(
    params: ExportParams,
    result: dict,
    scale: float | None,
    scale_source: str,
    planarity: float | None,
    axis: list[float] | None,
) -> tuple[dict[str, Any], list[str]]:
    faces_in = int(result.get("faces_in") or 0)
    faces_out = int(result.get("faces_out") or 0)
    dropped = int(result.get("dropped_faces") or 0)
    dimensions = [float(v) for v in (result.get("dimensions") or [0.0, 0.0, 0.0])]

    metrics: dict[str, Any] = {
        "faces_in": faces_in,
        "faces_out": faces_out,
        "components": int(result.get("components") or 1),
        "dropped_faces": dropped,
        "vertices_healed": int(result.get("vertices_healed") or 0),
        "scale_source": scale_source,
        "turntable_frames": int(result.get("turntable_frames") or 0),
        "formats": sorted((result.get("files") or {}).keys()),
    }
    # Only when something actually measured it. Reporting "1.0 mm per unit" for an
    # unscaled export states a scale that was never established.
    if scale and scale_source in ("baseline", "manual"):
        metrics["scale_mm_per_unit"] = round(scale, 6)
    if planarity is not None:
        metrics["camera_planarity"] = round(planarity, 4)
    if axis is not None:
        metrics["up_axis"] = [round(v, 4) for v in axis]

    units = "mm" if scale_source in ("baseline", "manual") else "model units"
    metrics["dimensions"] = [round(v, 3) for v in dimensions]
    metrics["dimension_units"] = units

    warnings: list[str] = []

    if scale_source == "none":
        warnings.append(
            "this model is in arbitrary units, not millimetres. Nothing in this capture "
            "could give it a real size -- a fixed rig with a measured baseline can, and "
            "so can typing in one measured dimension -- so anything printed from the STL "
            "will come out at whatever size the slicer guesses. That is the one failure "
            "here that looks exactly like success."
        )

    if planarity is not None and planarity > orient.PLANAR_RATIO_WARN:
        warnings.append(
            f"the camera centres are only loosely planar ({planarity:.3f}), which "
            "happens when a capture covers more than one height. The up axis is still "
            "the right one, but it is fitted rather than measured, so check the model "
            "is standing up before printing it."
        )

    if faces_in and dropped / faces_in > DROPPED_FRACTION_WARN:
        warnings.append(
            f"keeping the largest connected shell dropped {dropped:,} of {faces_in:,} "
            "triangles. Some floating debris is normal around a reflective surface, but "
            "that much suggests the subject itself came out in pieces -- look at the "
            "mesh matte-shaded before trusting this export."
        )

    if faces_out and faces_out < THIN_FACES_WARN:
        warnings.append(
            f"the exported mesh has only {faces_out:,} triangles, which is past the "
            "point where surface detail survives. Raise target_faces."
        )

    return metrics, warnings
