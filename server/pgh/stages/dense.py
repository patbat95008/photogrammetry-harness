"""Stage 5: turn known poses into a dense sampling of the surface.

Three tools in sequence, each of which has to be handled on its own terms:

1. ``colmap image_undistorter`` rectifies the selected images so the lens model
   becomes a plain pinhole. OpenMVS assumes that and will not undistort for itself.
2. ``InterfaceCOLMAP`` converts the undistorted workspace into a ``.mvs`` scene.
3. ``DensifyPointCloud`` estimates a depth map per view and fuses them.

**This is the stage that runs out of memory,** and it runs out of *system* RAM long
before it troubles the GPU -- depth maps are held per neighbourhood at full working
resolution. ``resolution_level`` is the knob: each level halves the working image
size, so level 2 costs about a quarter of level 1. It is the first parameter for that
reason.

**It is also the stage that fills the disk.** The per-view ``.dmap`` files are around
nine tenths of what a run consumes and are never read again after fusion, so they are
deleted by default -- but counted and measured first, because "freed 41 GB" is worth
knowing and a silent deletion is not.

**Masks go through a second undistortion pass** (HANDOVER 6.8). COLMAP does not
undistort masks, but OpenMVS consumes undistorted images, so the masks are staged
under the image names, rectified by a second ``image_undistorter`` call sharing one
frozen ``UndistortGeometry`` with the first, and thresholded at 127. The shared object
is the point: two call sites that merely happen to pass the same flags stay identical
only until someone edits one of them, and a mask rectified even slightly differently
is wrong precisely around ears and hair -- the places you would look last and trust
most. ``_check_mask_siblings`` then looks at the result, because a smeared mask writes
and runs without complaint.

The thresholded masks are written **beside** each undistorted image as
``<stem>.mask.png`` and read via ``--ignore-mask-label 0``. Not through
``-m/--mask-path``: that flattens every camera group into one directory keyed on the
bare filename, and the shared slot clock guarantees ``cam_high/000042`` and
``cam_eye/000042`` collide there -- which looks perfect on a single-camera run and
silently applies one camera's masks to the other on a rig.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import psutil
from pydantic import Field

from .. import ply
from ..manifest import RunManifest, StageId, StageState
from ..vendor import openmvs
from ..vendor import colmap
from .base import Stage, StageContext, StageParams, StageResult
from .shell import PeakMemory, run_openmvs, run_tool

#: Warn when fewer than this many dense points come out per registered view; it
#: usually means the depth maps mostly failed rather than that the object is small.
POINTS_PER_VIEW_WARN = 2000


class DenseParams(StageParams):
    resolution_level: int = Field(
        1,
        ge=0,
        le=4,
        description=(
            "How many times to halve the images before estimating depth. 0 is full "
            "resolution and the most detail; each step up cuts memory and time to "
            "roughly a quarter. Raise this first if the stage exhausts memory."
        ),
    )
    max_resolution: int = Field(
        2560,
        ge=640,
        le=8192,
        description="Images larger than this are scaled down before processing.",
    )
    min_resolution: int = Field(
        640,
        ge=320,
        le=4096,
        description="Images are never scaled below this, whatever the level.",
    )
    number_views: int = Field(
        8,
        ge=0,
        le=32,
        description=(
            "Neighbouring views used to estimate each depth map. More is steadier "
            "and slower; 0 means every available neighbour."
        ),
    )
    number_views_fuse: int = Field(
        2,
        ge=1,
        le=8,
        description=(
            "How many views must agree before a point survives fusion. Raise it to "
            "suppress noise and floating fragments, at the cost of thinning genuine "
            "but poorly-seen surfaces."
        ),
    )
    undistort_max_image_size: int = Field(
        -1,
        description=(
            "Cap on the undistorted images, or -1 for full size. This also sets the "
            "resolution the texture would later be sampled at."
        ),
    )
    delete_depth_maps: bool = Field(
        True,
        description=(
            "Delete the per-view depth maps after fusion. They are most of what a run "
            "consumes on disk and are not needed again."
        ),
    )
    preview_point_cap: int = Field(
        1_500_000,
        ge=50_000,
        le=10_000_000,
        description="Points kept in the browser preview. The full cloud is untouched.",
        json_schema_extra={"affects_fingerprint": False},
    )


class DenseStage(Stage):
    id = StageId.DENSE
    label = "Dense cloud"
    description = "Estimate depth per view and fuse it into a dense point cloud."
    params_model = DenseParams

    def external_inputs(self, manifest: RunManifest) -> dict[str, Any]:
        sparse = manifest.stages[StageId.SPARSE]
        return {
            "sparse_fingerprint": sparse.fingerprint,
            "registered": sparse.metrics.get("images_registered"),
            "colmap": _version(colmap.version),
            "openmvs": _version(openmvs.version),
        }

    def preflight(self, manifest: RunManifest) -> list[str]:
        problems: list[str] = []
        sparse = manifest.stages[StageId.SPARSE]
        if not sparse.artifacts.get("model"):
            problems.append("no sparse model: run the align stage first")
        if not sparse.metrics.get("images_registered"):
            problems.append("the sparse model registered no images")

        mask = manifest.stages[StageId.MASK]
        if mask.state is not StageState.SKIPPED and mask.artifacts.get("masks"):
            # Masks are used now (HANDOVER 6.8 is implemented), so what matters is
            # that the files are actually there. A recorded artifact whose directory
            # has gone is the case that would otherwise reconstruct the background
            # back in without saying anything.
            views = mask.artifacts.get("masks_openmvs")
            if not views:
                problems.append(
                    "the mask stage ran before this run understood how to feed masks "
                    "to OpenMVS, so it recorded no OpenMVS-named view. Re-run the "
                    "mask stage."
                )
        return problems

    def run(self, ctx: StageContext) -> StageResult:
        params: DenseParams = ctx.params  # type: ignore[assignment]
        run_dir = ctx.run_dir
        scratch = ctx.scratch / "dense"
        scratch.mkdir(parents=True, exist_ok=True)

        model_dir = _best_model(run_dir, ctx.manifest)
        images_dir = run_dir / "sparse" / "images"
        undistorted = scratch / "undistorted"

        peak = PeakMemory()
        peak.start()
        try:
            # One geometry object, used by both passes. HANDOVER 6.8: if the mask
            # pass rectifies even slightly differently, the mask lands beside the
            # image it belongs to rather than on it.
            geometry = colmap.UndistortGeometry(
                max_image_size=params.undistort_max_image_size
            )

            ctx.progress("undistorting images", fraction=0.02)
            run_tool(
                ctx,
                colmap.image_undistorter(
                    image_path=images_dir,
                    input_path=model_dir,
                    output_path=undistorted,
                    geometry=geometry,
                ),
                cwd=scratch,
                label="colmap image_undistorter",
            )
            _check_undistorted(undistorted)

            mask_source = _mask_source(run_dir, ctx.manifest)
            masked = mask_source is not None
            if masked:
                ctx.progress("undistorting masks", fraction=0.10)
                farm = scratch / "mask_images"
                staged = _build_mask_farm(mask_source, images_dir, farm)
                ctx.logger.info("staged %d masks for undistortion", staged)
                run_tool(
                    ctx,
                    colmap.image_undistorter(
                        image_path=farm,
                        input_path=model_dir,
                        output_path=scratch / "undistorted_masks",
                        geometry=geometry,
                        # Not geometry, so it may differ: a crisp mask edge costs
                        # nothing and the threshold below absorbs the rest.
                        jpeg_quality=100,
                    ),
                    cwd=scratch,
                    label="colmap image_undistorter (masks)",
                )
                written = _threshold_masks(
                    scratch / "undistorted_masks" / "images",
                    undistorted / "images",
                )
                ctx.logger.info("wrote %d sibling .mask.png files", written)
                _check_mask_siblings(undistorted / "images")
                shutil.rmtree(farm, ignore_errors=True)
                shutil.rmtree(scratch / "undistorted_masks", ignore_errors=True)

            scene = scratch / "scene.mvs"
            ctx.progress("converting the scene", fraction=0.15)
            run_openmvs(
                ctx,
                openmvs.interface_colmap(
                    input_dir=undistorted, output_file=scene, image_folder="images"
                ),
                cwd=scratch,
                tool="InterfaceCOLMAP",
                label="InterfaceCOLMAP",
                span=(0.15, 0.2),
            )
            if not scene.exists():
                raise RuntimeError(
                    "InterfaceCOLMAP produced no scene file. Its log is in the stage "
                    "log above -- it writes nothing to the console, so that is the "
                    "only place its complaint appears."
                )

            dense_scene = scratch / "scene_dense.mvs"
            ctx.progress("estimating depth maps", fraction=0.2)
            run_openmvs(
                ctx,
                openmvs.densify_point_cloud(
                    input_file=scene,
                    output_file=dense_scene,
                    resolution_level=params.resolution_level,
                    max_resolution=params.max_resolution,
                    min_resolution=params.min_resolution,
                    number_views=params.number_views,
                    number_views_fuse=params.number_views_fuse,
                    # >= 0 or no mask is read at all: -1 means "estimate a lens
                    # distortion mask", not "use the files next to the images".
                    ignore_mask_label=0 if masked else None,
                ),
                cwd=scratch,
                tool="DensifyPointCloud",
                label="DensifyPointCloud",
                span=(0.2, 0.9),
            )
        finally:
            peak.stop()

        cloud = _find_cloud(scratch)
        if cloud is None:
            raise RuntimeError(
                "DensifyPointCloud produced no point cloud. The usual cause is "
                "running out of memory during depth estimation -- raise "
                "resolution_level and try again."
            )

        ctx.progress("building the preview", fraction=0.92)
        # Log the header before parsing. Scratch is discarded when a stage fails, so
        # if the read goes wrong this is the only surviving evidence of what the
        # engine actually wrote.
        try:
            ctx.logger.info(
                "%s header: %s",
                cloud.name,
                ply.read_header_text(cloud).replace("\n", " | "),
            )
        except ply.PlyError as exc:
            ctx.logger.warning("could not read the header of %s: %s", cloud.name, exc)

        total_points = ply.count_vertices(cloud)
        preview_points = ply.write_preview(
            cloud, scratch / "preview.ply", max_points=params.preview_point_cap
        )

        depth_maps, freed = _sweep_depth_maps(ctx, scratch, params.delete_depth_maps)
        _commit(run_dir / "dense", scratch)

        metrics, warnings = _summarise(
            ctx.manifest, params, total_points, preview_points, depth_maps, freed,
            peak.peak_gb, masked=masked,
        )
        return StageResult(
            artifacts={
                "cloud": f"dense/{cloud.name}",
                "preview": "dense/preview.ply",
                "scene": "dense/scene_dense.mvs",
                "undistorted": "dense/undistorted",
            },
            metrics=metrics,
            warnings=warnings,
            tool_versions={
                "colmap": _version(colmap.version),
                "openmvs": _version(openmvs.version),
            },
        )


# -- helpers -----------------------------------------------------------------


def _version(fn) -> str:
    try:
        return fn()
    except RuntimeError:
        return "missing"


def _best_model(run_dir: Path, manifest: RunManifest) -> Path:
    """The submodel the align stage settled on, or the largest one on disk."""
    recorded = manifest.stages[StageId.SPARSE].artifacts.get("model_best")
    if recorded:
        candidate = run_dir / recorded
        if candidate.is_dir():
            return candidate

    model_root = run_dir / "sparse" / "model"
    submodels = [p for p in model_root.iterdir() if p.is_dir()] if model_root.is_dir() else []
    if not submodels:
        raise RuntimeError(f"no sparse submodel found under {model_root}")
    return max(submodels, key=lambda p: sum(f.stat().st_size for f in p.iterdir() if f.is_file()))


def _mask_area_fraction(manifest: RunManifest) -> float | None:
    """How much of each frame the masks keep, as the mask stage measured it."""
    value = manifest.stages[StageId.MASK].metrics.get("mean_area_fraction")
    try:
        area = float(value)
    except (TypeError, ValueError):
        return None
    return area if 0.0 < area < 1.0 else None


def _mask_source(run_dir: Path, manifest: RunManifest) -> Path | None:
    """The canonical mask directory for this run, or None if it has no masks."""
    record = manifest.stages[StageId.MASK]
    if record.state is StageState.SKIPPED:
        return None
    relative = record.artifacts.get("masks")
    if not relative:
        return None
    directory = run_dir / relative
    return directory if directory.is_dir() else None


def _build_mask_farm(masks: Path, images_dir: Path, target: Path) -> int:
    """Stage the masks under the image names, so the undistorter will map them.

    ``image_undistorter`` looks up each name from the sparse model's images.txt, so
    every file here has to be called exactly what its photograph is called --
    ``cam_high/000042.jpg`` -- whatever is actually inside it. PNG bytes are written
    under that name because COLMAP sniffs the magic rather than trusting the
    extension, and a lossless round trip keeps the edge exact; the threshold
    afterwards is what makes it not matter if that ever stops being true.
    """
    shutil.rmtree(target, ignore_errors=True)
    target.mkdir(parents=True, exist_ok=True)
    staged = 0
    for image in sorted(images_dir.rglob("*")):
        if not image.is_file():
            continue
        relative = image.relative_to(images_dir)
        mask = masks / relative.parent / f"{relative.stem}.png"
        if not mask.is_file():
            continue
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(mask, destination)
        except OSError:
            shutil.copy2(mask, destination)
        staged += 1
    return staged


def _threshold_masks(undistorted_masks: Path, undistorted_images: Path) -> int:
    """Threshold at 127 and write each mask beside the image it belongs to.

    Beside, rather than into one folder passed with ``-m``: that flag flattens every
    camera group into a single directory keyed on the bare filename, and the shared
    slot clock guarantees two cameras collide there. ``ignore_mask_label`` reads these
    siblings per-image instead, which the v2.4.0 help describes as masks "next to each
    image with '.mask.png'".

    The threshold is HANDOVER 6.8's, and it is what makes the round trip through
    whatever the undistorter chose to write harmless.
    """
    written = 0
    for mask in sorted(undistorted_masks.rglob("*")):
        if not mask.is_file():
            continue
        image = cv2.imread(str(mask), cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
        binary = np.where(image >= 127, 255, 0).astype(np.uint8)
        target = undistorted_images / mask.relative_to(undistorted_masks)
        target = target.with_name(f"{target.stem}.mask.png")
        target.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(target), binary)
        written += 1
    return written


def _check_mask_siblings(undistorted_images: Path) -> None:
    """Fail loudly if the second pass produced something other than a clean mask.

    A mask that came back smeared -- interpolated, half-transparent, or resampled
    against a different geometry -- still writes without complaint and still lets the
    densify run. It goes wrong only at the boundary, which is exactly where the ears
    and the hair are, and nothing downstream would ever mention it. So look at the
    pixel distribution: a real mask is almost entirely black and white.
    """
    masks = sorted(undistorted_images.rglob("*.mask.png"))
    if not masks:
        raise RuntimeError(
            "the mask undistortion pass wrote no masks beside the images, so the "
            "densify would silently reconstruct the background back in"
        )
    sample = masks[: min(12, len(masks))]
    for mask in sample:
        image = cv2.imread(str(mask), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise RuntimeError(f"could not read back the undistorted mask {mask}")
        middling = float(((image > 32) & (image < 223)).mean())
        if middling > 0.02:
            raise RuntimeError(
                f"{mask.name} is {middling:.1%} mid-grey after thresholding, which "
                f"means the mask was smeared rather than rectified. The two "
                f"image_undistorter passes have to share one UndistortGeometry "
                f"(HANDOVER 6.8); if they have drifted apart, the mask is wrong "
                f"precisely at the boundary and nothing downstream will say so."
            )


def _check_undistorted(undistorted: Path) -> None:
    """Fail early and specifically if the undistorted workspace is not what we expect."""
    images = undistorted / "images"
    sparse = undistorted / "sparse"
    if not images.is_dir() or not any(images.rglob("*")):
        raise RuntimeError(f"image_undistorter wrote no images to {images}")
    if not sparse.is_dir():
        raise RuntimeError(f"image_undistorter wrote no model to {sparse}")


def _find_cloud(scratch: Path) -> Path | None:
    candidates = sorted(scratch.glob("*_dense.ply")) or sorted(scratch.glob("*.ply"))
    candidates = [p for p in candidates if p.name != "preview.ply"]
    return candidates[0] if candidates else None


def _sweep_depth_maps(
    ctx: StageContext, scratch: Path, delete: bool
) -> tuple[int, int]:
    """Count the depth maps, and delete them if asked. Returns (count, bytes freed)."""
    dmaps = list(scratch.rglob("*.dmap"))
    total = sum(p.stat().st_size for p in dmaps)
    if not delete:
        return len(dmaps), 0
    for path in dmaps:
        try:
            path.unlink()
        except OSError:
            pass
    if dmaps:
        ctx.logger.info(
            "deleted %d depth maps, freeing %.1f GB", len(dmaps), total / 1e9
        )
    return len(dmaps), total


def _commit(target: Path, scratch: Path) -> None:
    if target.exists():
        shutil.rmtree(target, ignore_errors=True)
    scratch.replace(target)


def _summarise(
    manifest: RunManifest,
    params: DenseParams,
    total_points: int,
    preview_points: int,
    depth_maps: int,
    freed: int,
    peak_gb: float,
    *,
    masked: bool = False,
) -> tuple[dict[str, Any], list[str]]:
    registered = manifest.stages[StageId.SPARSE].metrics.get("images_registered") or 0
    per_view = total_points / registered if registered else 0.0

    metrics: dict[str, Any] = {
        "num_points": total_points,
        "preview_points": preview_points,
        # Whether the background was excluded, and the label that made OpenMVS read
        # the masks at all -- -1, its default, means "estimate one" and ignores them.
        "masked": masked,
        "ignore_mask_label": 0 if masked else None,
        "points_per_view": round(per_view),
        "resolution_level": params.resolution_level,
        "depth_maps": depth_maps,
        "depth_maps_deleted": params.delete_depth_maps,
        "bytes_freed": freed,
        "peak_system_memory_gb": round(peak_gb, 1),
    }

    warnings: list[str] = []
    # On a masked run the subject really IS a small part of each frame, so the bar has
    # to come down with it. cup-1 masked to 6.5% of frame produced 1,409 points a view
    # and was a perfectly good cloud; judged against the unmasked figure it read as a
    # failure, which is a warning that trains you to ignore warnings.
    area = _mask_area_fraction(manifest) if masked else None
    floor = POINTS_PER_VIEW_WARN * area if area else POINTS_PER_VIEW_WARN
    if registered and per_view < floor:
        detail = (
            f" The masks keep about {area:.0%} of each frame, so the bar for this run "
            f"is {round(floor)} rather than {POINTS_PER_VIEW_WARN}."
            if area
            else ""
        )
        warnings.append(
            f"only about {round(per_view)} points per registered view. Depth "
            "estimation mostly failed rather than the object being small: the usual "
            "causes are too few overlapping neighbours, a textureless or shiny "
            "surface, or a resolution level so high there is nothing left to match."
            + detail
        )

    total_ram = psutil.virtual_memory().total / 1e9
    if peak_gb > total_ram * 0.9:
        warnings.append(
            f"memory peaked at {peak_gb:.0f} GB of {total_ram:.0f} GB. This stage "
            "exhausts system RAM before it troubles the GPU; raise resolution_level "
            "before running anything larger."
        )

    if params.delete_depth_maps and freed:
        metrics["freed_gb"] = round(freed / 1e9, 1)

    return metrics, warnings
