"""Stage 4: recover where every camera was, and a sparse point cloud with them.

This is the stage that decides whether the capture worked. Everything after it is
refinement; nothing after it can rescue a bad solve.

The layout it builds matters as much as the flags it passes:

    sparse/images/<camera_group>/<slot:06d>.jpg

Hardlinks, not copies -- the selected frames already exist under ``frames/`` and a
few hundred full-resolution stills is real disk. Keeping one folder per camera lets
``--ImageReader.single_camera_per_folder 1`` fit one intrinsic per physical camera
instead of averaging two lenses into one wrong model.

**Reading the result.** Registration rate is the headline: COLMAP silently drops
images it cannot place, so a model that looks fine can be built from half the frames.
Multiple submodels means the orbit broke into pieces that never recognised each other
-- almost always a loop that did not close. And for a chair spin with masking skipped,
a *confident* model is the failure: the solver locks onto the stationary room and
leaves the head as noise, which is exactly what the mask stage exists to prevent.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from .. import ply
from ..config import DATA_DIR
from ..manifest import CaptureMode, RunManifest, StageId, StageState
from ..proc import capture
from ..registry import colmap_dialect
from ..vendor import colmap
from .base import Stage, StageContext, StageParams, StageResult
from .shell import counter_progress, run_tool

#: The vocabulary tree loop detection needs. Fetch with scripts/fetch-vocab-tree.ps1.
VOCAB_TREE = DATA_DIR / "vocab_tree_flickr100K_words32K.bin"


def vocab_tree_status(path: Path = VOCAB_TREE) -> str:
    """Whether a usable vocabulary tree is present: 'ok', 'legacy' or 'missing'.

    COLMAP replaced FLANN with faiss for its visual index in May 2025, and the
    prebuilt trees still published at demuc.de are the old FLANN format. Feeding one
    to a current build does not fail cleanly -- it aborts the process with
    STATUS_STACK_BUFFER_OVERRUN (0xC0000409) partway through matching, after feature
    extraction has already been paid for.

    The new format opens with a version field of 1 or 2, which is exactly what
    COLMAP checks. A FLANN tree instead opens with its word count (32762 for the
    32K tree), so four bytes are enough to tell them apart before spending anything.
    """
    if not path.exists():
        return "missing"
    try:
        with path.open("rb") as fh:
            version = int.from_bytes(fh.read(4), "little")
    except OSError:
        return "missing"
    return "ok" if version in (1, 2) else "legacy"


#: Below this fraction of images registered, the model is not describing the capture.
REGISTRATION_WARN = 0.9

#: Mean reprojection error above this suggests a bad intrinsic or a non-rigid scene.
REPROJECTION_WARN_PX = 1.5


class SparseParams(StageParams):
    matcher: Literal["auto", "sequential", "exhaustive"] = Field(
        "auto",
        description=(
            "How image pairs are chosen. 'sequential' matches neighbours in time plus "
            "loop-closure candidates and is the right choice for a continuous orbit. "
            "'exhaustive' compares every pair -- slower, but it needs no vocabulary "
            "tree and cannot miss a loop. 'auto' uses sequential when a usable tree "
            "is present and exhaustive otherwise, which today means exhaustive: the "
            "published trees are the old FLANN format and this COLMAP needs faiss."
        ),
    )
    sequential_overlap: int = Field(
        10,
        ge=1,
        le=100,
        description="How many following frames each frame is matched against.",
    )
    loop_detection: bool = Field(
        True,
        description=(
            "Let frames far apart in time match if they look at the same thing. This "
            "is what closes an orbit: the far side of the circuit sees the near side, "
            "but hundreds of frames later."
        ),
    )
    camera_model: Literal["OPENCV", "SIMPLE_RADIAL", "RADIAL", "PINHOLE"] = Field(
        "OPENCV",
        description=(
            "Lens model fitted per camera. OPENCV carries two radial and two "
            "tangential terms and suits phone lenses. SIMPLE_RADIAL has fewer "
            "parameters and is steadier when there are few images to constrain them."
        ),
    )
    single_camera_per_folder: bool = Field(
        True,
        description=(
            "Fit one set of intrinsics per camera folder. Turn this off only if the "
            "lens or zoom changed during the take."
        ),
    )
    max_image_size: int = Field(
        3200,
        ge=512,
        le=12000,
        description="Images are scaled to this before features are detected.",
    )
    max_num_features: int = Field(
        8192, ge=1024, le=65536, description="Cap on SIFT features per image."
    )
    use_gpu: bool = Field(True, description="Detect and match features on the GPU.")
    filter_stationary_matches: bool = Field(
        True,
        description=(
            "Discard matches between frames where nothing moved. They have no "
            "baseline, so they pull the solution toward a degenerate one."
        ),
    )
    min_model_size: int = Field(
        10,
        ge=3,
        description="Smallest number of images COLMAP will call a reconstruction.",
    )


class SparseStage(Stage):
    id = StageId.SPARSE
    label = "Align"
    description = "Solve camera poses and a sparse point cloud with COLMAP."
    params_model = SparseParams

    def external_inputs(self, manifest: RunManifest) -> dict[str, Any]:
        select = manifest.stages[StageId.SELECT]
        mask = manifest.stages[StageId.MASK]
        return {
            "selected": select.metrics.get("selected"),
            "selection_fingerprint": select.fingerprint,
            # Whether masks exist is a real input to this stage, and it does NOT
            # follow from the mask stage's fingerprint: that hashes its params and
            # inputs, which are identical whether it ran or was skipped. Without this,
            # unskipping masking would leave the sparse model looking fresh.
            "masks": (
                "none"
                if mask.state is StageState.SKIPPED
                else mask.artifacts.get("masks", "none")
            ),
            "colmap": _colmap_version(),
            "dialect": colmap_dialect(),
            "vocab_tree": vocab_tree_status(),
        }

    def preflight(self, manifest: RunManifest) -> list[str]:
        problems: list[str] = []
        select = manifest.stages[StageId.SELECT]
        selected = select.metrics.get("selected") or 0
        if not select.artifacts.get("selection"):
            problems.append("no selection: run the select stage first")
        elif selected < 3:
            problems.append(
                f"only {selected} frames selected; structure from motion needs at "
                "least three views and realistically dozens"
            )
        if colmap_dialect().get("available") is False:
            problems.append("COLMAP was not found; check the doctor page")
        return problems

    def run(self, ctx: StageContext) -> StageResult:
        params: SparseParams = ctx.params  # type: ignore[assignment]
        run_dir = ctx.run_dir
        scratch = ctx.scratch / "sparse"
        scratch.mkdir(parents=True, exist_ok=True)

        images_dir = scratch / "images"
        database = scratch / "database.db"
        model_root = scratch / "model"
        model_root.mkdir(parents=True, exist_ok=True)

        submitted = _build_image_farm(ctx, run_dir, images_dir)
        ctx.logger.info("linked %d selected frames into the image farm", len(submitted))

        dialect = colmap_dialect()
        matcher = _choose_matcher(params, ctx)

        ctx.progress("detecting features", fraction=0.02)
        run_tool(
            ctx,
            colmap.feature_extractor(
                database_path=database,
                image_path=images_dir,
                dialect=dialect,
                camera_model=params.camera_model,
                single_camera_per_folder=params.single_camera_per_folder,
                max_image_size=params.max_image_size,
                max_num_features=params.max_num_features,
                use_gpu=params.use_gpu,
            ),
            cwd=scratch,
            label="colmap feature_extractor",
            on_line=counter_progress(ctx, "detecting features", scale=(0.02, 0.25)),
        )

        ctx.progress(f"matching ({matcher})", fraction=0.25)
        if matcher == "sequential":
            argv = colmap.sequential_matcher(
                database_path=database,
                dialect=dialect,
                overlap=params.sequential_overlap,
                loop_detection=params.loop_detection,
                vocab_tree_path=(
                    VOCAB_TREE if vocab_tree_status() == "ok" else None
                ),
                use_gpu=params.use_gpu,
                filter_stationary_matches=params.filter_stationary_matches,
            )
        else:
            argv = colmap.exhaustive_matcher(
                database_path=database,
                dialect=dialect,
                use_gpu=params.use_gpu,
                filter_stationary_matches=params.filter_stationary_matches,
            )
        run_tool(
            ctx,
            argv,
            cwd=scratch,
            label=f"colmap {matcher}_matcher",
            on_line=counter_progress(ctx, "matching", scale=(0.25, 0.6)),
        )

        ctx.progress("solving", fraction=0.6)
        run_tool(
            ctx,
            colmap.mapper(
                database_path=database,
                image_path=images_dir,
                output_path=model_root,
                min_model_size=params.min_model_size,
            ),
            cwd=scratch,
            label="colmap mapper",
            on_line=counter_progress(ctx, "solving", scale=(0.6, 0.92)),
        )

        submodels = sorted(p for p in model_root.iterdir() if p.is_dir())
        if not submodels:
            raise RuntimeError(
                "COLMAP produced no reconstruction at all. Nothing registered, which "
                "usually means the frames do not overlap enough, the subject moved "
                "relative to its background, or the scene is too smooth to feature."
            )

        ctx.progress("reading the model", fraction=0.92)
        best, model, sizes = _pick_best(ctx, scratch, submodels)
        analyzer = _analyze(best)

        cloud_points = _write_outputs(ctx, scratch, best, model, submitted)
        # Record which submodel won before committing: the dense stage would
        # otherwise have to re-derive it, and "largest folder on disk" is a proxy for
        # "most registered images", not the same thing.
        best_name = best.name
        _commit(run_dir / "sparse", scratch)

        metrics, warnings = _summarise(
            ctx.manifest, model, analyzer, submitted, sizes, matcher, cloud_points
        )
        return StageResult(
            artifacts={
                "model": "sparse/model",
                "model_best": f"sparse/model/{best_name}",
                "images": "sparse/images",
                "poses": "sparse/poses.json",
                "registration": "sparse/registration.jsonl",
                "cloud": "sparse/points.ply",
            },
            metrics=metrics,
            warnings=warnings,
            tool_versions={"colmap": _colmap_version()},
        )


# -- setup -------------------------------------------------------------------


def _colmap_version() -> str:
    try:
        return colmap.version()
    except RuntimeError:
        return "missing"


def _choose_matcher(params: SparseParams, ctx: StageContext) -> str:
    """Decide how to match, refusing to hand COLMAP a tree that will crash it."""
    status = vocab_tree_status()

    if params.matcher == "sequential":
        if status == "legacy":
            raise RuntimeError(
                "the vocabulary tree in data/ is the old FLANN format, and this "
                "COLMAP aborts rather than rejects it. No faiss-format tree is "
                "published yet. Set the matcher to 'exhaustive' or 'auto'."
            )
        if status == "missing":
            ctx.logger.warning(
                "sequential matching requested but there is no vocabulary tree at %s, "
                "so loop detection is off and frames on opposite sides of the orbit "
                "will never be compared.",
                VOCAB_TREE,
            )
        return "sequential"

    if params.matcher == "exhaustive":
        return "exhaustive"

    if status == "ok":
        return "sequential"
    reason = (
        "the vocabulary tree is the old FLANN format, which this COLMAP cannot read"
        if status == "legacy"
        else "there is no vocabulary tree"
    )
    ctx.logger.info("matching exhaustively: %s", reason)
    return "exhaustive"


def _build_image_farm(
    ctx: StageContext, run_dir: Path, images_dir: Path
) -> list[str]:
    """Hardlink the selected frames into a per-camera tree. Returns their names."""
    selection = run_dir / "select" / "selection.jsonl"
    if not selection.exists():
        raise RuntimeError("select/selection.jsonl is missing; re-run the select stage")

    names: list[str] = []
    for line in selection.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if not record["selected"]:
            continue
        source = run_dir / record["file"]
        if not source.exists():
            continue
        target = images_dir / record["camera_group"] / Path(record["file"]).name
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(source, target)
        except OSError:
            shutil.copy2(source, target)
        # COLMAP names images relative to image_path, with forward slashes.
        names.append(f"{record['camera_group']}/{target.name}")

    if not names:
        raise RuntimeError("the selection contains no usable frames")
    ctx.cancel.raise_if_cancelled()
    return names


# -- reading the result ------------------------------------------------------


def _pick_best(
    ctx: StageContext, scratch: Path, submodels: list[Path]
) -> tuple[Path, colmap.SparseModel, list[int]]:
    """Choose the submodel with the most registered images and parse it.

    COLMAP writes each disconnected reconstruction as its own numbered folder. More
    than one is a finding, not a detail, so the sizes are carried out for the summary.
    """
    parsed: list[tuple[Path, colmap.SparseModel]] = []
    for index, submodel in enumerate(submodels):
        text_dir = scratch / "text" / submodel.name
        text_dir.mkdir(parents=True, exist_ok=True)
        run_tool(
            ctx,
            colmap.model_converter(
                input_path=submodel, output_path=text_dir, output_type="TXT"
            ),
            cwd=scratch,
            label=f"colmap model_converter[{index}]",
        )
        parsed.append((submodel, colmap.read_text_model(text_dir)))

    parsed.sort(key=lambda pair: len(pair[1].images), reverse=True)
    sizes = sorted((len(m.images) for _, m in parsed), reverse=True)
    best, model = parsed[0]
    ctx.logger.info(
        "%d submodel(s); largest has %d registered images", len(parsed), len(model.images)
    )
    return best, model, sizes


def _analyze(model_dir: Path) -> dict[str, float]:
    """Run model_analyzer for its summary statistics; failure is not fatal."""
    try:
        result = capture(
            colmap.model_analyzer(path=model_dir), cwd=model_dir.parent, timeout=300
        )
    except RuntimeError:
        return {}
    return colmap.parse_analyzer(result.combined)


def _write_outputs(
    ctx: StageContext,
    scratch: Path,
    best: Path,
    model: colmap.SparseModel,
    submitted: list[str],
) -> int:
    """Write the point cloud, the camera poses, and the per-image registration table."""
    cloud = scratch / "points.ply"
    run_tool(
        ctx,
        colmap.model_converter(input_path=best, output_path=cloud, output_type="PLY"),
        cwd=scratch,
        label="colmap model_converter[ply]",
    )

    points = 0
    if cloud.exists():
        try:
            # Normalise into the fixed preview layout. A sparse cloud is small enough
            # to serve whole, so the cap is a ceiling rather than a real decimation.
            points = ply.write_preview(cloud, scratch / "preview.ply", max_points=2_000_000)
        except ply.PlyError as exc:
            ctx.logger.warning("could not normalise the sparse cloud: %s", exc)

    registered = {image.name: image for image in model.images}
    poses = []
    for image in model.images:
        camera = model.cameras.get(image.camera_id)
        poses.append(
            {
                "name": image.name,
                "camera_id": image.camera_id,
                "qvec": list(image.qvec),
                "tvec": list(image.tvec),
                "center": list(image.center()),
                "num_points": image.num_points,
                "width": camera.width if camera else None,
                "height": camera.height if camera else None,
                "focal_px": camera.focal_px if camera else None,
            }
        )

    (scratch / "poses.json").write_text(
        json.dumps(
            {
                "cameras": [
                    {
                        "camera_id": c.camera_id,
                        "model": c.model,
                        "width": c.width,
                        "height": c.height,
                        "params": c.params,
                    }
                    for c in model.cameras.values()
                ],
                "images": poses,
            },
            indent=1,
        ),
        encoding="utf-8",
    )

    with (scratch / "registration.jsonl").open("w", encoding="utf-8") as fh:
        for name in submitted:
            image = registered.get(name)
            fh.write(
                json.dumps(
                    {
                        "name": name,
                        "registered": image is not None,
                        "num_points": image.num_points if image else 0,
                    }
                )
                + "\n"
            )
    return points


def _commit(target: Path, scratch: Path) -> None:
    if target.exists():
        shutil.rmtree(target, ignore_errors=True)
    scratch.replace(target)


def _summarise(
    manifest: RunManifest,
    model: colmap.SparseModel,
    analyzer: dict[str, float],
    submitted: list[str],
    sizes: list[int],
    matcher: str,
    cloud_points: int,
) -> tuple[dict[str, Any], list[str]]:
    registered = len(model.images)
    rate = registered / len(submitted) if submitted else 0.0
    reprojection = analyzer.get(
        "mean_reprojection_error_px", model.mean_reprojection_error_px
    )

    metrics: dict[str, Any] = {
        "matcher": matcher,
        "images_submitted": len(submitted),
        "images_registered": registered,
        "registration_rate": round(rate, 4),
        "num_points3D": model.num_points3D,
        "mean_track_length": (
            round(model.mean_track_length, 2) if model.mean_track_length else None
        ),
        "mean_reprojection_error_px": (
            round(reprojection, 3) if reprojection else None
        ),
        "mean_observations_per_image": analyzer.get("mean_observations_per_image"),
        "num_submodels": len(sizes),
        "submodel_sizes": sizes,
        "preview_points": cloud_points,
        "cameras": [
            {
                "camera_id": c.camera_id,
                "model": c.model,
                "width": c.width,
                "height": c.height,
                "focal_px": round(c.focal_px, 1) if c.focal_px else None,
            }
            for c in model.cameras.values()
        ],
    }

    warnings: list[str] = []

    if rate < REGISTRATION_WARN:
        missing = len(submitted) - registered
        warnings.append(
            f"only {registered} of {len(submitted)} frames registered ({rate:.0%}); "
            f"{missing} were dropped. COLMAP discards images it cannot place without "
            "failing, so a model built from half the frames still looks like a "
            "success. Usual causes: too little overlap between views, motion blur, or "
            "a subject that moved relative to its surroundings."
        )

    if len(sizes) > 1:
        warnings.append(
            f"COLMAP produced {len(sizes)} disconnected models (sizes {sizes}). The "
            "capture broke into pieces that never recognised each other as the same "
            "scene, which is what an orbit that did not close looks like. More "
            "overlap, or loop detection, is the fix."
        )

    if reprojection and reprojection > REPROJECTION_WARN_PX:
        warnings.append(
            f"mean reprojection error is {reprojection:.2f} px. Above about "
            f"{REPROJECTION_WARN_PX} px the fitted camera model is not describing the "
            "lens well, or the scene was not rigid while it was filmed."
        )

    if model.mean_track_length is not None and model.mean_track_length < 3.0:
        warnings.append(
            f"points are seen by only {model.mean_track_length:.1f} images on average. "
            "Short tracks triangulate weakly and give the dense stage little to agree "
            "on; it usually means consecutive frames are too far apart."
        )

    if (
        manifest.capture.mode is CaptureMode.SUBJECT_ROTATES
        and manifest.stages[StageId.MASK].state is StageState.SKIPPED
    ):
        warnings.append(
            "the subject rotated while the cameras stayed put, and masking was "
            "skipped. This model is very probably locked onto the room rather than "
            "the subject: the background is static in the world, so the solver "
            "prefers it and treats the subject as noise. Look at the camera "
            "positions -- if they scatter instead of forming an arc, that is what "
            "happened, and the mask stage is the fix."
        )

    return metrics, warnings
