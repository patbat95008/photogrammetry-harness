"""Stage 2: choose which extracted frames are worth reconstructing.

Thousands of frames is not better than a few hundred. Matching cost grows with the
square of the image count, and a blurry or duplicated frame does not merely waste
that time -- it contributes wrong correspondences that bundle adjustment then tries
to satisfy.

Three rejections, in the order they are applied:

**Duplicates** are already flagged in ``frames.jsonl`` by the extract stage, so they
are read rather than recomputed. Two views with no baseline between them have nothing
to triangulate.

**Blur** is measured as the variance of the Laplacian -- how much high-frequency
detail survives -- and deliberately *not* over the whole frame. A shot with a sharp
background and a motion-blurred subject scores well globally and is useless here, so
the measurement is restricted to the region the subject occupies.

**Coverage** thins what is left so the kept frames are spread evenly rather than
clustered wherever the camera happened to linger. Note the honest limitation: at this
point in the pipeline there are no poses, so "evenly spread" means evenly spread in
*slot order*, which equals evenly spread in angle only if rotation was reasonably
constant. The extract stage's degrees-per-second reading is the check on that.

Manual overrides are applied last and always win, because the operator can see things
none of these measures can.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
from pydantic import Field

from ..manifest import RunManifest, StageId
from .base import Stage, StageContext, StageParams, StageResult

#: Below roughly this many registered views, COLMAP's incremental mapper tends to
#: fail to close an orbit rather than closing it badly.
THIN_MODEL_WARN = 60

#: A sharp frame and a soft frame in the same take normally differ several-fold in
#: Laplacian variance. Much less spread than this and the take is uniformly soft, so
#: the percentile floor is discarding frames for no real reason.
SHARPNESS_SPREAD_WARN = 2.0

#: Fraction of the frame's width and height treated as "the subject" in centre mode.
CENTRE_FRACTION = 0.5


class SelectParams(StageParams):
    target_count: int | None = Field(
        200,
        ge=8,
        description=(
            "Roughly how many frames to keep, thinned evenly across the take. "
            "Leave empty to keep everything that passes the other checks."
        ),
    )
    sharpness_floor_percentile: float = Field(
        20.0,
        ge=0.0,
        le=90.0,
        description=(
            "Reject this percentage of the softest frames. Relative rather than "
            "absolute, because what counts as sharp depends on lens, light and subject."
        ),
    )
    drop_duplicates: bool = Field(
        True,
        description=(
            "Drop frames extraction flagged as near-identical to the one before. "
            "Two views with no baseline between them cannot be triangulated."
        ),
    )
    subject_region: Literal["auto", "center", "full"] = Field(
        "auto",
        description=(
            "Where sharpness is measured. 'center' uses the middle of the frame, "
            "which is where an orbited subject sits; 'full' scores the whole image "
            "and will happily rate a sharp background over a blurred subject. "
            "'auto' means centre for now, and the mask outline once masking exists."
        ),
    )
    coverage: Literal["even_slots", "off"] = Field(
        "even_slots",
        description=(
            "Thin the survivors evenly over the take so no part of the orbit is "
            "over-represented. Even in slot order, which is even in angle only if "
            "rotation was steady -- check degrees per second on the extract stage."
        ),
    )
    overrides: dict[str, Literal["keep", "reject"]] = Field(
        default_factory=dict,
        description="Manual accept/reject per frame, keyed '<camera group>/<slot>'.",
        # Rendered by the contact sheet, not by the generic params form, which has no
        # widget for a mapping. Still hashed: an override changes the output.
        json_schema_extra={"widget": "hidden"},
    )


class SelectStage(Stage):
    id = StageId.SELECT
    label = "Select frames"
    description = "Keep the sharp, non-redundant frames that cover the take evenly."
    params_model = SelectParams

    def external_inputs(self, manifest: RunManifest) -> dict[str, Any]:
        """The frame set this ran against, identified by what extract reported.

        Cheap on purpose: this is called on every page load, so it leans on the
        upstream stage's own fingerprint rather than re-reading the frame index.
        """
        extract = manifest.stages[StageId.EXTRACT]
        return {
            "frame_count": extract.metrics.get("frame_count"),
            "camera_groups": extract.metrics.get("camera_groups"),
        }

    def preflight(self, manifest: RunManifest) -> list[str]:
        problems: list[str] = []
        extract = manifest.stages[StageId.EXTRACT]
        if not extract.artifacts.get("frames_index"):
            problems.append("no frame index: run the extract stage first")
        if not extract.metrics.get("frame_count"):
            problems.append("extract produced no frames")
        return problems

    def run(self, ctx: StageContext) -> StageResult:
        params: SelectParams = ctx.params  # type: ignore[assignment]
        run_dir = ctx.run_dir

        frames = _load_frames(run_dir)
        if not frames:
            raise RuntimeError("frames.jsonl is empty; re-run the extract stage")

        ctx.logger.info("scoring %d frames for sharpness", len(frames))
        _score_sharpness(ctx, run_dir, frames, params)

        decisions = _decide(frames, params)

        scratch = ctx.scratch / "select"
        scratch.mkdir(parents=True, exist_ok=True)
        _write_selection(scratch / "selection.jsonl", decisions)
        _commit(run_dir / "select", scratch)

        metrics, warnings = _summarise(decisions, params)
        return StageResult(
            artifacts={"selection": "select/selection.jsonl"},
            metrics=metrics,
            warnings=warnings,
        )


# -- reading -----------------------------------------------------------------


def _load_frames(run_dir: Path) -> list[dict[str, Any]]:
    index = run_dir / "frames" / "frames.jsonl"
    if not index.exists():
        return []
    return [
        json.loads(line)
        for line in index.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _imread(path: Path) -> np.ndarray | None:
    """Read an image as greyscale.

    Goes through ``np.fromfile`` rather than ``cv2.imread`` because OpenCV's own
    reader takes a byte string for the path and mangles anything outside the active
    code page on Windows.
    """
    try:
        buffer = np.fromfile(path, dtype=np.uint8)
    except OSError:
        return None
    if buffer.size == 0:
        return None
    return cv2.imdecode(buffer, cv2.IMREAD_GRAYSCALE)


def _crop(image: np.ndarray, region: str) -> np.ndarray:
    if region == "full":
        return image
    height, width = image.shape[:2]
    half = CENTRE_FRACTION / 2.0
    y0, y1 = int(height * (0.5 - half)), int(height * (0.5 + half))
    x0, x1 = int(width * (0.5 - half)), int(width * (0.5 + half))
    crop = image[y0:y1, x0:x1]
    return crop if crop.size else image


def _score_sharpness(
    ctx: StageContext,
    run_dir: Path,
    frames: list[dict[str, Any]],
    params: SelectParams,
) -> None:
    """Attach a ``sharpness`` value to every frame record, in place."""
    region = "center" if params.subject_region == "auto" else params.subject_region
    total = len(frames)

    for done, frame in enumerate(frames):
        if done % 10 == 0:
            ctx.progress("measuring sharpness", current=done, total=total)
        ctx.cancel.raise_if_cancelled()

        image = _imread(run_dir / frame["file"])
        if image is None:
            frame["sharpness"] = 0.0
            frame["unreadable"] = True
            continue
        patch = _crop(image, region)
        # Variance of the Laplacian: the spread of the second derivative, which
        # collapses toward zero as an image loses high-frequency detail.
        frame["sharpness"] = float(cv2.Laplacian(patch, cv2.CV_64F).var())

    ctx.progress("measuring sharpness", current=total, total=total)


# -- deciding ----------------------------------------------------------------


def _decide(
    frames: list[dict[str, Any]], params: SelectParams
) -> list[dict[str, Any]]:
    """Turn scored frames into keep/reject decisions with a reason for each."""
    by_group: dict[str, list[dict[str, Any]]] = {}
    for frame in frames:
        by_group.setdefault(frame["camera_group"], []).append(frame)

    decisions: list[dict[str, Any]] = []
    for group, group_frames in by_group.items():
        group_frames.sort(key=lambda f: f["slot"])
        scores = [f["sharpness"] for f in group_frames]
        floor = (
            float(np.percentile(scores, params.sharpness_floor_percentile))
            if scores and params.sharpness_floor_percentile > 0
            else 0.0
        )

        survivors: list[dict[str, Any]] = []
        rejected: list[tuple[dict[str, Any], str]] = []

        for frame in group_frames:
            if frame.get("unreadable"):
                rejected.append((frame, "unreadable"))
            elif params.drop_duplicates and frame.get("duplicate_of") is not None:
                rejected.append((frame, "duplicate"))
            elif frame["sharpness"] < floor:
                rejected.append((frame, "blurry"))
            else:
                survivors.append(frame)

        kept = _thin(survivors, params)
        kept_slots = {id(f) for f in kept}
        for frame in survivors:
            if id(frame) not in kept_slots:
                rejected.append((frame, "coverage"))

        for frame in kept:
            decisions.append(_decision(frame, group, True, ""))
        for frame, reason in rejected:
            decisions.append(_decision(frame, group, False, reason))

    _apply_overrides(decisions, params)
    decisions.sort(key=lambda d: (d["camera_group"], d["slot"]))
    _rank(decisions)
    return decisions


def _decision(
    frame: dict[str, Any], group: str, selected: bool, reason: str
) -> dict[str, Any]:
    return {
        "slot": frame["slot"],
        "camera_group": group,
        "file": frame["file"],
        "sharpness": round(frame["sharpness"], 2),
        "selected": selected,
        "reason": reason,
        "overridden": False,
    }


def _thin(survivors: list[dict[str, Any]], params: SelectParams) -> list[dict[str, Any]]:
    """Reduce to roughly ``target_count`` frames, evenly spaced in slot order.

    Uses evenly spaced positions rather than a fixed stride so the first and last
    frames are always kept: dropping either end shortens the angular arc, and on a
    take with barely enough overlap that is what stops the loop closing.
    """
    target = params.target_count
    if params.coverage == "off" or target is None or len(survivors) <= target:
        return list(survivors)
    if target <= 1:
        return survivors[:1]
    positions = np.linspace(0, len(survivors) - 1, target).round().astype(int)
    return [survivors[i] for i in sorted(set(positions.tolist()))]


def _apply_overrides(decisions: list[dict[str, Any]], params: SelectParams) -> None:
    """Manual decisions beat every automatic rule, and say so in the reason."""
    if not params.overrides:
        return
    index = {f"{d['camera_group']}/{d['slot']}": d for d in decisions}
    for key, verdict in params.overrides.items():
        decision = index.get(key)
        if decision is None:
            continue
        decision["selected"] = verdict == "keep"
        decision["reason"] = "" if verdict == "keep" else "manual"
        decision["overridden"] = True


def _rank(decisions: list[dict[str, Any]]) -> None:
    """Record each frame's sharpness percentile within its camera group."""
    by_group: dict[str, list[dict[str, Any]]] = {}
    for decision in decisions:
        by_group.setdefault(decision["camera_group"], []).append(decision)
    for group_decisions in by_group.values():
        scores = np.array([d["sharpness"] for d in group_decisions], dtype=float)
        order = scores.argsort().argsort()
        span = max(1, len(group_decisions) - 1)
        for decision, rank in zip(group_decisions, order):
            decision["sharpness_percentile"] = round(100.0 * rank / span, 1)


# -- output ------------------------------------------------------------------


def _write_selection(path: Path, decisions: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for decision in decisions:
            fh.write(json.dumps(decision) + "\n")


def _commit(target: Path, scratch: Path) -> None:
    """Swap the finished output into place. Re-running replaces prior results."""
    if target.exists():
        shutil.rmtree(target, ignore_errors=True)
    scratch.replace(target)


def _summarise(
    decisions: list[dict[str, Any]], params: SelectParams
) -> tuple[dict[str, Any], list[str]]:
    selected = [d for d in decisions if d["selected"]]
    reasons: dict[str, int] = {}
    for decision in decisions:
        if not decision["selected"]:
            reasons[decision["reason"]] = reasons.get(decision["reason"], 0) + 1

    per_group = {
        group: sum(1 for d in selected if d["camera_group"] == group)
        for group in sorted({d["camera_group"] for d in decisions})
    }
    kept_scores = [d["sharpness"] for d in selected]
    all_scores = [d["sharpness"] for d in decisions]

    metrics: dict[str, Any] = {
        "input_frames": len(decisions),
        "selected": len(selected),
        "rejected": len(decisions) - len(selected),
        "rejected_by_reason": reasons,
        "selected_per_group": per_group,
        "overridden": sum(1 for d in decisions if d["overridden"]),
        "sharpness_p50": round(float(np.median(all_scores)), 2) if all_scores else 0.0,
        "sharpness_min_kept": round(min(kept_scores), 2) if kept_scores else 0.0,
    }

    warnings: list[str] = []

    if len(selected) < THIN_MODEL_WARN:
        warnings.append(
            f"only {len(selected)} frames selected. Below roughly {THIN_MODEL_WARN} "
            "views the mapper usually cannot close an orbit at all, rather than "
            "closing it badly. Raise the target count, or lower the sharpness floor."
        )

    if all_scores:
        low = float(np.percentile(all_scores, 10))
        high = float(np.percentile(all_scores, 90))
        if low > 0 and high / low < SHARPNESS_SPREAD_WARN:
            warnings.append(
                f"the sharpest and softest frames differ by only {high / low:.1f}x, "
                "so there is no sharp subset to find -- the whole take is uniformly "
                "soft. The percentile floor is discarding frames that are no worse "
                "than the ones it keeps. Fix this at capture: more light, faster "
                "shutter, slower movement."
            )

    if len(per_group) > 1:
        counts = list(per_group.values())
        if min(counts) * 2 < max(counts):
            warnings.append(
                f"cameras contributed very unevenly ({per_group}). The thinner camera "
                "was either softer throughout or recorded fewer usable frames, and a "
                "camera with too few views may not register at all."
            )

    if reasons.get("unreadable"):
        warnings.append(
            f"{reasons['unreadable']} frame(s) could not be read from disk. The "
            "extract stage's output may be incomplete; re-running it is the fix."
        )

    return metrics, warnings
