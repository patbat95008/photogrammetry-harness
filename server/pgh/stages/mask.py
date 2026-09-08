"""Stage 3: mark which pixels belong to the subject and which must be ignored.

For a chair spin this is not a quality improvement, it is the stage that makes the
capture work at all. When the subject rotates and the cameras stay put, the room is
static in the world but *moves relative to the head*. The reconstruction solves for
one rigid scene, so the room is the part behaving inconsistently -- left in, the
solver locks onto it and the head never resolves. See ``CAPTURE_NOTES``, and note the
inversion: for an orbit around a still subject the background is rigid *with* the
subject, supplies extra features, and masking it out makes alignment harder.

**One click, then tracking.** SAM 2.1 is a video segmenter: it is given a point on
the subject in one frame and propagates the mask through the rest. Clicks live in the
params as ``prompts``, keyed ``"<group>/<slot>"``, so they are hashed like any other
parameter and a re-run reproduces the same masks.

**Why the propagation is chunked.** The video predictor's inference state grows with
sequence length, and it holds every decoded frame. Measured here on 241 frames of
``cup-1`` with both offloads on: 1.66 GB of VRAM but around 3 GB of *system* RAM, and
that is the number that scales. Chunking bounds it, and it is also what makes progress
and cancellation granular rather than one opaque wait.

**Chunks overlap by exactly one frame, and that overlap is load-bearing.** Each chunk
after the first is seeded with the previous chunk's last mask via ``add_new_mask``,
which attaches a mask to a frame *index*. Unless the seeded frame really is index 0 of
the next chunk, the mask of frame 149 gets attached to frame 150 and every subsequent
mask is one frame stale -- which still looks like a mask, and is wrong everywhere.

**What is carried forward is the cleaned mask, not the dilated one.** Cleanup (closing
holes, dropping specks) genuinely improves the seed: a stray blob the tracker latched
onto is not handed on to compound. Dilation is different -- it is an output
concession, widening the edge so wisps of hair survive, and feeding it back would grow
the mask by ``dilate_px`` at *every chunk boundary*. So the two are separate steps and
only the first is carried.

**Masks are written once and viewed three ways** (HANDOVER 6.7). One canonical
greyscale PNG per frame, then per-engine hardlink farms: COLMAP wants ``.png`` appended
to the whole filename, OpenMVS wants the extension replaced by ``.mask.png``. They
differ by one dot, which is why both live in named functions with tests on them.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
from pydantic import BaseModel, Field

from .. import sam2rt
from ..manifest import CaptureMode, RunManifest, StageId
from .base import Stage, StageContext, StageParams, StageResult

#: Written into the stage page so the operator sees, in the moment, why masking is
#: mandatory here and optional there. Lifted verbatim from ``planned.py`` when the
#: stage stopped being planned -- it is the most useful sentence on the page.
CAPTURE_NOTES: dict[CaptureMode, str] = {
    CaptureMode.SUBJECT_ROTATES: (
        "This run has the subject rotating while the cameras stayed put, so masking "
        "is REQUIRED. The background is static in the room but moves relative to the "
        "subject, which makes it the part behaving inconsistently. Left in, the "
        "solver locks onto the room and the subject never resolves."
    ),
    CaptureMode.CAMERA_ORBITS: (
        "This run has the cameras orbiting a still subject, so masking is OPTIONAL "
        "and usually best skipped. The background is rigid with respect to the "
        "subject, so it supplies extra features and helps the orbit close. Mask only "
        "if you specifically want the background excluded from the final mesh."
    ),
}

#: A mask covering more than this fraction of the frame has almost certainly tracked
#: the room rather than the subject, which is the same as no mask at all.
RUNAWAY_AREA_WARN = 0.75
#: Below this the tracker has probably lost the subject rather than found a small one.
THIN_AREA_WARN = 0.02
#: Frame-to-frame agreement below this is a track that jumped rather than moved.
DRIFT_IOU_WARN = 0.8


class MaskPrompt(BaseModel):
    """One click, on one frame, in normalised coordinates."""

    model_config = {"extra": "forbid"}

    x: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "Horizontal position as a fraction of the frame width. Normalised rather "
            "than in pixels so a click survives re-extracting at a different size."
        ),
    )
    y: float = Field(
        ..., ge=0.0, le=1.0, description="Vertical position as a fraction of the frame height."
    )
    include: bool = Field(
        True,
        description=(
            "A point on the thing being tracked. Clear it to push the boundary back "
            "off something the mask wrongly swallowed -- the chair back, on a spin."
        ),
    )


class MaskParams(StageParams):
    model: Literal["tiny", "small", "base_plus", "large"] = Field(
        "large",
        description=(
            "Which SAM 2.1 checkpoint to track with. Large is the default and the one "
            "this pipeline is calibrated on; the smaller ones are for a quick look, "
            "not for a final run. Each checkpoint is paired with its own config in "
            "config.py -- a mismatched pair loads without raising and masks poorly."
        ),
    )
    prompts: dict[str, list[MaskPrompt]] = Field(
        default_factory=dict,
        description=(
            "Click points per frame, keyed '<camera group>/<slot>'. Every camera "
            "group needs at least one in its first chunk; extra keys correct a frame "
            "where the track went wrong."
        ),
        # Rendered by the mask page, which needs the frame under the clicks. Still
        # hashed: the clicks are the whole input to this stage.
        json_schema_extra={"widget": "hidden"},
    )
    clicks_mark: Literal["subject", "background"] = Field(
        "subject",
        description=(
            "Whether the clicks select the thing to keep or the thing to throw away. "
            "Clicking a plain wall is sometimes easier than clicking hair. The mask "
            "file is 255-means-keep either way -- both engines are wired to that "
            "polarity, and flipping the file would mask out the subject silently."
        ),
    )
    chunk_frames: int = Field(
        150,
        ge=16,
        le=1000,
        description=(
            "Frames per propagation pass, each seeded from the last mask of the one "
            "before. Measured on cup-1 with offloading on, 241 frames peaked at 1.7 GB "
            "of video memory but around 3 GB of system memory -- so this bounds system "
            "memory and progress granularity rather than VRAM. Lower it if the machine "
            "starts swapping."
        ),
    )
    offload_to_cpu: bool = Field(
        True,
        description=(
            "Hold the decoded frames and the tracker's memory in system RAM rather "
            "than on the card. Costs perhaps a tenth of the speed and is the "
            "difference between under 2 GB of VRAM and tens of GB."
        ),
    )
    close_holes_px: int = Field(
        5,
        ge=0,
        le=64,
        description=(
            "Close gaps up to this wide inside the mask. SAM 2's own hole filling "
            "needs a compiled extension that is not built here, so it is skipped with "
            "a warning and does nothing -- and the gaps it would have closed are the "
            "ones between strands of hair."
        ),
    )
    largest_component_only: bool = Field(
        True,
        description=(
            "Keep only the biggest connected blob. The tracker occasionally grabs a "
            "speck of similar-looking background across the room; left in, it "
            "reconstructs as a floating fragment."
        ),
    )
    dilate_px: int = Field(
        2,
        ge=-64,
        le=64,
        description=(
            "Grow the mask edge by this many pixels, or shrink it with a negative "
            "value. A little growth keeps the wisps of hair and the rim of an ear "
            "that the boundary cuts through; shrinking trims a halo of background "
            "that came with them. Applied on the way out only, never fed back into "
            "the tracker, or it would compound at every chunk boundary."
        ),
    )
    min_area_fraction: float = Field(
        0.002,
        ge=0.0,
        le=0.5,
        description=(
            "A mask smaller than this fraction of the frame is recorded as a lost "
            "track rather than as a very small subject, so the stage can name the "
            "frame where tracking failed instead of quietly writing empty masks."
        ),
    )
    overlay_px: int = Field(
        384,
        ge=128,
        le=1024,
        description="Width of the review overlays. Previews only.",
        json_schema_extra={"affects_fingerprint": False},
    )


class MaskStage(Stage):
    id = StageId.MASK
    label = "Mask"
    description = "Mark which pixels belong to the subject and which must be ignored."
    params_model = MaskParams
    # Extract, not select. Masking every extracted frame costs seconds at 0.1s a
    # frame, and it means adjusting a selection knob does not throw away every mask.
    # It also gives SAM 2 consecutive frames, which is what it tracks best over;
    # the coverage-thinned selection has gaps of varying size.
    depends_on = [StageId.EXTRACT]

    def note(self, manifest: RunManifest) -> str | None:
        return CAPTURE_NOTES.get(manifest.capture.mode)

    def external_inputs(self, manifest: RunManifest) -> dict[str, Any]:
        extract = manifest.stages[StageId.EXTRACT]
        return {
            "extract_fingerprint": extract.fingerprint,
            "frame_count": extract.metrics.get("frame_count"),
            "camera_groups": extract.metrics.get("camera_groups"),
            "sam2": sam2rt.sam2_version(),
        }

    def preflight(self, manifest: RunManifest) -> list[str]:
        problems: list[str] = []
        extract = manifest.stages[StageId.EXTRACT]
        groups = extract.metrics.get("camera_groups") or []
        if not groups:
            problems.append("the extract stage recorded no camera groups; re-run it")

        try:
            params = MaskParams.model_validate(manifest.stages[StageId.MASK].params or {})
        except Exception:
            params = MaskParams()

        try:
            sam2rt.import_sam2()
            sam2rt.resolve_model(params.model)
        except RuntimeError as exc:
            problems.append(str(exc))

        total = int(extract.metrics.get("frame_count") or 0)
        for group in groups:
            slots = sorted(
                int(key.split("/", 1)[1])
                for key in params.prompts
                if key.split("/", 1)[0] == group and params.prompts[key]
            )
            if not slots:
                problems.append(
                    f"{group} has no click points. Open the mask page and click the "
                    f"subject on its first frame -- a camera with no prompt would be "
                    f"left unmasked while the other is masked, and the solver would "
                    f"still lock onto the room through it."
                )
                continue
            # The first chunk has to be conditioned on something. A prompt that only
            # appears later leaves the propagation with nothing to track from.
            if total and slots[0] >= params.chunk_frames:
                problems.append(
                    f"{group}'s earliest click is on slot {slots[0]}, which falls "
                    f"beyond the first chunk of {params.chunk_frames} frames. "
                    f"Propagation starts at the first frame and would have nothing to "
                    f"track. Click the subject near the start of {group} as well."
                )
        return problems

    def run(self, ctx: StageContext) -> StageResult:
        params: MaskParams = ctx.params  # type: ignore[assignment]
        run_dir = ctx.run_dir
        scratch = ctx.scratch / "mask"
        scratch.mkdir(parents=True, exist_ok=True)

        frames = _load_frames(run_dir)
        by_group = _by_group(frames)
        plans = {g: plan_chunks(sorted(f["slot"] for f in fs), params.chunk_frames)
                 for g, fs in by_group.items()}
        # Unique frames, not the sum of chunk lengths: chunks overlap by one, and the
        # seam frame is propagated twice but written once.
        total = sum(len(entries) for entries in by_group.values())
        if not total:
            raise RuntimeError("frames.jsonl lists no frames; re-run the extract stage")

        ctx.progress(f"loading SAM 2.1 {params.model}", fraction=0.0)
        predictor = sam2rt.load_video_predictor(params.model)

        records: list[dict[str, Any]] = []
        started = time.monotonic()
        done = 0
        transcoded = 0

        for group in sorted(plans):
            lookup = {f["slot"]: f for f in by_group[group]}
            seed: np.ndarray | None = None
            previous: np.ndarray | None = None

            for index, chunk in enumerate(plans[group]):
                ctx.cancel.raise_if_cancelled()
                chunk_dir = ctx.scratch / "chunks" / group / f"{index:03d}"
                transcoded += _stage_chunk(run_dir, chunk, lookup, chunk_dir)

                state = predictor.init_state(
                    video_path=str(chunk_dir),
                    offload_video_to_cpu=params.offload_to_cpu,
                    offload_state_to_cpu=params.offload_to_cpu,
                )
                if seed is not None:
                    predictor.add_new_mask(state, frame_idx=0, obj_id=1, mask=seed)

                clicks = _prompts_in_chunk(params, group, chunk, lookup)
                for local, points, labels in clicks:
                    predictor.add_new_points_or_box(
                        state, frame_idx=local, obj_id=1, points=points, labels=labels
                    )
                if seed is None and not clicks:
                    raise RuntimeError(
                        f"{group} chunk {index} has neither a click nor a mask carried "
                        f"forward, so there is nothing to track. This is a bug in "
                        f"chunk planning rather than something you did."
                    )

                last_clean: np.ndarray | None = None
                was_seeded = seed is not None
                for local, _ids, logits in predictor.propagate_in_video(state):
                    ctx.cancel.raise_if_cancelled()
                    raw = np.asarray((logits[0, 0] > 0.0).cpu().numpy(), dtype=bool)
                    if params.clicks_mark == "background":
                        raw = ~raw
                    clean = clean_mask(raw, params)
                    out = dilate_mask(clean, params.dilate_px)
                    last_clean = clean

                    # Frame 0 of a seeded chunk is the seam: the previous chunk already
                    # tracked and wrote it. Re-deriving it here from its own seed would
                    # put a second, slightly different row in masks.jsonl for one frame.
                    if was_seeded and local == 0:
                        continue

                    slot = chunk[local]
                    _write_canonical(scratch, group, slot, out)
                    _write_overlay(scratch, run_dir, lookup[slot], group, slot, out,
                                   params.overlay_px)
                    records.append(
                        _record(group, slot, lookup[slot], out, previous, index,
                                seeded=(seed is not None and local == 0),
                                prompted=any(c[0] == local for c in clicks),
                                params=params)
                    )
                    previous = out
                    done += 1
                    if done % 5 == 0 or done == total:
                        ctx.progress(
                            f"{group}, chunk {index + 1} of {len(plans[group])}",
                            current=done,
                            total=total,
                        )

                # Carry the cleaned mask, not the dilated one -- see the module
                # docstring. Read it before the state is dropped.
                seed = last_clean
                predictor.reset_state(state)
                del state
                shutil.rmtree(chunk_dir, ignore_errors=True)

        records.sort(key=lambda r: (r["camera_group"], r["slot"]))
        views = _write_views(scratch, records)
        _write_records(scratch / "masks.jsonl", records)
        _commit(run_dir / "mask", scratch)

        elapsed = time.monotonic() - started
        metrics, warnings = _summarise(ctx.manifest, records, params, plans, elapsed,
                                       transcoded, views)
        return StageResult(
            artifacts={
                "masks": "mask/canonical",
                "masks_colmap": "mask/colmap",
                "masks_openmvs": "mask/openmvs",
                "overlays": "mask/overlay",
                "index": "mask/masks.jsonl",
            },
            metrics=metrics,
            warnings=warnings,
            tool_versions={"sam2": sam2rt.sam2_version()},
        )


# -- naming ------------------------------------------------------------------

def colmap_view_name(image_name: str) -> str:
    """``000042.jpg`` -> ``000042.jpg.png`` (HANDOVER 6.7).

    COLMAP appends ``.png`` to the *whole* filename, extension included. OpenMVS
    replaces the extension instead. The two differ by one dot and are invisible in a
    diff, which is why they are functions with tests rather than inline f-strings.
    """
    return f"{image_name}.png"


def openmvs_view_name(image_name: str) -> str:
    """``000042.jpg`` -> ``000042.mask.png`` (HANDOVER 6.7)."""
    return f"{Path(image_name).stem}.mask.png"


def prompt_key(group: str, slot: int) -> str:
    return f"{group}/{slot}"


# -- chunking ----------------------------------------------------------------

def plan_chunks(slots: list[int], size: int) -> list[list[int]]:
    """Split slots into chunks of ``size`` that **overlap by exactly one frame**.

    The overlap is the mechanism, not tidiness. Each chunk after the first is seeded
    with the previous chunk's final mask, and ``add_new_mask`` attaches that mask to a
    frame *index* -- so the seeded frame has to genuinely be index 0 of the next
    chunk. Without the overlap the mask of frame 149 is attached to frame 150 and
    every mask after it is one frame stale, which looks entirely plausible and is
    wrong from there on.
    """
    if not slots or size < 2:
        return [list(slots)] if slots else []

    chunks: list[list[int]] = []
    start = 0
    while start < len(slots):
        chunk = slots[start : start + size]
        chunks.append(chunk)
        if start + size >= len(slots):
            break
        start += size - 1  # step back one, so the last frame reappears as the next first
    return chunks


# -- mask arithmetic ---------------------------------------------------------

def clean_mask(mask: np.ndarray, params: MaskParams) -> np.ndarray:
    """Close small holes and drop stray blobs. Never dilates -- see the docstring."""
    out = np.asarray(mask, dtype=bool)
    if params.close_holes_px > 0:
        size = params.close_holes_px * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        out = cv2.morphologyEx(
            out.astype(np.uint8), cv2.MORPH_CLOSE, kernel
        ).astype(bool)
    if params.largest_component_only and out.any():
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            out.astype(np.uint8), connectivity=8
        )
        if count > 2:  # 0 is the background label
            biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            out = labels == biggest
    return out


def dilate_mask(mask: np.ndarray, pixels: int) -> np.ndarray:
    """Grow (or, negative, shrink) the mask edge."""
    if not pixels:
        return np.asarray(mask, dtype=bool)
    size = abs(pixels) * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    op = cv2.dilate if pixels > 0 else cv2.erode
    return op(np.asarray(mask, dtype=np.uint8), kernel).astype(bool)


def to_bytes(mask: np.ndarray) -> np.ndarray:
    """A mask as 8-bit, exactly {0, 255}.

    Both engines compare against an exact value -- COLMAP keeps non-zero pixels,
    OpenMVS ignores pixels equal to the label -- so a stray 254 from a resize or a
    lossy write is a hole nothing reports.
    """
    return np.where(np.asarray(mask, dtype=bool), 255, 0).astype(np.uint8)


def iou(a: np.ndarray | None, b: np.ndarray | None) -> float | None:
    if a is None or b is None or a.shape != b.shape:
        return None
    union = np.logical_or(a, b).sum()
    if not union:
        return None
    return round(float(np.logical_and(a, b).sum() / union), 4)


# -- reading and writing -----------------------------------------------------

def _load_frames(run_dir: Path) -> list[dict[str, Any]]:
    index = run_dir / "frames" / "frames.jsonl"
    if not index.exists():
        raise RuntimeError("frames/frames.jsonl is missing; re-run the extract stage")
    frames = [
        json.loads(line)
        for line in index.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not frames:
        raise RuntimeError("frames.jsonl is empty; re-run the extract stage")
    return frames


def _by_group(frames: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for frame in frames:
        grouped.setdefault(frame["camera_group"], []).append(frame)
    for entries in grouped.values():
        entries.sort(key=lambda f: f["slot"])
    return grouped


def _stage_chunk(
    run_dir: Path, chunk: list[int], lookup: dict[int, dict], target: Path
) -> int:
    """Put this chunk's frames where SAM 2 can read them. Returns frames transcoded.

    SAM 2 reads a directory of JPEGs and sorts them by ``int(stem)``, so the slot
    names carry through untouched and ``chunk[local_index]`` is the slot. Frames are
    hardlinked when they are already JPEG; a run extracted to PNG -- which the extract
    stage allows -- is transcoded instead, because SAM 2 skips every other extension
    and would otherwise report an empty directory rather than an unreadable one.
    """
    shutil.rmtree(target, ignore_errors=True)
    target.mkdir(parents=True, exist_ok=True)
    transcoded = 0
    for slot in chunk:
        source = run_dir / lookup[slot]["file"]
        destination = target / f"{slot:06d}.jpg"
        if source.suffix.lower() in (".jpg", ".jpeg"):
            try:
                os.link(source, destination)
            except OSError:
                shutil.copy2(source, destination)
        else:
            image = cv2.imread(str(source), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"could not read {source} to hand to SAM 2")
            cv2.imwrite(str(destination), image, [cv2.IMWRITE_JPEG_QUALITY, 95])
            transcoded += 1
    return transcoded


def _prompts_in_chunk(
    params: MaskParams, group: str, chunk: list[int], lookup: dict[int, dict]
) -> list[tuple[int, np.ndarray, np.ndarray]]:
    """Click points that fall inside this chunk, at their *local* frame index.

    A prompt for a slot that no longer exists -- the frames were re-extracted at a
    different rate, say -- is ignored rather than raising: the click is stale input,
    not a broken run.
    """
    positions = {slot: local for local, slot in enumerate(chunk)}
    found: list[tuple[int, np.ndarray, np.ndarray]] = []
    for key, clicks in sorted(params.prompts.items()):
        if "/" not in key or not clicks:
            continue
        key_group, _, raw_slot = key.partition("/")
        if key_group != group or not raw_slot.isdigit():
            continue
        slot = int(raw_slot)
        if slot not in positions or slot not in lookup:
            continue
        frame = lookup[slot]
        width, height = int(frame["w"]), int(frame["h"])
        points = np.array(
            [[click.x * width, click.y * height] for click in clicks], dtype=np.float32
        )
        labels = np.array([1 if click.include else 0 for click in clicks], dtype=np.int32)
        found.append((positions[slot], points, labels))
    return found


def _write_canonical(scratch: Path, group: str, slot: int, mask: np.ndarray) -> None:
    target = scratch / "canonical" / group
    target.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(target / f"{slot:06d}.png"), to_bytes(mask))


def _write_overlay(
    scratch: Path,
    run_dir: Path,
    frame: dict[str, Any],
    group: str,
    slot: int,
    mask: np.ndarray,
    width: int,
) -> None:
    """A small tinted thumbnail, so the review strip is one fetch per frame."""
    image = cv2.imread(str(run_dir / frame["file"]), cv2.IMREAD_COLOR)
    if image is None:
        return
    height = max(1, round(image.shape[0] * width / image.shape[1]))
    small = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    # Nearest neighbour: the mask is binary and interpolating it would invent the
    # partial values that to_bytes exists to prevent.
    thumb = cv2.resize(to_bytes(mask), (width, height), interpolation=cv2.INTER_NEAREST)
    tint = np.zeros_like(small)
    tint[:, :, 1] = 255
    selected = thumb > 127
    small[selected] = (0.55 * small[selected] + 0.45 * tint[selected]).astype(np.uint8)

    target = scratch / "overlay" / group
    target.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(target / f"{slot:06d}.webp"), small, [cv2.IMWRITE_WEBP_QUALITY, 80])


def _write_views(scratch: Path, records: list[dict[str, Any]]) -> dict[str, int]:
    """Hardlink the canonical masks into the two per-engine naming conventions.

    Driven from the records rather than by globbing the canonical tree, because both
    engines name a mask after its **image** and only the record knows what that image
    was called. COLMAP appends to the whole filename, so a run extracted to PNG needs
    ``000042.png.png`` -- assuming ``.jpg`` there would produce a tree of names COLMAP
    silently never matches, and a missing mask is not something it reports.

    Hardlinks rather than copies: the same bytes under three names costs one copy of
    the pixels. Falls back to copying on the one case ``os.link`` refuses, a target on
    a different volume.
    """
    counts = {"colmap": 0, "openmvs": 0}
    for record in records:
        group = record["camera_group"]
        source = scratch / "canonical" / group / f"{record['slot']:06d}.png"
        if not source.is_file():
            continue
        image_name = Path(record["file"]).name
        for view, rename in (("colmap", colmap_view_name), ("openmvs", openmvs_view_name)):
            target = scratch / view / group / rename(image_name)
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(source, target)
            except OSError:
                shutil.copy2(source, target)
            counts[view] += 1
    return counts


def _record(
    group: str,
    slot: int,
    frame: dict[str, Any],
    mask: np.ndarray,
    previous: np.ndarray | None,
    chunk: int,
    *,
    seeded: bool,
    prompted: bool,
    params: MaskParams,
) -> dict[str, Any]:
    area = float(mask.mean())
    rows = np.any(mask, axis=1)
    columns = np.any(mask, axis=0)
    if rows.any():
        y0, y1 = int(np.argmax(rows)), int(len(rows) - np.argmax(rows[::-1]) - 1)
        x0, x1 = int(np.argmax(columns)), int(len(columns) - np.argmax(columns[::-1]) - 1)
        bbox = [x0, y0, x1, y1]
    else:
        bbox = None

    edges = []
    if mask.shape[0] and mask.shape[1]:
        if mask[0].any():
            edges.append("top")
        if mask[-1].any():
            edges.append("bottom")
        if mask[:, 0].any():
            edges.append("left")
        if mask[:, -1].any():
            edges.append("right")

    return {
        "slot": slot,
        "camera_group": group,
        "file": frame["file"],
        "mask": f"mask/canonical/{group}/{slot:06d}.png",
        "overlay": f"mask/overlay/{group}/{slot:06d}.webp",
        "chunk": chunk,
        "seeded": seeded,
        "prompted": prompted,
        "area_fraction": round(area, 5),
        "bbox": bbox,
        "touches_border": edges,
        "empty": bool(area < params.min_area_fraction),
        "iou_prev": iou(previous, mask),
    }


def _write_records(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def _commit(target: Path, scratch: Path) -> None:
    shutil.rmtree(target, ignore_errors=True)
    scratch.replace(target)


def _summarise(
    manifest: RunManifest,
    records: list[dict[str, Any]],
    params: MaskParams,
    plans: dict[str, list[list[int]]],
    elapsed: float,
    transcoded: int,
    views: dict[str, int],
) -> tuple[dict[str, Any], list[str]]:
    areas = [r["area_fraction"] for r in records]
    empty = [r for r in records if r["empty"]]
    border = [r for r in records if r["touches_border"]]
    drifted = [
        r for r in records if r["iou_prev"] is not None and r["iou_prev"] < DRIFT_IOU_WARN
    ]
    mean_area = round(sum(areas) / len(areas), 5) if areas else 0.0

    metrics: dict[str, Any] = {
        "frames_masked": len(records),
        "masked_per_group": {
            group: sum(1 for r in records if r["camera_group"] == group)
            for group in sorted(plans)
        },
        "chunks_per_group": {group: len(chunks) for group, chunks in sorted(plans.items())},
        "model": params.model,
        "mean_area_fraction": mean_area,
        "min_area_fraction": round(min(areas), 5) if areas else 0.0,
        "max_area_fraction": round(max(areas), 5) if areas else 0.0,
        "empty_masks": len(empty),
        "border_touching": len(border),
        "drifting_frames": len(drifted),
        "views_written": views,
        "frames_transcoded": transcoded,
        "seconds_per_frame": round(elapsed / len(records), 3) if records else None,
    }
    if empty:
        metrics["lost_track_slots"] = [
            f"{r['camera_group']}/{r['slot']}" for r in empty[:20]
        ]

    warnings: list[str] = []
    if empty:
        first = empty[0]
        warnings.append(
            f"the mask vanished on {len(empty)} frames, first at "
            f"{first['camera_group']}/{first['slot']:06d}. SAM 2 lost the subject "
            f"there, and every frame after it in that chunk is tracking nothing -- "
            f"those frames contribute no features at all, so COLMAP will not register "
            f"them and the dense stage will produce no depth for them. Click the "
            f"subject on that frame and run this again."
        )
    if mean_area > RUNAWAY_AREA_WARN:
        warnings.append(
            f"the mask covers {mean_area * 100:.0f}% of the frame on average, which is "
            f"almost all of it. The clicks very probably landed on the room rather "
            f"than on the subject, and a mask that keeps the background is the same as "
            f"no mask -- the solver will still lock onto the static room. Check the "
            f"cutout view before running alignment."
        )
    elif 0 < mean_area < THIN_AREA_WARN:
        warnings.append(
            f"the mask covers only {mean_area * 100:.1f}% of the frame. Feature "
            f"extraction has almost no area to work in, and an orbit built from that "
            f"few features usually breaks into disconnected submodels rather than "
            f"failing outright."
        )
    if border:
        warnings.append(
            f"the mask touches the frame edge on {len(border)} frames, so the subject "
            f"was partly out of shot there. The reconstruction simply stops at the "
            f"crop, which reads as a flat cut through the subject rather than as "
            f"missing data."
        )
    if drifted:
        first = drifted[0]
        warnings.append(
            f"the mask changed shape abruptly between neighbouring frames "
            f"{len(drifted)} times, first at {first['camera_group']}/{first['slot']:06d}. "
            f"Neighbouring frames are a fraction of a second apart, so the track "
            f"jumped rather than followed -- usually onto something behind the "
            f"subject. Look at those frames in the review strip before trusting them."
        )
    if manifest.capture.mode is CaptureMode.CAMERA_ORBITS:
        warnings.append(
            "this run has the cameras orbiting a still subject, where the background "
            "is rigid with the subject and supplies features that help the orbit "
            "close. Masking it out is right only if you specifically want the "
            "background out of the mesh; it will make alignment harder, not easier."
        )
    return metrics, warnings
