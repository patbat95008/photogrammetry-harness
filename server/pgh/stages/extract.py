"""Stage 1: source video to a slot-indexed frame set.

The output layout is chosen to feed the rest of the pipeline directly:

    frames/<camera_group>/<slot:06d>.jpg

One folder per physical camera, so ``--ImageReader.single_camera_per_folder 1``
gives COLMAP one intrinsic per camera. Filenames are shared timeline slots, so
``cam_high/000042.jpg`` and ``cam_eye/000042.jpg`` are the same instant and can
later be declared a rig frame without re-extracting anything.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Literal

import numpy as np
from PIL import Image
from pydantic import Field

from ..manifest import RunManifest, StageId, TimelineSummary
from ..timeline import SegmentPlan, compute_segment_plan, residual_p95
from ..vendor import ffmpeg
from .base import Stage, StageContext, StageParams, StageResult

#: dHash Hamming distance at or below this counts as a duplicate frame. Duplicates
#: are poison for structure-from-motion: two identical views have zero baseline, so
#: the triangulation between them is degenerate.
DUPLICATE_HAMMING = 2

#: Warn above this apparent rotation rate. Faster than roughly this and rolling
#: shutter skew starts to distort the geometry in a way nothing downstream can fix.
FAST_ROTATION_DEG_PER_S = 15.0

#: Warn if mean frame brightness varies by more than this across the take, which
#: means auto-exposure was hunting and the texture will show seams.
LUMA_RANGE_WARN = 0.15


class ExtractParams(StageParams):
    fps: float = Field(6.0, gt=0, le=60, description="Output frames per second of run time")
    trim_in_s: float | None = Field(None, description="Start, in run-timeline seconds")
    trim_out_s: float | None = Field(None, description="End, in run-timeline seconds")
    format: Literal["jpg", "png"] = "jpg"
    jpeg_qscale: int = Field(2, ge=1, le=31, description="ffmpeg -q:v, lower is better")
    max_dimension: int | None = Field(
        2560,
        description=(
            "Longest side after rotation. 2560 rather than native 4K: COLMAP's SIFT "
            "downscales above ~3200px anyway, and densifying 500 4K images exhausts "
            "system RAM long before it troubles the GPU."
        ),
    )
    rotation: Literal["auto", "none", "cw90", "ccw90", "180"] = "auto"
    color_handling: Literal["auto", "passthrough", "tonemap"] = "auto"
    detect_duplicates: bool = True

    # Cosmetic: these change previews only, never the extracted frames, so they are
    # excluded from the fingerprint and do not invalidate anything downstream.
    thumbnail_px: int = Field(256, json_schema_extra={"affects_fingerprint": False})


class ExtractStage(Stage):
    id = StageId.EXTRACT
    label = "Extract frames"
    description = "Cut the source clips into a slot-indexed, time-aligned frame set."
    params_model = ExtractParams
    depends_on: list[StageId] = []

    def external_inputs(self, manifest: RunManifest) -> dict[str, Any]:
        """Source identity, grouping and sync offsets all change the output frames."""
        return {
            "clips": [
                {
                    "clip_id": c.clip_id,
                    "camera_group": c.camera_group,
                    "segment_id": c.segment_id,
                    "identity": (
                        c.source_identity.model_dump() if c.source_identity else None
                    ),
                    "offset": round(c.time_offset_s, 6),
                    "rotation": c.probe.effective_rotation,
                    "is_hdr": c.probe.is_hdr,
                }
                for c in manifest.enabled_clips()
            ],
            "ffmpeg": _ffmpeg_version(),
        }

    def preflight(self, manifest: RunManifest) -> list[str]:
        problems: list[str] = []
        if not manifest.enabled_clips():
            problems.append("no enabled clips: add at least one source video")
        for clip in manifest.enabled_clips():
            if not Path(clip.source_path).exists():
                problems.append(f"source file is missing: {clip.source_path}")
            if not clip.probe.packets_path:
                problems.append(f"{clip.clip_id} has no packet timeline; re-add the clip")
        groups = [c.camera_group for c in manifest.enabled_clips()]
        if len(groups) != len(set(groups)):
            problems.append(
                "two enabled clips share a camera group. Each physical camera needs "
                "its own group, or COLMAP will fit one intrinsic to both."
            )
        return problems

    # -- the work ----------------------------------------------------------

    def run(self, ctx: StageContext) -> StageResult:
        params: ExtractParams = ctx.params  # type: ignore[assignment]
        manifest = ctx.manifest
        run_dir = ctx.run_dir

        pts_by_clip = _load_packet_timelines(manifest, run_dir)
        plans = _plan_segments(manifest, pts_by_clip, params)

        total_frames = sum(
            plan.slot_count * len(plan.frames) for plan in plans.values()
        )
        ctx.logger.info(
            "planned %d slots across %d segment(s) -> %d frames",
            sum(p.slot_count for p in plans.values()),
            len(plans),
            total_frames,
        )
        for plan in plans.values():
            for warning in plan.warnings:
                ctx.logger.warning(warning)

        frames_scratch = ctx.scratch / "frames"
        thumbs_scratch = ctx.scratch / "thumbs"
        records: list[dict[str, Any]] = []
        done = 0

        for plan in plans.values():
            for clip_id, slot_frames in plan.frames.items():
                clip = manifest.clip(clip_id)
                assert clip is not None
                ctx.progress(
                    f"extracting {clip.camera_group}", current=done, total=total_frames
                )

                produced = _extract_clip(ctx, clip, slot_frames, params, frames_scratch)
                clip_records = _postprocess(
                    ctx,
                    clip,
                    slot_frames,
                    produced,
                    params,
                    thumbs_scratch / clip.camera_group,
                )
                records.extend(clip_records)
                done += len(slot_frames)
                ctx.progress(
                    f"extracted {clip.camera_group}", current=done, total=total_frames
                )

        _commit(run_dir, frames_scratch, thumbs_scratch)
        _write_frame_records(run_dir, records)

        metrics, warnings = _summarise(manifest, plans, records, params)
        _update_manifest_timeline(manifest, plans, metrics)

        return StageResult(
            artifacts={
                "frames": "frames",
                "thumbs": "thumbs",
                "frames_index": "frames/frames.jsonl",
            },
            metrics=metrics,
            warnings=warnings,
            tool_versions={"ffmpeg": _ffmpeg_version()},
        )


# -- planning ----------------------------------------------------------------


def _load_packet_timelines(manifest: RunManifest, run_dir: Path) -> dict[str, list[float]]:
    timelines: dict[str, list[float]] = {}
    for clip in manifest.enabled_clips():
        if not clip.probe.packets_path:
            continue
        path = run_dir / clip.probe.packets_path
        timelines[clip.clip_id] = json.loads(path.read_text(encoding="utf-8"))
    return timelines


def _plan_segments(
    manifest: RunManifest, pts_by_clip: dict[str, list[float]], params: ExtractParams
) -> dict[str, SegmentPlan]:
    """One plan per segment, with disjoint slot ranges.

    Segments are separate takes -- the main two-camera spin, then the crown pass --
    so they share no instants. Giving each its own block of slot indices keeps
    filenames globally unique while preserving the within-segment pairing that the
    rig depends on.
    """
    plans: dict[str, SegmentPlan] = {}
    slot_base = 0

    segment_ids = [s.segment_id for s in manifest.segments] or ["seg0"]
    for segment_id in segment_ids:
        clips = manifest.enabled_clips(segment_id)
        if not clips:
            continue
        plan = compute_segment_plan(
            clips,
            pts_by_clip,
            fps=params.fps,
            slot_base=slot_base,
            trim_in_s=params.trim_in_s,
            trim_out_s=params.trim_out_s,
            segment_id=segment_id,
        )
        plans[segment_id] = plan
        # Round the next base up so slot blocks stay visually distinct in listings.
        slot_base += ((plan.slot_count // 1000) + 1) * 1000

    if not plans:
        raise RuntimeError("no enabled clips in any segment")
    return plans


# -- ffmpeg ------------------------------------------------------------------


def _resolve_rotation(clip, params: ExtractParams) -> int:
    if params.rotation == "auto":
        return clip.probe.effective_rotation
    return {"none": 0, "cw90": 90, "ccw90": 270, "180": 180}[params.rotation]


def _build_filters(
    clip, params: ExtractParams, start_pts: float, end_pts: float, src_period: float
) -> str:
    """Order matters: trim -> tonemap -> rotate -> scale -> pixel format.

    Rotating before scaling means ``max_dimension`` applies to the image as
    displayed, which is what the number is understood to mean.
    """
    filters: list[str] = []

    # Bound the work with a trim filter rather than -t. Because -copyts keeps the
    # source timestamps, -t would measure the cut from zero rather than from the
    # first frame we want, and silently truncate the tail of the range.
    # A little slack at each end: one source frame before, so the fps filter has an
    # input at or before its first output instant, and two output periods after, so
    # the final slot is never lost to rounding. Surplus frames are discarded later.
    filters.append(
        f"trim=start={max(start_pts - src_period, 0.0):.6f}"
        f":end={end_pts + 2.0 / params.fps:.6f}"
    )

    # Shift this clip's timeline so that our first wanted instant sits at zero, then
    # sample. This is the crux of the whole stage, and it is NOT what
    # `fps=...:start_time=` does: that option anchors its sampling grid at zero and
    # merely skips earlier frames, so asking for start_time=0.6 at 6 fps yields
    # frames at 0.667, 0.833, ... -- a third of a frame off the requested phase.
    # Two cameras with different offsets would each snap to the same absolute grid
    # and end up sampling different instants under the same slot number: a silent
    # misalignment that survives all the way to reconstruction. Shifting first makes
    # the grid land exactly on start_pts + n/fps in the clip's own timebase.
    filters.append(f"setpts=PTS-{start_pts:.6f}/TB")

    # Decimate before the expensive per-pixel work below, so tone-mapping and
    # scaling touch only the frames actually kept.
    filters.append(f"fps={params.fps}:round=near")

    tonemap = params.color_handling == "tonemap" or (
        params.color_handling == "auto" and clip.probe.is_hdr
    )
    if tonemap:
        filters.extend(ffmpeg.hdr_tonemap_chain())

    filters.extend(ffmpeg.rotation_filter(_resolve_rotation(clip, params)))

    if params.max_dimension:
        d = params.max_dimension
        filters.append(
            f"scale=w='min({d},iw)':h='min({d},ih)'"
            f":force_original_aspect_ratio=decrease:flags=lanczos"
        )
        filters.append("scale=trunc(iw/2)*2:trunc(ih/2)*2")

    filters.append("format=yuvj420p" if params.format == "jpg" else "format=rgb24")
    return ",".join(filters)


def _extract_clip(
    ctx: StageContext,
    clip,
    slot_frames: list,
    params: ExtractParams,
    frames_scratch: Path,
) -> list[Path]:
    """Run ffmpeg for one clip and return the produced files, in output order."""
    out_dir = frames_scratch / clip.camera_group
    out_dir.mkdir(parents=True, exist_ok=True)

    start_pts = slot_frames[0].timeline_time - clip.time_offset_s
    end_pts = slot_frames[-1].timeline_time - clip.time_offset_s
    src_period = 1.0 / (clip.probe.avg_frame_rate or 30.0)

    # Seek coarsely before -i (fast on a multi-GB file), then back off two seconds
    # so the decoder has a keyframe in hand before the first frame we actually want.
    seek = max(start_pts - 2.0, 0.0)

    argv: list[str | Path] = [
        ffmpeg.ffmpeg_path(),
        "-hide_banner", "-nostdin", "-y",
        "-loglevel", "warning",
        "-noautorotate",  # an INPUT option; ffmpeg rejects it after the output URL
    ]
    if seek > 0:
        argv += ["-ss", f"{seek:.6f}"]
    argv += [
        "-copyts",  # keep source timestamps so fps:start_time stays meaningful
        "-i", clip.source_path,
        "-map", "0:v:0",
        "-an", "-sn", "-dn",
        "-vf", _build_filters(clip, params, start_pts, end_pts, src_period),
        "-fps_mode", "passthrough",  # the fps filter already set the rate
    ]
    if params.format == "jpg":
        argv += ["-q:v", str(params.jpeg_qscale)]
    argv += ["-f", "image2", "-start_number", "0", out_dir / f"%06d.{params.format}"]

    ctx.logger.info("ffmpeg %s", " ".join(str(a) for a in argv[1:]))

    def on_line(line: str) -> None:
        if line.strip():
            ctx.logger.info("[ffmpeg] %s", line)

    code = ffmpeg_stream(argv, cwd=out_dir, on_line=on_line, ctx=ctx)
    if code != 0:
        raise RuntimeError(f"ffmpeg exited {code} while extracting {clip.clip_id}")

    produced = sorted(out_dir.glob(f"*.{params.format}"))

    # The files carry ffmpeg's own output counter, and output frame n is the instant
    # start_pts + n/fps. Mapping that order onto slot order is only sound if every
    # planned slot was actually produced. A shortfall means the assumption broke, and
    # renaming anyway would misalign the two cameras invisibly now and catastrophically
    # at the alignment stage -- so fail loudly instead of guessing.
    if len(produced) < len(slot_frames):
        raise RuntimeError(
            f"{clip.clip_id}: ffmpeg produced {len(produced)} frames but the timeline "
            f"planned {len(slot_frames)}. Refusing to guess the slot mapping, because "
            f"a wrong guess would misalign the cameras invisibly. "
            f"(fps={params.fps}, start_pts={start_pts:.4f}, end_pts={end_pts:.4f})"
        )

    # The trim window is deliberately a little generous at the tail, so a couple of
    # frames past the last slot are expected. They belong to no slot; drop them.
    for surplus in produced[len(slot_frames) :]:
        surplus.unlink()
    produced = produced[: len(slot_frames)]

    renamed: list[Path] = []
    for source_file, slot_frame in zip(produced, slot_frames):
        target = out_dir / f"{slot_frame.slot:06d}.{params.format}"
        if source_file != target:
            source_file.replace(target)
        renamed.append(target)
    return renamed


def ffmpeg_stream(argv, *, cwd: Path, on_line, ctx: StageContext) -> int:
    """Run ffmpeg, honouring cancellation."""
    from ..proc import stream

    def check(line: str) -> None:
        on_line(line)
        ctx.cancel.raise_if_cancelled()

    return stream(argv, cwd=cwd, on_line=check)


# -- per-frame analysis ------------------------------------------------------


def _dhash(image: Image.Image, size: int = 8) -> int:
    """64-bit difference hash: robust to compression, sensitive to real change."""
    small = image.convert("L").resize((size + 1, size), Image.Resampling.LANCZOS)
    pixels = np.asarray(small, dtype=np.int16)
    bits = pixels[:, 1:] > pixels[:, :-1]
    value = 0
    for bit in bits.flatten():
        value = (value << 1) | int(bit)
    return value


def _hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def _postprocess(
    ctx: StageContext,
    clip,
    slot_frames: list,
    files: list[Path],
    params: ExtractParams,
    thumb_dir: Path,
) -> list[dict[str, Any]]:
    """One decode per frame, several uses: thumbnail, dHash, exposure statistics."""
    thumb_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    previous_hash: int | None = None
    previous_slot: int | None = None

    for slot_frame, path in zip(slot_frames, files):
        ctx.cancel.raise_if_cancelled()
        with Image.open(path) as image:
            image.load()
            width, height = image.size
            grey = np.asarray(image.convert("L"), dtype=np.uint8)
            dhash = _dhash(image)
            thumb = image.convert("RGB")
            thumb.thumbnail((params.thumbnail_px, params.thumbnail_px))
            thumb.save(thumb_dir / f"{slot_frame.slot:06d}.webp", quality=80, method=4)

        duplicate_of = None
        if (
            params.detect_duplicates
            and previous_hash is not None
            and _hamming(dhash, previous_hash) <= DUPLICATE_HAMMING
        ):
            duplicate_of = previous_slot

        records.append(
            {
                "slot": slot_frame.slot,
                "clip_id": clip.clip_id,
                "camera_group": clip.camera_group,
                "segment_id": clip.segment_id,
                "file": f"frames/{clip.camera_group}/{path.name}",
                "src_pts": round(slot_frame.src_pts, 6),
                "timeline_time_s": round(slot_frame.timeline_time, 6),
                "sync_residual_s": round(slot_frame.residual, 6),
                "sync_suspect": slot_frame.suspect,
                "w": width,
                "h": height,
                "bytes": path.stat().st_size,
                "dhash": f"{dhash:016x}",
                "duplicate_of": duplicate_of,
                "mean_luma": round(float(grey.mean()) / 255.0, 4),
                "clipped_frac": round(float((grey >= 250).mean()), 4),
            }
        )
        previous_hash, previous_slot = dhash, slot_frame.slot

    return records


# -- commit and summary ------------------------------------------------------


def _commit(run_dir: Path, frames_scratch: Path, thumbs_scratch: Path) -> None:
    """Swap the finished output into place. Re-running replaces prior results."""
    for scratch, name in ((frames_scratch, "frames"), (thumbs_scratch, "thumbs")):
        target = run_dir / name
        if target.exists():
            shutil.rmtree(target, ignore_errors=True)
        if scratch.exists():
            scratch.replace(target)


def _write_frame_records(run_dir: Path, records: list[dict[str, Any]]) -> None:
    """Bulk per-frame data as a sidecar; project.json holds only summary state."""
    path = run_dir / "frames" / "frames.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for record in sorted(records, key=lambda r: (r["camera_group"], r["slot"])):
            fh.write(json.dumps(record) + "\n")


def _summarise(
    manifest: RunManifest,
    plans: dict[str, SegmentPlan],
    records: list[dict[str, Any]],
    params: ExtractParams,
) -> tuple[dict[str, Any], list[str]]:
    warnings: list[str] = []
    for plan in plans.values():
        warnings.extend(plan.warnings)

    duplicates = sum(1 for r in records if r["duplicate_of"] is not None)
    suspects = sum(1 for r in records if r["sync_suspect"])
    lumas = [r["mean_luma"] for r in records]
    luma_range = (max(lumas) - min(lumas)) if lumas else 0.0
    p95 = max((residual_p95(plan) for plan in plans.values()), default=0.0)

    degrees_per_second = _estimate_rotation_rate(manifest, plans)

    if duplicates:
        share = duplicates / max(len(records), 1)
        warnings.append(
            f"{duplicates} near-identical consecutive frames ({share:.0%}). These are "
            "degenerate zero-baseline views -- nothing to triangulate between -- and "
            "will be dropped at selection. Common causes: the phone repeated a frame, "
            "the extraction rate is higher than the scene changes, or the subject "
            "barely moved."
        )
    if luma_range > LUMA_RANGE_WARN:
        warnings.append(
            f"frame brightness varies by {luma_range * 100:.0f}% across the take, which "
            "means auto-exposure was hunting. Expect visible seams in the final "
            "texture; lock exposure before the next shoot."
        )
    if degrees_per_second and degrees_per_second > FAST_ROTATION_DEG_PER_S:
        warnings.append(
            f"camera and subject rotate relative to each other at about "
            f"{degrees_per_second:.0f} deg/s. Above ~15 deg/s, rolling-shutter skew "
            "degrades the geometry and nothing downstream can undo it. Move slower -- "
            "around 45s per revolution."
        )
    if suspects:
        warnings.append(
            f"{suspects} slots had no source frame close to the target instant; "
            "the two cameras may not be perfectly paired there"
        )

    metrics: dict[str, Any] = {
        "frame_count": len(records),
        "slot_count": sum(p.slot_count for p in plans.values()),
        "segment_count": len(plans),
        "camera_groups": sorted({r["camera_group"] for r in records}),
        "fps": params.fps,
        "duplicates": duplicates,
        "sync_suspect_slots": suspects,
        "sync_residual_p95_s": round(p95, 5),
        "mean_luma_range": round(luma_range, 4),
        "degrees_per_second": degrees_per_second,
        "bytes_on_disk": sum(r["bytes"] for r in records),
        "segments": {
            sid: {
                "slot_base": plan.slot_base,
                "slot_count": plan.slot_count,
                "t_start": round(plan.t_start, 4),
                "t_end": round(plan.t_end, 4),
            }
            for sid, plan in plans.items()
        },
    }
    return metrics, warnings


def _estimate_rotation_rate(
    manifest: RunManifest, plans: dict[str, SegmentPlan]
) -> float | None:
    """Degrees per second of relative rotation, if it can be known at all.

    Applies to both capture modes: rolling-shutter skew depends on motion *between*
    the camera and the scene, so it makes no difference which of the two was moving.

    Reported only when the revolution count was recorded. Guessing one would produce
    a confident-looking number derived from nothing, and the warning it feeds is only
    worth acting on if it is real.
    """
    revolutions = manifest.capture.revolutions
    if not revolutions or revolutions <= 0:
        return None
    total_seconds = sum(plan.t_end - plan.t_start for plan in plans.values())
    if total_seconds <= 0:
        return None
    return round(revolutions * 360.0 / total_seconds, 1)


def _update_manifest_timeline(
    manifest: RunManifest, plans: dict[str, SegmentPlan], metrics: dict[str, Any]
) -> None:
    first = next(iter(plans.values()))
    manifest.timeline = TimelineSummary(
        fps=first.fps,
        t_start=first.t_start,
        t_end=first.t_end,
        total_slots=metrics["slot_count"],
        sync_residual_p95_s=metrics["sync_residual_p95_s"],
        degrees_per_second=metrics["degrees_per_second"],
    )
    for segment_id, plan in plans.items():
        segment = manifest.segment(segment_id)
        if segment is not None:
            segment.slot_base = plan.slot_base
            segment.slot_count = plan.slot_count


def _ffmpeg_version() -> str:
    try:
        return ffmpeg.version()
    except RuntimeError:
        return "missing"
