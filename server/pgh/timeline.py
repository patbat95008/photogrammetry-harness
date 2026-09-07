"""The shared run timeline: which source frame becomes which slot.

This is the piece that makes two simultaneously-recorded clips usable as a rigid
camera rig. Both cameras are static in the room while the subject rotates, so in
the subject's reference frame -- the one COLMAP reconstructs -- the two cameras
orbit the head in lockstep at a constant relative pose. COLMAP can exploit that,
but only if it can tell which two images were taken at the same instant, and it
does that by **matching filenames across per-camera folders**.

So slot index is a shared clock, not a per-clip frame counter:
``frames/cam_high/000042.jpg`` and ``frames/cam_eye/000042.jpg`` must be the same
moment. Everything here exists to make that true.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass, field

from .manifest import Clip

#: A slot whose nearest source frame is further away than this many source-frame
#: periods is flagged: it usually means the phone dropped frames there.
SUSPECT_RESIDUAL_PERIODS = 0.75


@dataclass(slots=True)
class SlotFrame:
    slot: int
    clip_id: str
    camera_group: str
    #: Timestamp of the chosen frame in the clip's own timebase.
    src_pts: float
    #: The instant this slot represents on the shared run timeline.
    timeline_time: float
    #: How far the chosen frame is from the ideal instant, in seconds.
    residual: float
    suspect: bool = False


@dataclass(slots=True)
class SegmentPlan:
    segment_id: str
    fps: float
    t_start: float
    t_end: float
    slot_base: int
    slot_count: int
    #: clip_id -> one SlotFrame per slot, in slot order.
    frames: dict[str, list[SlotFrame]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def slots(self) -> range:
        return range(self.slot_base, self.slot_base + self.slot_count)


def _median_period(pts: list[float]) -> float:
    """Mean inter-frame gap across a clip's packet timeline."""
    if len(pts) < 2:
        return 0.0
    return (pts[-1] - pts[0]) / (len(pts) - 1)


def _nearest(sorted_values: list[float], target: float) -> float:
    """Closest value in a sorted list."""
    index = bisect.bisect_left(sorted_values, target)
    if index == 0:
        return sorted_values[0]
    if index >= len(sorted_values):
        return sorted_values[-1]
    before, after = sorted_values[index - 1], sorted_values[index]
    return after if (after - target) < (target - before) else before


def compute_segment_plan(
    clips: list[Clip],
    pts_by_clip: dict[str, list[float]],
    *,
    fps: float,
    slot_base: int = 0,
    trim_in_s: float | None = None,
    trim_out_s: float | None = None,
    segment_id: str = "seg0",
) -> SegmentPlan:
    """Lay out one segment's slots and pick the source frame for each.

    The window is the **intersection** of the clips' coverage: the latest start and
    the earliest end. A slot that only some cameras can see would break the rig
    pairing, so slots that are not covered by every enabled clip are never created.
    """
    if not clips:
        raise ValueError("a segment needs at least one enabled clip")
    if fps <= 0:
        raise ValueError("fps must be positive")

    missing = [c.clip_id for c in clips if not pts_by_clip.get(c.clip_id)]
    if missing:
        raise ValueError(f"no packet timeline available for: {', '.join(missing)}")

    warnings: list[str] = []

    # Convert each clip's own timestamps onto the shared run timeline.
    starts, ends = [], []
    for clip in clips:
        pts = pts_by_clip[clip.clip_id]
        period = _median_period(pts)
        starts.append(pts[0] + clip.time_offset_s)
        # Hold the end back so the last slot is actually producible. ffmpeg's fps
        # filter, sampling with round=near, cannot commit to an output frame until it
        # has seen an input frame past the halfway point to the *next* output
        # instant; at end-of-stream it discards that pending frame instead. So a slot
        # needs one source frame plus half an output period of coverage beyond it.
        # Planning one fewer slot is much better than planning one that cannot be
        # produced, which makes extraction refuse to map slots at all.
        ends.append(pts[-1] + clip.time_offset_s - period - 0.5 / fps)

    t_start = max(starts)
    t_end = min(ends)

    if trim_in_s is not None:
        t_start = max(t_start, trim_in_s)
    if trim_out_s is not None:
        t_end = min(t_end, trim_out_s)

    if t_end <= t_start:
        raise ValueError(
            "the enabled clips do not overlap in time. Check the sync offsets: "
            f"usable window computed as {t_start:.3f}s to {t_end:.3f}s"
        )

    overlap = t_end - t_start
    span = max(ends) - min(starts)
    if span > 0 and overlap < span * 0.5:
        warnings.append(
            f"clips overlap for only {overlap:.1f}s of a {span:.1f}s span; "
            "most of the footage is unusable because not every camera covers it"
        )

    slot_count = int(overlap * fps) + 1
    plan = SegmentPlan(
        segment_id=segment_id,
        fps=fps,
        t_start=t_start,
        t_end=t_end,
        slot_base=slot_base,
        slot_count=slot_count,
        warnings=warnings,
    )

    for clip in clips:
        pts = pts_by_clip[clip.clip_id]
        # Inter-frame gap, used to judge whether a residual is suspicious.
        period = _median_period(pts) or (1.0 / fps)
        tolerance = period * SUSPECT_RESIDUAL_PERIODS

        chosen: list[SlotFrame] = []
        suspect_count = 0
        for i in range(slot_count):
            timeline_time = t_start + i / fps
            target_pts = timeline_time - clip.time_offset_s
            src_pts = _nearest(pts, target_pts)
            residual = src_pts - target_pts
            suspect = abs(residual) > tolerance
            suspect_count += suspect
            chosen.append(
                SlotFrame(
                    slot=slot_base + i,
                    clip_id=clip.clip_id,
                    camera_group=clip.camera_group,
                    src_pts=src_pts,
                    timeline_time=timeline_time,
                    residual=residual,
                    suspect=suspect,
                )
            )

        plan.frames[clip.clip_id] = chosen
        if suspect_count:
            warnings.append(
                f"{clip.clip_id}: {suspect_count} of {slot_count} slots have no source "
                "frame close to the target instant, which usually means dropped frames"
            )

    return plan


def residual_p95(plan: SegmentPlan) -> float:
    """95th-percentile absolute sync residual across every clip in the segment."""
    residuals = sorted(
        abs(frame.residual) for frames in plan.frames.values() for frame in frames
    )
    if not residuals:
        return 0.0
    index = min(int(len(residuals) * 0.95), len(residuals) - 1)
    return residuals[index]


def source_pts_list(plan: SegmentPlan, clip_id: str) -> list[float]:
    return [frame.src_pts for frame in plan.frames[clip_id]]
