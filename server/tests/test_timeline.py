"""The shared run timeline.

The property under test throughout: for any slot k, every enabled clip in a
segment must point at the same instant in the real world. That is the entire basis
on which the two cameras can later be declared a rigid rig, and it is decided here
rather than at the alignment stage where a mistake would be unrecoverable.
"""

from __future__ import annotations

import pytest

from pgh.manifest import Clip, ClipProbe
from pgh.timeline import compute_segment_plan, residual_p95


def make_clip(clip_id: str, offset: float = 0.0, group: str | None = None) -> Clip:
    return Clip(
        clip_id=clip_id,
        camera_group=group or clip_id,
        source_path=f"D:/footage/{clip_id}.mp4",
        time_offset_s=offset,
        probe=ClipProbe(avg_frame_rate=60.0),
    )


def pts_series(count: int, fps: float = 60.0, start: float = 0.0) -> list[float]:
    return [start + i / fps for i in range(count)]


# -- the core property -------------------------------------------------------


def test_paired_slots_refer_to_the_same_world_instant():
    """cam_high/000042 and cam_eye/000042 must be the same moment.

    cam_eye started 0.62 s before cam_high, so the same world event sits 0.62 s
    later in cam_eye's own timebase and its offset is -0.62.
    """
    high = make_clip("cam_high", offset=0.0)
    eye = make_clip("cam_eye", offset=-0.62)
    pts = {"cam_high": pts_series(600), "cam_eye": pts_series(600)}

    plan = compute_segment_plan([high, eye], pts, fps=6.0)

    for a, b in zip(plan.frames["cam_high"], plan.frames["cam_eye"]):
        assert a.slot == b.slot
        assert a.timeline_time == pytest.approx(b.timeline_time)
        # Their source timestamps must differ by exactly the offset difference.
        assert (a.src_pts - b.src_pts) == pytest.approx(-0.62, abs=1 / 60)


def test_window_is_the_intersection_not_the_union():
    """A slot only some cameras cover would break the pairing, so it is not created."""
    a = make_clip("a", offset=0.0)
    b = make_clip("b", offset=-2.0)  # b's footage lands 2s earlier on the run clock
    pts = {"a": pts_series(600), "b": pts_series(600)}

    plan = compute_segment_plan([a, b], pts, fps=10.0)

    # a covers run time 0..9.983; b covers -2..7.983. The intersection is 0..7.983,
    # less the one-source-frame end guard (see the tail-coverage test below).
    assert plan.t_start == pytest.approx(0.0)
    assert plan.t_end == pytest.approx(7.983 - 1 / 60 - 0.5 / 10.0, abs=0.01)
    for frames in plan.frames.values():
        assert len(frames) == plan.slot_count


def test_last_slot_has_source_coverage_beyond_it():
    """No slot may sit inside the final frame of any clip.

    ffmpeg's fps filter will not reliably emit an output frame at end-of-stream when
    there is no input frame past the requested instant. Planning such a slot means
    ffmpeg returns one frame fewer than planned, and extraction refuses to map slots
    at all -- so the planner holds the window back by one source frame instead.
    """
    clip = make_clip("solo")
    pts = pts_series(600, fps=60)  # last frame at 9.9833
    plan = compute_segment_plan([clip], {"solo": pts}, fps=6.0)

    last = plan.frames["solo"][-1]
    assert last.src_pts < pts[-1], "the final slot must not consume the final frame"
    assert plan.t_end <= pts[-1] - 1 / 60 - 0.5 / 6.0 + 1e-9


def test_slot_times_are_evenly_spaced_at_the_requested_fps():
    clip = make_clip("solo")
    plan = compute_segment_plan([clip], {"solo": pts_series(600)}, fps=6.0)

    times = [f.timeline_time for f in plan.frames["solo"]]
    gaps = [b - a for a, b in zip(times, times[1:])]
    assert all(gap == pytest.approx(1 / 6) for gap in gaps)


def test_slot_base_offsets_a_second_segment():
    """Separate passes get disjoint slot blocks so filenames stay globally unique."""
    clip = make_clip("crown")
    plan = compute_segment_plan(
        [clip], {"crown": pts_series(300)}, fps=6.0, slot_base=1000, segment_id="seg1"
    )
    assert plan.frames["crown"][0].slot == 1000
    assert plan.slots.start == 1000


# -- residuals and dropped frames --------------------------------------------


def test_residuals_are_within_half_a_source_frame():
    clip = make_clip("solo")
    plan = compute_segment_plan([clip], {"solo": pts_series(600, fps=60)}, fps=7.0)

    for frame in plan.frames["solo"]:
        assert abs(frame.residual) <= (0.5 / 60) + 1e-9
        assert not frame.suspect


def test_dropped_source_frames_are_flagged():
    """A phone throttling under load drops frames; those slots cannot be trusted."""
    clip = make_clip("solo")
    pts = pts_series(600)
    del pts[200:260]  # a one-second hole

    plan = compute_segment_plan([clip], {"solo": pts}, fps=10.0)

    assert any(f.suspect for f in plan.frames["solo"])
    assert plan.warnings, "a gap this size should produce a warning"


def test_residual_p95_reports_worst_case_not_average():
    clip = make_clip("solo")
    plan = compute_segment_plan([clip], {"solo": pts_series(600)}, fps=6.0)
    assert 0.0 <= residual_p95(plan) <= 0.5 / 60 + 1e-9


# -- trimming ----------------------------------------------------------------


def test_trim_narrows_the_window():
    clip = make_clip("solo")
    plan = compute_segment_plan(
        [clip], {"solo": pts_series(600)}, fps=10.0, trim_in_s=2.0, trim_out_s=5.0
    )
    assert plan.t_start == pytest.approx(2.0)
    assert plan.t_end == pytest.approx(5.0)
    assert plan.slot_count == 31  # 2.0s to 5.0s inclusive at 10 fps
    assert plan.frames["solo"][0].timeline_time == pytest.approx(2.0)


# -- failure modes -----------------------------------------------------------


def test_non_overlapping_clips_fail_with_an_explanation():
    a = make_clip("a", offset=0.0)
    b = make_clip("b", offset=-100.0)
    pts = {"a": pts_series(600), "b": pts_series(600)}

    with pytest.raises(ValueError, match="do not overlap"):
        compute_segment_plan([a, b], pts, fps=6.0)


def test_missing_packet_timeline_is_reported_by_name():
    a = make_clip("a")
    b = make_clip("b")
    with pytest.raises(ValueError, match="b"):
        compute_segment_plan([a, b], {"a": pts_series(100)}, fps=6.0)


def test_no_clips_is_an_error():
    with pytest.raises(ValueError, match="at least one"):
        compute_segment_plan([], {}, fps=6.0)


def test_zero_fps_is_rejected():
    clip = make_clip("solo")
    with pytest.raises(ValueError, match="fps must be positive"):
        compute_segment_plan([clip], {"solo": pts_series(100)}, fps=0)


def test_poor_overlap_produces_a_warning():
    """Mostly-disjoint clips still work, but the user should be told why so little
    footage survived."""
    a = make_clip("a", offset=0.0)
    b = make_clip("b", offset=-8.0)
    pts = {"a": pts_series(600), "b": pts_series(600)}

    plan = compute_segment_plan([a, b], pts, fps=6.0)

    assert any("overlap" in w for w in plan.warnings)
