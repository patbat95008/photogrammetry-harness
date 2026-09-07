"""Frame selection keeps the frames worth reconstructing, and says why it dropped the rest.

The load-bearing properties:

* a genuinely blurred frame scores below a sharp one, measured where the subject is
  rather than over the whole image;
* thinning for coverage keeps both ends of the take, because dropping either shortens
  the angular arc and that is what stops a loop closing;
* a manual override beats every automatic rule, since the operator can see things
  none of these measures can.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import pytest

from pgh.stages.select import SelectParams, _crop, _decide, _summarise, _thin


def frame(
    slot: int,
    *,
    sharpness: float = 500.0,
    group: str = "cam",
    duplicate_of: int | None = None,
) -> dict[str, Any]:
    return {
        "slot": slot,
        "camera_group": group,
        "file": f"frames/{group}/{slot:06d}.jpg",
        "sharpness": sharpness,
        "duplicate_of": duplicate_of,
    }


def selected_slots(decisions: list[dict[str, Any]]) -> list[int]:
    return sorted(d["slot"] for d in decisions if d["selected"])


def reason_for(decisions: list[dict[str, Any]], slot: int) -> str:
    return next(d["reason"] for d in decisions if d["slot"] == slot)


# -- sharpness ---------------------------------------------------------------


def test_blur_lowers_the_laplacian_variance() -> None:
    """The measure itself: a blurred image must score below its sharp original."""
    rng = np.random.RandomState(0)
    sharp = (rng.rand(200, 200) * 255).astype(np.uint8)
    blurred = cv2.GaussianBlur(sharp, (0, 0), 3.0)

    sharp_score = cv2.Laplacian(sharp, cv2.CV_64F).var()
    blurred_score = cv2.Laplacian(blurred, cv2.CV_64F).var()

    assert blurred_score < sharp_score / 5, "blur must be clearly separable from detail"


def test_centre_crop_ignores_the_edges_of_the_frame() -> None:
    """A sharp background around a blurred subject must not read as a sharp frame."""
    image = np.zeros((200, 200), dtype=np.uint8)
    rng = np.random.RandomState(1)
    image[:, :] = 128
    image[0:40, :] = (rng.rand(40, 200) * 255).astype(np.uint8)  # busy edge only

    whole = cv2.Laplacian(_crop(image, "full"), cv2.CV_64F).var()
    centre = cv2.Laplacian(_crop(image, "center"), cv2.CV_64F).var()

    assert centre < whole / 10, "detail outside the centre must not count as sharpness"


def test_full_region_returns_the_whole_image() -> None:
    image = np.zeros((40, 60), dtype=np.uint8)
    assert _crop(image, "full").shape == (40, 60)


def test_centre_crop_is_smaller_but_not_empty() -> None:
    image = np.zeros((40, 60), dtype=np.uint8)
    crop = _crop(image, "center")
    assert crop.size > 0
    assert crop.shape[0] < 40 and crop.shape[1] < 60


# -- decisions ---------------------------------------------------------------


def test_duplicates_are_dropped_with_their_reason() -> None:
    frames = [frame(0), frame(1, duplicate_of=0), frame(2)]
    decisions = _decide(frames, SelectParams(target_count=None))

    assert selected_slots(decisions) == [0, 2]
    assert reason_for(decisions, 1) == "duplicate"


def test_duplicates_can_be_kept_when_asked() -> None:
    frames = [frame(0), frame(1, duplicate_of=0)]
    decisions = _decide(frames, SelectParams(drop_duplicates=False, target_count=None))
    assert selected_slots(decisions) == [0, 1]


def test_the_softest_frames_fall_below_the_floor() -> None:
    frames = [frame(i, sharpness=float(100 * (i + 1))) for i in range(10)]
    decisions = _decide(
        frames, SelectParams(sharpness_floor_percentile=30.0, target_count=None)
    )

    assert reason_for(decisions, 0) == "blurry"
    assert reason_for(decisions, 9) == ""
    assert len(selected_slots(decisions)) == 7


def test_a_zero_floor_rejects_nothing_for_blur() -> None:
    frames = [frame(i, sharpness=float(i + 1)) for i in range(10)]
    decisions = _decide(
        frames, SelectParams(sharpness_floor_percentile=0.0, target_count=None)
    )
    assert not any(d["reason"] == "blurry" for d in decisions)


def test_unreadable_frames_are_rejected_and_named() -> None:
    frames = [frame(0), {**frame(1), "unreadable": True, "sharpness": 0.0}]
    decisions = _decide(frames, SelectParams(target_count=None))
    assert reason_for(decisions, 1) == "unreadable"


def test_each_camera_gets_its_own_sharpness_floor() -> None:
    """One soft camera must not drag the whole of a sharper one below the threshold."""
    frames = [frame(i, sharpness=1000.0, group="high") for i in range(10)]
    frames += [frame(i, sharpness=50.0, group="eye") for i in range(10)]

    decisions = _decide(
        frames, SelectParams(sharpness_floor_percentile=30.0, target_count=None)
    )
    kept_eye = [d for d in decisions if d["camera_group"] == "eye" and d["selected"]]
    assert kept_eye, "the softer camera must still contribute frames"


# -- coverage thinning -------------------------------------------------------


def test_thinning_keeps_both_ends_of_the_take() -> None:
    """Dropping either end shortens the arc, which is what breaks loop closure."""
    survivors = [frame(i) for i in range(100)]
    kept = _thin(survivors, SelectParams(target_count=10))

    slots = [f["slot"] for f in kept]
    assert slots[0] == 0, "the first frame must survive thinning"
    assert slots[-1] == 99, "the last frame must survive thinning"
    assert len(kept) <= 10


def test_thinning_spreads_evenly_rather_than_taking_a_prefix() -> None:
    survivors = [frame(i) for i in range(100)]
    slots = [f["slot"] for f in _thin(survivors, SelectParams(target_count=11))]

    gaps = [b - a for a, b in zip(slots, slots[1:])]
    assert max(gaps) - min(gaps) <= 1, f"spacing should be near-uniform, got {gaps}"


def test_thinning_leaves_a_short_take_alone() -> None:
    survivors = [frame(i) for i in range(5)]
    assert len(_thin(survivors, SelectParams(target_count=200))) == 5


def test_coverage_off_keeps_everything() -> None:
    survivors = [frame(i) for i in range(50)]
    assert len(_thin(survivors, SelectParams(coverage="off", target_count=10))) == 50


def test_no_target_keeps_everything() -> None:
    survivors = [frame(i) for i in range(50)]
    assert len(_thin(survivors, SelectParams(target_count=None))) == 50


def test_thinned_frames_say_so() -> None:
    frames = [frame(i) for i in range(50)]
    decisions = _decide(frames, SelectParams(target_count=10, sharpness_floor_percentile=0.0))
    assert any(d["reason"] == "coverage" for d in decisions if not d["selected"])


# -- overrides ---------------------------------------------------------------


def test_a_manual_reject_beats_an_automatic_keep() -> None:
    frames = [frame(0), frame(1)]
    decisions = _decide(
        frames,
        SelectParams(target_count=None, overrides={"cam/1": "reject"}),
    )

    assert selected_slots(decisions) == [0]
    assert reason_for(decisions, 1) == "manual"
    assert next(d for d in decisions if d["slot"] == 1)["overridden"]


def test_a_manual_keep_beats_every_automatic_rejection() -> None:
    """Including a duplicate, which the operator may know is not really one."""
    frames = [frame(0), frame(1, duplicate_of=0, sharpness=1.0)]
    decisions = _decide(
        frames,
        SelectParams(
            sharpness_floor_percentile=90.0,
            overrides={"cam/1": "keep"},
            target_count=None,
        ),
    )

    assert 1 in selected_slots(decisions)
    assert next(d for d in decisions if d["slot"] == 1)["overridden"]


def test_an_override_for_a_frame_that_no_longer_exists_is_ignored() -> None:
    """Re-extracting at a different rate leaves stale keys; they must not crash."""
    decisions = _decide(
        [frame(0)], SelectParams(target_count=None, overrides={"cam/999": "keep"})
    )
    assert selected_slots(decisions) == [0]


# -- summary -----------------------------------------------------------------


def test_summary_counts_rejections_by_reason() -> None:
    frames = [frame(0), frame(1, duplicate_of=0), frame(2, sharpness=1.0)]
    decisions = _decide(frames, SelectParams(sharpness_floor_percentile=40.0, target_count=None))
    metrics, _ = _summarise(decisions, SelectParams())

    assert metrics["input_frames"] == 3
    assert metrics["rejected_by_reason"]["duplicate"] == 1
    assert metrics["selected"] + metrics["rejected"] == 3


def test_a_thin_selection_is_warned_about() -> None:
    frames = [frame(i) for i in range(12)]
    decisions = _decide(frames, SelectParams(target_count=None, sharpness_floor_percentile=0.0))
    _, warnings = _summarise(decisions, SelectParams())
    assert any("close an orbit" in w for w in warnings)


def test_a_uniformly_soft_take_is_called_out() -> None:
    """The floor is discarding frames no worse than the ones it keeps."""
    frames = [frame(i, sharpness=100.0 + i * 0.1) for i in range(80)]
    decisions = _decide(frames, SelectParams(target_count=None, sharpness_floor_percentile=10.0))
    _, warnings = _summarise(decisions, SelectParams())
    assert any("uniformly" in w for w in warnings)


def test_a_healthy_take_produces_no_warnings() -> None:
    rng = np.random.RandomState(2)
    frames = [frame(i, sharpness=float(rng.uniform(100, 900))) for i in range(150)]
    decisions = _decide(frames, SelectParams(target_count=None, sharpness_floor_percentile=10.0))
    _, warnings = _summarise(decisions, SelectParams())
    assert warnings == [], warnings


def test_lopsided_cameras_are_reported() -> None:
    frames = [frame(i, group="high") for i in range(80)]
    frames += [frame(i, group="eye") for i in range(10)]
    decisions = _decide(frames, SelectParams(target_count=None, sharpness_floor_percentile=0.0))
    _, warnings = _summarise(decisions, SelectParams())
    assert any("unevenly" in w for w in warnings)


@pytest.mark.parametrize("percentile", [0.0, 25.0, 50.0, 90.0])
def test_percentiles_are_recorded_for_every_frame(percentile: float) -> None:
    frames = [frame(i, sharpness=float(i * 10 + 1)) for i in range(20)]
    decisions = _decide(
        frames, SelectParams(sharpness_floor_percentile=percentile, target_count=None)
    )
    assert all("sharpness_percentile" in d for d in decisions)
    assert all(0.0 <= d["sharpness_percentile"] <= 100.0 for d in decisions)
