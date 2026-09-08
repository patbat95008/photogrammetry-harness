"""Recovering up and scale from the solved cameras.

Pure geometry against synthetic rings, so the assertions are exact rather than
approximate. The reason this is worth testing hard is that both quantities fail
silently: a model exported upside down still opens, and a model exported at the wrong
scale still prints -- just at the wrong size, discovered after the print.
"""

from __future__ import annotations

import numpy as np
import pytest

from pgh.orient import (
    PLANAR_RATIO_WARN,
    baseline_scale,
    plane_fit,
    rig_pairs,
    up_axis,
)


def ring(
    count: int = 24,
    *,
    radius: float = 3.0,
    normal: tuple[float, float, float] = (0.0, 0.0, 1.0),
    centre: tuple[float, float, float] = (0.0, 0.0, 0.0),
    heights: tuple[float, ...] = (0.0,),
) -> list[dict]:
    """Camera centres on one or more circles in the plane perpendicular to ``normal``."""
    axis = np.asarray(normal, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    # Any two vectors perpendicular to the axis span the plane.
    seed = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(axis, seed)
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)

    poses = []
    slot = 0
    for height in heights:
        for i in range(count):
            angle = 2 * np.pi * i / count
            position = (
                np.asarray(centre)
                + radius * (np.cos(angle) * u + np.sin(angle) * v)
                + height * axis
            )
            poses.append({"name": f"cam/{slot:06d}.jpg", "center": position.tolist()})
            slot += 1
    return poses


# -- the plane fit -----------------------------------------------------------


@pytest.mark.parametrize(
    "normal",
    [(0.0, 0.0, 1.0), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (0.3, -0.5, 0.81)],
)
def test_the_axis_of_an_orbit_is_recovered(normal) -> None:
    found, _ = plane_fit(np.array([p["center"] for p in ring(normal=normal)]))
    expected = np.asarray(normal) / np.linalg.norm(normal)

    # Up to sign: the fit cannot know which way up, and flip_x answers that.
    assert abs(float(np.dot(found, expected))) == pytest.approx(1.0, abs=1e-9)


def test_a_flat_orbit_reads_as_planar() -> None:
    _, planarity = plane_fit(np.array([p["center"] for p in ring()]))
    assert planarity < PLANAR_RATIO_WARN


def test_a_two_height_orbit_still_finds_the_axis_but_fits_it_loosely() -> None:
    """The capture checklist encourages shooting more than one height, so this happens.

    The axis stays right, which is what matters; the point of reporting planarity is to
    say the fit was loose rather than presenting it as exact.
    """
    poses = ring(normal=(0.0, 0.0, 1.0), heights=(-1.2, 1.2))
    found, planarity = plane_fit(np.array([p["center"] for p in poses]))

    assert abs(float(found[2])) == pytest.approx(1.0, abs=1e-6)
    assert planarity > PLANAR_RATIO_WARN


def test_the_sign_is_stable_rather_than_left_to_the_decomposition() -> None:
    """Otherwise the same run could export upside down depending on the machine."""
    poses = ring(normal=(0.0, 0.0, 1.0))
    first, _ = plane_fit(np.array([p["center"] for p in poses]))
    reversed_order, _ = plane_fit(np.array([p["center"] for p in reversed(poses)]))

    assert np.allclose(first, reversed_order)


def test_too_few_cameras_is_refused_with_a_reason() -> None:
    with pytest.raises(ValueError, match="at least three"):
        plane_fit(np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]))


# -- the flip ----------------------------------------------------------------


def test_the_flip_reverses_the_axis_and_nothing_else() -> None:
    poses = ring(normal=(0.2, 0.3, 0.93))
    upright, planarity_a = up_axis(poses, flip=False)
    flipped, planarity_b = up_axis(poses, flip=True)

    assert np.allclose(upright, -flipped)
    assert planarity_a == planarity_b


# -- scale -------------------------------------------------------------------


def rig_poses(separation: float, *, count: int = 12) -> list[dict]:
    """Two camera groups on fixed mounts, so matching slots are one baseline apart."""
    poses = []
    for i in range(count):
        angle = 2 * np.pi * i / count
        base = np.array([3.0 * np.cos(angle), 3.0 * np.sin(angle), 0.0])
        poses.append({"name": f"cam_eye/{i:06d}.jpg", "center": base.tolist()})
        poses.append(
            {
                "name": f"cam_high/{i:06d}.jpg",
                "center": (base + np.array([0.0, 0.0, separation])).tolist(),
            }
        )
    return poses


def test_paired_slots_are_found_across_camera_groups() -> None:
    assert len(rig_pairs(rig_poses(0.5, count=12))) == 12


def test_a_single_camera_has_no_pairs() -> None:
    assert rig_pairs(ring()) == []


def test_the_measured_baseline_becomes_millimetres_per_unit() -> None:
    """0.5 model units between the lenses, measured at 120 mm, is 240 mm per unit."""
    assert baseline_scale(rig_poses(0.5), 120.0) == pytest.approx(240.0)


def test_one_badly_solved_pair_does_not_rescale_the_export() -> None:
    """The median is why. A mean would let a single outlier move the whole model."""
    poses = rig_poses(0.5, count=12)
    poses[1]["center"] = [50.0, 50.0, 50.0]  # one camera flung across the scene

    assert baseline_scale(poses, 120.0) == pytest.approx(240.0)


def test_no_scale_without_a_rig() -> None:
    """A single orbiting camera has no measurable pair, and a guess would be worse."""
    assert baseline_scale(ring(), 120.0) is None


def test_no_scale_without_a_measurement() -> None:
    assert baseline_scale(rig_poses(0.5), None) is None
    assert baseline_scale(rig_poses(0.5), 0.0) is None


def test_coincident_cameras_yield_no_scale_rather_than_infinity() -> None:
    assert baseline_scale(rig_poses(0.0), 120.0) is None
