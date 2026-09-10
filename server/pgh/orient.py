"""Recovering which way is up, and how big things are, from the solved cameras.

COLMAP's world frame is arbitrary: the cup comes out lying on its side, and nothing in
the reconstruction says otherwise. But two facts are recoverable from the poses the
alignment already wrote, and both are needed before a mesh can be exported as something
you could print.

**Up.** The camera centres of an orbit lie in a plane, and that plane's normal is the up
axis -- for both capture modes, because in either one the cameras go round the subject
rather than over it. Fitting the plane is a singular value decomposition of the centred
positions: the direction of least variance is the normal.

**Which way up.** The axis is recoverable; its *sign* is not. A plane normal is equally
valid negated, and no amount of looking at the cameras decides it. That half is answered
by ``capture.flip_x``, the toggle in the cloud viewer -- a person looked at the model and
said which way up it goes. So the maths here produces a sign-stable axis and the caller
applies the person's answer to it.

**Size.** Photogrammetry recovers shape but never scale. For a fixed two-camera rig the
tape-measured ``capture.baseline_mm`` between the lenses is the one measurement that
cannot be reconstructed afterwards, and paired slots across camera groups are the same
instant, so the distance between those two camera centres in model units *is* that
baseline. For a single orbiting camera there is no such pair and nothing here can help;
the export stage asks for a measured dimension instead rather than guessing.

**What this module does not do is build the rotation.** It hands out an axis, and the
rotation onto +Z is built inside Blender, because that is the only place the frame is
known: importers apply their own axis conventions -- glTF is Y-up and Blender's importer
bakes a 90 degree turn into the object transform to fix it -- so a matrix composed here
would be applied on top of a frame it did not account for. That is not hypothetical; it
exported the cup standing on its edge until it was found.

Everything is a pure function over numpy arrays so it can be tested against synthetic
rings without Blender, COLMAP, or a run directory -- the same shape as ``timeline.py``
and ``sync.py``.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

import numpy as np

#: Below this ratio of least to greatest variance, the camera centres are convincingly
#: planar. An orbit shot at two heights, which the capture checklist actively
#: encourages, sits above it -- the axis is still the right one, but it is worth saying
#: the fit was loose rather than reporting it as though it were exact.
PLANAR_RATIO_WARN = 0.15

#: Below this ratio of middle to greatest variance the camera centres span a line rather
#: than a plane, so there is no plane to fit and no up axis in them.
COLLINEAR_RATIO = 1e-3


def _centres(poses: list[dict]) -> np.ndarray:
    return np.asarray([p["center"] for p in poses], dtype=np.float64)


def camera_motion(poses: list[dict], scene_centre: Sequence[float] | None) -> dict[str, float]:
    """How far each camera group travelled, relative to its distance from the scene.

    One number per camera group: the mean distance of its centres from their own centroid,
    divided by their mean distance from the scene. Dividing by the *standoff* rather than by
    the subject's size is what makes it comparable between captures -- a camera taken once
    around its subject scores about 1.0 whether the subject is a coffee cup or a person,
    while a camera that never moved scores near zero however many frames it recorded and
    however confidently they registered.

    Per *group* rather than over all cameras, because a two-camera rig separated by its
    baseline looks like motion if you pool them: the pair spans the baseline while neither
    camera has moved at all. Each camera has to have gone somewhere on its own.

    Returns an empty mapping when there is nothing to measure against -- no points, or
    cameras sitting on top of the scene centroid -- rather than inventing a ratio.
    """
    if scene_centre is None:
        return {}
    centre = np.asarray(scene_centre, dtype=np.float64)

    grouped: dict[str, list[np.ndarray]] = defaultdict(list)
    for pose in poses:
        name = str(pose["name"])
        group = name.partition("/")[0] if "/" in name else ""
        grouped[group].append(np.asarray(pose["center"], dtype=np.float64))

    motion: dict[str, float] = {}
    for group, centres in grouped.items():
        if len(centres) < 2:
            continue
        stack = np.asarray(centres)
        standoff = float(np.linalg.norm(stack - centre, axis=1).mean())
        if standoff <= 0:
            continue
        spread = float(np.linalg.norm(stack - stack.mean(axis=0), axis=1).mean())
        motion[group] = spread / standoff
    return motion


def plane_fit(centres: np.ndarray) -> tuple[np.ndarray, float]:
    """The best-fit plane through a set of points: its unit normal, and how flat it is.

    Returns ``(normal, planarity)`` where planarity is the ratio of the smallest
    singular value to the largest -- 0 for points exactly in a plane, rising as they
    stop being. A single orbit gives something near 0.02; an orbit shot at two heights
    gives a larger number and a normal that is still correct.

    The sign of the normal is made deterministic rather than left to the decomposition,
    which is free to return either: without this the same run could export the right way
    up on one machine and upside down on another. The convention is arbitrary -- largest
    component positive -- and the caller decides the true sign from ``flip_x``.
    """
    if len(centres) < 3:
        raise ValueError(
            f"a plane needs at least three camera centres, and this model has "
            f"{len(centres)}. Without them there is no up axis to recover."
        )

    _, singular, vectors = np.linalg.svd(centres - centres.mean(axis=0), full_matrices=False)
    normal = vectors[-1]
    if normal[int(np.argmax(np.abs(normal)))] < 0:
        normal = -normal
    normal = normal / np.linalg.norm(normal)

    if singular[0] <= 0:
        return normal, 1.0

    # Least-over-greatest is 0 for a perfect plane and *also* 0 for a straight line, which
    # is a real capture and not a hypothetical: a subject-rotates run with masking skipped
    # collapses its cameras onto two points, and this returned 0.000003 -- flatter-looking
    # than a genuine orbit's 0.043, from a plane fitted through a line. Two spanning
    # directions are what makes a plane, so the middle singular value decides whether there
    # was a plane to fit at all, and a degenerate fit reports the worst score rather than
    # the best one.
    if float(singular[1] / singular[0]) < COLLINEAR_RATIO:
        return normal, 1.0

    return normal, float(singular[-1] / singular[0])


def up_axis(poses: list[dict], *, flip: bool = False) -> tuple[np.ndarray, float]:
    """The up axis of a reconstruction, and the planarity of the fit that found it.

    ``flip`` negates it, and is where ``capture.flip_x`` is applied: the axis comes from
    the geometry, the direction it points comes from the person who looked at the model.
    """
    normal, planarity = plane_fit(_centres(poses))
    return (-normal if flip else normal), planarity


def rig_pairs(poses: list[dict]) -> list[tuple[np.ndarray, np.ndarray]]:
    """Camera centres that were recorded at the same instant by different cameras.

    Slot naming is the shared clock (HANDOVER 4.1): ``cam_high/000042.jpg`` and
    ``cam_eye/000042.jpg`` are the same moment, so their two centres are separated by
    the rig's baseline whatever the rig was pointing at.
    """
    by_slot: dict[str, list[tuple[str, np.ndarray]]] = defaultdict(list)
    for pose in poses:
        name = str(pose["name"])
        if "/" not in name:
            continue
        group, _, filename = name.partition("/")
        slot = filename.rsplit(".", 1)[0]
        by_slot[slot].append((group, np.asarray(pose["center"], dtype=np.float64)))

    pairs: list[tuple[np.ndarray, np.ndarray]] = []
    for entries in by_slot.values():
        groups = {group for group, _ in entries}
        if len(groups) < 2:
            continue
        # Two cameras is the rig this harness describes; with more, any consistent pair
        # measures the same baseline, so take the first two groups in name order.
        ordered = sorted(entries, key=lambda item: item[0])
        first_group = ordered[0][0]
        other = next(item for item in ordered if item[0] != first_group)
        pairs.append((ordered[0][1], other[1]))
    return pairs


def baseline_scale(poses: list[dict], baseline_mm: float | None) -> float | None:
    """Millimetres per model unit, from a measured rig baseline. None if there is no rig.

    The median across every paired slot rather than one pair or the mean: a handful of
    slots always solve badly, and one of them being out by a factor of two would
    otherwise rescale the whole export.

    Returns None rather than a guess when there is no pair to measure, when the baseline
    was never measured, or when the paired cameras came out on top of each other -- all
    three mean the same thing, which is that this run has no route to absolute size.
    """
    if not baseline_mm or baseline_mm <= 0:
        return None

    pairs = rig_pairs(poses)
    if not pairs:
        return None

    distances = np.array([np.linalg.norm(a - b) for a, b in pairs])
    distances = distances[distances > 1e-9]
    if len(distances) == 0:
        return None

    return float(baseline_mm / np.median(distances))
