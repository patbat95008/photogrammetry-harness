"""Crash recovery and capture-mode behaviour."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pgh.manifest import CaptureMode, SegmentKind, StageId, StageState
from pgh.stages.mask import MaskStage
from pgh.stages.planned import planned_for
from pgh.store import RunStore


@pytest.fixture
def run_store(tmp_path: Path) -> RunStore:
    return RunStore(root=tmp_path / "runs")


def test_interrupted_stage_is_marked_failed(run_store):
    """The worker lives in this process, so RUNNING across a restart is impossible.

    Left alone it would show a progress bar forever and refuse to start again,
    because the runner rejects a duplicate submission for a stage already running.
    """
    run = run_store.create("crashed")
    run.update(lambda m: setattr(m.stages[StageId.EXTRACT], "state", StageState.RUNNING))
    scratch = run.stage_scratch("extract")
    scratch.mkdir(parents=True, exist_ok=True)
    (scratch / "half-written.jpg").write_bytes(b"partial")

    repaired = RunStore(root=run_store.root).recover()

    assert repaired == [f"{run.run_id}/extract"]
    record = run_store.get(run.run_id).io.load().stages[StageId.EXTRACT]
    assert record.state is StageState.FAILED
    assert "interrupted" in record.error
    assert not scratch.exists(), "incomplete scratch output must not survive"


def test_recovery_leaves_healthy_runs_alone(run_store):
    run = run_store.create("fine")
    run.update(lambda m: setattr(m.stages[StageId.EXTRACT], "state", StageState.DONE))

    assert RunStore(root=run_store.root).recover() == []
    assert (
        run_store.get(run.run_id).io.load().stages[StageId.EXTRACT].state
        is StageState.DONE
    )


def test_recovery_survives_a_corrupt_manifest(run_store):
    """One unreadable run must not stop the others being repaired."""
    good = run_store.create("good")
    good.update(lambda m: setattr(m.stages[StageId.EXTRACT], "state", StageState.RUNNING))

    broken = run_store.root / "20260101-broken"
    broken.mkdir(parents=True)
    (broken / "project.json").write_text("{not json", encoding="utf-8")

    repaired = RunStore(root=run_store.root).recover()
    assert repaired == [f"{good.run_id}/extract"]


# -- capture mode ------------------------------------------------------------


def test_chair_spin_requires_masking(run_store):
    """Subject rotating, cameras fixed: the background moves relative to the subject."""
    run = run_store.create("spin")
    manifest = run.load()
    assert manifest.capture.mode is CaptureMode.SUBJECT_ROTATES
    assert manifest.capture.needs_background_mask

    # The prose lived in planned.py until M9 was built and now comes from the
    # stage itself. Implementing a stage must not delete its explanation.
    note = MaskStage().note(manifest)
    assert note and "REQUIRED" in note


def test_handheld_orbit_does_not_require_masking(run_store):
    """Cameras orbiting a still subject: the background is rigid with it and helps."""
    run = run_store.create("orbit")
    manifest = run.update(
        lambda m: setattr(m.capture, "mode", CaptureMode.CAMERA_ORBITS)
    )
    assert not manifest.capture.needs_background_mask

    note = MaskStage().note(manifest)
    assert note and "OPTIONAL" in note


def test_orientation_flip_survives_a_reload(run_store):
    """The viewer's flip is read back on the next visit and by the export stage."""
    run = run_store.create("upside down")
    assert run.load().capture.flip_x, "COLMAP's frame is arbitrary; default to flipping"

    run.update(lambda m: setattr(m.capture, "flip_x", False))

    assert run_store.get(run.run_id).io.load().capture.flip_x is False


def test_a_manifest_written_before_the_flip_existed_still_loads(run_store):
    """Adding the field must not strand runs made before it."""
    run = run_store.create("older")
    raw = json.loads((run.dir / "project.json").read_text(encoding="utf-8"))
    del raw["capture"]["flip_x"]
    (run.dir / "project.json").write_text(json.dumps(raw), encoding="utf-8")

    assert run_store.get(run.run_id).io.load().capture.flip_x is True


def test_only_fixed_mounts_support_a_rig():
    """One phone in each hand has no constant relative pose, so no rig can be claimed."""
    assert SegmentKind.RIG.supports_rig
    assert not SegmentKind.INDEPENDENT.supports_rig
    assert not SegmentKind.SINGLE.supports_rig


def test_planned_copy_is_removed_once_a_stage_is_built() -> None:
    """The other half of the pairing, which was not previously enforced.

    Leaving an entry behind after building a stage is invisible: api/stages.py only
    reads PLANNED when the registry has no Stage object, so the stale copy sits there
    describing work that is finished and nothing ever renders it.
    """
    from pgh.stages import registry as stage_registry
    from pgh.stages.planned import PLANNED

    for stage_id in list(PLANNED):
        assert stage_registry.get(stage_id) is None, (
            f"{stage_id} is implemented, so its entry in planned.py is dead copy"
        )


def test_planned_stages_cover_every_unimplemented_stage(run_store):
    from pgh.stages.registry import registry

    manifest = run_store.create("coverage").load()
    for stage_id in StageId:
        if stage_id in registry.implemented():
            continue
        assert planned_for(stage_id, manifest) is not None, f"{stage_id} has no stub copy"
