"""The staleness invalidation matrix.

Pure Python: no ffmpeg, no COLMAP, no GPU. This engine decides when work is thrown
away and when it is kept, so it is tested before any real stage exists.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import Field

from pgh.fingerprint import apply_evaluation, compute_fingerprints, evaluate
from pgh.manifest import RunManifest, StageId, StageState
from pgh.stages.base import Stage, StageParams, StageResult
from pgh.stages.registry import StageRegistry


# -- fakes -------------------------------------------------------------------


class ExtractLikeParams(StageParams):
    fps: float = 6.0
    jpeg_qscale: int = 2
    # Cosmetic: re-renders a preview, does not invalidate any artifact.
    thumbnail_px: int = Field(256, json_schema_extra={"affects_fingerprint": False})


class SelectLikeParams(StageParams):
    target_count: int = 300


class FakeExtract(Stage):
    id = StageId.EXTRACT
    label = "Extract"
    params_model = ExtractLikeParams
    depends_on: list[StageId] = []

    def external_inputs(self, manifest: RunManifest) -> dict[str, Any]:
        # Mirrors the real stage: source identity, grouping, and sync offsets.
        return {
            "clips": [
                {
                    "clip_id": c.clip_id,
                    "camera_group": c.camera_group,
                    "identity": c.source_identity.model_dump() if c.source_identity else None,
                    "offset": c.time_offset_s,
                }
                for c in manifest.enabled_clips()
            ]
        }

    def run(self, ctx):  # pragma: no cover - never executed in these tests
        return StageResult()


class FakeSelect(Stage):
    id = StageId.SELECT
    label = "Select"
    params_model = SelectLikeParams
    depends_on = [StageId.EXTRACT]

    def run(self, ctx):  # pragma: no cover
        return StageResult()


class FakeMask(Stage):
    id = StageId.MASK
    label = "Mask"
    depends_on = [StageId.SELECT]

    def run(self, ctx):  # pragma: no cover
        return StageResult()


@pytest.fixture
def registry() -> StageRegistry:
    reg = StageRegistry()
    reg.register(FakeExtract())
    reg.register(FakeSelect())
    reg.register(FakeMask())
    return reg


@pytest.fixture
def manifest() -> RunManifest:
    from pgh.manifest import Clip, SourceIdentity

    m = RunManifest(run_id="test")
    m.clips = [
        Clip(
            clip_id="cam_high",
            camera_group="cam_high",
            source_path=r"D:\footage\high.mp4",
            source_identity=SourceIdentity(size_bytes=1000, mtime_ns=1, digest="aaa"),
        ),
        Clip(
            clip_id="cam_eye",
            camera_group="cam_eye",
            source_path=r"D:\footage\eye.mp4",
            source_identity=SourceIdentity(size_bytes=2000, mtime_ns=2, digest="bbb"),
        ),
    ]
    m.stages[StageId.EXTRACT].params = ExtractLikeParams().model_dump()
    m.stages[StageId.SELECT].params = SelectLikeParams().model_dump()
    return m


def mark_done(manifest: RunManifest, registry: StageRegistry, *stage_ids: StageId) -> None:
    """Simulate a successful run: store the current fingerprint and mark DONE."""
    computed = compute_fingerprints(manifest, registry)
    for stage_id in stage_ids:
        manifest.stages[stage_id].state = StageState.DONE
        manifest.stages[stage_id].fingerprint = computed[stage_id]


def states(manifest: RunManifest, registry: StageRegistry) -> dict[StageId, StageState]:
    return {sid: ev.state for sid, ev in evaluate(manifest, registry).items()}


# -- the matrix --------------------------------------------------------------


def test_fresh_after_run(manifest, registry):
    mark_done(manifest, registry, StageId.EXTRACT, StageId.SELECT, StageId.MASK)
    s = states(manifest, registry)
    assert s[StageId.EXTRACT] is StageState.DONE
    assert s[StageId.SELECT] is StageState.DONE
    assert s[StageId.MASK] is StageState.DONE


def test_param_change_marks_self_stale(manifest, registry):
    mark_done(manifest, registry, StageId.EXTRACT, StageId.SELECT, StageId.MASK)

    manifest.stages[StageId.EXTRACT].params["fps"] = 12.0

    s = states(manifest, registry)
    assert s[StageId.EXTRACT] is StageState.STALE


def test_param_change_propagates_downstream(manifest, registry):
    mark_done(manifest, registry, StageId.EXTRACT, StageId.SELECT, StageId.MASK)

    manifest.stages[StageId.EXTRACT].params["fps"] = 12.0

    s = states(manifest, registry)
    assert s[StageId.SELECT] is StageState.STALE, "select depends on extract"
    assert s[StageId.MASK] is StageState.STALE, "staleness must cross the whole chain"


def test_downstream_param_change_does_not_affect_upstream(manifest, registry):
    mark_done(manifest, registry, StageId.EXTRACT, StageId.SELECT, StageId.MASK)

    manifest.stages[StageId.SELECT].params["target_count"] = 150

    s = states(manifest, registry)
    assert s[StageId.EXTRACT] is StageState.DONE, "extract is upstream and untouched"
    assert s[StageId.SELECT] is StageState.STALE
    assert s[StageId.MASK] is StageState.STALE


def test_cosmetic_param_invalidates_nothing(manifest, registry):
    """thumbnail_px re-renders a preview; it must not throw away the frames."""
    mark_done(manifest, registry, StageId.EXTRACT, StageId.SELECT, StageId.MASK)

    manifest.stages[StageId.EXTRACT].params["thumbnail_px"] = 512

    s = states(manifest, registry)
    assert s[StageId.EXTRACT] is StageState.DONE
    assert s[StageId.SELECT] is StageState.DONE
    assert s[StageId.MASK] is StageState.DONE


def test_early_cutoff_on_revert(manifest, registry):
    """Change a param, change it back: downstream work must survive.

    This is the property that makes the harness pleasant to poke at. It only holds
    because fingerprints are content-derived rather than run counters.
    """
    mark_done(manifest, registry, StageId.EXTRACT, StageId.SELECT, StageId.MASK)
    original = compute_fingerprints(manifest, registry)

    manifest.stages[StageId.EXTRACT].params["fps"] = 12.0
    assert states(manifest, registry)[StageId.MASK] is StageState.STALE

    manifest.stages[StageId.EXTRACT].params["fps"] = 6.0

    assert compute_fingerprints(manifest, registry) == original
    s = states(manifest, registry)
    assert s[StageId.EXTRACT] is StageState.DONE
    assert s[StageId.MASK] is StageState.DONE, "reverting must restore, not re-run"


def test_new_clip_invalidates_extract_and_downstream(manifest, registry):
    from pgh.manifest import Clip, SourceIdentity

    mark_done(manifest, registry, StageId.EXTRACT, StageId.SELECT, StageId.MASK)

    manifest.clips.append(
        Clip(
            clip_id="cam_crown",
            camera_group="cam_crown",
            segment_id="seg1",
            source_path=r"D:\footage\crown.mp4",
            source_identity=SourceIdentity(size_bytes=3000, mtime_ns=3, digest="ccc"),
        )
    )

    s = states(manifest, registry)
    assert s[StageId.EXTRACT] is StageState.STALE
    assert s[StageId.MASK] is StageState.STALE


def test_swapping_the_source_file_invalidates(manifest, registry):
    """Re-exporting a clip at the same path must not be mistaken for the same input."""
    mark_done(manifest, registry, StageId.EXTRACT, StageId.SELECT, StageId.MASK)

    manifest.clips[0].source_identity.digest = "different"

    assert states(manifest, registry)[StageId.EXTRACT] is StageState.STALE


def test_sync_offset_change_invalidates_extract(manifest, registry):
    """Offsets define the shared timeline, so they must re-cut the frames."""
    mark_done(manifest, registry, StageId.EXTRACT, StageId.SELECT, StageId.MASK)

    manifest.clips[1].time_offset_s = 0.42

    assert states(manifest, registry)[StageId.EXTRACT] is StageState.STALE


def test_disabling_a_clip_invalidates_extract(manifest, registry):
    mark_done(manifest, registry, StageId.EXTRACT, StageId.SELECT, StageId.MASK)

    manifest.clips[1].enabled = False

    assert states(manifest, registry)[StageId.EXTRACT] is StageState.STALE


def test_unrelated_manifest_edits_do_not_invalidate(manifest, registry):
    """Renaming the run or adding a note must not throw away hours of compute."""
    mark_done(manifest, registry, StageId.EXTRACT, StageId.SELECT, StageId.MASK)

    manifest.name = "bust take 2"
    manifest.capture.notes = "spun a bit fast"
    manifest.capture.baseline_mm = 412.0

    s = states(manifest, registry)
    completed = (StageId.EXTRACT, StageId.SELECT, StageId.MASK)
    assert all(s[stage_id] is StageState.DONE for stage_id in completed)


# -- blocking ----------------------------------------------------------------


def test_stage_is_blocked_until_dependency_completes(manifest, registry):
    evals = evaluate(manifest, registry)

    assert evals[StageId.EXTRACT].blocked_by == []
    assert evals[StageId.EXTRACT].runnable
    assert evals[StageId.SELECT].blocked_by == [StageId.EXTRACT]
    assert not evals[StageId.SELECT].runnable


def test_stale_dependency_still_permits_running(manifest, registry):
    """A stale upstream is re-runnable output, not missing output.

    The UI warns, but does not forbid -- being able to run a later stage against
    slightly outdated inputs is genuinely useful while iterating.
    """
    mark_done(manifest, registry, StageId.EXTRACT)
    manifest.stages[StageId.EXTRACT].params["fps"] = 9.0

    evals = evaluate(manifest, registry)
    assert evals[StageId.EXTRACT].state is StageState.STALE
    assert evals[StageId.SELECT].blocked_by == []


def test_running_state_is_preserved(manifest, registry):
    manifest.stages[StageId.EXTRACT].state = StageState.RUNNING
    assert states(manifest, registry)[StageId.EXTRACT] is StageState.RUNNING


def test_failed_stage_blocks_downstream(manifest, registry):
    manifest.stages[StageId.EXTRACT].state = StageState.FAILED
    evals = evaluate(manifest, registry)
    assert evals[StageId.SELECT].blocked_by == [StageId.EXTRACT]


# -- engine mechanics --------------------------------------------------------


def test_unimplemented_stages_still_participate(manifest, registry):
    """dense/mesh/export have no Stage object yet but must still be in the DAG."""
    evals = evaluate(manifest, registry)
    for stage_id in (StageId.SPARSE, StageId.DENSE, StageId.MESH, StageId.EXPORT):
        assert stage_id in evals


def test_fingerprints_are_stable_across_key_order(manifest, registry):
    before = compute_fingerprints(manifest, registry)
    params = manifest.stages[StageId.EXTRACT].params
    manifest.stages[StageId.EXTRACT].params = dict(reversed(list(params.items())))
    assert compute_fingerprints(manifest, registry) == before


def test_apply_evaluation_writes_state_back(manifest, registry):
    mark_done(manifest, registry, StageId.EXTRACT, StageId.SELECT, StageId.MASK)
    manifest.stages[StageId.EXTRACT].params["fps"] = 30.0

    apply_evaluation(manifest, registry)

    assert manifest.stages[StageId.EXTRACT].state is StageState.STALE
    assert manifest.stages[StageId.MASK].state is StageState.STALE
    assert manifest.stages[StageId.MASK].stale_reason == "upstream"
