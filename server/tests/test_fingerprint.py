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


# -- skipping ----------------------------------------------------------------
#
# Masking is optional when the cameras orbit a still subject: the background is
# rigid with the subject, so masking it out throws away features that help the orbit
# close. Skipping is therefore a decision about the run, and dependents have to treat
# it as satisfied -- otherwise the align stage is simply unreachable.


class FakeSparse(Stage):
    """Stands in for the align stage, whose real dependency edge is the awkward one."""

    id = StageId.SPARSE
    label = "Align"
    depends_on = [StageId.SELECT, StageId.MASK]

    def external_inputs(self, manifest: RunManifest) -> dict[str, Any]:
        # Mirrors the real stage: whether masks exist is an input in its own right,
        # because the mask stage's fingerprint is identical whether it ran or was
        # skipped -- it hashes params and inputs, not outcome.
        mask = manifest.stages[StageId.MASK]
        return {
            "masks": (
                "none"
                if mask.state is StageState.SKIPPED
                else mask.artifacts.get("masks", "none")
            )
        }

    def run(self, ctx):  # pragma: no cover
        return StageResult()


@pytest.fixture
def registry_with_sparse(registry: StageRegistry) -> StageRegistry:
    registry.register(FakeSparse())
    return registry


def blocked(manifest: RunManifest, registry: StageRegistry, stage_id: StageId) -> list[StageId]:
    return evaluate(manifest, registry)[stage_id].blocked_by


def test_a_pending_mask_blocks_align(manifest, registry_with_sparse):
    """The situation that makes SKIPPED necessary rather than merely tidy."""
    mark_done(manifest, registry_with_sparse, StageId.EXTRACT, StageId.SELECT)
    assert blocked(manifest, registry_with_sparse, StageId.SPARSE) == [StageId.MASK]
    assert not evaluate(manifest, registry_with_sparse)[StageId.SPARSE].runnable


def test_skipping_the_mask_unblocks_align(manifest, registry_with_sparse):
    mark_done(manifest, registry_with_sparse, StageId.EXTRACT, StageId.SELECT)
    manifest.stages[StageId.MASK].state = StageState.SKIPPED

    assert blocked(manifest, registry_with_sparse, StageId.SPARSE) == []
    assert evaluate(manifest, registry_with_sparse)[StageId.SPARSE].runnable


def test_a_skipped_stage_stays_skipped(manifest, registry_with_sparse):
    """Evaluation must not quietly reinterpret the state as pending or done."""
    mark_done(manifest, registry_with_sparse, StageId.EXTRACT, StageId.SELECT)
    manifest.stages[StageId.MASK].state = StageState.SKIPPED
    assert states(manifest, registry_with_sparse)[StageId.MASK] is StageState.SKIPPED


def test_failed_and_cancelled_still_block(manifest, registry_with_sparse):
    """Only an explicit skip counts; a stage that fell over is not satisfied."""
    mark_done(manifest, registry_with_sparse, StageId.EXTRACT, StageId.SELECT)
    for state in (StageState.FAILED, StageState.CANCELLED, StageState.PENDING):
        manifest.stages[StageId.MASK].state = state
        assert blocked(manifest, registry_with_sparse, StageId.SPARSE) == [StageId.MASK], state


def test_unskipping_blocks_align_again(manifest, registry_with_sparse):
    mark_done(manifest, registry_with_sparse, StageId.EXTRACT, StageId.SELECT)
    manifest.stages[StageId.MASK].state = StageState.SKIPPED
    assert blocked(manifest, registry_with_sparse, StageId.SPARSE) == []

    manifest.stages[StageId.MASK].state = StageState.PENDING
    assert blocked(manifest, registry_with_sparse, StageId.SPARSE) == [StageId.MASK]


def test_running_align_over_a_skipped_mask_goes_stale_when_masks_arrive(
    manifest, registry_with_sparse
):
    """The easy bug: the mask stage's own fingerprint does not change when it runs.

    It hashes params, external inputs and upstream fingerprints, all of which are the
    same whether the stage was skipped or executed. Without align declaring mask
    availability as one of *its* inputs, producing masks would leave a
    background-including reconstruction looking perfectly fresh.
    """
    mark_done(manifest, registry_with_sparse, StageId.EXTRACT, StageId.SELECT)
    manifest.stages[StageId.MASK].state = StageState.SKIPPED
    mark_done(manifest, registry_with_sparse, StageId.SPARSE)
    assert states(manifest, registry_with_sparse)[StageId.SPARSE] is StageState.DONE

    # Now masking actually runs and produces something.
    manifest.stages[StageId.MASK].state = StageState.DONE
    manifest.stages[StageId.MASK].artifacts = {"masks": "masks"}
    manifest.stages[StageId.MASK].fingerprint = compute_fingerprints(
        manifest, registry_with_sparse
    )[StageId.MASK]

    assert states(manifest, registry_with_sparse)[StageId.SPARSE] is StageState.STALE, (
        "align must go stale when masks appear, or it silently keeps a model built "
        "from the unmasked images"
    )


def test_skipping_again_restores_the_earlier_fingerprint(manifest, registry_with_sparse):
    """Content-derived, not a counter: skip, unskip, skip must be back where it began."""
    mark_done(manifest, registry_with_sparse, StageId.EXTRACT, StageId.SELECT)
    manifest.stages[StageId.MASK].state = StageState.SKIPPED
    original = compute_fingerprints(manifest, registry_with_sparse)

    manifest.stages[StageId.MASK].state = StageState.DONE
    manifest.stages[StageId.MASK].artifacts = {"masks": "masks"}
    assert compute_fingerprints(manifest, registry_with_sparse) != original

    manifest.stages[StageId.MASK].state = StageState.SKIPPED
    manifest.stages[StageId.MASK].artifacts = {}
    assert compute_fingerprints(manifest, registry_with_sparse) == original


# -- params normalisation ----------------------------------------------------


def test_omitted_params_hash_like_explicit_defaults(manifest, registry):
    """Opening a stage page and pressing Save must not invalidate anything.

    A stage run before anyone touched its form stores {}; the UI then saves every
    field explicitly. Both describe the same run. Hashing the raw dict would make
    that Save throw away every finished stage downstream.
    """
    manifest.stages[StageId.SELECT].params = {}
    with_nothing = compute_fingerprints(manifest, registry)

    manifest.stages[StageId.SELECT].params = SelectLikeParams().model_dump()
    with_defaults = compute_fingerprints(manifest, registry)

    assert with_nothing == with_defaults


def test_a_partial_params_dict_hashes_like_the_full_one(manifest, registry):
    """Adding a parameter to a model must not invalidate runs that predate it."""
    manifest.stages[StageId.EXTRACT].params = {"fps": 6.0}
    partial = compute_fingerprints(manifest, registry)

    manifest.stages[StageId.EXTRACT].params = ExtractLikeParams(fps=6.0).model_dump()
    full = compute_fingerprints(manifest, registry)

    assert partial == full


def test_a_real_change_still_invalidates(manifest, registry):
    """The normalisation must not blunt the thing the engine exists to do."""
    manifest.stages[StageId.EXTRACT].params = {}
    before = compute_fingerprints(manifest, registry)
    manifest.stages[StageId.EXTRACT].params = {"fps": 12.0}
    assert compute_fingerprints(manifest, registry) != before


def test_unparseable_stored_params_do_not_raise(manifest, registry):
    """This runs on every page load; a bad stored value must not break the page."""
    manifest.stages[StageId.EXTRACT].params = {"fps": "not a number"}
    assert compute_fingerprints(manifest, registry)  # no exception


def test_cosmetic_params_stay_cosmetic_after_normalisation(manifest, registry):
    manifest.stages[StageId.EXTRACT].params = {}
    before = compute_fingerprints(manifest, registry)
    manifest.stages[StageId.EXTRACT].params = {"thumbnail_px": 512}
    assert compute_fingerprints(manifest, registry) == before


def test_the_orientation_flip_invalidates_nothing(manifest, registry):
    """It is a presentation and export choice, not a reconstruction input.

    Nothing between extract and dense reads it, so turning the model over in the
    viewer must never throw away hours of solving.
    """
    before = compute_fingerprints(manifest, registry)
    manifest.capture.flip_x = not manifest.capture.flip_x
    assert compute_fingerprints(manifest, registry) == before
