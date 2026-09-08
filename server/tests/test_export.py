"""The export stage: what it refuses, what it claims, and what it admits it cannot know.

Pure Python -- no Blender. The geometry it relies on is tested in ``test_orient.py``;
what is tested here is the reasoning around it, which is mostly about *not* asserting
things that were never established. Both of this stage's real failure modes look exactly
like success: a model exported upside down still opens, and a model exported at the wrong
size still prints.
"""

from __future__ import annotations

import pytest

from pgh.manifest import RunManifest, StageId, StageState
from pgh.stages.export import ExportParams, ExportStage, _summarise
from pgh.vendor import blender


@pytest.fixture
def stage() -> ExportStage:
    return ExportStage()


@pytest.fixture
def manifest() -> RunManifest:
    """A run whose mesh finished, which is the only state export can run from."""
    m = RunManifest(run_id="cup")
    m.stages[StageId.MESH].state = StageState.DONE
    m.stages[StageId.MESH].fingerprint = "meshfp"
    m.stages[StageId.MESH].artifacts = {
        "mesh": "mesh/mesh_textured.glb",
        "mesh_untextured": "mesh/mesh.ply",
    }
    m.stages[StageId.SPARSE].fingerprint = "sparsefp"
    m.stages[StageId.SPARSE].artifacts = {"poses": "sparse/poses.json"}
    return m


@pytest.fixture(autouse=True)
def blender_present(monkeypatch: pytest.MonkeyPatch) -> None:
    """Blender is the one optional engine, so the tests say which case they are in."""
    monkeypatch.setattr(blender, "available", lambda: True)
    monkeypatch.setattr(blender, "version", lambda: "5.2")


# -- preflight ---------------------------------------------------------------


def test_preflight_passes_on_a_finished_mesh(stage, manifest) -> None:
    assert stage.preflight(manifest) == []


def test_preflight_blocks_without_a_mesh(stage, manifest) -> None:
    manifest.stages[StageId.MESH].artifacts = {}
    assert any("run the mesh stage first" in p for p in stage.preflight(manifest))


def test_preflight_blocks_without_camera_poses(stage, manifest) -> None:
    """The up axis comes from where the cameras were, so this is not optional."""
    manifest.stages[StageId.SPARSE].artifacts = {}
    assert any("poses.json" in p for p in stage.preflight(manifest))


def test_preflight_blocks_when_blender_is_missing(stage, manifest, monkeypatch) -> None:
    """It is required=False in config.py precisely because only this stage needs it."""
    monkeypatch.setattr(blender, "available", lambda: False)
    problems = stage.preflight(manifest)

    assert any("Blender was not found" in p for p in problems)
    assert any("tools.local.toml" in p for p in problems)


def test_preflight_blocks_manual_scale_with_no_measurement(stage, manifest) -> None:
    manifest.stages[StageId.EXPORT].params = {"scale_mode": "manual"}
    assert any("known_dimension_mm" in p for p in stage.preflight(manifest))


def test_preflight_accepts_manual_scale_with_a_measurement(stage, manifest) -> None:
    manifest.stages[StageId.EXPORT].params = {
        "scale_mode": "manual",
        "known_dimension_mm": 95.0,
    }
    assert stage.preflight(manifest) == []


def test_preflight_survives_params_it_cannot_parse(stage, manifest) -> None:
    """Saved params can predate a field. Preflight returns reasons; it must not raise."""
    manifest.stages[StageId.EXPORT].params = {"a_field_that_was_removed": 7}
    assert isinstance(stage.preflight(manifest), list)


# -- fingerprint inputs ------------------------------------------------------


def test_the_orientation_flip_invalidates_this_stage(stage, manifest) -> None:
    """The one stage that reads flip_x, and therefore the only one it may invalidate.

    Turning the model over in the viewer has to re-export -- the exported file is the
    thing that would otherwise disagree with what was reviewed -- while never touching
    anything upstream. HANDOVER 8 asks for the key to be added here at creation for
    exactly that reason.
    """
    before = stage.external_inputs(manifest)
    manifest.capture.flip_x = not manifest.capture.flip_x

    assert stage.external_inputs(manifest) != before


def test_the_measured_baseline_invalidates_this_stage(stage, manifest) -> None:
    """It is the scale of everything exported, so a corrected tape measure must re-run."""
    before = stage.external_inputs(manifest)
    manifest.capture.baseline_mm = 120.0

    assert stage.external_inputs(manifest) != before


def test_external_inputs_are_stable_across_calls(stage, manifest) -> None:
    assert stage.external_inputs(manifest) == stage.external_inputs(manifest)


# -- what the report claims --------------------------------------------------


def summarise(params, result, scale, source, planarity=0.02, axis=(0.0, 0.0, 1.0)):
    return _summarise(params, result, scale, source, planarity, list(axis))


def result(**over):
    base = {
        "faces_in": 497926,
        "faces_out": 495490,
        "components": 7,
        "dropped_faces": 2436,
        "vertices_healed": 88778,
        "dimensions": [6.7, 6.9, 2.0],
        "files": {"glb": "model.glb", "stl": "model.stl"},
        "turntable_frames": 36,
    }
    base.update(over)
    return base


def test_an_unscaled_export_says_so_plainly(manifest) -> None:
    """The failure that looks exactly like success, so the warning has to be blunt."""
    metrics, warnings = summarise(ExportParams(), result(), None, "none")

    assert metrics["dimension_units"] == "model units"
    assert any("arbitrary units" in w for w in warnings)
    assert any("wrong size" in w or "whatever size" in w for w in warnings)


def test_an_unscaled_export_does_not_claim_a_millimetre_scale(manifest) -> None:
    """Reporting 1.0 mm per unit states a scale that nothing ever measured."""
    metrics, _ = summarise(ExportParams(), result(), 1.0, "none")
    assert "scale_mm_per_unit" not in metrics


def test_a_measured_export_reports_its_scale_and_millimetres(manifest) -> None:
    metrics, warnings = summarise(ExportParams(), result(), 240.0, "baseline")

    assert metrics["scale_mm_per_unit"] == 240.0
    assert metrics["dimension_units"] == "mm"
    assert not any("arbitrary units" in w for w in warnings)


def test_a_loose_plane_fit_is_named(manifest) -> None:
    _, warnings = summarise(ExportParams(), result(), 240.0, "baseline", planarity=0.42)
    assert any("loosely planar" in w for w in warnings)


def test_a_tight_plane_fit_is_not_mentioned(manifest) -> None:
    _, warnings = summarise(ExportParams(), result(), 240.0, "baseline", planarity=0.04)
    assert not any("loosely planar" in w for w in warnings)


def test_losing_most_of_the_model_to_cleanup_is_named(manifest) -> None:
    """The trap from HANDOVER 6.23, in the shape it would take if it came back."""
    _, warnings = summarise(
        ExportParams(), result(faces_out=2400, dropped_faces=495526), 240.0, "baseline"
    )
    assert any("largest connected shell dropped" in w for w in warnings)


def test_the_cup_run_is_not_flagged_for_the_debris_it_did_drop(manifest) -> None:
    """2,436 of 497,926 is normal floating debris and must read as unremarkable."""
    _, warnings = summarise(ExportParams(), result(), 240.0, "baseline")
    assert not any("largest connected shell dropped" in w for w in warnings)


def test_decimating_past_where_detail_survives_is_named(manifest) -> None:
    _, warnings = summarise(
        ExportParams(target_faces=500), result(faces_out=500), 240.0, "baseline"
    )
    assert any("target_faces" in w for w in warnings)


def test_the_report_records_what_was_written(manifest) -> None:
    metrics, _ = summarise(ExportParams(), result(), 240.0, "baseline")

    assert metrics["formats"] == ["glb", "stl"]
    assert metrics["turntable_frames"] == 36
    assert metrics["vertices_healed"] == 88778
    assert metrics["up_axis"] == [0.0, 0.0, 1.0]


def test_an_unoriented_export_reports_no_axis(manifest) -> None:
    """orient_model off is a deliberate choice, not a recovered axis to report."""
    metrics, _ = _summarise(ExportParams(orient_model=False), result(), None, "none", None, None)

    assert "up_axis" not in metrics
    assert "camera_planarity" not in metrics


# -- reading Blender's answer out of its noise -------------------------------


NOISY = [
    "Blender 5.2.0 LTS (hash fbe6228777e7 built 2026-07-14 01:35:40)",
    "Read prefs: implicit default",
    "PGH healed 88778 split vertices at threshold 1.2e-05",
    "PGH components=7 kept=495490 dropped=2436",
    "Fra:1 Mem:12.00M | Time:00:00.42 | Compositing",
    'PGH_RESULT {"faces_out": 495490, "components": 7}',
    "Info: Saved image 'frame_035.png'",
    "Blender quit",
]


def test_the_result_is_found_among_blenders_chatter() -> None:
    assert blender.parse_result(NOISY) == {"faces_out": 495490, "components": 7}


def test_a_script_that_never_finished_reports_nothing() -> None:
    """Which is how the stage tells "it crashed" from "it worked but said little"."""
    assert blender.parse_result([line for line in NOISY if "PGH_RESULT" not in line]) is None


def test_the_last_result_line_wins() -> None:
    """A script reporting progress and then a result must not be read as stopping early."""
    lines = NOISY + ['PGH_RESULT {"faces_out": 1}']
    assert blender.parse_result(lines) == {"faces_out": 1}


def test_a_malformed_result_line_is_skipped_not_fatal() -> None:
    lines = ['PGH_RESULT {"faces_out": 495490}', "PGH_RESULT {not json"]
    assert blender.parse_result(lines) == {"faces_out": 495490}


def test_a_mention_of_the_prefix_mid_line_is_not_a_result() -> None:
    assert blender.parse_result(["writing PGH_RESULT to the log"]) is None


# -- what a download actually needs -------------------------------------------


def test_an_obj_is_offered_with_its_material_and_texture():
    """A .obj on its own is an untextured mesh. All three files or none."""
    from pgh.stages.export import _artifacts

    artifacts = _artifacts(
        {
            "files": {"glb": "model.glb", "obj": "model.obj", "stl": "model.stl"},
            "sidecars": {"obj_material": "model.mtl", "obj_texture": "atlas.png"},
            "turntable_frames": 36,
        }
    )

    assert artifacts["model_obj"] == "export/model.obj"
    assert artifacts["sidecar_obj_material"] == "export/model.mtl"
    assert artifacts["sidecar_obj_texture"] == "export/atlas.png"


def test_a_material_is_not_recorded_as_a_model():
    """The page lists models by the model_ prefix, so a .mtl there reads as one."""
    from pgh.stages.export import _artifacts

    artifacts = _artifacts(
        {"files": {"glb": "model.glb"}, "sidecars": {"obj_material": "model.mtl"}}
    )

    models = [k for k in artifacts if k.startswith("model_")]
    assert models == ["model_glb"]


def test_a_glb_only_export_records_no_sidecars():
    from pgh.stages.export import _artifacts

    artifacts = _artifacts({"files": {"glb": "model.glb"}, "turntable_frames": 0})

    assert not any(k.startswith("sidecar_") for k in artifacts)
    assert "turntable" not in artifacts


def test_the_report_carries_the_size_the_viewer_has_to_gate_on():
    """Blender writes normals the mesh stage's file does not, so an export is much
    bigger than the mesh it came from -- 68 MB against 23 MB on cup-1. Without this
    the viewer's 40 MB prompt cannot fire on the page that most needs it."""
    metrics, _warnings = _summarise(
        ExportParams(), result(), None, "none", 0.04, [0.0, 0.0, 1.0], 67_902_144
    )

    assert metrics["model_bytes"] == 67_902_144
