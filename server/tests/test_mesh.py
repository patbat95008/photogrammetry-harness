"""The mesh stage's preflight, and the decisions it is easy to undo by accident.

Pure Python: no OpenMVS, no GPU, no subprocess. What is tested here is everything that
happens around the three tool invocations, because those three lines are the only part
that cannot be exercised without a 64 GB machine and five minutes.

The recurring theme is that this stage's failures are quiet ones. Reconstructing from
the wrong cloud, chaining on a file the tool never wrote, mistaking a statistic for
progress, and picking up the texture atlas instead of the mesh all produce a run that
finishes and reports success.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pgh.manifest import RunManifest, StageId, StageState
from pgh.stages.mesh import (
    RECONSTRUCT_PHASES,
    TEXTURE_PHASES,
    MeshParams,
    MeshStage,
    _count_recorder,
    _find_mesh,
    _find_textures,
    _phase_mapper,
    _summarise,
)


@pytest.fixture
def stage() -> MeshStage:
    return MeshStage()


@pytest.fixture
def manifest() -> RunManifest:
    """A run whose dense stage finished, which is the only state mesh can run from."""
    m = RunManifest(run_id="cup")
    dense = m.stages[StageId.DENSE]
    dense.state = StageState.DONE
    dense.fingerprint = "abc123"
    dense.metrics = {"num_points": 738_015}
    dense.artifacts = {
        "scene": "dense/scene_dense.mvs",
        "cloud": "dense/scene_dense.ply",
        "undistorted": "dense/undistorted",
        "preview": "dense/preview.ply",
    }
    return m


# -- preflight ---------------------------------------------------------------


def test_preflight_passes_on_a_finished_dense_run(stage, manifest) -> None:
    assert stage.preflight(manifest) == []


def test_preflight_blocks_without_a_dense_scene(stage, manifest) -> None:
    del manifest.stages[StageId.DENSE].artifacts["scene"]
    assert any("run the dense stage first" in p for p in stage.preflight(manifest))


def test_preflight_blocks_when_the_dense_stage_recorded_no_cloud(stage, manifest) -> None:
    """The message has to name the file, because the .mvs looks like it would do.

    Reconstructing from the .mvs alone succeeds -- it carries the 50k sparse points --
    and produces a mesh from a fifteenth of the data with nothing calling it an error.
    """
    del manifest.stages[StageId.DENSE].artifacts["cloud"]
    problems = stage.preflight(manifest)
    assert any("scene_dense.ply" in p for p in problems)
    assert any("sparse cloud" in p for p in problems)


def test_preflight_blocks_without_the_undistorted_images(stage, manifest) -> None:
    del manifest.stages[StageId.DENSE].artifacts["undistorted"]
    assert any("relative to that folder" in p for p in stage.preflight(manifest))


def test_preflight_blocks_on_an_empty_cloud(stage, manifest) -> None:
    manifest.stages[StageId.DENSE].metrics = {"num_points": 0}
    assert any("no surface to find" in p for p in stage.preflight(manifest))


def test_preflight_does_not_repeat_the_mask_block(stage, manifest) -> None:
    """A deliberate absence, pinned so nobody adds it back as an obvious omission.

    The dense stage already refuses to run when masks are present, and it is a hard
    dependency of this one, so a masked run cannot arrive here without HANDOVER 6.8
    having been implemented. A block here would be a message nothing could exercise.
    """
    mask = manifest.stages[StageId.MASK]
    mask.state = StageState.DONE
    mask.artifacts = {"masks": "masks"}

    assert stage.preflight(manifest) == []


# -- fingerprint inputs ------------------------------------------------------


def test_external_inputs_are_stable_across_calls(stage, manifest) -> None:
    """Anything volatile in here makes the stage look stale on every page load."""
    assert stage.external_inputs(manifest) == stage.external_inputs(manifest)


def test_external_inputs_do_not_include_the_orientation_flip(stage, manifest) -> None:
    """flip_x belongs to the export stage's fingerprint, and only to it.

    Turning the model over in the viewer must never throw away a reconstruction. See
    HANDOVER 8: it is added to export's external_inputs at that stage's creation, so it
    can never rehash work that already exists.
    """
    before = stage.external_inputs(manifest)
    manifest.capture.flip_x = not manifest.capture.flip_x

    assert stage.external_inputs(manifest) == before


def test_the_dense_fingerprint_is_an_input(stage, manifest) -> None:
    before = stage.external_inputs(manifest)
    manifest.stages[StageId.DENSE].fingerprint = "different"

    assert stage.external_inputs(manifest) != before


# -- discovering what the tools wrote ----------------------------------------


def test_the_mesh_is_discovered_not_assumed(tmp_path: Path) -> None:
    """The extension comes from --export-type, not from the name given to -o."""
    (tmp_path / "mesh_textured.glb").write_bytes(b"glb")
    (tmp_path / "mesh_textured_0.png").write_bytes(b"png")
    (tmp_path / "TextureMesh-2609.log").write_text("log")

    assert _find_mesh(tmp_path, "mesh_textured") == tmp_path / "mesh_textured.glb"


def test_the_texture_atlas_is_not_mistaken_for_the_mesh(tmp_path: Path) -> None:
    """<stem>_0.png shares a prefix with the mesh but is a different stem."""
    (tmp_path / "mesh_textured_0.png").write_bytes(b"png")

    assert _find_mesh(tmp_path, "mesh_textured") is None


def test_an_obj_is_found_rather_than_its_material_file(tmp_path: Path) -> None:
    (tmp_path / "mesh_textured.obj").write_text("o")
    (tmp_path / "mesh_textured.mtl").write_text("m")

    assert _find_mesh(tmp_path, "mesh_textured") == tmp_path / "mesh_textured.obj"


def test_the_sidecar_texture_is_recorded(tmp_path: Path) -> None:
    """A .glb is not self-contained; the viewer needs the .png served beside it."""
    mesh = tmp_path / "mesh_textured.glb"
    mesh.write_bytes(b"glb")
    (tmp_path / "mesh_textured_0.png").write_bytes(b"png")

    found = _find_textures(tmp_path, "mesh_textured", mesh)

    assert [p.name for p in found] == ["mesh_textured_0.png"]


def test_the_mesh_itself_is_not_listed_as_its_own_texture(tmp_path: Path) -> None:
    mesh = tmp_path / "mesh_textured.ply"
    mesh.write_bytes(b"ply")

    assert _find_textures(tmp_path, "mesh_textured", mesh) == []


# -- progress ----------------------------------------------------------------


def test_phase_progress_starts_by_naming_what_it_is_doing() -> None:
    mapper = _phase_mapper(RECONSTRUCT_PHASES, "tetrahedralization")

    assert mapper("Scene loaded in interface format") == (0.0, "tetrahedralization")


def test_phase_progress_advances_on_a_marker() -> None:
    mapper = _phase_mapper(RECONSTRUCT_PHASES, "tetrahedralization")
    mapper("first line")

    found = mapper("Delaunay tetrahedralization completed: 738015 points -> 577954 vertices")

    assert found == (0.20, "weighting")


def test_an_unrecognised_line_moves_nothing() -> None:
    mapper = _phase_mapper(TEXTURE_PHASES, "assigning views")
    mapper("first line")

    assert mapper("some line the table does not know about") is None


def test_texture_progress_gives_view_assignment_nearly_the_whole_span() -> None:
    """It took 4m2s of a 4m17s run. Treating it as half would park the bar at 50%."""
    marker, fraction, _ = TEXTURE_PHASES[0]
    assert "Assigning the best view" in marker
    assert fraction >= 0.9


# -- counts ------------------------------------------------------------------


def test_counts_are_taken_from_the_last_line_that_reports_them() -> None:
    """The scene load reports zeroes before any mesh exists."""
    counts: dict[str, tuple[int, int]] = {}
    record = _count_recorder(counts, "reconstruct")

    record("\t50189 points, 0 vertices, 0 faces")
    record("Mesh reconstruction completed: 249632 vertices, 498780 faces (21s6ms)")
    record("Mesh 'mesh.ply' saved: 249022 vertices, 497926 faces (54ms)")

    assert counts["reconstruct"] == (249022, 497926)


def test_the_scene_load_alone_records_nothing() -> None:
    """Zeroes are not a measurement, and recording them would report an empty mesh."""
    counts: dict[str, tuple[int, int]] = {}
    _count_recorder(counts, "texture")("\t50189 points, 0 vertices, 0 faces")

    assert counts == {}


# -- warnings ----------------------------------------------------------------


def summarise(manifest, params, faces, *, sizes=None, timings=None, peak=1.0):
    return _summarise(
        manifest,
        params,
        {"vertices": faces // 2, "faces": faces, "raw_vertices": 1, "raw_faces": faces},
        sizes or {"mesh_bytes": 1_000_000, "texture_bytes": 1_000_000},
        1,
        timings or {},
        peak,
    )


def test_warns_when_almost_nothing_came_out(manifest) -> None:
    _, warnings = summarise(manifest, MeshParams(), 12)
    assert any("it is an empty one" in w for w in warnings)
    assert any("crop_to_roi" in w for w in warnings)


def test_warns_when_the_face_count_is_absurd(manifest) -> None:
    _, warnings = summarise(manifest, MeshParams(), 5_000_000)
    assert any("target_face_num" in w for w in warnings)


def test_a_normal_mesh_gets_neither_size_warning(manifest) -> None:
    """The cup came out at 498k faces, so that must read as unremarkable."""
    _, warnings = summarise(manifest, MeshParams(), 497_926)
    assert not any("empty one" in w or "Nothing downstream needs" in w for w in warnings)


def test_always_says_when_refinement_was_skipped(manifest) -> None:
    """The absence of a refinement pass is a fact about the surface, not a non-event."""
    _, warnings = summarise(manifest, MeshParams(), 497_926)
    assert any("no pass against the photographs" in w for w in warnings)


def test_says_nothing_about_skipping_when_it_did_not_skip(manifest) -> None:
    _, warnings = summarise(manifest, MeshParams(refine=True), 497_926)
    assert not any("no pass against the photographs" in w for w in warnings)


def test_warns_when_refinement_crawled_on_the_processor(manifest) -> None:
    _, warnings = summarise(
        manifest, MeshParams(refine=True), 497_926, timings={"refine": 3600}
    )
    assert any("refine_on_gpu" in w for w in warnings)


def test_warns_when_the_result_is_too_big_to_open(manifest) -> None:
    _, warnings = summarise(
        manifest,
        MeshParams(),
        497_926,
        sizes={"mesh_bytes": 30_000_000, "texture_bytes": 30_000_000},
    )
    assert any("max_texture_size" in w for w in warnings)


def test_metrics_report_the_ratio_worth_sanity_checking(manifest) -> None:
    """Faces per thousand dense points. The cup ran at 675, which is the shape to expect."""
    metrics, _ = summarise(manifest, MeshParams(), 497_926)
    assert metrics["faces_per_thousand_points"] == pytest.approx(674.7, abs=0.5)
