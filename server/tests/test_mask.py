"""The mask stage, without a GPU and without SAM 2.

``stages/mask.py`` imports ``sam2rt`` and never imports ``sam2`` itself, which is what
lets the whole propagation path be driven here against a fake predictor. The
assertions that matter most are the ones about *seeding*: a chunk seeded from the
wrong frame, or from a mask that has been dilated once per boundary, still produces a
perfectly plausible-looking mask for every frame. Nothing downstream would report it,
and nobody would see it by eye.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pgh.manifest import CaptureMode, RunManifest, StageId
from pgh.stages.mask import (
    MaskParams,
    MaskPrompt,
    MaskStage,
    _prompts_in_chunk,
    _summarise,
    _write_views,
    clean_mask,
    colmap_view_name,
    dilate_mask,
    iou,
    openmvs_view_name,
    plan_chunks,
    to_bytes,
)


# -- chunk planning ----------------------------------------------------------
#
# The overlap is the mechanism that carries a track across a chunk boundary. If it
# is lost, every mask after the first boundary is one frame stale.


def test_chunks_overlap_by_exactly_one_frame():
    chunks = plan_chunks(list(range(240)), 150)
    assert chunks == [list(range(150)), list(range(149, 240))]


@pytest.mark.parametrize("count", [17, 150, 151, 299, 300, 301, 720])
@pytest.mark.parametrize("size", [16, 50, 150])
def test_the_seeded_frame_is_always_index_zero_of_the_next_chunk(count, size):
    """add_new_mask attaches a mask to a frame INDEX, so this has to hold exactly."""
    chunks = plan_chunks(list(range(count)), size)
    for earlier, later in zip(chunks, chunks[1:]):
        assert earlier[-1] == later[0], "the carried mask would land on the wrong frame"


@pytest.mark.parametrize("count", [17, 150, 151, 299, 300, 720])
@pytest.mark.parametrize("size", [16, 50, 150])
def test_every_frame_is_masked_exactly_once_apart_from_the_seams(count, size):
    chunks = plan_chunks(list(range(count)), size)
    seen = [slot for chunk in chunks for slot in chunk]
    assert set(seen) == set(range(count)), "a frame would be left with no mask"
    # Only the seam frames repeat, and each of them exactly twice.
    repeated = {slot for slot in seen if seen.count(slot) > 1}
    assert repeated == {chunk[-1] for chunk in chunks[:-1]}


def test_no_chunk_exceeds_the_requested_size():
    """The chunk size is a memory bound, so overshooting it is not cosmetic."""
    for chunk in plan_chunks(list(range(1000)), 150):
        assert len(chunk) <= 150


def test_a_take_shorter_than_one_chunk_is_a_single_chunk():
    assert plan_chunks(list(range(150)), 150) == [list(range(150))]
    assert plan_chunks([7], 150) == [[7]]


def test_no_frames_means_no_chunks():
    assert plan_chunks([], 150) == []


def test_the_seam_frame_is_the_only_one_that_repeats():
    """It is propagated twice and written once, so the bar counts unique frames."""
    chunks = plan_chunks(list(range(241)), 150)
    seen = [slot for chunk in chunks for slot in chunk]
    assert len(seen) == 241 + (len(chunks) - 1)
    assert len(set(seen)) == 241


# -- the two filename conventions (HANDOVER 6.7) -----------------------------


def test_colmap_appends_png_to_the_whole_filename():
    assert colmap_view_name("000042.jpg") == "000042.jpg.png"


def test_openmvs_replaces_the_extension():
    assert openmvs_view_name("000042.jpg") == "000042.mask.png"


def test_the_two_conventions_differ():
    """They differ by one dot, which is why neither is written inline anywhere."""
    assert colmap_view_name("000042.jpg") != openmvs_view_name("000042.jpg")


def test_colmap_naming_follows_the_image_extension():
    """A run extracted to PNG needs 000042.png.png, not 000042.jpg.png."""
    assert colmap_view_name("000042.png") == "000042.png.png"
    assert openmvs_view_name("000042.png") == "000042.mask.png"


def test_views_are_named_after_the_image_not_the_mask(tmp_path):
    """The mask file is always .png; the name it is filed under is the image's."""
    canonical = tmp_path / "canonical" / "cam_high"
    canonical.mkdir(parents=True)
    (canonical / "000042.png").write_bytes(b"not really a png")
    records = [
        {"camera_group": "cam_high", "slot": 42, "file": "frames/cam_high/000042.jpg"}
    ]

    counts = _write_views(tmp_path, records)

    assert counts == {"colmap": 1, "openmvs": 1}
    assert (tmp_path / "colmap" / "cam_high" / "000042.jpg.png").is_file()
    assert (tmp_path / "openmvs" / "cam_high" / "000042.mask.png").is_file()


def test_a_png_run_gets_png_named_colmap_masks(tmp_path):
    canonical = tmp_path / "canonical" / "cam_high"
    canonical.mkdir(parents=True)
    (canonical / "000042.png").write_bytes(b"x")
    records = [
        {"camera_group": "cam_high", "slot": 42, "file": "frames/cam_high/000042.png"}
    ]

    _write_views(tmp_path, records)

    assert (tmp_path / "colmap" / "cam_high" / "000042.png.png").is_file()


# -- mask arithmetic ---------------------------------------------------------


def test_a_mask_is_written_as_exactly_zero_or_255():
    """Both engines compare against an exact value; a stray 254 is a silent hole."""
    out = to_bytes(np.array([[True, False], [False, True]]))
    assert out.dtype == np.uint8
    assert set(np.unique(out)) == {0, 255}


def test_closing_fills_a_hole_that_sam2_would_have_left():
    mask = np.ones((80, 80), dtype=bool)
    mask[38:42, 38:42] = False
    params = MaskParams(close_holes_px=5, largest_component_only=False)

    assert clean_mask(mask, params).all()


def test_closing_can_be_turned_off():
    mask = np.ones((80, 80), dtype=bool)
    mask[38:42, 38:42] = False
    params = MaskParams(close_holes_px=0, largest_component_only=False)

    assert not clean_mask(mask, params).all()


def test_the_largest_component_survives_and_a_speck_does_not():
    mask = np.zeros((100, 100), dtype=bool)
    mask[10:60, 10:60] = True   # the subject
    mask[90:93, 90:93] = True   # a speck of matching background across the room
    params = MaskParams(close_holes_px=0, largest_component_only=True)

    cleaned = clean_mask(mask, params)

    assert cleaned[10:60, 10:60].all()
    assert not cleaned[90:93, 90:93].any()


def test_dilation_grows_and_erosion_shrinks():
    mask = np.zeros((100, 100), dtype=bool)
    mask[40:60, 40:60] = True

    assert dilate_mask(mask, 3).sum() > mask.sum()
    assert dilate_mask(mask, -3).sum() < mask.sum()
    assert dilate_mask(mask, 0).sum() == mask.sum()


def test_iou_is_one_for_identical_masks_and_none_when_empty():
    mask = np.zeros((10, 10), dtype=bool)
    mask[2:5, 2:5] = True

    assert iou(mask, mask) == 1.0
    assert iou(np.zeros((10, 10), bool), np.zeros((10, 10), bool)) is None
    assert iou(None, mask) is None


# -- prompts -----------------------------------------------------------------


def _lookup(slots, w=1000, h=2000):
    return {s: {"slot": s, "w": w, "h": h, "file": f"frames/g/{s:06d}.jpg"} for s in slots}


def test_a_click_converts_to_pixels_against_that_frames_size():
    params = MaskParams(prompts={"g/5": [MaskPrompt(x=0.25, y=0.5)]})

    found = _prompts_in_chunk(params, "g", [3, 4, 5, 6], _lookup([3, 4, 5, 6]))

    (local, points, labels) = found[0]
    assert local == 2, "the prompt must be attached at its LOCAL index in the chunk"
    assert points.tolist() == [[250.0, 1000.0]]
    assert labels.tolist() == [1]


def test_an_exclude_click_is_labelled_zero():
    params = MaskParams(
        prompts={"g/3": [MaskPrompt(x=0.5, y=0.5), MaskPrompt(x=0.1, y=0.1, include=False)]}
    )

    (_local, _points, labels) = _prompts_in_chunk(params, "g", [3], _lookup([3]))[0]

    assert labels.tolist() == [1, 0]


def test_a_prompt_for_another_camera_group_is_not_applied():
    params = MaskParams(prompts={"other/3": [MaskPrompt(x=0.5, y=0.5)]})

    assert _prompts_in_chunk(params, "g", [3], _lookup([3])) == []


def test_a_prompt_for_a_slot_that_no_longer_exists_is_ignored():
    """Re-extracting at a different rate leaves stale clicks; that is not an error."""
    params = MaskParams(prompts={"g/999": [MaskPrompt(x=0.5, y=0.5)]})

    assert _prompts_in_chunk(params, "g", [3, 4], _lookup([3, 4])) == []


def test_a_prompt_outside_this_chunk_is_left_for_its_own_chunk():
    params = MaskParams(prompts={"g/200": [MaskPrompt(x=0.5, y=0.5)]})

    assert _prompts_in_chunk(params, "g", [0, 1, 2], _lookup([0, 1, 2])) == []


def test_a_malformed_prompt_key_does_not_break_the_run():
    params = MaskParams(prompts={"nonsense": [MaskPrompt(x=0.5, y=0.5)]})

    assert _prompts_in_chunk(params, "g", [0], _lookup([0])) == []


# -- the propagation loop, against a fake predictor ---------------------------


class FakePredictor:
    """Records how it was seeded, which is the thing worth asserting on."""

    def __init__(self, shape=(40, 30)):
        self.shape = shape
        self.seeds: list[tuple[int, np.ndarray]] = []
        self.points: list[int] = []
        self.chunk_sizes: list[int] = []
        self._frames = 0

    def init_state(self, video_path, **kwargs):
        self._frames = len(list(Path(video_path).glob("*.jpg")))
        self.chunk_sizes.append(self._frames)
        return {"frames": self._frames}

    def add_new_mask(self, state, frame_idx, obj_id, mask):
        self.seeds.append((frame_idx, np.array(mask, dtype=bool)))

    def add_new_points_or_box(self, state, frame_idx, obj_id, points, labels):
        self.points.append(frame_idx)

    def propagate_in_video(self, state, **kwargs):
        for index in range(state["frames"]):
            block = np.zeros(self.shape, dtype=bool)
            block[5:20, 5:20] = True
            yield index, [1], _FakeLogits(block)

    def reset_state(self, state):
        return None


class _FakeLogits:
    """Mimics the (objects, 1, H, W) tensor SAM 2 yields, down to .cpu().numpy()."""

    def __init__(self, mask):
        self._mask = mask

    def __getitem__(self, index):
        return self

    def __gt__(self, _other):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self._mask


@pytest.fixture
def masked_run(tmp_path, monkeypatch):
    """A tiny two-chunk run with real frame files, so the stage can be driven."""
    import cv2

    from pgh.stages import mask as mask_module

    run_dir = tmp_path / "run"
    frames_dir = run_dir / "frames" / "g"
    frames_dir.mkdir(parents=True)

    records = []
    for slot in range(20):
        cv2.imwrite(str(frames_dir / f"{slot:06d}.jpg"), np.full((40, 30, 3), 128, np.uint8))
        records.append(
            {"slot": slot, "camera_group": "g", "file": f"frames/g/{slot:06d}.jpg",
             "w": 30, "h": 40}
        )
    (run_dir / "frames" / "frames.jsonl").write_text(
        "\n".join(json.dumps(r) for r in records), encoding="utf-8"
    )

    predictor = FakePredictor()
    monkeypatch.setattr(mask_module.sam2rt, "load_video_predictor",
                        lambda *a, **k: predictor)
    monkeypatch.setattr(mask_module.sam2rt, "sam2_version", lambda: "test")
    return run_dir, predictor


def _run_stage(run_dir, tmp_path, params, manifest):
    import logging
    import threading

    from pgh.stages.base import StageContext

    class _Cancel:
        def raise_if_cancelled(self):
            return None

    ctx = StageContext(
        run_dir=run_dir,
        manifest=manifest,
        params=params,
        logger=logging.getLogger("test"),
        cancel=_Cancel(),
        report=lambda *_a, **_k: None,
        scratch=tmp_path / "scratch",
        proc_cancel=threading.Event(),
    )
    (tmp_path / "scratch").mkdir(parents=True, exist_ok=True)
    return MaskStage().run(ctx)


def test_the_first_chunk_is_seeded_by_clicks_and_never_by_a_mask(masked_run, tmp_path):
    run_dir, predictor = masked_run
    params = MaskParams(chunk_frames=16, prompts={"g/0": [MaskPrompt(x=0.5, y=0.5)]})

    _run_stage(run_dir, tmp_path, params, RunManifest(run_id="test"))

    assert predictor.points == [0], "the click belongs on the first frame"
    # Two chunks over ten frames, so exactly one carry-forward.
    assert len(predictor.seeds) == 1


def test_every_later_chunk_is_seeded_at_frame_zero(masked_run, tmp_path):
    """Seeding at any other index silently shifts the whole track by a frame."""
    run_dir, predictor = masked_run
    params = MaskParams(chunk_frames=16, prompts={"g/0": [MaskPrompt(x=0.5, y=0.5)]})

    _run_stage(run_dir, tmp_path, params, RunManifest(run_id="test"))

    assert predictor.seeds, "later chunks were never seeded at all"
    assert all(frame_idx == 0 for frame_idx, _mask in predictor.seeds)


def test_the_carried_mask_is_cleaned_but_not_dilated(masked_run, tmp_path):
    """Dilation is an output concession; fed back it would grow at every boundary."""
    run_dir, predictor = masked_run
    params = MaskParams(
        chunk_frames=16, dilate_px=3, close_holes_px=0, largest_component_only=False,
        prompts={"g/0": [MaskPrompt(x=0.5, y=0.5)]},
    )

    _run_stage(run_dir, tmp_path, params, RunManifest(run_id="test"))

    block = np.zeros((40, 30), dtype=bool)
    block[5:20, 5:20] = True
    _frame_idx, carried = predictor.seeds[0]
    assert carried.sum() == block.sum(), "the seed was dilated before being carried"


def test_every_frame_gets_a_mask_and_both_engine_views(masked_run, tmp_path):
    run_dir, _predictor = masked_run
    params = MaskParams(chunk_frames=16, prompts={"g/0": [MaskPrompt(x=0.5, y=0.5)]})

    result = _run_stage(run_dir, tmp_path, params, RunManifest(run_id="test"))

    assert result.metrics["frames_masked"] == 20
    assert result.metrics["views_written"] == {"colmap": 20, "openmvs": 20}
    for slot in range(20):
        assert (run_dir / "mask" / "canonical" / "g" / f"{slot:06d}.png").is_file()
        assert (run_dir / "mask" / "colmap" / "g" / f"{slot:06d}.jpg.png").is_file()
        assert (run_dir / "mask" / "openmvs" / "g" / f"{slot:06d}.mask.png").is_file()


def test_the_sidecar_carries_one_row_per_frame_and_no_bulk_in_metrics(masked_run, tmp_path):
    run_dir, _predictor = masked_run
    params = MaskParams(chunk_frames=16, prompts={"g/0": [MaskPrompt(x=0.5, y=0.5)]})

    result = _run_stage(run_dir, tmp_path, params, RunManifest(run_id="test"))

    rows = [
        json.loads(line)
        for line in (run_dir / "mask" / "masks.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert len(rows) == 20
    assert rows[0]["camera_group"] == "g"
    assert "area_fraction" in rows[0] and "iou_prev" in rows[0]
    # project.json is rewritten on every change, so per-frame data stays out of it.
    assert not any(isinstance(v, list) and len(v) > 25 for v in result.metrics.values())


def test_background_clicks_invert_the_written_mask(masked_run, tmp_path):
    """The file is 255-means-keep whichever way the clicks were made."""
    run_dir, _predictor = masked_run
    # Both runs post-process identically, so the only difference is the polarity.
    plain = dict(chunk_frames=16, close_holes_px=0, largest_component_only=False,
                 dilate_px=0, prompts={"g/0": [MaskPrompt(x=0.5, y=0.5)]})
    subject = MaskParams(**plain)
    kept = _run_stage(
        run_dir, tmp_path, subject, RunManifest(run_id="test")
    ).metrics["mean_area_fraction"]

    background = MaskParams(clicks_mark="background", **plain)
    inverted = _run_stage(
        run_dir, tmp_path / "b", background, RunManifest(run_id="test")
    ).metrics["mean_area_fraction"]

    assert inverted == pytest.approx(1.0 - kept, abs=0.05)


# -- preflight ---------------------------------------------------------------


def _manifest_with_groups(groups, frames=241, params=None):
    manifest = RunManifest(run_id="test")
    extract = manifest.stages[StageId.EXTRACT]
    extract.metrics = {"camera_groups": groups, "frame_count": frames}
    manifest.stages[StageId.MASK].params = params or {}
    return manifest


def test_preflight_names_a_camera_group_with_no_clicks(monkeypatch):
    from pgh import sam2rt

    monkeypatch.setattr(sam2rt, "import_sam2", lambda: None)
    monkeypatch.setattr(sam2rt, "resolve_model", lambda key: (Path("x"), "y"))
    manifest = _manifest_with_groups(
        ["cam_high", "cam_eye"],
        params={"prompts": {"cam_high/0": [{"x": 0.5, "y": 0.5}]}},
    )

    problems = MaskStage().preflight(manifest)

    assert any("cam_eye has no click points" in p for p in problems)
    assert not any("cam_high has no click points" in p for p in problems)


def test_preflight_rejects_a_first_click_beyond_the_first_chunk(monkeypatch):
    """Propagation starts at frame 0, so a late-only prompt tracks nothing."""
    from pgh import sam2rt

    monkeypatch.setattr(sam2rt, "import_sam2", lambda: None)
    monkeypatch.setattr(sam2rt, "resolve_model", lambda key: (Path("x"), "y"))
    manifest = _manifest_with_groups(
        ["g"],
        params={"chunk_frames": 150, "prompts": {"g/200": [{"x": 0.5, "y": 0.5}]}},
    )

    problems = MaskStage().preflight(manifest)

    assert any("beyond the first chunk" in p for p in problems)


def test_preflight_passes_with_a_click_on_every_group(monkeypatch):
    from pgh import sam2rt

    monkeypatch.setattr(sam2rt, "import_sam2", lambda: None)
    monkeypatch.setattr(sam2rt, "resolve_model", lambda key: (Path("x"), "y"))
    manifest = _manifest_with_groups(
        ["cam_high", "cam_eye"],
        params={
            "prompts": {
                "cam_high/0": [{"x": 0.5, "y": 0.5}],
                "cam_eye/0": [{"x": 0.5, "y": 0.5}],
            }
        },
    )

    assert MaskStage().preflight(manifest) == []


def test_preflight_reports_the_sam2_import_failure_verbatim(monkeypatch):
    """The shadowing message is the diagnosis; paraphrasing it loses the cause."""
    from pgh import sam2rt

    def boom():
        raise RuntimeError("a directory named 'sam2' is shadowing the installed package")

    monkeypatch.setattr(sam2rt, "import_sam2", boom)
    manifest = _manifest_with_groups(["g"], params={"prompts": {"g/0": [{"x": 0.5, "y": 0.5}]}})

    problems = MaskStage().preflight(manifest)

    assert any("shadowing" in p for p in problems)


# -- fingerprints ------------------------------------------------------------


def test_external_inputs_are_stable_across_calls():
    manifest = _manifest_with_groups(["g"])
    stage = MaskStage()

    assert stage.external_inputs(manifest) == stage.external_inputs(manifest)


def test_external_inputs_carry_nothing_volatile():
    """Free VRAM or a file mtime here would make the stage stale on every page load."""
    inputs = MaskStage().external_inputs(_manifest_with_groups(["g"]))

    assert set(inputs) == {"extract_fingerprint", "frame_count", "camera_groups", "sam2"}


def test_the_overlay_size_is_cosmetic():
    """Changing a preview size must not invalidate a run's masks."""
    field = MaskParams.model_fields["overlay_px"]
    assert (field.json_schema_extra or {}).get("affects_fingerprint") is False


def test_the_clicks_are_hashed():
    """They are the entire input to this stage, so they must affect the fingerprint."""
    field = MaskParams.model_fields["prompts"]
    extra = field.json_schema_extra or {}
    assert extra.get("widget") == "hidden"
    assert extra.get("affects_fingerprint") is not False


def test_the_stage_depends_on_extract_not_select():
    """Masks must survive a select tweak; see the class comment."""
    assert MaskStage().depends_on == [StageId.EXTRACT]


# -- summaries ---------------------------------------------------------------


def _records(count=10, **overrides):
    rows = []
    for slot in range(count):
        row = {
            "slot": slot, "camera_group": "g", "file": f"frames/g/{slot:06d}.jpg",
            "area_fraction": 0.2, "touches_border": [], "empty": False,
            "iou_prev": 0.99, "bbox": [1, 1, 2, 2], "chunk": 0,
            "seeded": False, "prompted": False,
        }
        row.update(overrides)
        rows.append(row)
    return rows


def _summary(records, params=None, mode=CaptureMode.SUBJECT_ROTATES):
    manifest = RunManifest(run_id="test")
    manifest.capture.mode = mode
    return _summarise(
        manifest, records, params or MaskParams(), {"g": [list(range(10))]},
        1.0, 0, {"colmap": len(records), "openmvs": len(records)},
    )


def test_a_healthy_take_produces_no_warnings():
    _metrics, warnings = _summary(_records())
    assert warnings == []


def test_an_empty_mask_names_the_frame_to_re_prompt():
    records = _records()
    records[4]["empty"] = True
    records[4]["area_fraction"] = 0.0

    _metrics, warnings = _summary(records)

    assert any("000004" in w and "lost the subject" in w for w in warnings)


def test_a_runaway_mask_says_the_clicks_probably_hit_the_room():
    _metrics, warnings = _summary(_records(area_fraction=0.95))

    assert any("landed on the room" in w for w in warnings)


def test_a_tiny_mask_warns_about_too_few_features():
    _metrics, warnings = _summary(_records(area_fraction=0.005, empty=False))

    assert any("almost no area" in w for w in warnings)


def test_a_border_touching_mask_says_the_subject_was_out_of_shot():
    _metrics, warnings = _summary(_records(touches_border=["left"]))

    assert any("partly out of shot" in w for w in warnings)


def test_an_abrupt_shape_change_is_reported_as_a_jump():
    records = _records()
    records[6]["iou_prev"] = 0.1

    _metrics, warnings = _summary(records)

    assert any("jumped rather than followed" in w for w in warnings)


def test_masking_an_orbit_warns_that_it_makes_alignment_harder():
    _metrics, warnings = _summary(_records(), mode=CaptureMode.CAMERA_ORBITS)

    assert any("rigid with the subject" in w for w in warnings)


def test_masking_a_chair_spin_does_not_warn():
    _metrics, warnings = _summary(_records(), mode=CaptureMode.SUBJECT_ROTATES)

    assert not any("rigid with the subject" in w for w in warnings)
