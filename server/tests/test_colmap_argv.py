"""COLMAP argv builders speak the dialect the binary actually accepts.

COLMAP 4.2 carries two option namespaces at once. Pipeline and GPU options moved to
``FeatureMatching.*`` / ``FeatureExtraction.*``; algorithm tuning stayed on
``SiftMatching.*`` / ``SiftExtraction.*``. Verified against the binary in ``tools/``:
``--FeatureExtraction.max_image_size`` and ``--SiftExtraction.max_num_features`` are
*both* correct at the same time.

That is the trap these tests exist for. A blanket rename in either direction produces
a command line COLMAP rejects, or worse, silently ignores. The builders ask
``registry.colmap_dialect()`` instead of assuming, and the assertions below are
written against both dialects so a builder cannot quietly hard-code one.

The other invariant is logging. glog defaults ``--log_target`` to
``stderr_and_file``, so a builder that forgets to redirect it produces a stage whose
progress never reaches the UI and which therefore looks hung for its whole run.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pgh.vendor import colmap

#: What the doctor reports for the vendored 4.2.0 build.
MODERN = {
    "available": True,
    "feature_matching_ns": True,
    "sift_matching_ns": True,
    "feature_extraction_ns": True,
    "sift_extraction_ns": True,
    "single_camera_per_folder": True,
    "mask_path": True,
    "filter_stationary_matches": True,
}

#: An older build that only understands the Sift* namespaces.
LEGACY = {
    "available": True,
    "feature_matching_ns": False,
    "sift_matching_ns": True,
    "feature_extraction_ns": False,
    "sift_extraction_ns": True,
    "single_camera_per_folder": False,
    "mask_path": False,
    "filter_stationary_matches": False,
}

ALL_DIALECTS = [pytest.param(MODERN, id="modern"), pytest.param(LEGACY, id="legacy")]


@pytest.fixture(autouse=True)
def fake_binary(monkeypatch: pytest.MonkeyPatch) -> None:
    """Build argv without needing COLMAP installed; only the flags are under test."""
    monkeypatch.setattr(colmap, "colmap_path", lambda: Path("colmap.exe"))


def flags(argv: list) -> list[str]:
    return [str(a) for a in argv]


def flag_value(argv: list, name: str) -> str | None:
    items = flags(argv)
    return items[items.index(name) + 1] if name in items else None


def builders(dialect: dict) -> list[list]:
    """Every builder that shells out, so invariants can be asserted across all of them."""
    return [
        colmap.feature_extractor(
            database_path=Path("db"), image_path=Path("img"), dialect=dialect
        ),
        colmap.exhaustive_matcher(database_path=Path("db"), dialect=dialect),
        colmap.sequential_matcher(
            database_path=Path("db"), dialect=dialect, vocab_tree_path=Path("v.bin")
        ),
        colmap.mapper(
            database_path=Path("db"), image_path=Path("img"), output_path=Path("out")
        ),
        colmap.model_converter(
            input_path=Path("in"), output_path=Path("out"), output_type="TXT"
        ),
        colmap.model_analyzer(path=Path("m")),
        colmap.image_undistorter(
            image_path=Path("img"), input_path=Path("in"), output_path=Path("out")
        ),
    ]


@pytest.mark.parametrize("dialect", ALL_DIALECTS)
def test_every_builder_redirects_glog_to_stdout(dialect: dict) -> None:
    """Otherwise progress goes to stderr and a file, and the stage looks hung."""
    for argv in builders(dialect):
        items = flags(argv)
        assert flag_value(argv, "--log_target") == "stdout", items[1]
        assert flag_value(argv, "--log_color") == "0", items[1]


@pytest.mark.parametrize("dialect", ALL_DIALECTS)
def test_booleans_are_passed_as_one_and_zero(dialect: dict) -> None:
    """COLMAP takes 1/0, not bare presence and not 'True'."""
    argv = colmap.feature_extractor(
        database_path=Path("db"),
        image_path=Path("img"),
        dialect=dialect,
        use_gpu=False,
    )
    for item in flags(argv):
        assert item not in ("True", "False"), "python bools must not reach the argv"
    gpu = flag_value(argv, "--FeatureExtraction.use_gpu") or flag_value(
        argv, "--SiftExtraction.use_gpu"
    )
    assert gpu == "0"


def test_modern_dialect_splits_across_both_namespaces() -> None:
    """The whole point: max_image_size moved, max_num_features did not."""
    argv = colmap.feature_extractor(
        database_path=Path("db"),
        image_path=Path("img"),
        dialect=MODERN,
        max_image_size=1600,
        max_num_features=4096,
    )
    assert flag_value(argv, "--FeatureExtraction.max_image_size") == "1600"
    assert flag_value(argv, "--SiftExtraction.max_num_features") == "4096"
    assert "--SiftExtraction.max_image_size" not in flags(argv)


def test_legacy_dialect_keeps_everything_on_sift() -> None:
    argv = colmap.feature_extractor(
        database_path=Path("db"),
        image_path=Path("img"),
        dialect=LEGACY,
        max_image_size=1600,
    )
    assert flag_value(argv, "--SiftExtraction.max_image_size") == "1600"
    assert "--FeatureExtraction.max_image_size" not in flags(argv)


def test_stationary_filter_sits_on_two_view_geometry() -> None:
    """Not on SiftMatching, where it would be accepted-looking and wrong."""
    argv = colmap.exhaustive_matcher(
        database_path=Path("db"), dialect=MODERN, filter_stationary_matches=True
    )
    assert flag_value(argv, "--TwoViewGeometry.filter_stationary_matches") == "1"


def test_unsupported_options_are_omitted_not_guessed() -> None:
    """A build without an option must not be handed it anyway."""
    argv = colmap.exhaustive_matcher(
        database_path=Path("db"), dialect=LEGACY, filter_stationary_matches=True
    )
    assert not any("filter_stationary_matches" in f for f in flags(argv))

    extractor = colmap.feature_extractor(
        database_path=Path("db"),
        image_path=Path("img"),
        dialect=LEGACY,
        single_camera_per_folder=True,
    )
    assert not any("single_camera_per_folder" in f for f in flags(extractor))


def test_one_camera_per_folder_when_the_build_supports_it() -> None:
    argv = colmap.feature_extractor(
        database_path=Path("db"),
        image_path=Path("img"),
        dialect=MODERN,
        single_camera_per_folder=True,
    )
    assert flag_value(argv, "--ImageReader.single_camera_per_folder") == "1"


def test_loop_detection_is_off_without_a_vocabulary_tree() -> None:
    """COLMAP does nothing useful with loop detection on and no tree to consult."""
    argv = colmap.sequential_matcher(
        database_path=Path("db"),
        dialect=MODERN,
        loop_detection=True,
        vocab_tree_path=None,
    )
    assert not any("loop_detection" in f for f in flags(argv))
    assert not any("vocab_tree_path" in f for f in flags(argv))


def test_loop_detection_is_armed_when_a_tree_is_given() -> None:
    argv = colmap.sequential_matcher(
        database_path=Path("db"),
        dialect=MODERN,
        loop_detection=True,
        vocab_tree_path=Path("tree.bin"),
    )
    assert flag_value(argv, "--SequentialMatching.loop_detection") == "1"
    assert flag_value(argv, "--SequentialMatching.vocab_tree_path") == "tree.bin"


def test_masks_are_only_passed_when_the_build_reads_them() -> None:
    modern = colmap.feature_extractor(
        database_path=Path("db"),
        image_path=Path("img"),
        dialect=MODERN,
        mask_path=Path("masks"),
    )
    assert flag_value(modern, "--ImageReader.mask_path") == "masks"

    legacy = colmap.feature_extractor(
        database_path=Path("db"),
        image_path=Path("img"),
        dialect=LEGACY,
        mask_path=Path("masks"),
    )
    assert not any("mask_path" in f for f in flags(legacy))


# -- HANDOVER 6.8: the two undistort passes must agree exactly ---------------
#
# Masks are undistorted in a second pass, and if its geometry differs at all the
# mask lands a pixel or two from the image it belongs to. That is invisible except
# at the boundary, and the boundary is where ears and hair are.

GEOMETRY_FLAGS = (
    "--max_image_size",
    "--blank_pixels",
    "--min_scale",
    "--max_scale",
    "--roi_min_x",
    "--roi_min_y",
    "--roi_max_x",
    "--roi_max_y",
)


def _geometry_flags(argv: list) -> dict[str, str]:
    return {name: flag_value(argv, name) for name in GEOMETRY_FLAGS}


def test_two_passes_from_one_geometry_are_identical_where_it_matters() -> None:
    """The image pass and the mask pass, built from the same object."""
    geometry = colmap.UndistortGeometry(max_image_size=2000, roi_max_x=0.9)

    images = colmap.image_undistorter(
        image_path=Path("img"),
        input_path=Path("model"),
        output_path=Path("dense"),
        geometry=geometry,
    )
    masks = colmap.image_undistorter(
        image_path=Path("mask_farm"),
        input_path=Path("model"),
        output_path=Path("dense_masks"),
        geometry=geometry,
        jpeg_quality=100,
    )

    assert _geometry_flags(images) == _geometry_flags(masks)


def test_every_geometry_flag_is_emitted_even_at_its_default() -> None:
    """Identical-by-omission stays identical only until someone passes one of them.

    The builder used to emit max_image_size alone and rely on the binary's defaults
    for the rest, so two call sites agreed by accident rather than by construction.
    """
    argv = colmap.image_undistorter(
        image_path=Path("img"), input_path=Path("model"), output_path=Path("dense")
    )

    for name in GEOMETRY_FLAGS:
        assert flag_value(argv, name) is not None, f"{name} is left to the binary"


def test_the_geometry_cannot_be_edited_after_it_is_shared() -> None:
    """Both passes hold the same object, so it must not be mutable."""
    import dataclasses

    geometry = colmap.UndistortGeometry()
    with pytest.raises(dataclasses.FrozenInstanceError):
        geometry.max_image_size = 1000  # type: ignore[misc]


def test_jpeg_quality_is_not_part_of_the_geometry() -> None:
    """It changes the encoding, not where a pixel lands, so it may safely differ."""
    geometry = colmap.UndistortGeometry()
    plain = colmap.image_undistorter(
        image_path=Path("img"), input_path=Path("m"), output_path=Path("o"),
        geometry=geometry,
    )
    crisp = colmap.image_undistorter(
        image_path=Path("img"), input_path=Path("m"), output_path=Path("o"),
        geometry=geometry, jpeg_quality=100,
    )

    assert _geometry_flags(plain) == _geometry_flags(crisp)
    assert flag_value(crisp, "--jpeg_quality") == "100"
    assert flag_value(plain, "--jpeg_quality") is None


def test_undistorter_still_names_its_output_type() -> None:
    argv = colmap.image_undistorter(
        image_path=Path("img"), input_path=Path("model"), output_path=Path("dense"),
        geometry=colmap.UndistortGeometry(max_image_size=2000),
    )
    assert flag_value(argv, "--max_image_size") == "2000"
    assert flag_value(argv, "--output_type") == "COLMAP"


# -- model parsing -----------------------------------------------------------


def test_reads_a_text_model(tmp_path: Path) -> None:
    """images.txt puts each image on two lines: pose, then its observations."""
    (tmp_path / "cameras.txt").write_text(
        "# comment\n1 OPENCV 1080 2320 2784.0 2784.0 540.0 1160.0 0.1 -0.2 0.0 0.0\n",
        encoding="utf-8",
    )
    (tmp_path / "images.txt").write_text(
        "# comment\n"
        "1 1.0 0.0 0.0 0.0 1.0 2.0 3.0 1 cup/000000.jpg\n"
        "10.0 20.0 5 30.0 40.0 -1 50.0 60.0 7\n"
        "2 1.0 0.0 0.0 0.0 4.0 5.0 6.0 1 cup/000001.jpg\n"
        "11.0 21.0 -1\n",
        encoding="utf-8",
    )
    (tmp_path / "points3D.txt").write_text(
        "# comment\n"
        "5 1.0 2.0 3.0 200 100 50 0.42 1 0 2 0\n"
        "7 4.0 5.0 6.0 10 20 30 0.84 1 1 2 1 3 2\n",
        encoding="utf-8",
    )

    model = colmap.read_text_model(tmp_path)

    assert len(model.images) == 2
    assert model.images[0].name == "cup/000000.jpg"
    assert model.images[0].num_points == 2, "the -1 observation is not triangulated"
    assert model.images[1].num_points == 0
    assert model.cameras[1].model == "OPENCV"
    assert model.cameras[1].focal_px == pytest.approx(2784.0)
    assert model.num_points3D == 2
    assert model.mean_track_length == pytest.approx(2.5)
    assert model.mean_reprojection_error_px == pytest.approx(0.63)


def test_camera_centre_is_the_inverse_transform() -> None:
    """COLMAP stores camera-from-world; a viewer needs where the camera actually was."""
    image = colmap.RegisteredImage(
        image_id=1,
        name="a.jpg",
        camera_id=1,
        qvec=(1.0, 0.0, 0.0, 0.0),  # identity rotation
        tvec=(1.0, 2.0, 3.0),
    )
    assert image.center() == pytest.approx((-1.0, -2.0, -3.0))


def test_missing_model_files_yield_an_empty_model(tmp_path: Path) -> None:
    model = colmap.read_text_model(tmp_path)
    assert model.images == []
    assert model.num_points3D == 0


def test_parses_analyzer_statistics() -> None:
    text = "\n".join(
        [
            "Cameras: 1",
            "Images: 183",
            "Registered images: 181",
            "Points: 41234",
            "Observations: 210000",
            "Mean track length: 5.0921",
            "Mean observations per image: 1160.22",
            "Mean reprojection error: 0.61234px",
            "Unrelated: skip me",
        ]
    )
    stats = colmap.parse_analyzer(text)
    assert stats["registered_images"] == 181
    assert stats["num_points"] == 41234
    assert stats["mean_track_length"] == pytest.approx(5.0921)
    assert stats["mean_reprojection_error_px"] == pytest.approx(0.61234)
    assert "skip me" not in stats
