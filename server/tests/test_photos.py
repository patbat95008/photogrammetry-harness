"""A folder of photographs reads as a capture source.

Two properties matter more than the rest.

**Order.** Sequential matching compares frames by index, so index order has to be
viewpoint order. Lexical sort puts ``IMG_10`` before ``IMG_9``, which scrambles the
sequence in a way nothing downstream can detect -- the reconstruction simply gets
worse for no visible reason.

**EXIF.** COLMAP seeds focal length from EXIF and otherwise guesses it from the image
dimensions. Whatever the ingest does to a photograph, the metadata has to survive it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from pgh import photos


def write_photo(
    path: Path,
    *,
    size: tuple[int, int] = (160, 120),
    orientation: int | None = None,
    focal: float | None = 4.25,
    camera: str | None = "TestCorp Model Q",
    seed: int = 0,
) -> Path:
    rng = np.random.RandomState(seed)
    image = Image.fromarray((rng.rand(size[1], size[0], 3) * 255).astype(np.uint8))
    exif = image.getexif()
    if orientation is not None:
        exif[274] = orientation
    if focal is not None:
        exif[37386] = focal
    if camera is not None:
        make, _, model = camera.partition(" ")
        exif[271] = make
        exif[272] = model
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, quality=92, exif=exif)
    return path


# -- ordering ----------------------------------------------------------------


@pytest.mark.parametrize(
    "names,expected",
    [
        (["IMG_10.jpg", "IMG_9.jpg"], ["IMG_9.jpg", "IMG_10.jpg"]),
        (["a2.jpg", "a10.jpg", "a1.jpg"], ["a1.jpg", "a2.jpg", "a10.jpg"]),
        (["DSC00100.jpg", "DSC00099.jpg"], ["DSC00099.jpg", "DSC00100.jpg"]),
        (["b.jpg", "A.jpg"], ["A.jpg", "b.jpg"]),
    ],
)
def test_natural_order_puts_nine_before_ten(names: list[str], expected: list[str]) -> None:
    assert sorted(names, key=photos.natural_key) == expected


def test_scan_returns_photographs_in_natural_order(tmp_path: Path) -> None:
    for i in (1, 2, 9, 10, 11):
        write_photo(tmp_path / f"IMG_{i}.jpg", seed=i)

    found = photos.scan(tmp_path)
    assert [p.path.name for p in found.photos] == [
        "IMG_1.jpg",
        "IMG_2.jpg",
        "IMG_9.jpg",
        "IMG_10.jpg",
        "IMG_11.jpg",
    ]


# -- probing -----------------------------------------------------------------


def test_reads_dimensions_focal_length_and_camera(tmp_path: Path) -> None:
    write_photo(tmp_path / "a.jpg", size=(320, 240))
    found = photos.scan(tmp_path)

    assert len(found) == 1
    photo = found.photos[0]
    assert (photo.width, photo.height) == (320, 240)
    assert photo.focal_mm == pytest.approx(4.25)
    assert photo.camera == "TestCorp Model Q"
    assert found.with_focal == 1


def test_detects_an_orientation_flag(tmp_path: Path) -> None:
    write_photo(tmp_path / "upright.jpg")
    write_photo(tmp_path / "sideways.jpg", orientation=6, seed=1)

    found = photos.scan(tmp_path)
    assert found.needs_reorientation, "a rotated photograph must be noticed at scan time"


def test_upright_photographs_need_no_reorientation(tmp_path: Path) -> None:
    for i in range(3):
        write_photo(tmp_path / f"p{i}.jpg", seed=i)
    assert not photos.scan(tmp_path).needs_reorientation


def test_detects_mixed_dimensions(tmp_path: Path) -> None:
    write_photo(tmp_path / "a.jpg", size=(320, 240))
    write_photo(tmp_path / "b.jpg", size=(300, 240), seed=1)

    found = photos.scan(tmp_path)
    assert found.mixed_dimensions
    assert any("not all the same size" in w for w in found.warnings)


def test_uniform_dimensions_are_not_flagged(tmp_path: Path) -> None:
    for i in range(3):
        write_photo(tmp_path / f"p{i}.jpg", size=(320, 240), seed=i)
    assert not photos.scan(tmp_path).mixed_dimensions


def test_lists_every_camera_that_contributed(tmp_path: Path) -> None:
    write_photo(tmp_path / "a.jpg", camera="Canon R5")
    write_photo(tmp_path / "b.jpg", camera="Nikon Z6", seed=1)

    found = photos.scan(tmp_path)
    assert found.cameras == ["Canon R5", "Nikon Z6"]
    assert any("more than one camera" in w for w in found.warnings)


# -- diagnostics -------------------------------------------------------------


def test_missing_exif_focal_is_reported(tmp_path: Path) -> None:
    """COLMAP falls back to guessing focal length, which converges worse."""
    for i in range(25):
        write_photo(tmp_path / f"p{i:03d}.jpg", focal=None, camera=None, seed=i)

    found = photos.scan(tmp_path)
    assert found.with_focal == 0
    assert any("no EXIF focal length" in w for w in found.warnings)


def test_a_handful_of_photographs_is_flagged_as_too_few(tmp_path: Path) -> None:
    for i in range(5):
        write_photo(tmp_path / f"p{i}.jpg", seed=i)
    assert any("only 5 photographs" in w for w in photos.scan(tmp_path).warnings)


def test_enough_photographs_is_not_flagged(tmp_path: Path) -> None:
    for i in range(25):
        write_photo(tmp_path / f"p{i:03d}.jpg", seed=i)
    warnings = photos.scan(tmp_path).warnings
    assert not any("photographs. Reconstruction needs" in w for w in warnings)


def test_undecodable_formats_are_named_rather_than_skipped(tmp_path: Path) -> None:
    """A silently ignored half of a shoot is worse than a refusal."""
    write_photo(tmp_path / "good.jpg")
    (tmp_path / "raw1.cr2").write_bytes(b"not really a raw file")
    (tmp_path / "phone.heic").write_bytes(b"nor this")

    found = photos.scan(tmp_path)
    assert len(found) == 1
    assert sorted(found.unsupported) == ["phone.heic", "raw1.cr2"]
    assert any("could not be read" in w for w in found.warnings)


def test_non_image_files_are_ignored_quietly(tmp_path: Path) -> None:
    """A stray notes.txt is not a problem worth reporting."""
    write_photo(tmp_path / "a.jpg")
    (tmp_path / "notes.txt").write_text("shoot log", encoding="utf-8")
    (tmp_path / "capture.mp4").write_bytes(b"video")

    found = photos.scan(tmp_path)
    assert len(found) == 1
    assert found.unsupported == []


def test_a_corrupt_image_is_reported_not_fatal(tmp_path: Path) -> None:
    write_photo(tmp_path / "good.jpg")
    (tmp_path / "truncated.jpg").write_bytes(b"\xff\xd8\xff\xe0 garbage")

    found = photos.scan(tmp_path)
    assert len(found) == 1
    assert "truncated.jpg" in found.unsupported


def test_scanning_a_missing_directory_is_empty_not_an_error(tmp_path: Path) -> None:
    found = photos.scan(tmp_path / "nope")
    assert len(found) == 0
    assert found.warnings == []


def test_subdirectories_are_not_walked(tmp_path: Path) -> None:
    """A photo set is one folder: recursing would sweep in unrelated shoots."""
    write_photo(tmp_path / "a.jpg")
    write_photo(tmp_path / "nested" / "b.jpg", seed=1)

    assert len(photos.scan(tmp_path)) == 1


# -- the stills path must not disturb video runs -----------------------------


def test_a_video_clip_hashes_without_any_photo_keys() -> None:
    """Adding the stills path must not invalidate existing video reconstructions.

    The extract stage declares its clips as external inputs. Adding a key
    unconditionally changes the hash of every video run ever made, marking hours of
    completed COLMAP and OpenMVS work stale for a source that has not changed.
    """
    from pgh.manifest import Clip, SourceIdentity, SourceKind
    from pgh.stages.extract import _clip_inputs

    clip = Clip(
        clip_id="cam_high",
        camera_group="cam_high",
        source_path=r"D:\footage\high.mp4",
        source_identity=SourceIdentity(size_bytes=10, mtime_ns=1, digest="aaa"),
    )
    assert clip.kind is SourceKind.VIDEO

    inputs = _clip_inputs(clip)
    assert set(inputs) == {
        "clip_id",
        "camera_group",
        "segment_id",
        "identity",
        "offset",
        "rotation",
        "is_hdr",
    }, "a video clip must contribute exactly the keys it always has"


def test_a_photo_clip_carries_its_folder_contents(tmp_path: Path) -> None:
    from pgh.manifest import Clip, SourceKind
    from pgh.stages.extract import _clip_inputs

    write_photo(tmp_path / "a.jpg")
    write_photo(tmp_path / "b.jpg", seed=1)
    clip = Clip(
        clip_id="set",
        camera_group="set",
        source_path=str(tmp_path),
        kind=SourceKind.PHOTOS,
    )

    inputs = _clip_inputs(clip)
    assert inputs["kind"] == "photos"
    assert inputs["photos"]["count"] == 2

    # Adding a photograph has to change the hash, or the stage never re-runs.
    before = inputs["photos"]
    write_photo(tmp_path / "c.jpg", seed=2)
    assert _clip_inputs(clip)["photos"] != before
