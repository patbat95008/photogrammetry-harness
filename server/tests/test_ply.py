"""Point clouds survive the trip from an engine to the browser.

The property this protects: whatever COLMAP or OpenMVS wrote, the preview the viewer
loads has exactly ``x y z red green blue`` in that order and no more. Both engines
attach extra per-point data -- normals, view counts, confidence -- in whatever order
the version that wrote it chose, and three.js's PLYLoader is given a fixed layout
precisely so it never has to guess.

The other property is that decimation is a cap, not a suggestion: a dense cup orbit
is tens of millions of points, and a viewer that tries to fetch all of them is a hung
browser tab.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pgh.ply import (
    PlyError,
    PointCloud,
    count_vertices,
    read_header_text,
    read_point_cloud,
    write_point_cloud,
    write_preview,
)


def make_cloud(n: int, seed: int = 0) -> PointCloud:
    rng = np.random.RandomState(seed)
    return PointCloud(
        xyz=rng.rand(n, 3).astype(np.float32),
        rgb=(rng.rand(n, 3) * 255).astype(np.uint8),
    )


def write_raw(path: Path, properties: list[tuple[str, str]], rows: np.ndarray) -> None:
    """Write a PLY with an arbitrary property layout, as the engines do."""
    # numpy's .descr writes byte order even for single-byte types ("|u1"), which is
    # not what a PLY header calls them.
    type_names = {"f4": "float", "u1": "uchar", "u4": "uint", "f8": "double"}
    header = ["ply", "format binary_little_endian 1.0", f"element vertex {len(rows)}"]
    header += [f"property {type_names[t.lstrip('<>|=')]} {n}" for n, t in properties]
    header.append("end_header\n")
    path.write_bytes("\n".join(header).encode("ascii") + rows.tobytes())


def test_round_trip_preserves_positions_and_colours(tmp_path: Path) -> None:
    cloud = make_cloud(500)
    path = tmp_path / "cloud.ply"
    write_point_cloud(path, cloud)

    back = read_point_cloud(path)
    assert len(back) == 500
    assert np.allclose(back.xyz, cloud.xyz), "positions must survive the round trip"
    assert (back.rgb == cloud.rgb).all(), "colours must survive the round trip"


def test_header_vertex_count_matches_body(tmp_path: Path) -> None:
    path = tmp_path / "cloud.ply"
    write_point_cloud(path, make_cloud(37))
    assert count_vertices(path) == 37


def test_extra_properties_are_ignored_not_misread(tmp_path: Path) -> None:
    """The OpenMVS shape: normals between the coordinates and the colours.

    Reading these by fixed offset rather than by name would silently return normals
    as positions -- a cloud that looks plausible and is wrong.
    """
    n = 64
    dtype = np.dtype(
        [
            ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
            ("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4"),
            ("red", "u1"), ("green", "u1"), ("blue", "u1"), ("views", "u1"),
        ]
    )
    rows = np.zeros(n, dtype=dtype)
    cloud = make_cloud(n, seed=3)
    for i, axis in enumerate("xyz"):
        rows[axis] = cloud.xyz[:, i]
    for i, channel in enumerate(("red", "green", "blue")):
        rows[channel] = cloud.rgb[:, i]
    rows["nx"] = 99.0  # would be read as a position if offsets were assumed

    path = tmp_path / "dense.ply"
    write_raw(path, [(n_, t) for n_, t in dtype.descr], rows)

    back = read_point_cloud(path)
    assert np.allclose(back.xyz, cloud.xyz)
    assert (back.rgb == cloud.rgb).all()


def test_cloud_without_colour_reads_as_grey(tmp_path: Path) -> None:
    dtype = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4")])
    rows = np.zeros(8, dtype=dtype)
    path = tmp_path / "plain.ply"
    write_raw(path, [(n, t) for n, t in dtype.descr], rows)

    back = read_point_cloud(path)
    assert back.rgb.shape == (8, 3), "a colourless cloud must still yield colours"
    assert (back.rgb == 160).all()


def test_float_colours_are_scaled_to_bytes(tmp_path: Path) -> None:
    """Some writers store colour as 0-1 floats; truncating those yields black."""
    dtype = np.dtype(
        [("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
         ("red", "<f4"), ("green", "<f4"), ("blue", "<f4")]
    )
    rows = np.zeros(4, dtype=dtype)
    rows["red"] = 1.0
    rows["green"] = 0.5
    rows["blue"] = 0.0
    path = tmp_path / "float_colour.ply"
    write_raw(path, [(n, t) for n, t in dtype.descr], rows)

    back = read_point_cloud(path)
    assert back.rgb[0][0] == 255
    assert 120 <= back.rgb[0][1] <= 130
    assert back.rgb[0][2] == 0


@pytest.mark.parametrize("cap", [1, 10, 250, 999])
def test_decimation_never_exceeds_the_cap(tmp_path: Path, cap: int) -> None:
    source = tmp_path / "big.ply"
    write_point_cloud(source, make_cloud(1000, seed=1))

    target = tmp_path / "preview.ply"
    written = write_preview(source, target, max_points=cap)

    assert written <= cap, f"asked for at most {cap} points, wrote {written}"
    assert count_vertices(target) == written


def test_decimation_leaves_a_small_cloud_alone(tmp_path: Path) -> None:
    source = tmp_path / "small.ply"
    write_point_cloud(source, make_cloud(120, seed=2))
    target = tmp_path / "preview.ply"
    assert write_preview(source, target, max_points=5000) == 120


def test_decimation_samples_the_whole_cloud(tmp_path: Path) -> None:
    """A stride, not a prefix: taking the first N points would crop the object."""
    n = 1000
    xyz = np.zeros((n, 3), dtype=np.float32)
    xyz[:, 0] = np.linspace(0.0, 100.0, n)  # spread along one axis
    source = tmp_path / "line.ply"
    write_point_cloud(source, PointCloud(xyz, np.zeros((n, 3), np.uint8)))

    target = tmp_path / "preview.ply"
    write_preview(source, target, max_points=50)
    kept = read_point_cloud(target)

    assert kept.xyz[:, 0].max() > 95.0, "the far end of the cloud must be represented"
    assert kept.xyz[:, 0].min() < 5.0, "the near end of the cloud must be represented"


def test_empty_cloud_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "empty.ply"
    write_point_cloud(path, PointCloud(np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint8)))
    assert count_vertices(path) == 0
    assert len(read_point_cloud(path)) == 0


def test_ascii_ply_is_refused_with_an_explanation(tmp_path: Path) -> None:
    """Silently returning nothing here would look like an empty reconstruction."""
    path = tmp_path / "ascii.ply"
    path.write_bytes(
        b"ply\nformat ascii 1.0\nelement vertex 1\n"
        b"property float x\nproperty float y\nproperty float z\nend_header\n0 0 0\n"
    )
    with pytest.raises(PlyError, match="binary_little_endian"):
        read_point_cloud(path)


def test_non_ply_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "notaply.ply"
    path.write_bytes(b"this is not a point cloud")
    with pytest.raises(PlyError, match="magic"):
        read_point_cloud(path)


# -- variable-length records -------------------------------------------------
#
# OpenMVS attaches the list of views that saw each point, so records differ in
# length and nothing can be memory-mapped. This is what the dense cup orbit actually
# produced, and reading it as a fixed stride returns garbage rather than failing.


def write_with_view_lists(path: Path, cloud: PointCloud, views: list[int]) -> None:
    """The OpenMVS dense layout: xyz, normals, rgb, then a per-point list of views."""
    import struct

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {len(cloud)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property float nx\nproperty float ny\nproperty float nz\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "property list uchar int views\n"
        "end_header\n"
    ).encode("ascii")

    body = bytearray()
    for i in range(len(cloud)):
        body += struct.pack("<fff", *cloud.xyz[i])
        body += struct.pack("<fff", 0.0, 1.0, 0.0)
        body += struct.pack("<BBB", *cloud.rgb[i])
        body += struct.pack("<B", views[i])
        body += b"".join(struct.pack("<i", v) for v in range(views[i]))
    path.write_bytes(header + bytes(body))


def test_reads_points_past_a_variable_length_list(tmp_path: Path) -> None:
    cloud = make_cloud(200, seed=5)
    # Deliberately uneven: a constant list length would still work by accident.
    views = [(i % 7) + 1 for i in range(200)]
    path = tmp_path / "scene_dense.ply"
    write_with_view_lists(path, cloud, views)

    back = read_point_cloud(path)
    assert len(back) == 200, "every point must be found despite the varying stride"
    assert np.allclose(back.xyz, cloud.xyz, atol=1e-6)
    assert (back.rgb == cloud.rgb).all()


def test_variable_length_records_respect_the_cap(tmp_path: Path) -> None:
    cloud = make_cloud(500, seed=6)
    path = tmp_path / "scene_dense.ply"
    write_with_view_lists(path, cloud, [(i % 5) + 1 for i in range(500)])

    target = tmp_path / "preview.ply"
    written = write_preview(path, target, max_points=100)
    assert written <= 100
    assert count_vertices(target) == written

    # And the preview is the plain layout, whatever the source looked like.
    kept = read_point_cloud(target)
    assert len(kept) == written


def test_header_text_is_available_for_diagnostics(tmp_path: Path) -> None:
    """When a cloud cannot be read, the header is what says why."""
    cloud = make_cloud(4, seed=7)
    path = tmp_path / "scene_dense.ply"
    write_with_view_lists(path, cloud, [1, 2, 3, 4])
    text = read_header_text(path)
    assert "property list uchar int views" in text
    assert text.startswith("ply")


def test_reads_the_real_openmvs_dense_layout(tmp_path: Path) -> None:
    """The exact header OpenMVS 2.4.0 wrote for the cup orbit.

    Two things here defeat any offset-based reader: colour sits *before* the normals
    rather than after, and there are *two* variable-length lists, not one. Working by
    property name is what makes this survivable.
    """
    import struct

    n = 128
    cloud = make_cloud(n, seed=11)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float32 x\nproperty float32 y\nproperty float32 z\n"
        "property uint8 red\nproperty uint8 green\nproperty uint8 blue\n"
        "property float32 nx\nproperty float32 ny\nproperty float32 nz\n"
        "property list uint8 uint32 view_indices\n"
        "property list uint8 float32 view_weights\n"
        "end_header\n"
    ).encode("ascii")

    body = bytearray()
    for i in range(n):
        views = (i % 6) + 2
        body += struct.pack("<fff", *cloud.xyz[i])
        body += struct.pack("<BBB", *cloud.rgb[i])
        body += struct.pack("<fff", 0.0, 0.0, 1.0)
        body += struct.pack("<B", views)
        body += b"".join(struct.pack("<I", v) for v in range(views))
        body += struct.pack("<B", views)
        body += b"".join(struct.pack("<f", 0.5) for _ in range(views))

    path = tmp_path / "scene_dense.ply"
    path.write_bytes(header + bytes(body))

    assert count_vertices(path) == n
    back = read_point_cloud(path)
    assert len(back) == n
    assert np.allclose(back.xyz, cloud.xyz, atol=1e-6)
    assert (back.rgb == cloud.rgb).all(), "colour must not be read as normals"
