"""Reading point clouds out of the engines, and writing previews the browser can hold.

COLMAP and OpenMVS both emit PLY, but not the same PLY: the dense cloud carries
normals, per-point view counts and sometimes confidence, in whatever order the
version that wrote it chose. A dense cup orbit is tens of millions of points and
several hundred megabytes -- fetching that into a browser tab is not a viewer, it is
a hang.

So each stage writes a **preview** beside its full-resolution output: binary
little-endian, exactly ``x y z`` float32 plus ``red green blue`` uchar, nothing else,
decimated to a cap. One fixed layout means three.js's stock ``PLYLoader`` parses it
with no bespoke reader and no guessing at property order, and the full file stays on
disk for anyone who wants it.

Decimation is a uniform stride rather than a random sample: strided points stay
evenly spread over the surface, where a random draw of the same size visibly clumps.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

#: PLY scalar type names mapped to numpy, including the aliases the spec allows.
_PLY_TYPES: dict[str, str] = {
    "char": "i1", "int8": "i1",
    "uchar": "u1", "uint8": "u1",
    "short": "i2", "int16": "i2",
    "ushort": "u2", "uint16": "u2",
    "int": "i4", "int32": "i4",
    "uint": "u4", "uint32": "u4",
    "float": "f4", "float32": "f4",
    "double": "f8", "float64": "f8",
}

_COLOR_NAMES = (("red", "green", "blue"), ("r", "g", "b"), ("diffuse_red", "diffuse_green", "diffuse_blue"))


@dataclass(slots=True)
class PointCloud:
    #: (N, 3) float32 positions.
    xyz: np.ndarray
    #: (N, 3) uint8 colours. Mid-grey where the source carried none.
    rgb: np.ndarray

    def __len__(self) -> int:
        return int(self.xyz.shape[0])


class PlyError(RuntimeError):
    """A PLY we cannot read. Raised with what was expected and what was found."""


@dataclass(slots=True)
class Property:
    """One vertex property. ``count_type`` is set only for variable-length lists."""

    name: str
    dtype: str
    count_type: str | None = None

    @property
    def is_list(self) -> bool:
        return self.count_type is not None


@dataclass(slots=True)
class Header:
    vertex_count: int
    properties: list[Property]
    fmt: str
    data_offset: int
    text: str

    @property
    def has_lists(self) -> bool:
        return any(p.is_list for p in self.properties)

    @property
    def names(self) -> list[str]:
        return [p.name for p in self.properties]


def _parse_header(fh) -> Header:
    """Read the header, describing the vertex element.

    Later elements (faces, edges) are ignored: a point cloud has none, and a mesh's
    vertices are still the first block.

    List properties are described rather than rejected. OpenMVS attaches a per-point
    list of the views that saw it, which makes the record length vary from point to
    point -- so the reader needs to know that before it decides how to read.
    """
    if fh.readline().strip() != b"ply":
        raise PlyError("not a PLY file: missing the 'ply' magic line")

    fmt = ""
    counts: list[tuple[str, int]] = []
    properties: list[Property] = []
    current = ""
    lines: list[str] = ["ply"]

    while True:
        raw = fh.readline()
        if not raw:
            raise PlyError("PLY header ended without 'end_header'")
        line = raw.strip().decode("ascii", errors="replace")
        lines.append(line)
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "format":
            fmt = parts[1]
        elif parts[0] == "element":
            current = parts[1]
            counts.append((parts[1], int(parts[2])))
        elif parts[0] == "property" and current == "vertex":
            if parts[1] == "list":
                count_type = _PLY_TYPES.get(parts[2])
                item_type = _PLY_TYPES.get(parts[3])
                if count_type is None or item_type is None:
                    raise PlyError(f"unsupported PLY list property: {line}")
                properties.append(
                    Property(name=parts[4], dtype=item_type, count_type=count_type)
                )
            else:
                dtype = _PLY_TYPES.get(parts[1])
                if dtype is None:
                    raise PlyError(f"unsupported PLY property type: {parts[1]}")
                properties.append(Property(name=parts[2], dtype=dtype))
        elif parts[0] == "end_header":
            break

    vertex_count = next((n for name, n in counts if name == "vertex"), 0)
    return Header(
        vertex_count=vertex_count,
        properties=properties,
        fmt=fmt,
        data_offset=fh.tell(),
        text="\n".join(lines),
    )


def read_header_text(path: Path) -> str:
    """The header as written, for logging when a cloud cannot be read."""
    with path.open("rb") as fh:
        return _parse_header(fh).text


def _colour_triple(names: list[str]) -> tuple[str, str, str] | None:
    for triple in _COLOR_NAMES:
        if all(c in names for c in triple):
            return triple
    return None


def _as_bytes(channel: np.ndarray) -> np.ndarray:
    """Colour may arrive as 0-1 floats; truncating those would render everything black."""
    if channel.dtype.kind == "f":
        channel = np.clip(channel * 255.0, 0, 255)
    return channel.astype(np.uint8)


def read_point_cloud(path: Path, *, max_points: int | None = None) -> PointCloud:
    """Read the vertices of a PLY, keeping only position and colour.

    ``max_points`` applies a uniform stride, so a forty-million-point cloud costs one
    pass rather than forty million points of RAM.
    """
    with path.open("rb") as fh:
        header = _parse_header(fh)

    if header.vertex_count == 0:
        return PointCloud(np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint8))
    if header.fmt != "binary_little_endian":
        raise PlyError(
            f"expected binary_little_endian PLY, found {header.fmt!r}. Both COLMAP "
            "and OpenMVS write binary; an ASCII file here means something re-saved it."
        )

    for axis in ("x", "y", "z"):
        if axis not in header.names:
            raise PlyError(f"PLY vertex element has no {axis!r} property")

    if header.has_lists:
        return _read_variable_stride(path, header, max_points)
    return _read_fixed_stride(path, header, max_points)


def _read_fixed_stride(
    path: Path, header: Header, max_points: int | None
) -> PointCloud:
    """Every record the same length, so numpy can map the file directly."""
    dtype = np.dtype([(p.name, "<" + p.dtype) for p in header.properties])
    data = np.memmap(
        path, dtype=dtype, mode="r", offset=header.data_offset,
        shape=(header.vertex_count,),
    )

    stride = 1
    if max_points is not None and max_points > 0 and header.vertex_count > max_points:
        stride = int(np.ceil(header.vertex_count / max_points))
    view = data[::stride]

    xyz = np.stack(
        [np.asarray(view["x"]), np.asarray(view["y"]), np.asarray(view["z"])], axis=1
    ).astype(np.float32)

    triple = _colour_triple(header.names)
    if triple is None:
        rgb = np.full((xyz.shape[0], 3), 160, dtype=np.uint8)
    else:
        rgb = np.stack([_as_bytes(np.asarray(view[c])) for c in triple], axis=1)

    del data
    return PointCloud(np.ascontiguousarray(xyz), np.ascontiguousarray(rgb))


def _read_variable_stride(
    path: Path, header: Header, max_points: int | None
) -> PointCloud:
    """Records of differing length, because a list property is in the way.

    OpenMVS attaches the list of views that saw each point, so the record length
    depends on how many cameras happened to see it. Nothing can be memory-mapped and
    every record has to be walked to find where the next one starts -- but the values
    we want are scalars, so only the lists need skipping rather than decoding.
    """
    import struct

    fixed: list[tuple[str, str, int]] = []
    for prop in header.properties:
        size = np.dtype(prop.dtype).itemsize
        fixed.append((prop.name, prop.dtype, size))

    triple = _colour_triple(header.names)
    wanted = {"x", "y", "z"} | (set(triple) if triple else set())

    stride = 1
    if max_points is not None and max_points > 0 and header.vertex_count > max_points:
        stride = int(np.ceil(header.vertex_count / max_points))

    struct_codes = {
        "i1": "b", "u1": "B", "i2": "h", "u2": "H",
        "i4": "i", "u4": "I", "f4": "f", "f8": "d",
    }

    blob = path.read_bytes()
    pos = header.data_offset
    size_of = len(blob)

    kept_xyz: list[tuple[float, float, float]] = []
    kept_rgb: list[tuple[int, int, int]] = []

    for index in range(header.vertex_count):
        if pos >= size_of:
            break
        values: dict[str, float] = {}
        keep = index % stride == 0
        for prop, (name, dtype, size) in zip(header.properties, fixed):
            if prop.is_list:
                count_size = np.dtype(prop.count_type).itemsize
                count = int.from_bytes(blob[pos : pos + count_size], "little")
                pos += count_size + count * size
                continue
            if keep and name in wanted:
                values[name] = struct.unpack_from(
                    "<" + struct_codes[dtype], blob, pos
                )[0]
            pos += size
        if keep:
            kept_xyz.append((values["x"], values["y"], values["z"]))
            if triple:
                kept_rgb.append(
                    tuple(int(values[c]) for c in triple)  # type: ignore[arg-type]
                )

    xyz = np.asarray(kept_xyz, dtype=np.float32).reshape(-1, 3)
    if triple:
        # Whether colour needs scaling is a fact about the file, not about the array
        # this loop happened to build: everything collected here is a Python float,
        # so inspecting the array's dtype would rescale integer colour to white.
        colour_prop = next(p for p in header.properties if p.name == triple[0])
        raw = np.asarray(kept_rgb, dtype=np.float64).reshape(-1, 3)
        if np.dtype(colour_prop.dtype).kind == "f":
            raw = np.clip(raw * 255.0, 0, 255)
        rgb = raw.astype(np.uint8)
    else:
        rgb = np.full((xyz.shape[0], 3), 160, dtype=np.uint8)

    return PointCloud(np.ascontiguousarray(xyz), np.ascontiguousarray(rgb))


def write_point_cloud(path: Path, cloud: PointCloud) -> int:
    """Write the fixed preview layout. Returns the number of points written."""
    n = len(cloud)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"comment written by the photogrammetry harness\n"
        f"element vertex {n}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    ).encode("ascii")

    record = np.zeros(
        n,
        dtype=np.dtype(
            [
                ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                ("red", "u1"), ("green", "u1"), ("blue", "u1"),
            ]
        ),
    )
    if n:
        record["x"] = cloud.xyz[:, 0]
        record["y"] = cloud.xyz[:, 1]
        record["z"] = cloud.xyz[:, 2]
        record["red"] = cloud.rgb[:, 0]
        record["green"] = cloud.rgb[:, 1]
        record["blue"] = cloud.rgb[:, 2]

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        fh.write(header)
        fh.write(record.tobytes())
    return n


def write_preview(source: Path, target: Path, *, max_points: int) -> int:
    """Decimate an engine's PLY into the preview layout. Returns points written."""
    return write_point_cloud(target, read_point_cloud(source, max_points=max_points))


def count_vertices(path: Path) -> int:
    """Read just the vertex count out of a PLY header."""
    with path.open("rb") as fh:
        return _parse_header(fh).vertex_count
