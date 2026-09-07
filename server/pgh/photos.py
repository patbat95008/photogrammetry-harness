"""Reading a folder of photographs as a capture source.

A folder of stills is the other way into this pipeline, and in reconstruction terms
the better one: no rolling shutter, no variable frame rate, no display matrix, no
inter-frame compression, and real EXIF. When a video run produces a bad model, a
photo run is how you tell "the reconstruction chain is wrong" apart from "extraction
is feeding it bad frames".

Two things here matter more than they look:

**Ordering.** Photos are ordered by natural sort, so ``IMG_9.jpg`` precedes
``IMG_10.jpg``. Plain lexical order puts 10 before 9, and since sequential matching
compares neighbours *by index*, a scrambled order silently destroys the assumption
that adjacent indices are adjacent viewpoints.

**EXIF focal length.** COLMAP seeds each camera's focal length from EXIF and falls
back to ``1.2 x max(width, height)`` when it is missing -- a guess that bundle
adjustment can usually recover from, but not always, and not quickly. Anything that
rewrites these images has to carry EXIF across.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, ExifTags

#: Extensions worth offering. HEIC needs pillow-heif, which is not installed; it is
#: listed so the scanner can say so rather than silently ignoring half a shoot.
PHOTO_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
UNSUPPORTED_SUFFIXES = {".heic", ".heif", ".dng", ".cr2", ".nef", ".arw", ".raf"}

_EXIF_TAGS = {v: k for k, v in ExifTags.TAGS.items()}
_ORIENTATION = _EXIF_TAGS.get("Orientation", 274)
_FOCAL_LENGTH = _EXIF_TAGS.get("FocalLength", 37386)
_FOCAL_35MM = _EXIF_TAGS.get("FocalLengthIn35mmFilm", 41989)
_MODEL = _EXIF_TAGS.get("Model", 272)
_MAKE = _EXIF_TAGS.get("Make", 271)

_NUMBER_RE = re.compile(r"(\d+)")


def natural_key(name: str) -> tuple[object, ...]:
    """Sort key that orders embedded numbers numerically: IMG_9 before IMG_10."""
    return tuple(
        int(part) if part.isdigit() else part.lower()
        for part in _NUMBER_RE.split(name)
        if part != ""
    )


@dataclass(slots=True)
class Photo:
    path: Path
    width: int
    height: int
    bytes: int
    #: EXIF orientation tag, 1 when absent or already upright.
    orientation: int = 1
    focal_mm: float | None = None
    focal_35mm: float | None = None
    camera: str = ""


@dataclass(slots=True)
class PhotoSet:
    directory: Path
    photos: list[Photo] = field(default_factory=list)
    #: Files that look like photographs but cannot be opened with what is installed.
    unsupported: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.photos)

    @property
    def needs_reorientation(self) -> bool:
        return any(p.orientation not in (0, 1) for p in self.photos)

    @property
    def mixed_dimensions(self) -> bool:
        return len({(p.width, p.height) for p in self.photos}) > 1

    @property
    def cameras(self) -> list[str]:
        return sorted({p.camera for p in self.photos if p.camera})

    @property
    def with_focal(self) -> int:
        return sum(1 for p in self.photos if p.focal_mm or p.focal_35mm)


def _read_one(path: Path) -> Photo | None:
    try:
        with Image.open(path) as image:
            width, height = image.size
            exif = image.getexif()
    except Exception:
        return None

    make = str(exif.get(_MAKE, "") or "").strip()
    model = str(exif.get(_MODEL, "") or "").strip()
    camera = f"{make} {model}".strip()

    def _num(tag: int) -> float | None:
        value = exif.get(tag)
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    orientation = exif.get(_ORIENTATION, 1)
    try:
        orientation = int(orientation)
    except (TypeError, ValueError):
        orientation = 1

    return Photo(
        path=path,
        width=width,
        height=height,
        bytes=path.stat().st_size,
        orientation=orientation,
        focal_mm=_num(_FOCAL_LENGTH),
        focal_35mm=_num(_FOCAL_35MM),
        camera=camera,
    )


def scan(directory: Path) -> PhotoSet:
    """Read every usable photograph in ``directory``, in natural filename order."""
    result = PhotoSet(directory=directory)
    if not directory.is_dir():
        return result

    entries = sorted(
        (p for p in directory.iterdir() if p.is_file()),
        key=lambda p: natural_key(p.name),
    )
    for entry in entries:
        suffix = entry.suffix.lower()
        if suffix in UNSUPPORTED_SUFFIXES:
            result.unsupported.append(entry.name)
            continue
        if suffix not in PHOTO_SUFFIXES:
            continue
        photo = _read_one(entry)
        if photo is None:
            result.unsupported.append(entry.name)
            continue
        result.photos.append(photo)

    result.warnings = _diagnose(result)
    return result


def _diagnose(photo_set: PhotoSet) -> list[str]:
    """Problems worth raising before hours of reconstruction, in plain sentences."""
    problems: list[str] = []
    count = len(photo_set)

    if photo_set.unsupported:
        shown = ", ".join(photo_set.unsupported[:3])
        more = f" and {len(photo_set.unsupported) - 3} more" if len(photo_set.unsupported) > 3 else ""
        problems.append(
            f"{len(photo_set.unsupported)} file(s) could not be read ({shown}{more}). "
            "RAW and HEIC need a decoder that is not installed here -- export to JPEG "
            "or TIFF first, at full resolution."
        )

    if 0 < count < 20:
        problems.append(
            f"only {count} photographs. Reconstruction needs overlapping views from "
            "many angles; below about 20 the mapper usually cannot triangulate a "
            "coherent model at all."
        )

    if photo_set.mixed_dimensions:
        sizes = sorted({f"{p.width}x{p.height}" for p in photo_set.photos})
        problems.append(
            f"the photographs are not all the same size ({', '.join(sizes[:4])}). "
            "Mixed dimensions usually mean cropping or mixed cameras, and a crop "
            "changes the effective focal length, so one shared camera model no "
            "longer describes them all."
        )

    if count and photo_set.with_focal == 0:
        problems.append(
            "no EXIF focal length on any photograph. COLMAP will fall back to "
            "guessing it from the image size, which converges more slowly and "
            "sometimes not at all. Avoid tools that strip metadata."
        )

    if len(photo_set.cameras) > 1:
        problems.append(
            f"photographs come from more than one camera ({', '.join(photo_set.cameras)}). "
            "Give each camera its own group so they do not share one set of intrinsics."
        )

    if photo_set.needs_reorientation:
        rotated = sum(1 for p in photo_set.photos if p.orientation not in (0, 1))
        problems.append(
            f"{rotated} photograph(s) carry an EXIF rotation flag. They will be "
            "physically rotated during ingest, because COLMAP does not apply that "
            "flag reliably and a sideways subset is hard to spot later."
        )

    return problems
