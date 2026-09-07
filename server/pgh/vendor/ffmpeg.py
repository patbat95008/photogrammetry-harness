"""ffmpeg and ffprobe: argv builders and probing.

Everything that knows ffmpeg's command-line grammar lives here, so the stages stay
readable and the awkward bits -- rotation, HDR, frame-exact seeking -- are decided
in exactly one place.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config import TOOL_SPECS, resolve_tool
from ..manifest import ClipProbe
from ..proc import capture

#: A phone advertising 60 fps often delivers 59.94 with the odd dropped frame under
#: thermal load. Beyond this much spread in the inter-frame gaps, naive stride
#: sampling silently drifts, so the clip is flagged as variable frame rate.
VFR_STDEV_RATIO = 0.10

HDR_TRANSFERS = {"arib-std-b67", "smpte2084"}  # HLG and PQ


def _tool(key: str) -> Path:
    spec = next(s for s in TOOL_SPECS if s.key == key)
    path = resolve_tool(spec)
    if path is None:
        raise RuntimeError(f"{spec.label} not found; check the doctor page")
    return path


def ffmpeg_path() -> Path:
    return _tool("ffmpeg")


def ffprobe_path() -> Path:
    return _tool("ffprobe")


# -- probing -----------------------------------------------------------------


def probe_raw(source: Path, cwd: Path) -> dict[str, Any]:
    """Full ffprobe JSON, persisted verbatim as evidence and never re-derived."""
    result = capture(
        [
            ffprobe_path(),
            "-v", "error",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            "-show_entries", "stream_side_data",
            "-i", source,
        ],
        cwd=cwd,
        timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr.strip()[:400]}")
    return json.loads(result.stdout)


def probe_packets(source: Path, cwd: Path) -> list[float]:
    """Presentation timestamps of every video packet, in display order.

    Uses ``-show_packets`` rather than ``-show_frames``: packets carry ``pts_time``
    without decoding, so a 4 GB clip is read in a second or two instead of a minute.
    """
    result = capture(
        [
            ffprobe_path(),
            "-v", "error",
            "-select_streams", "v:0",
            "-show_packets",
            "-show_entries", "packet=pts_time",
            "-print_format", "csv=p=0",
            "-i", source,
        ],
        cwd=cwd,
        timeout=600,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe packet scan failed: {result.stderr.strip()[:400]}")

    times: list[float] = []
    for line in result.stdout.splitlines():
        value = line.strip().rstrip(",")
        if not value or value == "N/A":
            continue
        try:
            times.append(float(value))
        except ValueError:
            continue
    # Packets arrive in decode order when B-frames are present; pts is display time.
    times.sort()
    return times


def _parse_rate(value: str | None) -> float | None:
    """ffprobe frame rates are rationals like '60000/1001'."""
    if not value or value in ("0/0", "N/A"):
        return None
    if "/" in value:
        num, _, den = value.partition("/")
        try:
            denominator = float(den)
            return float(num) / denominator if denominator else None
        except ValueError:
            return None
    try:
        return float(value)
    except ValueError:
        return None


def extract_rotation(stream: dict[str, Any]) -> float:
    """Rotation in ffprobe's convention: counter-clockwise degrees to reach display."""
    for side_data in stream.get("side_data_list") or []:
        if side_data.get("side_data_type") == "Display Matrix":
            rotation = side_data.get("rotation")
            if rotation is not None:
                return float(rotation)
    # Older Android and .mov files carry it as a container tag instead.
    tag = (stream.get("tags") or {}).get("rotate")
    if tag is not None:
        try:
            return -float(tag)  # the tag is clockwise; normalise to ffprobe's sign
        except ValueError:
            pass
    return 0.0


def effective_rotation(probe_rotation: float) -> int:
    """Clockwise degrees the harness must apply to reach upright.

    This is the table people get backwards. ffprobe reports the rotation needed to
    *display* the video, counter-clockwise-positive; a portrait phone clip is
    typically -90, meaning "rotate 90 clockwise to display".
    """
    normalised = round(probe_rotation) % 360
    return {0: 0, 270: 90, 90: 270, 180: 180}.get(normalised, 0)


def rotation_filter(effective: int) -> list[str]:
    """The transpose filter chain for a given clockwise rotation."""
    return {
        0: [],
        90: ["transpose=1"],
        180: ["transpose=1", "transpose=1"],
        270: ["transpose=2"],
    }[effective]


def detect_vfr(pts_times: list[float]) -> tuple[bool, float | None]:
    """Flag variable frame rate, and return the median inter-frame gap."""
    if len(pts_times) < 10:
        return False, None
    deltas = [b - a for a, b in zip(pts_times, pts_times[1:]) if b > a]
    if len(deltas) < 5:
        return False, None
    median = statistics.median(deltas)
    if median <= 0:
        return False, None
    stdev = statistics.pstdev(deltas)
    return (stdev / median) > VFR_STDEV_RATIO, median


@dataclass(slots=True)
class ProbeResult:
    probe: ClipProbe
    raw: dict[str, Any]
    pts_times: list[float]


def probe_clip(source: Path, cwd: Path) -> ProbeResult:
    """Everything the pipeline needs to know about a source clip."""
    raw = probe_raw(source, cwd)
    streams = raw.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    if video is None:
        raise RuntimeError("no video stream found in this file")

    pts_times = probe_packets(source, cwd)
    is_vfr, median_delta = detect_vfr(pts_times)

    rotation = extract_rotation(video)
    effective = effective_rotation(rotation)

    width = int(video.get("width") or 0) or None
    height = int(video.get("height") or 0) or None
    # Report the displayed dimensions, which is what the user sees and what the
    # extracted frames will actually be.
    if effective in (90, 270) and width and height:
        width, height = height, width

    duration = raw.get("format", {}).get("duration")
    transfer = video.get("color_transfer")

    probe = ClipProbe(
        duration_s=float(duration) if duration else (pts_times[-1] if pts_times else None),
        codec_name=video.get("codec_name"),
        pix_fmt=video.get("pix_fmt"),
        width=width,
        height=height,
        avg_frame_rate=_parse_rate(video.get("avg_frame_rate"))
        or (1.0 / median_delta if median_delta else None),
        r_frame_rate=_parse_rate(video.get("r_frame_rate")),
        nb_frames=len(pts_times) or None,
        color_transfer=transfer,
        color_primaries=video.get("color_primaries"),
        color_space=video.get("color_space"),
        rotation=rotation,
        effective_rotation=effective,
        is_hdr=transfer in HDR_TRANSFERS,
        is_vfr_suspected=is_vfr,
        has_audio=any(s.get("codec_type") == "audio" for s in streams),
    )
    return ProbeResult(probe=probe, raw=raw, pts_times=pts_times)


# -- poster frames -----------------------------------------------------------


def extract_poster(
    source: Path, dest: Path, *, at_seconds: float, effective_rot: int, is_hdr: bool,
    max_width: int = 640, cwd: Path | None = None,
) -> None:
    """Grab one representative frame, correctly oriented and tone-mapped.

    Shown side by side per clip in the UI. Rotation bugs must surface here, at clip
    ingest, rather than at the alignment stage where they look like a solver failure.
    """
    filters: list[str] = []
    if is_hdr:
        filters.extend(hdr_tonemap_chain())
    filters.extend(rotation_filter(effective_rot))
    filters.append(f"scale={max_width}:-2:flags=lanczos")

    dest.parent.mkdir(parents=True, exist_ok=True)
    result = capture(
        [
            ffmpeg_path(),
            "-hide_banner", "-nostdin", "-y",
            "-loglevel", "error",
            "-noautorotate",
            "-ss", f"{max(at_seconds, 0):.3f}",
            "-i", source,
            "-map", "0:v:0",
            "-frames:v", "1",
            "-vf", ",".join(filters),
            "-q:v", "3",
            dest,
        ],
        cwd=cwd or dest.parent,
        timeout=120,
    )
    if result.returncode != 0 or not dest.exists():
        raise RuntimeError(f"poster extraction failed: {result.stderr.strip()[:400]}")


def hdr_tonemap_chain() -> list[str]:
    """Convert HLG/PQ to BT.709.

    Phones default to 10-bit HLG or HDR10. Written straight to JPEG the frames come
    out washed out and low contrast: SIFT still works, so alignment succeeds and
    nothing looks broken, but the final texture is grey and lifeless and the cause
    is invisible by then.
    """
    return [
        "zscale=t=linear:npl=100",
        "format=gbrpf32le",
        "zscale=p=bt709",
        "tonemap=tonemap=hable:desat=0",
        "zscale=t=bt709:m=bt709:r=tv",
        "format=yuv420p",
    ]


def version() -> str:
    result = capture([ffmpeg_path(), "-version"], cwd=Path.cwd(), timeout=20)
    first = result.stdout.splitlines()[0] if result.stdout else ""
    return first.replace("ffmpeg version ", "").split()[0] if first else "unknown"
