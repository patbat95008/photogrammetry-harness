"""Clip ingest: add a source video or photo set by server-side path, and probe it."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from .. import photos as photo_scan
from ..files import is_within_ingest_roots, source_identity
from ..manifest import (
    Clip,
    PhotoSetProbe,
    RunManifest,
    Segment,
    SegmentKind,
    SourceKind,
)
from ..store import RunHandle, slugify
from ..vendor import ffmpeg
from .deps import get_run

router = APIRouter(prefix="/api/runs/{run_id}", tags=["clips"])


class AddClipRequest(BaseModel):
    #: An absolute path on this machine. Not an upload: these files are multi-GB,
    #: and pushing one through the browser to a server on the same disk is absurd.
    source_path: str
    #: A video file, or a directory of photographs. Left unset it is inferred from
    #: whether source_path names a file or a folder, which is what the UI relies on.
    kind: SourceKind | None = None
    role: str = "cam"
    camera_group: str = ""
    segment_id: str = "seg0"
    segment_kind: SegmentKind = SegmentKind.RIG
    segment_label: str = ""


class UpdateClipRequest(BaseModel):
    role: str | None = None
    camera_group: str | None = None
    segment_id: str | None = None
    time_offset_s: float | None = None
    enabled: bool | None = None


@router.post("/clips", status_code=201)
def add_clip(body: AddClipRequest, run: RunHandle = Depends(get_run)) -> dict[str, Any]:
    source = Path(body.source_path)
    if not source.is_absolute():
        raise HTTPException(status_code=400, detail="source_path must be absolute")
    if not source.exists():
        raise HTTPException(status_code=404, detail=f"no such path: {source}")
    if not is_within_ingest_roots(source):
        raise HTTPException(
            status_code=403,
            detail="path is outside the configured ingest roots",
        )

    kind = body.kind or (
        SourceKind.PHOTOS if source.is_dir() else SourceKind.VIDEO
    )
    if kind is SourceKind.PHOTOS:
        return _add_photo_set(body, source, run)
    if not source.is_file():
        raise HTTPException(status_code=404, detail=f"not a file: {source}")

    probe_dir = run.path("probe")
    probe_dir.mkdir(parents=True, exist_ok=True)

    try:
        result = ffmpeg.probe_clip(source, cwd=probe_dir)
    except (RuntimeError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None

    manifest = run.load()
    clip_id = _unique_clip_id(manifest, body.role or source.stem)

    # Evidence, kept verbatim: the harness never re-derives what ffprobe said.
    raw_path = probe_dir / f"{clip_id}.ffprobe.json"
    raw_path.write_text(json.dumps(result.raw, indent=2), encoding="utf-8")
    packets_path = probe_dir / f"{clip_id}.packets.json"
    packets_path.write_text(json.dumps(result.pts_times), encoding="utf-8")

    probe = result.probe
    probe.raw_probe_path = f"probe/{raw_path.name}"
    probe.packets_path = f"probe/{packets_path.name}"

    clip = Clip(
        clip_id=clip_id,
        camera_group=slugify(body.camera_group) or clip_id,
        role=body.role,
        segment_id=body.segment_id,
        source_path=str(source),
        source_identity=source_identity(source),
        probe=probe,
    )

    def mutate(m: RunManifest) -> None:
        m.clips.append(clip)
        segment = m.segment(body.segment_id)
        if segment is None:
            segment = Segment(
                segment_id=body.segment_id,
                label=body.segment_label or body.segment_id,
                kind=body.segment_kind,
            )
            m.segments.append(segment)
        segment.clip_ids.append(clip.clip_id)

    run.update(mutate)

    _make_poster(run, clip)
    run.bus.publish("clip.added", clip_id=clip.clip_id)
    return clip.model_dump(mode="json")


def _add_photo_set(
    body: AddClipRequest, source: Path, run: RunHandle
) -> dict[str, Any]:
    """Register a folder of photographs as a clip.

    Deliberately not probed with ffprobe: there is no container, no timeline and no
    audio to sync against. What matters instead is how many images there are, whether
    they agree on size, and whether EXIF survived -- so that is what gets recorded.
    """
    photo_set = photo_scan.scan(source)
    if not photo_set.photos:
        detail = f"no readable photographs in {source}"
        if photo_set.unsupported:
            detail += (
                f" ({len(photo_set.unsupported)} file(s) need a decoder that is not "
                "installed -- export RAW or HEIC to JPEG first)"
            )
        raise HTTPException(status_code=422, detail=detail)

    manifest = run.load()
    clip_id = _unique_clip_id(manifest, body.role or source.name)
    first = photo_set.photos[0]

    clip = Clip(
        clip_id=clip_id,
        camera_group=slugify(body.camera_group) or clip_id,
        role=body.role,
        segment_id=body.segment_id,
        source_path=str(source),
        kind=SourceKind.PHOTOS,
        photos=PhotoSetProbe(
            count=len(photo_set),
            width=first.width,
            height=first.height,
            mixed_dimensions=photo_set.mixed_dimensions,
            needs_reorientation=photo_set.needs_reorientation,
            with_focal_length=photo_set.with_focal,
            focal_mm=first.focal_mm,
            focal_35mm=first.focal_35mm,
            cameras=photo_set.cameras,
            unsupported=photo_set.unsupported,
            bytes_total=sum(p.bytes for p in photo_set.photos),
        ),
    )

    def mutate(m: RunManifest) -> None:
        m.clips.append(clip)
        segment = m.segment(body.segment_id)
        if segment is None:
            segment = Segment(
                segment_id=body.segment_id,
                label=body.segment_label or body.segment_id,
                # A photo set is one pass by one camera, whatever the caller said:
                # there is no shared clock to make it simultaneous with anything.
                kind=SegmentKind.SINGLE,
            )
            m.segments.append(segment)
        segment.clip_ids.append(clip.clip_id)

    run.update(mutate)

    poster = run.path("probe", f"{clip_id}.poster.jpg")
    poster.parent.mkdir(parents=True, exist_ok=True)
    try:
        from PIL import Image, ImageOps

        with Image.open(first.path) as image:
            upright = ImageOps.exif_transpose(image).convert("RGB")
            upright.thumbnail((960, 960))
            upright.save(poster, quality=85)
    except Exception:  # a missing poster is cosmetic, never worth failing an add
        pass

    run.bus.publish("clip.added", clip_id=clip.clip_id)
    return clip.model_dump(mode="json")


@router.patch("/clips/{clip_id}")
def update_clip(
    clip_id: str, body: UpdateClipRequest, run: RunHandle = Depends(get_run)
) -> dict[str, Any]:
    manifest = run.load()
    if manifest.clip(clip_id) is None:
        raise HTTPException(status_code=404, detail=f"no such clip: {clip_id}")

    def mutate(m: RunManifest) -> None:
        clip = m.clip(clip_id)
        assert clip is not None
        if body.role is not None:
            clip.role = body.role
        if body.camera_group is not None:
            clip.camera_group = slugify(body.camera_group) or clip.clip_id
        if body.time_offset_s is not None:
            clip.time_offset_s = body.time_offset_s
            clip.sync_method = "manual"
        if body.enabled is not None:
            clip.enabled = body.enabled
        if body.segment_id is not None and body.segment_id != clip.segment_id:
            for segment in m.segments:
                if clip_id in segment.clip_ids:
                    segment.clip_ids.remove(clip_id)
            clip.segment_id = body.segment_id
            target = m.segment(body.segment_id)
            if target is None:
                target = Segment(segment_id=body.segment_id, label=body.segment_id)
                m.segments.append(target)
            target.clip_ids.append(clip_id)

    run.update(mutate)
    run.bus.publish("clip.updated", clip_id=clip_id)
    return run.load().clip(clip_id).model_dump(mode="json")  # type: ignore[union-attr]


@router.delete("/clips/{clip_id}", status_code=204)
def delete_clip(clip_id: str, run: RunHandle = Depends(get_run)) -> None:
    if run.load().clip(clip_id) is None:
        raise HTTPException(status_code=404, detail=f"no such clip: {clip_id}")

    def mutate(m: RunManifest) -> None:
        m.clips = [c for c in m.clips if c.clip_id != clip_id]
        for segment in m.segments:
            if clip_id in segment.clip_ids:
                segment.clip_ids.remove(clip_id)
        m.segments = [s for s in m.segments if s.clip_ids]

    run.update(mutate)
    run.bus.publish("clip.removed", clip_id=clip_id)


@router.get("/clips/{clip_id}/poster")
def get_poster(clip_id: str, run: RunHandle = Depends(get_run)) -> FileResponse:
    poster = run.path("probe", f"{clip_id}.poster.jpg")
    if not poster.exists():
        clip = run.load().clip(clip_id)
        if clip is None:
            raise HTTPException(status_code=404, detail=f"no such clip: {clip_id}")
        try:
            _make_poster(run, clip)
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from None
    return FileResponse(poster, media_type="image/jpeg")


# -- helpers -----------------------------------------------------------------


def _unique_clip_id(manifest: RunManifest, base: str) -> str:
    stem = slugify(base, fallback="clip")
    existing = {c.clip_id for c in manifest.clips}
    if stem not in existing:
        return stem
    n = 2
    while f"{stem}-{n}" in existing:
        n += 1
    return f"{stem}-{n}"


def _make_poster(run: RunHandle, clip: Clip) -> None:
    """A frame from a quarter of the way in -- past any lens-cap or setup wobble."""
    duration = clip.probe.duration_s or 0
    ffmpeg.extract_poster(
        Path(clip.source_path),
        run.path("probe", f"{clip.clip_id}.poster.jpg"),
        at_seconds=duration * 0.25,
        effective_rot=clip.probe.effective_rotation,
        is_hdr=clip.probe.is_hdr,
        cwd=run.path("probe"),
    )
