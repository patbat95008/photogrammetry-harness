"""Automatic alignment of simultaneously-recorded clips."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from ..manifest import RunManifest
from ..store import RunHandle
from ..sync import MIN_CONFIDENCE, align_clips
from .deps import get_run

router = APIRouter(prefix="/api/runs/{run_id}/sync", tags=["sync"])


class AutoSyncRequest(BaseModel):
    segment_id: str = "seg0"
    #: Defaults to the first enabled clip in the segment; it becomes the zero point.
    reference_clip_id: str | None = None


@router.post("/auto")
def auto_sync(body: AutoSyncRequest, run: RunHandle = Depends(get_run)) -> dict[str, Any]:
    manifest = run.load()
    clips = manifest.enabled_clips(body.segment_id)
    if len(clips) < 2:
        raise HTTPException(
            status_code=400,
            detail="need at least two enabled clips in the segment to sync",
        )

    reference = (
        manifest.clip(body.reference_clip_id) if body.reference_clip_id else clips[0]
    )
    if reference is None or reference.segment_id != body.segment_id:
        raise HTTPException(status_code=404, detail="reference clip not in this segment")

    without_audio = [c.clip_id for c in clips if not c.probe.has_audio]
    if without_audio:
        raise HTTPException(
            status_code=422,
            detail=(
                f"these clips have no audio track, so they cannot be synced "
                f"automatically: {', '.join(without_audio)}"
            ),
        )

    probe_dir = run.path("probe")
    probe_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, Any] = {}
    warnings: list[str] = []

    for clip in clips:
        if clip.clip_id == reference.clip_id:
            results[clip.clip_id] = {
                "offset_s": 0.0,
                "confidence": None,
                "reference": True,
            }
            continue
        try:
            result = align_clips(
                Path(reference.source_path), Path(clip.source_path), probe_dir
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from None

        results[clip.clip_id] = {
            "offset_s": round(result.offset_s, 4),
            "confidence": round(result.confidence, 1),
            "peak": round(result.peak, 3),
            "trustworthy": result.trustworthy,
            "reference": False,
        }
        if not result.trustworthy:
            warnings.append(
                f"{clip.clip_id}: confidence {result.confidence:.0f} is below "
                f"{MIN_CONFIDENCE:.0f}. Check the offset by eye, or clap sharply at "
                f"the start of both takes next time."
            )

    def mutate(m: RunManifest) -> None:
        for clip_id, info in results.items():
            target = m.clip(clip_id)
            if target is None:
                continue
            target.time_offset_s = info["offset_s"]
            target.sync_confidence = info.get("confidence")
            target.sync_method = "reference" if info["reference"] else "audio-xcorr"

    run.update(mutate)
    run.bus.publish("sync.updated", segment=body.segment_id, results=results)

    return {
        "reference_clip_id": reference.clip_id,
        "results": results,
        "warnings": warnings,
    }
