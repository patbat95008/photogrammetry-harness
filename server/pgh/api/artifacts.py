"""Serving extracted frames and their thumbnails."""

from __future__ import annotations

import json
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse

from ..store import RunHandle
from .deps import get_run

router = APIRouter(prefix="/api/runs/{run_id}", tags=["artifacts"])


@router.get("/frames")
def list_frames(
    run: RunHandle = Depends(get_run),
    group: str | None = Query(None, description="Restrict to one camera group"),
    limit: int = Query(2000, ge=1, le=20000),
    offset: int = Query(0, ge=0),
) -> dict[str, Any]:
    """The per-frame index written by the extract stage.

    Read from the ``frames.jsonl`` sidecar rather than the manifest: a few thousand
    frame records have no business in a document that is rewritten on every change.
    """
    index = run.path("frames", "frames.jsonl")
    if not index.exists():
        return {"frames": [], "total": 0, "groups": []}

    records = [
        json.loads(line) for line in index.read_text(encoding="utf-8").splitlines() if line
    ]
    groups = sorted({r["camera_group"] for r in records})
    if group:
        records = [r for r in records if r["camera_group"] == group]

    return {
        "frames": records[offset : offset + limit],
        "total": len(records),
        "groups": groups,
    }


@router.get("/frames/{group}/{slot}")
def get_frame(
    group: str,
    slot: int,
    run: RunHandle = Depends(get_run),
    size: Literal["thumb", "full"] = "thumb",
) -> FileResponse:
    directory = "thumbs" if size == "thumb" else "frames"
    suffix = ".webp" if size == "thumb" else None

    base = run.path(directory, group)
    # Resolve through the run directory so a crafted group name cannot escape it.
    try:
        base.resolve().relative_to(run.dir.resolve())
    except ValueError:
        raise HTTPException(status_code=400, detail="invalid group") from None

    if suffix:
        candidates = [base / f"{slot:06d}{suffix}"]
    else:
        candidates = [base / f"{slot:06d}{ext}" for ext in (".jpg", ".png")]

    for path in candidates:
        if path.exists():
            media = "image/webp" if path.suffix == ".webp" else f"image/{path.suffix[1:]}"
            return FileResponse(path, media_type=media.replace("image/jpg", "image/jpeg"))

    raise HTTPException(status_code=404, detail=f"no frame {slot} in {group}")
