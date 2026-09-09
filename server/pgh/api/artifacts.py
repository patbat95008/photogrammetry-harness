"""Serving extracted frames, thumbnails, and the files the later stages produce."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from fastapi.responses import FileResponse

from ..store import RunHandle
from .deps import get_run

router = APIRouter(prefix="/api/runs/{run_id}", tags=["artifacts"])

#: Artifact paths are stable but their CONTENTS are not: re-running a stage rewrites
#: mesh/mesh_textured.glb at the same URL. With no explicit directive a browser falls
#: back to heuristic freshness -- a fraction of the file's age -- and serves the
#: previous version from cache without asking, so a re-run stage shows its old output
#: and the pipeline looks broken rather than stale. "no-cache" does not mean "do not
#: store": it means store it and revalidate every time, which the ETag answers with a
#: bodiless 304. The turntable strip stays cheap and cannot show the wrong model.
ARTIFACT_CACHE_CONTROL = "no-cache"

#: What may be served out of a run directory, and as what. An allowlist rather than
#: a denylist: a run directory also holds a COLMAP database and tens of gigabytes of
#: depth maps, none of which any client has a reason to ask for.
ARTIFACT_MEDIA_TYPES: dict[str, str] = {
    ".ply": "application/octet-stream",
    # A .glb is not self-contained: TextureMesh writes the atlas beside it as a .png and
    # references it by relative URI, which resolves because both land in the same
    # directory. .gltf is deliberately absent -- it also needs a sidecar .bin, and
    # allowing that extension here would expose the dense stage's depth maps.
    ".glb": "model/gltf-binary",
    ".stl": "model/stl",
    ".json": "application/json",
    ".jsonl": "application/x-ndjson",
    ".txt": "text/plain; charset=utf-8",
    ".obj": "text/plain; charset=utf-8",
    ".mtl": "text/plain; charset=utf-8",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
}


def _safe_path(run: RunHandle, relative: str) -> Path:
    """Resolve a client-supplied path inside the run directory, or refuse.

    ``resolve()`` before comparing, so neither ``..`` nor a symlink can walk out.
    """
    target = (run.dir / relative).resolve()
    try:
        target.relative_to(run.dir.resolve())
    except ValueError:
        raise HTTPException(status_code=400, detail="path escapes the run") from None
    return target


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


def _serve(path: Path, media: str, request: Request) -> Response:
    """Serve a file, revalidating rather than re-sending when nothing changed.

    ``no-cache`` on its own would be correct and expensive: the browser asks every
    time, and Starlette answers every time with the whole file, because it sends an
    ETag but does not itself handle ``If-None-Match``. That is 4 MB per look at a mesh
    and 36 PNGs per turn of the turntable. Answering the conditional request with a
    bodiless 304 is what makes "always revalidate" affordable.

    The ETag is taken off the response Starlette built rather than recomputed here, so
    the two cannot drift apart if its derivation ever changes.
    """
    response = FileResponse(
        path,
        media_type=media,
        stat_result=path.stat(),
        headers={"Cache-Control": ARTIFACT_CACHE_CONTROL},
    )
    etag = response.headers.get("etag")
    if etag and etag in [
        tag.strip() for tag in (request.headers.get("if-none-match") or "").split(",")
    ]:
        return Response(
            status_code=304,
            headers={"ETag": etag, "Cache-Control": ARTIFACT_CACHE_CONTROL},
        )
    return response


@router.get("/frames/{group}/{slot}")
def get_frame(
    request: Request,
    group: str,
    slot: int,
    run: RunHandle = Depends(get_run),
    size: Literal["thumb", "full"] = "thumb",
) -> Response:
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
            return _serve(path, media.replace("image/jpg", "image/jpeg"), request)

    raise HTTPException(status_code=404, detail=f"no frame {slot} in {group}")


@router.get("/artifacts/{relative:path}")
def get_artifact(
    request: Request, relative: str, run: RunHandle = Depends(get_run)
) -> Response:
    """Serve a stage output by its run-relative path.

    This is how the point-cloud viewer fetches a preview PLY. ``FileResponse`` honours
    range requests, so a large cloud streams rather than arriving all at once.

    See ``ARTIFACT_CACHE_CONTROL`` for why the revalidation header is not optional.
    """
    target = _safe_path(run, relative)
    media = ARTIFACT_MEDIA_TYPES.get(target.suffix.lower())
    if media is None:
        raise HTTPException(
            status_code=400, detail=f"{target.suffix or 'that file type'} is not served"
        )
    if not target.is_file():
        raise HTTPException(status_code=404, detail=f"no artifact at {relative}")
    return _serve(target, media, request)
