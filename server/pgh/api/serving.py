"""Handing files to the browser without handing it a previous version.

**The rule this module exists to enforce:** almost every URL in this app is a stable
path whose contents change. `mesh/mesh_textured.glb` is rewritten by re-running the
mesh stage, a clip's poster is rewritten when its offset moves, and `index.html` is
rewritten by every UI build. A response with no cache directive is not "uncached" — the
browser falls back to *heuristic* freshness, a fraction of the file's age, and serves
what it already has without asking. The stage then appears to have done nothing, which
is a far more confusing failure than a slow page.

So: everything served from here revalidates. ``no-cache`` means "store it, but check
every time" rather than "do not store" — and the check is answered with a bodiless 304,
because Starlette sends an ETag but does not itself handle ``If-None-Match``. Without
that half, "always revalidate" would mean re-sending four megabytes every time somebody
glances at a mesh, and thirty-six PNGs per turn of the turntable.

The one exception is `/assets`, which is left alone deliberately: Vite content-hashes
those filenames, so a changed file is a *different URL* and a cached one can never be
wrong. See HANDOVER 6.31.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import Request, Response
from fastapi.responses import FileResponse

#: "Store it, but revalidate every time." Not ``no-store``: the point is to keep the
#: transfer cheap while making staleness impossible.
CACHE_CONTROL = "no-cache"


def serve_file(path: Path, request: Request, media: str | None = None) -> Response:
    """Serve a file, revalidating rather than re-sending when nothing has changed.

    The ETag is read off the response Starlette built rather than recomputed here, so
    the two cannot drift apart if its derivation ever changes.
    """
    response = FileResponse(
        path,
        media_type=media,
        stat_result=path.stat(),
        headers={"Cache-Control": CACHE_CONTROL},
    )
    etag = response.headers.get("etag")
    if etag and etag in _conditional_tags(request):
        return Response(
            status_code=304, headers={"ETag": etag, "Cache-Control": CACHE_CONTROL}
        )
    return response


def _conditional_tags(request: Request) -> list[str]:
    """``If-None-Match`` is a comma-separated list, not a single value."""
    header = request.headers.get("if-none-match") or ""
    return [tag.strip() for tag in header.split(",") if tag.strip()]
