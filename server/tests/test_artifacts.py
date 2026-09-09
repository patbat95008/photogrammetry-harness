"""Serving stage outputs, and the reason they must never be cached blind.

An artifact URL is a stable path whose *contents* change: re-running the mesh stage
rewrites ``mesh/mesh_textured.glb`` in place. With no cache directive a browser falls
back to heuristic freshness and serves the previous version without asking, so a stage
that has just been re-run shows its old output — which reads as the pipeline ignoring
the change rather than as a stale page.

No FastAPI test client here: `starlette.testclient` needs httpx, which is not a
dependency of this project. The helper takes a `Request`, so a hand-built scope is
enough and the tests stay dependency-free like the rest of the suite.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from starlette.requests import Request

from pgh.api.artifacts import ARTIFACT_MEDIA_TYPES
from pgh.api.serving import CACHE_CONTROL as ARTIFACT_CACHE_CONTROL
from pgh.api.serving import serve_file as _serve


def _request(headers: dict[str, str] | None = None) -> Request:
    raw = [(k.lower().encode(), v.encode()) for k, v in (headers or {}).items()]
    return Request({"type": "http", "method": "GET", "headers": raw, "path": "/"})


@pytest.fixture
def artifact(tmp_path: Path) -> Path:
    path = tmp_path / "mesh_textured.glb"
    path.write_bytes(b"first version of the mesh")
    return path


# -- always revalidate --------------------------------------------------------


def test_an_artifact_is_never_served_without_a_cache_directive(artifact):
    """Left unset, the browser invents a freshness lifetime from the file's age."""
    response = _serve(artifact, _request(), "model/gltf-binary")

    assert response.headers["cache-control"] == ARTIFACT_CACHE_CONTROL


def test_no_cache_means_revalidate_rather_than_do_not_store():
    """It is the directive that keeps the turntable cheap AND correct."""
    assert ARTIFACT_CACHE_CONTROL == "no-cache"
    assert "no-store" not in ARTIFACT_CACHE_CONTROL


def test_an_unchanged_artifact_revalidates_to_an_empty_304(artifact):
    """Otherwise 'always revalidate' costs the whole file on every look."""
    etag = _serve(artifact, _request(), "model/gltf-binary").headers["etag"]

    again = _serve(artifact, _request({"If-None-Match": etag}), "model/gltf-binary")

    assert again.status_code == 304
    assert again.body == b""
    assert again.headers["cache-control"] == ARTIFACT_CACHE_CONTROL


def test_a_rewritten_artifact_is_sent_again(artifact):
    """The case that started this: a re-run stage writes the same path.

    If this ever returns 304, the viewer shows the previous mesh and the stage looks
    as though it did nothing.
    """
    stale = _serve(artifact, _request(), "model/gltf-binary").headers["etag"]

    artifact.write_bytes(b"a different mesh entirely, after a re-run with masks")
    response = _serve(artifact, _request({"If-None-Match": stale}), "model/gltf-binary")

    assert response.status_code == 200
    assert response.headers["etag"] != stale


def test_a_conditional_request_listing_several_etags_still_matches(artifact):
    """If-None-Match is a comma-separated list, not a single value."""
    etag = _serve(artifact, _request(), "model/gltf-binary").headers["etag"]

    response = _serve(
        artifact, _request({"If-None-Match": f'"other", {etag}'}), "model/gltf-binary"
    )

    assert response.status_code == 304


def test_no_conditional_header_means_send_the_file(artifact):
    response = _serve(artifact, _request(), "model/gltf-binary")

    assert response.status_code == 200


# -- the allowlist ------------------------------------------------------------


def test_the_mesh_and_its_sidecars_are_all_servable():
    """A GLB needs its atlas, and an OBJ needs its material and texture."""
    for suffix in (".glb", ".png", ".obj", ".mtl", ".jpg", ".ply", ".stl"):
        assert suffix in ARTIFACT_MEDIA_TYPES


def test_gltf_is_deliberately_not_servable():
    """It needs a sidecar .bin, and allowing that would expose the depth maps."""
    assert ".gltf" not in ARTIFACT_MEDIA_TYPES
    assert ".bin" not in ARTIFACT_MEDIA_TYPES


# -- every other surface that could hand back something stale -----------------
#
# The mesh viewer was the symptom; it was never the only place. These pin the rest
# of the app so the same class of bug cannot come back through a different door.


def test_a_media_type_is_optional_so_the_shell_can_be_served_too(artifact):
    """index.html goes through the same helper; FileResponse guesses from the name."""
    response = _serve(artifact, _request())

    assert response.status_code == 200
    assert response.headers["cache-control"] == ARTIFACT_CACHE_CONTROL


def test_every_file_endpoint_uses_the_shared_helper():
    """A new endpoint that reaches for FileResponse directly reintroduces the bug.

    The poster endpoint did exactly that and was missed by the first fix: it is a
    stable URL whose contents change when a clip's offset moves.
    """
    import inspect

    from pgh.api import artifacts, clips
    from pgh import main

    for module in (artifacts, clips, main):
        source = inspect.getsource(module)
        body = "\n".join(
            line for line in source.splitlines() if not line.strip().startswith("#")
        )
        assert "FileResponse(" not in body, (
            f"{module.__name__} builds a FileResponse directly; use serving.serve_file "
            f"so the response revalidates"
        )


def _call(app, path: str) -> tuple[int, dict[str, str]]:
    """Drive the ASGI app directly for one GET, and return (status, headers).

    No test client: starlette's needs httpx, which this project does not depend on.
    The middleware is part of the ASGI stack, so calling the app is what exercises it
    -- asserting that a middleware object exists would pass even if it did nothing.
    """
    import asyncio

    captured: dict[str, object] = {}

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        if message["type"] == "http.response.start":
            captured["status"] = message["status"]
            captured["headers"] = {
                k.decode().lower(): v.decode() for k, v in message["headers"]
            }

    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.1"},
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "root_path": "",
        "headers": [(b"host", b"testserver")],
        "client": ("127.0.0.1", 1234),
        "server": ("127.0.0.1", 8756),
    }
    asyncio.run(app(scope, receive, send))
    return int(captured["status"]), captured["headers"]  # type: ignore[arg-type]


@pytest.mark.parametrize("path", ["/api/health", "/api/runs", "/api/doctor"])
def test_an_api_read_is_never_reusable(path):
    """JSON reads carry no validator, so a cache has nothing to revalidate against.

    A browser is free to invent a freshness lifetime for a response with no directive,
    and a cached stage detail would show a previous run's numbers under the current
    run's name.
    """
    from pgh.main import create_app

    status, headers = _call(create_app(), path)

    assert status == 200
    assert headers.get("cache-control") == "no-store"


def test_a_missing_api_endpoint_is_also_not_cacheable():
    """A 404 that gets cached is a route that stays broken after it is fixed."""
    from pgh.main import create_app

    status, headers = _call(create_app(), "/api/definitely-not-a-route")

    assert status == 404
    assert headers.get("cache-control") == "no-store"


def test_the_middleware_does_not_overwrite_a_deliberate_directive():
    """Artifacts revalidate, the SSE stream sets its own, the preview is no-store."""
    from pgh.api.serving import CACHE_CONTROL

    # An artifact must keep no-cache rather than being flattened to no-store: the
    # whole point there is that the browser may store it and ask.
    assert CACHE_CONTROL == "no-cache"
