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

from pgh.api.artifacts import ARTIFACT_CACHE_CONTROL, ARTIFACT_MEDIA_TYPES, _serve


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
    response = _serve(artifact, "model/gltf-binary", _request())

    assert response.headers["cache-control"] == ARTIFACT_CACHE_CONTROL


def test_no_cache_means_revalidate_rather_than_do_not_store():
    """It is the directive that keeps the turntable cheap AND correct."""
    assert ARTIFACT_CACHE_CONTROL == "no-cache"
    assert "no-store" not in ARTIFACT_CACHE_CONTROL


def test_an_unchanged_artifact_revalidates_to_an_empty_304(artifact):
    """Otherwise 'always revalidate' costs the whole file on every look."""
    etag = _serve(artifact, "model/gltf-binary", _request()).headers["etag"]

    again = _serve(artifact, "model/gltf-binary", _request({"If-None-Match": etag}))

    assert again.status_code == 304
    assert again.body == b""
    assert again.headers["cache-control"] == ARTIFACT_CACHE_CONTROL


def test_a_rewritten_artifact_is_sent_again(artifact):
    """The case that started this: a re-run stage writes the same path.

    If this ever returns 304, the viewer shows the previous mesh and the stage looks
    as though it did nothing.
    """
    stale = _serve(artifact, "model/gltf-binary", _request()).headers["etag"]

    artifact.write_bytes(b"a different mesh entirely, after a re-run with masks")
    response = _serve(artifact, "model/gltf-binary", _request({"If-None-Match": stale}))

    assert response.status_code == 200
    assert response.headers["etag"] != stale


def test_a_conditional_request_listing_several_etags_still_matches(artifact):
    """If-None-Match is a comma-separated list, not a single value."""
    etag = _serve(artifact, "model/gltf-binary", _request()).headers["etag"]

    response = _serve(
        artifact, "model/gltf-binary", _request({"If-None-Match": f'"other", {etag}'})
    )

    assert response.status_code == 304


def test_no_conditional_header_means_send_the_file(artifact):
    response = _serve(artifact, "model/gltf-binary", _request())

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
