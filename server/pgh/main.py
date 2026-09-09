"""FastAPI application factory.

Run in development with::

    .venv/Scripts/python.exe -m uvicorn pgh.main:app --reload --port 8756 --app-dir server
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from . import config
from .api import artifacts, clips, doctor, events, fs, mask, runs, stages, sync
from .api.serving import serve_file
from .jobs import runner
from .stages import registry as _stage_registry  # noqa: F401 -- registers stages
from .store import store

logger = logging.getLogger("pgh")

WEB_DIST = config.PROJECT_ROOT / "web" / "dist"

# The Vite dev server proxies /api, but keep these open for the case where the
# frontend is opened directly against a different port.
DEV_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = config.get_settings()
    settings.runs_root.mkdir(parents=True, exist_ok=True)
    logger.info("runs root: %s", settings.runs_root)

    if repaired := store.recover():
        logger.warning(
            "marked %d interrupted stage(s) as failed: %s",
            len(repaired),
            ", ".join(repaired),
        )

    runner.start()
    try:
        yield
    finally:
        runner.stop()


def create_app() -> FastAPI:
    app = FastAPI(
        title="Photogrammetry Harness",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=DEV_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def no_stale_api_reads(request: Request, call_next):
        """Never let a JSON read be answered from cache.

        Every GET under /api reports live state -- a stage's progress, a run's
        metrics, what the doctor can see -- and none of them carry a validator, so
        there is nothing for a cache to revalidate against. A browser is free to
        assign its own freshness lifetime to a response with no directive, and a
        cached stage detail would show the previous run's numbers under the current
        run's name.

        Only fills the header in when the handler has not set one, so the deliberate
        choices elsewhere stand: artifacts revalidate (api/serving.py), the SSE stream
        sets its own, and the mask preview is no-store.
        """
        response = await call_next(request)
        if request.url.path.startswith("/api/") and "cache-control" not in response.headers:
            response.headers["Cache-Control"] = "no-store"
        return response

    app.include_router(doctor.router)
    app.include_router(runs.router)
    app.include_router(stages.router)
    app.include_router(events.router)
    app.include_router(clips.router)
    app.include_router(sync.router)
    app.include_router(fs.router)
    app.include_router(artifacts.router)
    app.include_router(mask.router)

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    # Serve the built frontend when it exists; in development Vite serves it instead.
    if (WEB_DIST / "index.html").is_file():
        # /assets is deliberately NOT revalidated: Vite content-hashes those filenames,
        # so a changed file is a different URL and a cached one cannot be wrong.
        # index.html is the opposite -- it is the map from URL to hashed asset, it is
        # rewritten by every build, and a cached copy points at files a later build has
        # already deleted. That is a blank app rather than a stale one.
        app.mount("/assets", StaticFiles(directory=WEB_DIST / "assets"), name="assets")

        @app.get("/{full_path:path}", include_in_schema=False)
        def spa(request: Request, full_path: str) -> Response:
            """Serve the single-page app, falling back to index.html for deep links.

            The router lives in the browser, so a reload on /runs/x/clips arrives here
            as a path with no file behind it. StaticFiles alone would 404; the app has
            to be handed back so its own router can resolve the URL. Real files are
            still served directly.
            """
            # Registered API routes match before this one, so anything under /api
            # arriving here is a genuine miss. Falling back to the app would answer a
            # mistyped endpoint with 200 and a page of HTML, which surfaces later as a
            # JSON parse error a long way from the actual cause.
            if full_path == "api" or full_path.startswith("api/"):
                raise HTTPException(status_code=404, detail=f"no such endpoint: /{full_path}")

            candidate = (WEB_DIST / full_path).resolve()
            if full_path and candidate.is_file():
                try:
                    candidate.relative_to(WEB_DIST.resolve())
                    return serve_file(candidate, request)
                except ValueError:
                    pass  # escaped the dist directory; fall through to index.html
            return serve_file(WEB_DIST / "index.html", request, "text/html")

    return app


app = create_app()
