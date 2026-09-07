"""FastAPI application factory.

Run in development with::

    .venv/Scripts/python.exe -m uvicorn pgh.main:app --reload --port 8756 --app-dir server
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import config
from .api import artifacts, clips, doctor, events, fs, runs, stages, sync
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

    app.include_router(doctor.router)
    app.include_router(runs.router)
    app.include_router(stages.router)
    app.include_router(events.router)
    app.include_router(clips.router)
    app.include_router(sync.router)
    app.include_router(fs.router)
    app.include_router(artifacts.router)

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    # Serve the built frontend when it exists; in development Vite serves it instead.
    if (WEB_DIST / "index.html").is_file():
        app.mount("/assets", StaticFiles(directory=WEB_DIST / "assets"), name="assets")

        @app.get("/{full_path:path}", include_in_schema=False)
        def spa(full_path: str) -> FileResponse:
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
                    return FileResponse(candidate)
                except ValueError:
                    pass  # escaped the dist directory; fall through to index.html
            return FileResponse(WEB_DIST / "index.html")

    return app


app = create_app()
