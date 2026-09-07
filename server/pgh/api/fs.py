"""Server-side file browsing, restricted to the configured ingest roots."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException

from ..config import get_settings
from ..files import is_within_ingest_roots

router = APIRouter(prefix="/api/fs", tags=["fs"])

VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".mkv", ".avi", ".insv", ".mts", ".m2ts"}


@router.get("/roots")
def list_roots() -> list[dict[str, Any]]:
    return [
        {"path": str(root), "exists": root.exists()}
        for root in get_settings().ingest_roots
    ]


@router.get("/browse")
def browse(path: str | None = None) -> dict[str, Any]:
    """List one directory. Refuses anything outside the ingest roots."""
    if path is None:
        roots = [r for r in get_settings().ingest_roots if r.exists()]
        return {
            "path": None,
            "parent": None,
            "directories": [{"name": str(r), "path": str(r)} for r in roots],
            "files": [],
        }

    target = Path(path)
    if not target.is_absolute():
        raise HTTPException(status_code=400, detail="path must be absolute")
    if not is_within_ingest_roots(target):
        raise HTTPException(status_code=403, detail="path is outside the ingest roots")
    if not target.is_dir():
        raise HTTPException(status_code=404, detail=f"not a directory: {target}")

    directories: list[dict[str, Any]] = []
    files: list[dict[str, Any]] = []
    try:
        for entry in sorted(target.iterdir(), key=lambda p: p.name.lower()):
            try:
                if entry.is_dir():
                    directories.append({"name": entry.name, "path": str(entry)})
                elif entry.suffix.lower() in VIDEO_SUFFIXES:
                    files.append(
                        {
                            "name": entry.name,
                            "path": str(entry),
                            "size_bytes": entry.stat().st_size,
                        }
                    )
            except OSError:
                continue  # unreadable entries are skipped, not fatal
    except PermissionError:
        raise HTTPException(status_code=403, detail="permission denied") from None

    parent = str(target.parent) if is_within_ingest_roots(target.parent) else None
    return {"path": str(target), "parent": parent, "directories": directories, "files": files}
