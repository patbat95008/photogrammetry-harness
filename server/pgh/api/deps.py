"""Shared FastAPI dependencies."""

from __future__ import annotations

import asyncio

from fastapi import HTTPException, Path

from ..store import RunHandle, store


def get_run(run_id: str = Path(...)) -> RunHandle:
    handle = store.get(run_id)
    if handle is None:
        raise HTTPException(status_code=404, detail=f"no such run: {run_id}")
    # Publishing happens on the worker thread but must be delivered on the event
    # loop; binding here means any request path is enough to wire that up.
    try:
        handle.bus.bind_loop(asyncio.get_running_loop())
    except RuntimeError:  # pragma: no cover - no running loop (tests)
        pass
    return handle
