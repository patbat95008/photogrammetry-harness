"""Server-sent events: live progress and log lines for a run.

SSE rather than WebSockets because the traffic is one-directional and reconnection
matters: the browser sends ``Last-Event-ID`` automatically after a dropped
connection, and the bus replays exactly what was missed. Refreshing the page
mid-job resumes cleanly.
"""

from __future__ import annotations

import asyncio
from typing import AsyncIterator

from fastapi import APIRouter, Depends, Header, Request
from fastapi.responses import StreamingResponse

from ..store import RunHandle
from .deps import get_run

router = APIRouter(prefix="/api/runs/{run_id}", tags=["events"])

#: Comment frames keep proxies and browsers from timing out an idle stream.
HEARTBEAT_S = 15.0


@router.get("/events")
async def stream_events(
    request: Request,
    run: RunHandle = Depends(get_run),
    last_event_id: str | None = Header(default=None, alias="Last-Event-ID"),
) -> StreamingResponse:
    run.bus.bind_loop(asyncio.get_running_loop())

    since: int | None = None
    if last_event_id is not None:
        try:
            since = int(last_event_id)
        except ValueError:
            since = None

    async def generate() -> AsyncIterator[str]:
        yield ": connected\n\n"
        subscription = run.bus.subscribe(since_seq=since)
        try:
            while True:
                try:
                    event = await asyncio.wait_for(
                        subscription.__anext__(), timeout=HEARTBEAT_S
                    )
                except asyncio.TimeoutError:
                    if await request.is_disconnected():
                        break
                    yield ": ping\n\n"
                    continue
                except StopAsyncIteration:  # pragma: no cover
                    break
                yield event.to_sse()
        finally:
            await subscription.aclose()

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            # Vite's dev proxy and most reverse proxies buffer by default, which
            # would hold events until the stream closed.
            "X-Accel-Buffering": "no",
        },
    )
