"""Append-only event log plus an in-process bus, backing the SSE stream.

Events carry a monotonic sequence number so a browser that reconnects can send
``Last-Event-ID`` and be handed exactly what it missed. That is the whole reason
this is SSE rather than a WebSocket: reconnect-with-replay is free, and progress
only ever flows server to client.

Two kinds of traffic are deliberately treated differently:

* **Lifecycle and progress** events are persisted to ``events.ndjson``. They are
  small, and replaying them is what lets a refreshed page rebuild its state.
* **Log lines** are *not* persisted here. They already go to ``logs/<stage>.log``,
  and a dense-stereo run emits tens of thousands of them; duplicating that into the
  event log would bloat it for no gain. They are broadcast live only, and history
  is fetched from the log file.
"""

from __future__ import annotations

import asyncio
import json
import threading
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator

#: Log lines are broadcast but never written to events.ndjson.
EPHEMERAL_TYPES = frozenset({"stage.log"})

#: How many recent events to keep in memory for cheap replay without file IO.
RING_SIZE = 2000


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


@dataclass(slots=True)
class Event:
    seq: int
    ts: str
    type: str
    payload: dict[str, Any] = field(default_factory=dict)

    def to_sse(self) -> str:
        data = json.dumps({"seq": self.seq, "ts": self.ts, "type": self.type, **self.payload})
        return f"id: {self.seq}\nevent: {self.type}\ndata: {data}\n\n"


class EventBus:
    """One bus per run. Publishing is thread-safe; subscribing is asyncio."""

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.path = run_dir / "events.ndjson"
        self._lock = threading.Lock()
        self._seq = 0
        self._ring: deque[Event] = deque(maxlen=RING_SIZE)
        self._subscribers: set[asyncio.Queue[Event]] = set()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._restore_seq()

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Record the event loop that worker-thread publishes must hand off to."""
        self._loop = loop

    def _restore_seq(self) -> None:
        """Continue numbering across a server restart."""
        if not self.path.exists():
            return
        try:
            with self.path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        self._seq = max(self._seq, json.loads(line).get("seq", 0))
        except (OSError, json.JSONDecodeError):
            pass

    # -- publishing ---------------------------------------------------------

    def publish(self, type_: str, **payload: Any) -> Event:
        """Emit an event. Safe to call from the worker thread."""
        with self._lock:
            self._seq += 1
            event = Event(seq=self._seq, ts=_utcnow(), type=type_, payload=payload)
            if type_ not in EPHEMERAL_TYPES:
                # The ring is the replay cache, so it must mirror what was persisted.
                # Ephemeral events are broadcast live but never replayed -- otherwise
                # a reconnect would see different history depending on cache state.
                self._ring.append(event)
                self._append(event)
            subscribers = list(self._subscribers)

        loop = self._loop
        for queue in subscribers:
            if loop is not None and loop.is_running():
                loop.call_soon_threadsafe(self._offer, queue, event)
            else:  # pragma: no cover - only when publishing before startup
                self._offer(queue, event)
        return event

    @staticmethod
    def _offer(queue: asyncio.Queue[Event], event: Event) -> None:
        try:
            queue.put_nowait(event)
        except asyncio.QueueFull:  # pragma: no cover - slow consumer
            pass

    def _append(self, event: Event) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(event)) + "\n")

    # -- replay and subscription -------------------------------------------

    def replay(self, since_seq: int) -> list[Event]:
        """Every persisted event after ``since_seq``, in order."""
        with self._lock:
            if self._ring and self._ring[0].seq <= since_seq + 1:
                return [e for e in self._ring if e.seq > since_seq]

        if not self.path.exists():
            return []
        events: list[Event] = []
        try:
            with self.path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    raw = json.loads(line)
                    if raw.get("seq", 0) > since_seq:
                        events.append(Event(**raw))
        except (OSError, json.JSONDecodeError, TypeError):
            pass
        return events

    async def subscribe(self, since_seq: int | None = None) -> AsyncIterator[Event]:
        """Yield missed events, then live ones, until the client disconnects."""
        queue: asyncio.Queue[Event] = asyncio.Queue(maxsize=1000)
        with self._lock:
            self._subscribers.add(queue)
        try:
            if since_seq is not None:
                for event in self.replay(since_seq):
                    yield event
            while True:
                yield await queue.get()
        finally:
            with self._lock:
                self._subscribers.discard(queue)

    @property
    def current_seq(self) -> int:
        with self._lock:
            return self._seq
