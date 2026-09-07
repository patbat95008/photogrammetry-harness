"""Run directories: creation, manifest access, and per-run event buses."""

from __future__ import annotations

import re
import threading
from datetime import date
from pathlib import Path
from typing import Callable

from .config import get_settings
from .events import EventBus
from .manifest import ManifestIO, RunManifest, StageState, utcnow

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def slugify(value: str, *, fallback: str = "run") -> str:
    slug = _SLUG_RE.sub("-", value.strip().lower()).strip("-")
    return slug[:40] or fallback


class RunHandle:
    """One run: its directory, its manifest, and its event bus.

    The manifest is guarded by a lock and mutated only through :meth:`update`, so
    a background job and an HTTP request can never interleave writes. Progress
    updates deliberately do not go through here at all -- they are events only, so
    a running stage never rewrites ``project.json`` on a timer.
    """

    def __init__(self, run_dir: Path) -> None:
        self.dir = run_dir
        self.io = ManifestIO(run_dir)
        self.bus = EventBus(run_dir)
        self._lock = threading.RLock()
        self._manifest: RunManifest | None = None

    @property
    def run_id(self) -> str:
        return self.dir.name

    def load(self) -> RunManifest:
        with self._lock:
            if self._manifest is None:
                self._manifest = self.io.load()
            return self._manifest

    def update(self, mutate: Callable[[RunManifest], None]) -> RunManifest:
        """Mutate and persist the manifest atomically."""
        with self._lock:
            manifest = self.load()
            mutate(manifest)
            self.io.save(manifest)
            return manifest

    # -- standard subdirectories -------------------------------------------

    def path(self, *parts: str) -> Path:
        return self.dir.joinpath(*parts)

    @property
    def logs_dir(self) -> Path:
        return self.path("logs")

    @property
    def partial_dir(self) -> Path:
        return self.path(".partial")

    def stage_scratch(self, stage_id: str) -> Path:
        return self.partial_dir / stage_id


class RunStore:
    """Discovers and creates runs under the configured runs root."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = root or get_settings().runs_root
        self._handles: dict[str, RunHandle] = {}
        self._lock = threading.Lock()

    def _handle_for(self, run_dir: Path) -> RunHandle:
        with self._lock:
            handle = self._handles.get(run_dir.name)
            if handle is None:
                handle = RunHandle(run_dir)
                self._handles[run_dir.name] = handle
            return handle

    def list_ids(self) -> list[str]:
        if not self.root.exists():
            return []
        return sorted(
            (p.name for p in self.root.iterdir() if (p / ManifestIO.FILENAME).exists()),
            reverse=True,
        )

    def get(self, run_id: str) -> RunHandle | None:
        run_dir = self.root / run_id
        # Reject anything that tries to climb out of the runs root.
        if run_dir.parent != self.root or not (run_dir / ManifestIO.FILENAME).exists():
            return None
        return self._handle_for(run_dir)

    def create(self, name: str = "") -> RunHandle:
        """Create a run directory with a short, dated, unique id.

        Kept short on purpose: the dense stage builds paths like
        ``dense/undistorted/stereo/depth_maps/cam_high/000042.jpg.geometric.bin``
        beneath it, and COLMAP and OpenMVS may not honour long-path support.
        """
        self.root.mkdir(parents=True, exist_ok=True)
        stem = f"{date.today():%Y%m%d}-{slugify(name, fallback='run')}"
        run_id, suffix = stem, 1
        while (self.root / run_id).exists():
            suffix += 1
            run_id = f"{stem}-{suffix}"

        run_dir = self.root / run_id
        run_dir.mkdir(parents=True)
        (run_dir / "logs").mkdir(exist_ok=True)

        handle = self._handle_for(run_dir)
        manifest = RunManifest(run_id=run_id, name=name or run_id)
        handle.io.save(manifest)
        handle._manifest = manifest
        return handle

    def delete(self, run_id: str) -> bool:
        import shutil

        handle = self.get(run_id)
        if handle is None:
            return False
        shutil.rmtree(handle.dir, ignore_errors=True)
        with self._lock:
            self._handles.pop(run_id, None)
        return True


    def recover(self) -> list[str]:
        """Repair state left behind by a server that stopped mid-job.

        A stage recorded as RUNNING cannot still be running: the worker lives in this
        process, so a restart means the job died with it. Left alone, the stage would
        show a progress bar forever and refuse to start again. Its scratch directory
        is discarded too -- stages commit by atomic rename, so anything still in
        `.partial/` is by definition incomplete.
        """
        import shutil

        repaired: list[str] = []
        for run_id in self.list_ids():
            handle = self.get(run_id)
            if handle is None:
                continue
            try:
                manifest = handle.io.load()
            except (OSError, ValueError):
                continue

            interrupted = [
                sid
                for sid, record in manifest.stages.items()
                if record.state is StageState.RUNNING
            ]
            if not interrupted:
                continue

            def mutate(m: RunManifest, stages=interrupted) -> None:
                for sid in stages:
                    record = m.stages[sid]
                    record.state = StageState.FAILED
                    record.error = (
                        "interrupted: the harness stopped while this stage was running"
                    )
                    record.finished_at = utcnow()

            handle._manifest = manifest
            handle.update(mutate)
            shutil.rmtree(handle.partial_dir, ignore_errors=True)
            repaired.extend(f"{run_id}/{sid.value}" for sid in interrupted)

        return repaired


store = RunStore()
