"""Run CRUD and stage status."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from ..fingerprint import evaluate
from ..jobs import runner
from ..manifest import CaptureMode, RunManifest
from ..stages.registry import registry
from ..store import RunHandle, store
from .deps import get_run

router = APIRouter(prefix="/api/runs", tags=["runs"])


class CreateRunRequest(BaseModel):
    name: str = ""


class OrientationRequest(BaseModel):
    flip_x: bool


class UpdateRunRequest(BaseModel):
    name: str | None = None
    capture_mode: CaptureMode | None = None
    baseline_mm: float | None = None
    revolutions: float | None = None
    notes: str | None = None


def _directory_bytes(path: Path) -> int:
    total = 0
    for entry in path.rglob("*"):
        try:
            if entry.is_file():
                total += entry.stat().st_size
        except OSError:
            continue
    return total


def _disk_usage(run: RunHandle) -> dict[str, int]:
    """Bytes per top-level artifact directory.

    Surfaced because the dense stage produces tens of gigabytes of depth maps that
    are useless once fusion has run, and nobody deletes what they cannot see.
    """
    usage: dict[str, int] = {}
    for name in ("frames", "thumbs", "masks", "probe", "sparse", "dense", "mesh", "export"):
        directory = run.path(name)
        if directory.is_dir():
            usage[name] = _directory_bytes(directory)
    return usage


def _run_summary(manifest: RunManifest) -> dict[str, Any]:
    evaluations = evaluate(manifest, registry)
    return {
        "run_id": manifest.run_id,
        "name": manifest.name,
        "created_at": manifest.created_at,
        "updated_at": manifest.updated_at,
        "clip_count": len(manifest.clips),
        "total_slots": manifest.timeline.total_slots,
        "stages": {
            sid.value: {
                "state": ev.state.value,
                "stale_reason": ev.stale_reason,
                "blocked_by": [b.value for b in ev.blocked_by],
                "implemented": sid in registry.implemented(),
            }
            for sid, ev in evaluations.items()
        },
    }


@router.get("")
def list_runs() -> list[dict[str, Any]]:
    summaries = []
    for run_id in store.list_ids():
        handle = store.get(run_id)
        if handle is None:
            continue
        try:
            summaries.append(_run_summary(handle.load()))
        except Exception as exc:  # a corrupt run must not break the whole list
            summaries.append({"run_id": run_id, "error": str(exc)})
    return summaries


@router.post("", status_code=201)
def create_run(body: CreateRunRequest) -> dict[str, Any]:
    handle = store.create(body.name)
    return _run_summary(handle.load())


@router.get("/{run_id}")
def get_run_detail(run: RunHandle = Depends(get_run)) -> dict[str, Any]:
    manifest = run.load()
    evaluations = evaluate(manifest, registry)
    return {
        "manifest": manifest.model_dump(mode="json"),
        "evaluation": {
            sid.value: {
                "state": ev.state.value,
                "stale_reason": ev.stale_reason,
                "blocked_by": [b.value for b in ev.blocked_by],
                "runnable": ev.runnable,
                "implemented": sid in registry.implemented(),
                "fingerprint": ev.computed_fingerprint,
            }
            for sid, ev in evaluations.items()
        },
        "active_jobs": [j for j in runner.active() if j["run_id"] == manifest.run_id],
        "event_seq": run.bus.current_seq,
        "disk_usage": _disk_usage(run),
    }


@router.patch("/{run_id}")
def update_run(body: UpdateRunRequest, run: RunHandle = Depends(get_run)) -> dict[str, Any]:
    def mutate(manifest: RunManifest) -> None:
        if body.name is not None:
            manifest.name = body.name
        if body.capture_mode is not None:
            manifest.capture.mode = body.capture_mode
        if body.baseline_mm is not None:
            manifest.capture.baseline_mm = body.baseline_mm
        if body.revolutions is not None:
            manifest.capture.revolutions = body.revolutions
        if body.notes is not None:
            manifest.capture.notes = body.notes

    manifest = run.update(mutate)
    run.bus.publish("run.updated")
    return _run_summary(manifest)


@router.get("/{run_id}/orientation")
def get_orientation(run: RunHandle = Depends(get_run)) -> dict[str, bool]:
    """Which way up this run's model is.

    Its own endpoint rather than a field read off ``GET /api/runs/{run_id}``, because
    that one walks the run directory for disk usage -- tens of gigabytes of depth maps
    on a finished run -- and the cloud viewer asks for this on every mount.
    """
    return {"flip_x": run.load().capture.flip_x}


@router.put("/{run_id}/orientation")
def set_orientation(
    body: OrientationRequest, run: RunHandle = Depends(get_run)
) -> dict[str, bool]:
    """Record the flip chosen in the cloud viewer.

    Presentation and export only: no stage reads it as an input, so it is deliberately
    absent from every fingerprint and changing it invalidates nothing.
    """

    def mutate(manifest: RunManifest) -> None:
        manifest.capture.flip_x = body.flip_x

    manifest = run.update(mutate)
    run.bus.publish("run.updated")
    return {"flip_x": manifest.capture.flip_x}


@router.delete("/{run_id}", status_code=204)
def delete_run(run_id: str) -> None:
    if not store.delete(run_id):
        raise HTTPException(status_code=404, detail=f"no such run: {run_id}")
