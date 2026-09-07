"""Stage params, execution, cancellation, and log retrieval."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import ValidationError

from ..fingerprint import evaluate
from ..jobs import runner
from ..manifest import StageId
from ..stages.planned import planned_for
from ..stages.registry import registry
from ..store import RunHandle
from .deps import get_run

router = APIRouter(prefix="/api/runs/{run_id}/stages", tags=["stages"])


def _stage_id(stage_id: str) -> StageId:
    try:
        return StageId(stage_id)
    except ValueError:
        raise HTTPException(status_code=404, detail=f"no such stage: {stage_id}") from None


@router.get("/{stage_id}")
def get_stage(stage_id: str, run: RunHandle = Depends(get_run)) -> dict[str, Any]:
    sid = _stage_id(stage_id)
    manifest = run.load()
    evaluation = evaluate(manifest, registry)[sid]
    stage = registry.get(sid)

    return {
        "stage_id": sid.value,
        "label": stage.label if stage else sid.value.title(),
        "description": stage.description if stage else "",
        "implemented": stage is not None,
        "record": manifest.stages[sid].model_dump(mode="json"),
        "schema": registry.params_schema(sid),
        "defaults": (
            registry.default_params(sid).model_dump(mode="json") if stage else {}
        ),
        "state": evaluation.state.value,
        "stale_reason": evaluation.stale_reason,
        "blocked_by": [b.value for b in evaluation.blocked_by],
        "runnable": evaluation.runnable,
        "preflight": stage.preflight(manifest) if stage else [],
        "planned": None if stage else planned_for(sid, manifest),
    }


@router.put("/{stage_id}/params")
def set_params(
    stage_id: str, body: dict[str, Any], run: RunHandle = Depends(get_run)
) -> dict[str, Any]:
    sid = _stage_id(stage_id)
    model = registry.params_model(sid)
    if model is None:
        raise HTTPException(status_code=400, detail=f"{stage_id} is not implemented yet")

    try:
        validated = model.model_validate(body)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors()) from None

    def mutate(manifest):
        manifest.stages[sid].params = validated.model_dump(mode="json")

    run.update(mutate)
    # Params changed, so staleness may have shifted for this stage and everything
    # downstream of it. Recompute and tell the UI in one shot.
    manifest = run.load()
    evaluations = evaluate(manifest, registry)
    run.bus.publish(
        "stage.params",
        stage=sid.value,
        states={s.value: e.state.value for s, e in evaluations.items()},
    )
    return {
        "params": validated.model_dump(mode="json"),
        "states": {s.value: e.state.value for s, e in evaluations.items()},
    }


@router.post("/{stage_id}/run", status_code=202)
def run_stage(stage_id: str, run: RunHandle = Depends(get_run)) -> dict[str, Any]:
    sid = _stage_id(stage_id)
    stage = registry.get(sid)
    if stage is None:
        raise HTTPException(status_code=400, detail=f"{stage_id} is not implemented yet")

    manifest = run.load()
    if blockers := stage.preflight(manifest):
        raise HTTPException(status_code=409, detail=blockers)

    try:
        job = runner.submit(run, sid)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from None
    return job.to_dict()


@router.post("/{stage_id}/cancel")
def cancel_stage(stage_id: str, run: RunHandle = Depends(get_run)) -> dict[str, Any]:
    sid = _stage_id(stage_id)
    cancelled = runner.cancel_stage(run.run_id, sid)
    if not cancelled:
        raise HTTPException(status_code=409, detail=f"{stage_id} is not running")
    return {"cancelled": True, "stage_id": sid.value}


@router.get("/{stage_id}/log", response_class=PlainTextResponse)
def get_log(stage_id: str, run: RunHandle = Depends(get_run)) -> str:
    """The most recent log for this stage, for a page that loads mid-job."""
    sid = _stage_id(stage_id)
    record = run.load().stages[sid]
    if not record.log_path:
        return ""
    path = run.dir / record.log_path
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")
