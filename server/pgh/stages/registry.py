"""The stage registry: the concrete StageResolver the staleness engine runs against."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from ..manifest import STAGE_DEPENDENCIES, STAGE_ORDER, RunManifest, StageId
from .base import Stage, StageParams


class StageRegistry:
    """Holds the registered stages and answers the questions fingerprint.py asks."""

    def __init__(self, stages: dict[StageId, Stage] | None = None) -> None:
        self._stages: dict[StageId, Stage] = stages or {}

    def register(self, stage: Stage) -> Stage:
        self._stages[stage.id] = stage
        return stage

    def get(self, stage_id: StageId) -> Stage | None:
        return self._stages.get(stage_id)

    def implemented(self) -> set[StageId]:
        """Stages that are actually wired up; the rest render as stubs in the UI."""
        return set(self._stages)

    # -- StageResolver ------------------------------------------------------

    def stage_ids(self) -> list[StageId]:
        # The full DAG, not just the implemented stages -- an unimplemented stage
        # still has a state, still goes stale, and still blocks its dependents.
        return list(STAGE_ORDER)

    def dependencies(self, stage_id: StageId) -> list[StageId]:
        stage = self._stages.get(stage_id)
        if stage is not None and stage.depends_on:
            return list(stage.depends_on)
        return list(STAGE_DEPENDENCIES.get(stage_id, []))

    def params_model(self, stage_id: StageId) -> type[BaseModel] | None:
        stage = self._stages.get(stage_id)
        return stage.params_model if stage else None

    def external_inputs(self, stage_id: StageId, manifest: RunManifest) -> dict[str, Any]:
        stage = self._stages.get(stage_id)
        return stage.external_inputs(manifest) if stage else {}

    # -- helpers ------------------------------------------------------------

    def default_params(self, stage_id: StageId) -> StageParams:
        stage = self._stages.get(stage_id)
        return stage.default_params() if stage else StageParams()

    def params_schema(self, stage_id: StageId) -> dict[str, Any] | None:
        """JSON schema the frontend renders the params form from."""
        model = self.params_model(stage_id)
        return model.model_json_schema() if model else None


#: Process-wide registry. Concrete stages register themselves on import.
registry = StageRegistry()
