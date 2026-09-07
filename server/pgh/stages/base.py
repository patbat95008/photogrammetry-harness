"""The Stage contract every pipeline step implements.

A stage is: typed params, a set of declared external inputs, a preflight check,
and a ``run`` that writes artifacts into the run directory. Everything else --
scheduling, progress, logging, cancellation, staleness -- is handled by the
harness around it, so adding a stage means writing only the part that is specific
to that stage.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel

from ..manifest import RunManifest, StageId


class StageParams(BaseModel):
    """Base for a stage's typed parameters.

    Fields marked ``json_schema_extra={"affects_fingerprint": False}`` are cosmetic:
    changing them re-renders a preview but does not invalidate the stage's outputs
    or anything downstream.
    """

    model_config = {"extra": "forbid"}


class CancelledError(Exception):
    """Raised inside a stage when the user cancels the job."""


class CancelToken:
    """Cooperative cancellation, checked at stage-defined safe points."""

    def __init__(self) -> None:
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    @property
    def cancelled(self) -> bool:
        return self._cancelled

    def raise_if_cancelled(self) -> None:
        if self._cancelled:
            raise CancelledError("cancelled by user")


@dataclass(slots=True)
class Progress:
    """A single progress observation, forwarded to the UI over SSE."""

    #: 0.0 - 1.0 where known, else None for indeterminate work.
    fraction: float | None = None
    message: str = ""
    #: Counter-style progress, e.g. (42, 412) for "Processing image [42/412]".
    current: int | None = None
    total: int | None = None


ProgressCallback = Callable[[Progress], None]


@dataclass(slots=True)
class StageContext:
    """Everything a stage needs to do its work."""

    run_dir: Path
    manifest: RunManifest
    params: StageParams
    logger: logging.Logger
    cancel: CancelToken
    report: ProgressCallback
    #: Scratch directory that is atomically promoted on success and discarded on
    #: failure, so a crashed stage never leaves half-written artifacts in place.
    scratch: Path

    def progress(
        self,
        message: str = "",
        *,
        fraction: float | None = None,
        current: int | None = None,
        total: int | None = None,
    ) -> None:
        self.cancel.raise_if_cancelled()
        if fraction is None and current is not None and total:
            fraction = current / total
        self.report(
            Progress(fraction=fraction, message=message, current=current, total=total)
        )


@dataclass(slots=True)
class StageResult:
    """What a stage produced. Merged into its StageRecord on success."""

    artifacts: dict[str, str] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    tool_versions: dict[str, str] = field(default_factory=dict)


class Stage(ABC):
    """Base class for every pipeline stage."""

    id: StageId
    label: str
    description: str = ""
    params_model: type[StageParams] = StageParams
    depends_on: list[StageId] = []

    def external_inputs(self, manifest: RunManifest) -> dict[str, Any]:
        """Content this stage reads that is neither a param nor an upstream artifact.

        Whatever is returned here becomes part of the stage's fingerprint, so
        include the identity of source files and the versions of external tools --
        and nothing volatile, or the stage will appear stale on every page load.
        """
        return {}

    def preflight(self, manifest: RunManifest) -> list[str]:
        """Reasons this stage cannot run right now. Empty means good to go."""
        return []

    @abstractmethod
    def run(self, ctx: StageContext) -> StageResult:
        """Do the work. Must be idempotent: re-running replaces prior outputs."""
        raise NotImplementedError

    def default_params(self) -> StageParams:
        return self.params_model()
