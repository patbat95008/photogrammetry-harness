"""Canonical hashing and the staleness engine.

This is the load-bearing wall of the harness. A stage's fingerprint is derived
purely from its *content* inputs -- its fingerprint-affecting params, its declared
external inputs, and its upstream stages' fingerprints -- never from timestamps or
run counters. Two consequences follow, and both matter:

* Changing a param invalidates that stage and everything downstream of it.
* Changing a param **and changing it back** restores the original hash, so the
  downstream work stays valid. This is "early cutoff", and it is the difference
  between a harness that is pleasant to iterate in and one that punishes you for
  looking at a knob.

Cosmetic params (thumbnail size, sheet size) are excluded from the fingerprint by
declaring ``json_schema_extra={"affects_fingerprint": False}`` on the field.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Protocol

from pydantic import BaseModel, ValidationError

from .manifest import RunManifest, StageId, StageRecord, StageState

#: Truncated for readability in the UI and logs. 16 hex chars of SHA-256 is 64
#: bits -- far beyond any plausible accidental collision across a few dozen stages.
DIGEST_LENGTH = 16


def canonical_json(value: Any) -> str:
    """Deterministic JSON: sorted keys, no incidental whitespace.

    Dict ordering must never affect a hash, or fingerprints would change when
    unrelated code reorders a field.
    """
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )


def digest(*parts: Any) -> str:
    hasher = hashlib.sha256()
    for part in parts:
        hasher.update(canonical_json(part).encode("utf-8"))
        hasher.update(b"\x00")  # domain separator, so ["a","b"] != ["ab"]
    return hasher.hexdigest()[:DIGEST_LENGTH]


def fingerprint_fields(model: type[BaseModel]) -> set[str]:
    """Field names on a params model that participate in the fingerprint."""
    included: set[str] = set()
    for name, field in model.model_fields.items():
        extra = field.json_schema_extra or {}
        if isinstance(extra, dict) and extra.get("affects_fingerprint") is False:
            continue
        included.add(name)
    return included


def filter_params(model: type[BaseModel] | None, params: dict[str, Any]) -> dict[str, Any]:
    """Fill in the defaults, then drop the cosmetic params, before hashing.

    Normalising through the model is what makes an omitted parameter and an
    explicitly-defaulted one hash alike. A stage run before anyone opened its page
    stores ``{}``; the moment the UI saves that same form it stores every field. Both
    describe the same run, so hashing the raw dict would mean that merely pressing
    Save invalidated every finished stage downstream -- hours of reconstruction
    thrown away for a change that was not one.
    """
    if model is None:
        return dict(params)
    keep = fingerprint_fields(model)
    try:
        normalised = model.model_validate(params or {}).model_dump(mode="json")
    except ValidationError:
        # A stored value this model no longer accepts. Hash what is actually there
        # rather than raising: this runs on every page load, and the stage will be
        # re-stamped anyway the next time it runs.
        normalised = dict(params)
    return {k: v for k, v in normalised.items() if k in keep}


class StageResolver(Protocol):
    """What the staleness engine needs to know about the pipeline.

    Implemented by the real stage registry, and by trivial fakes in the tests so
    the engine can be exercised without any external tool.
    """

    def stage_ids(self) -> list[StageId]: ...

    def dependencies(self, stage_id: StageId) -> list[StageId]: ...

    def params_model(self, stage_id: StageId) -> type[BaseModel] | None: ...

    def external_inputs(self, stage_id: StageId, manifest: RunManifest) -> dict[str, Any]:
        """Content the stage reads that is not a param and not an upstream artifact.

        For extract this is the source clips' identity, offsets and grouping, plus
        the ffmpeg version. Anything returned here invalidates the stage when it changes.
        """
        ...


@dataclass(slots=True)
class StageEvaluation:
    stage_id: StageId
    computed_fingerprint: str
    state: StageState
    stale_reason: str | None = None
    #: Upstream stages that must succeed before this one can run.
    blocked_by: list[StageId] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.blocked_by is None:
            self.blocked_by = []

    @property
    def runnable(self) -> bool:
        return not self.blocked_by and self.state is not StageState.RUNNING


def _self_digest(
    stage_id: StageId, record: StageRecord, resolver: StageResolver, manifest: RunManifest
) -> tuple[str, str]:
    """Hash this stage's own inputs, ignoring upstream. Returns (params, external)."""
    model = resolver.params_model(stage_id)
    params_hash = digest(stage_id.value, filter_params(model, record.params))
    external_hash = digest(stage_id.value, resolver.external_inputs(stage_id, manifest))
    return params_hash, external_hash


def compute_fingerprints(
    manifest: RunManifest, resolver: StageResolver
) -> dict[StageId, str]:
    """Fingerprint every stage, chaining each one onto its dependencies.

    Uses each dependency's *computed* fingerprint rather than its stored one, so a
    change propagates the full length of the DAG in a single pass even when the
    intermediate stages have not been re-run yet.
    """
    computed: dict[StageId, str] = {}

    def resolve(stage_id: StageId, seen: frozenset[StageId]) -> str:
        if stage_id in computed:
            return computed[stage_id]
        if stage_id in seen:
            raise ValueError(f"cycle in stage DAG at {stage_id}")

        record = manifest.stages.get(stage_id) or StageRecord()
        params_hash, external_hash = _self_digest(stage_id, record, resolver, manifest)
        upstream = [
            resolve(dep, seen | {stage_id}) for dep in resolver.dependencies(stage_id)
        ]

        computed[stage_id] = digest(params_hash, external_hash, upstream)
        return computed[stage_id]

    for stage_id in resolver.stage_ids():
        resolve(stage_id, frozenset())

    return computed


def _stale_reason(
    stage_id: StageId,
    record: StageRecord,
    resolver: StageResolver,
    manifest: RunManifest,
    computed: dict[StageId, str],
) -> str:
    """Attribute staleness to the most specific cause, for a useful UI message."""
    params_hash, external_hash = _self_digest(stage_id, record, resolver, manifest)

    # Recompute what the fingerprint would have been if only upstream had moved.
    upstream = [computed[dep] for dep in resolver.dependencies(stage_id)]
    if digest(params_hash, external_hash, upstream) == record.fingerprint:
        return "unknown"  # should not happen; caller only asks when they differ

    # Isolate the cause by holding each component at its stored value in turn.
    # We cannot recover the stored components, so we test forward instead: if the
    # upstream chain alone explains the change, blame upstream.
    stored_upstream_matches = all(
        manifest.stages[dep].fingerprint == computed[dep]
        for dep in resolver.dependencies(stage_id)
        if dep in manifest.stages
    )
    if not stored_upstream_matches:
        return "upstream"
    return "params_or_inputs"


def evaluate(
    manifest: RunManifest, resolver: StageResolver
) -> dict[StageId, StageEvaluation]:
    """Recompute every stage's state from the manifest. Pure; writes nothing."""
    computed = compute_fingerprints(manifest, resolver)
    result: dict[StageId, StageEvaluation] = {}

    for stage_id in resolver.stage_ids():
        record = manifest.stages.get(stage_id) or StageRecord()
        fingerprint = computed[stage_id]

        # A stage can only run once every dependency has completed successfully --
        # or been explicitly skipped, which is a decision about the run, not an
        # omission. Masking is the motivating case; see StageState.SKIPPED.
        blocked_by = [
            dep
            for dep in resolver.dependencies(stage_id)
            if (manifest.stages.get(dep) or StageRecord()).state
            not in (StageState.DONE, StageState.STALE, StageState.SKIPPED)
        ]

        if record.state is StageState.RUNNING:
            state, reason = StageState.RUNNING, None
        elif record.state in (StageState.DONE, StageState.STALE):
            if record.fingerprint == fingerprint:
                state, reason = StageState.DONE, None
            else:
                state = StageState.STALE
                reason = _stale_reason(stage_id, record, resolver, manifest, computed)
        else:
            state, reason = record.state, None

        result[stage_id] = StageEvaluation(
            stage_id=stage_id,
            computed_fingerprint=fingerprint,
            state=state,
            stale_reason=reason,
            blocked_by=blocked_by,
        )

    return result


def apply_evaluation(
    manifest: RunManifest, resolver: StageResolver
) -> dict[StageId, StageEvaluation]:
    """Evaluate, then write the derived state back onto the manifest records."""
    evaluations = evaluate(manifest, resolver)
    for stage_id, ev in evaluations.items():
        record = manifest.stages.setdefault(stage_id, StageRecord())
        record.state = ev.state
        record.stale_reason = ev.stale_reason
    return evaluations
