"""Descriptions of stages that are not implemented yet.

A stub page that only says "not implemented" is useless. These entries let each
future stage explain what it will do, what it needs first, and which decisions
already made upstream will shape it -- so the pipeline is legible end to end even
while most of it is empty.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..manifest import CaptureMode, RunManifest, StageId


@dataclass(frozen=True)
class PlannedStage:
    label: str
    summary: str
    #: What this stage will do, in the order it will do it.
    plan: list[str] = field(default_factory=list)
    #: Notes that only apply in certain capture modes.
    conditional: dict[CaptureMode, str] = field(default_factory=dict)


PLANNED: dict[StageId, PlannedStage] = {
    # Empty: every stage in the DAG is implemented. The mask entry lived here until
    # M9 was built, and its capture-mode prose moved to stages/mask.py:CAPTURE_NOTES
    # rather than being deleted -- an implemented stage still needs to say why
    # masking is mandatory for a chair spin and a mistake for an orbit. The Stage.note
    # hook is where that goes now. Keep this module for the next unbuilt stage.
}


def planned_for(stage_id: StageId, manifest: RunManifest) -> dict[str, object] | None:
    """Describe an unimplemented stage, tailored to how this run was captured."""
    entry = PLANNED.get(stage_id)
    if entry is None:
        return None
    return {
        "label": entry.label,
        "summary": entry.summary,
        "plan": entry.plan,
        "note": entry.conditional.get(manifest.capture.mode),
    }
