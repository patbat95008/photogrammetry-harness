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
    StageId.MASK: PlannedStage(
        label="Mask",
        summary="Mark which pixels belong to the subject and which must be ignored.",
        plan=[
            "Click the subject on the first frame of each camera to seed SAM 2.",
            "Propagate the mask through the frames in timeline order, chunked to bound "
            "VRAM -- the video predictor's memory grows with sequence length.",
            "Store one canonical mask per frame, then generate per-engine filename "
            "views: COLMAP wants <name>.<ext>.png, OpenMVS wants <name>.mask.png.",
            "Offer an overlay editor for corrections around ears, hair and chin.",
        ],
        conditional={
            CaptureMode.SUBJECT_ROTATES: (
                "This run has the subject rotating while the cameras stay put, so "
                "masking is REQUIRED. The background is static in the room but moves "
                "relative to the subject, which makes it the part behaving "
                "inconsistently. Left in, the solver locks onto the room and the "
                "subject never resolves."
            ),
            CaptureMode.CAMERA_ORBITS: (
                "This run has the cameras orbiting a still subject, so masking is "
                "OPTIONAL and usually best skipped. The background is rigid with "
                "respect to the subject, so it supplies extra features and helps the "
                "orbit close. Mask only if you specifically want the background "
                "excluded from the final mesh."
            ),
        },
    ),
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
