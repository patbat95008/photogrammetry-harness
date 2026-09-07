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
    StageId.SELECT: PlannedStage(
        label="Select frames",
        summary="Whittle thousands of extracted frames down to the few hundred worth reconstructing.",
        plan=[
            "Score sharpness per frame (variance of Laplacian), restricted to the "
            "subject region rather than the whole image.",
            "Drop the near-identical frames already flagged during extraction: two "
            "views with no baseline between them have nothing to triangulate.",
            "Keep angular coverage even, so no part of the orbit is over- or "
            "under-represented.",
            "Prefer neutral frames -- eyes open, mouth closed -- since expression "
            "changes violate the rigid-scene assumption.",
            "Present a contact sheet with accept/reject and a reason per frame, with "
            "manual overrides folded into the stage fingerprint.",
        ],
    ),
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
    StageId.SPARSE: PlannedStage(
        label="Align",
        summary="Work out where each camera was, and recover a sparse point cloud.",
        plan=[
            "Build a hardlink farm of just the selected frames, preserving the "
            "per-camera folder layout so COLMAP assigns one intrinsic per camera.",
            "Extract and match features on the GPU, using sequential matching with "
            "loop detection to close the orbit.",
            "Run the mapper, then report which images failed to register.",
            "Show the sparse cloud with camera frusta, so a mis-registered view is "
            "visible rather than merely tabulated.",
        ],
        conditional={
            CaptureMode.SUBJECT_ROTATES: (
                "Two cameras on fixed mounts hold a constant relative pose in the "
                "subject's own reference frame, which is a rigid two-sensor rig. "
                "Declaring it constrains bundle adjustment hard and helps the loop "
                "close. This is why extraction names frames by shared timeline slot."
            ),
            CaptureMode.CAMERA_ORBITS: (
                "Handheld cameras do not hold a constant relative pose, so no rig will "
                "be declared here -- the two clips are solved as independent cameras."
            ),
        },
    ),
    StageId.DENSE: PlannedStage(
        label="Dense cloud",
        summary="Turn the sparse points and known camera poses into a dense surface sampling.",
        plan=[
            "Undistort the selected images -- and, separately, the masks, since COLMAP "
            "does not undistort masks and OpenMVS consumes undistorted images.",
            "Hand the scene to OpenMVS DensifyPointCloud, or COLMAP's own patch-match "
            "stereo, selectable per run.",
            "Expose resolution level prominently: this stage exhausts system RAM long "
            "before it troubles the GPU.",
            "Offer to delete the depth maps after fusion -- they are most of the tens "
            "of gigabytes a run consumes, and are never needed again.",
        ],
    ),
    StageId.MESH: PlannedStage(
        label="Mesh",
        summary="Build a surface from the dense cloud and paint it with the photographs.",
        plan=[
            "Reconstruct a mesh, refine it against the images, then texture it.",
            "Expect the crown of the head to be invented rather than measured unless a "
            "pass covering it was captured.",
            "Show the result texture-shaded, matte-shaded and as wireframe: a surface "
            "problem hidden by a convincing texture is the usual failure.",
        ],
    ),
    StageId.EXPORT: PlannedStage(
        label="Export",
        summary="Clean up, set real-world scale, and write files you can use.",
        plan=[
            "Isolate the largest connected component and drop floating fragments.",
            "Set absolute scale from the measured camera baseline -- photogrammetry "
            "recovers shape but never size.",
            "Orient and centre the bust, decimate to a sane polygon count, and export "
            "GLB/OBJ/STL via headless Blender.",
            "Render a turntable for the record.",
        ],
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
