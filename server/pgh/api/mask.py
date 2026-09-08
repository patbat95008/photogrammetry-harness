"""Single-frame mask preview, so a bad click is visible before a full run.

Without this the only way to find out what the clicks selected is to propagate the
whole take and look at the result. That is a minute for the cup and several for a
chair spin, and the failure it hides is not subtle: a single click on the test mug
selected one of the photographs *printed on* the mug, and SAM 2 ranked the mug itself
second. Nothing about that is visible from the click.

It answers one question -- what would this frame's mask be -- with the image
predictor rather than the video one, because building a video inference state to
segment a single frame costs seconds and a great deal of memory.
"""

from __future__ import annotations

import json
from typing import Any

import cv2
import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

from .. import sam2rt
from ..jobs import runner
from ..manifest import StageId
from ..stages.mask import MaskParams, MaskPrompt, clean_mask, dilate_mask, to_bytes
from ..store import RunHandle
from .deps import get_run

router = APIRouter(prefix="/api/runs/{run_id}/mask", tags=["mask"])


class PreviewRequest(BaseModel):
    camera_group: str
    slot: int = Field(0, ge=0)
    prompts: list[MaskPrompt] = Field(default_factory=list)
    params: dict[str, Any] = Field(default_factory=dict)


@router.post("/preview")
def preview(body: PreviewRequest, run: RunHandle = Depends(get_run)) -> Response:
    """Segment one frame and return the mask as a PNG."""
    if not body.prompts:
        raise HTTPException(status_code=400, detail="no click points to preview")

    # The preview must not fight a running stage for the card. A masking run holds
    # about 1.7 GB and a densify will take everything it can get.
    if runner.active():
        raise HTTPException(
            status_code=409,
            detail=(
                "something is running on the graphics card right now, and loading a "
                "second model would compete with it for memory. Wait for it to finish."
            ),
        )

    frame = _find_frame(run, body.camera_group, body.slot)
    image = cv2.imread(str(run.dir / frame["file"]), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=404, detail=f"could not read {frame['file']}")

    try:
        params = MaskParams.model_validate({**body.params, "prompts": {}})
    except Exception:
        params = MaskParams()

    height, width = image.shape[:2]
    points = np.array(
        [[p.x * width, p.y * height] for p in body.prompts], dtype=np.float32
    )
    labels = np.array([1 if p.include else 0 for p in body.prompts], dtype=np.int32)

    try:
        predictor = sam2rt.load_image_predictor(params.model)
        predictor.set_image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        masks, scores, _ = predictor.predict(
            point_coords=points, point_labels=labels, multimask_output=True
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    raw = np.asarray(masks[int(np.argmax(scores))], dtype=bool)
    if params.clicks_mark == "background":
        raw = ~raw
    mask = dilate_mask(clean_mask(raw, params), params.dilate_px)

    ok, encoded = cv2.imencode(".png", to_bytes(mask))
    if not ok:
        raise HTTPException(status_code=500, detail="could not encode the mask")
    return Response(
        content=encoded.tobytes(),
        media_type="image/png",
        headers={
            # So the page can show the number without decoding the image, and so a
            # mask that covers most of the frame is legible as a mistake.
            "X-Mask-Area": f"{float(mask.mean()):.5f}",
            "Cache-Control": "no-store",
        },
    )


def _find_frame(run: RunHandle, group: str, slot: int) -> dict[str, Any]:
    index = run.dir / "frames" / "frames.jsonl"
    if not index.exists():
        raise HTTPException(
            status_code=409, detail="this run has no frames yet; run the extract stage"
        )
    for line in index.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record["camera_group"] == group and int(record["slot"]) == slot:
            return record
    raise HTTPException(status_code=404, detail=f"no frame {group}/{slot}")


@router.post("/release")
def release(run: RunHandle = Depends(get_run)) -> dict[str, bool]:
    """Drop the cached preview model, handing its memory back."""
    _ = run
    sam2rt.release()
    return {"released": True}
