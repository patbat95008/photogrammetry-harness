"""Environment health endpoint."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from ..registry import doctor_report

router = APIRouter(prefix="/api", tags=["doctor"])


@router.get("/doctor")
def get_doctor() -> dict[str, Any]:
    """Report on every external dependency the pipeline relies on.

    Probes run on each request rather than being cached: the point of this page is
    to reflect the machine as it is right now, and the whole sweep costs well under
    a second.
    """
    return doctor_report()
