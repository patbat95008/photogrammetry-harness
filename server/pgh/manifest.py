"""``project.json`` -- the single source of truth for a run's state.

State only. Bulk per-frame data (a 3600-row frame table) lives in sidecar
``.jsonl`` files referenced by path: this document is rewritten whenever anything
changes, and must stay small enough that rewriting it is free.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

SCHEMA_VERSION = 1


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class StageId(StrEnum):
    EXTRACT = "extract"
    SELECT = "select"
    MASK = "mask"
    SPARSE = "sparse"
    DENSE = "dense"
    MESH = "mesh"
    EXPORT = "export"


#: The pipeline DAG, in dependency order. Currently a straight chain, but the
#: staleness engine treats it as a general DAG so a branch can be added later.
STAGE_ORDER: list[StageId] = [
    StageId.EXTRACT,
    StageId.SELECT,
    StageId.MASK,
    StageId.SPARSE,
    StageId.DENSE,
    StageId.MESH,
    StageId.EXPORT,
]

STAGE_DEPENDENCIES: dict[StageId, list[StageId]] = {
    StageId.EXTRACT: [],
    StageId.SELECT: [StageId.EXTRACT],
    StageId.MASK: [StageId.SELECT],
    StageId.SPARSE: [StageId.SELECT, StageId.MASK],
    StageId.DENSE: [StageId.SPARSE],
    StageId.MESH: [StageId.DENSE],
    StageId.EXPORT: [StageId.MESH],
}


class StageState(StrEnum):
    #: Never run.
    PENDING = "pending"
    #: Currently executing.
    RUNNING = "running"
    #: Completed, and its fingerprint still matches its inputs.
    DONE = "done"
    #: Completed, but a param or upstream artifact has changed since.
    STALE = "stale"
    FAILED = "failed"
    CANCELLED = "cancelled"


class CaptureMode(StrEnum):
    """Which of the two things was moving. This inverts several later decisions.

    SUBJECT_ROTATES -- you spin in a chair while the cameras stay put. The
    background is static in the room but moves relative to your head, so to the
    reconstruction it is the part behaving wrongly. Masking it out is **mandatory**;
    without it the solver locks onto the room and the head never resolves.

    CAMERA_ORBITS -- the classic walk-around, cameras moving, subject still. Now the
    background is rigid with respect to the subject, so it *helps*: more features,
    better loop closure. Masking it out here throws away good information.

    The mode also decides whether a rigid camera rig is meaningful -- see SegmentKind.
    """

    SUBJECT_ROTATES = "subject_rotates"
    CAMERA_ORBITS = "camera_orbits"


class SegmentKind(StrEnum):
    #: Multiple cameras recording simultaneously from FIXED mounts, so their relative
    #: pose is constant and they can be declared to COLMAP as one rigid rig.
    RIG = "rig"
    #: A single camera recording alone (e.g. the crown pass).
    SINGLE = "single"
    #: Multiple cameras recording at once but not rigidly coupled -- one phone in each
    #: hand, say. Frames still share a timeline, but the relative pose changes every
    #: frame, so a rig constraint would be a lie. Treated as independent cameras.
    INDEPENDENT = "independent"

    @property
    def supports_rig(self) -> bool:
        return self is SegmentKind.RIG


class SourceIdentity(BaseModel):
    """Cheap but sufficient identity for a multi-GB source file.

    Full-hashing a 4 GB clip takes ~15 s and buys nothing over size + mtime +
    head/tail digest, which catches every realistic case: replaced, re-exported,
    truncated, or transcoded.
    """

    size_bytes: int
    mtime_ns: int
    digest: str  # sha256 of first and last 8 MiB, plus the size


class ClipProbe(BaseModel):
    """The subset of ffprobe output the pipeline actually reasons about."""

    duration_s: float | None = None
    codec_name: str | None = None
    pix_fmt: str | None = None
    width: int | None = None
    height: int | None = None
    avg_frame_rate: float | None = None
    r_frame_rate: float | None = None
    nb_frames: int | None = None
    color_transfer: str | None = None
    color_primaries: str | None = None
    color_space: str | None = None
    #: Display-matrix rotation in degrees, ffprobe's counter-clockwise convention.
    rotation: float = 0.0
    #: Orientation actually applied at extract time, clockwise degrees.
    effective_rotation: int = 0
    is_hdr: bool = False
    is_vfr_suspected: bool = False
    has_audio: bool = False
    #: Path (relative to the run dir) of the verbatim ffprobe JSON kept as evidence.
    raw_probe_path: str | None = None
    packets_path: str | None = None


class Clip(BaseModel):
    clip_id: str
    #: One folder of frames, one COLMAP camera. Two clips from the same physical
    #: camera at the same settings may share a group; two different phones must not.
    camera_group: str
    role: str = "cam"
    segment_id: str = "seg0"
    source_path: str
    source_identity: SourceIdentity | None = None
    probe: ClipProbe = Field(default_factory=ClipProbe)
    #: Added to this clip's PTS to place it on the shared run timeline.
    time_offset_s: float = 0.0
    sync_confidence: float | None = None
    sync_method: str | None = None
    enabled: bool = True


class Segment(BaseModel):
    """A set of clips recorded at the same time, sharing one timeline.

    The two head-height cameras form one RIG segment. The crown pass, recorded
    separately, is its own SINGLE segment -- it shares no instants with the others,
    so it gets a disjoint block of slot indices and stays out of the rig config.
    """

    segment_id: str
    label: str = ""
    kind: SegmentKind = SegmentKind.RIG
    clip_ids: list[str] = Field(default_factory=list)
    #: First slot index owned by this segment. Segments never overlap.
    slot_base: int = 0
    slot_count: int = 0


class TimelineSummary(BaseModel):
    """Computed by extract; read by every later stage."""

    fps: float | None = None
    t_start: float | None = None
    t_end: float | None = None
    total_slots: int = 0
    sync_residual_p95_s: float | None = None
    degrees_per_second: float | None = None


class Capture(BaseModel):
    """Facts about the shoot that cannot be recovered from the footage."""

    mode: CaptureMode = CaptureMode.SUBJECT_ROTATES
    #: Tape-measured distance between the two camera lenses. The only route to
    #: absolute scale -- photogrammetry alone recovers shape but not size. Only
    #: meaningful for a fixed rig, where that distance stays constant.
    baseline_mm: float | None = None
    scale_reference_note: str = ""
    #: How many full revolutions the take covers. Without it the rotation-rate
    #: warning has nothing to calibrate against and can only guess.
    revolutions: float | None = None
    notes: str = ""

    @property
    def needs_background_mask(self) -> bool:
        return self.mode is CaptureMode.SUBJECT_ROTATES


class StageRecord(BaseModel):
    state: StageState = StageState.PENDING
    #: Hash of params + external inputs + upstream fingerprints at last success.
    fingerprint: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    started_at: str | None = None
    finished_at: str | None = None
    duration_s: float | None = None
    artifacts: dict[str, str] = Field(default_factory=dict)
    metrics: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    error: str | None = None
    log_path: str | None = None
    tool_versions: dict[str, str] = Field(default_factory=dict)
    #: Why the stage is stale, when it is: "params", "inputs", or "upstream".
    stale_reason: str | None = None


class RunManifest(BaseModel):
    schema_version: int = SCHEMA_VERSION
    run_id: str
    name: str = ""
    created_at: str = Field(default_factory=utcnow)
    updated_at: str = Field(default_factory=utcnow)
    capture: Capture = Field(default_factory=Capture)
    clips: list[Clip] = Field(default_factory=list)
    segments: list[Segment] = Field(default_factory=list)
    timeline: TimelineSummary = Field(default_factory=TimelineSummary)
    stages: dict[StageId, StageRecord] = Field(default_factory=dict)

    def model_post_init(self, _context: Any) -> None:
        for stage_id in STAGE_ORDER:
            self.stages.setdefault(stage_id, StageRecord())

    # -- lookups -------------------------------------------------------------

    def clip(self, clip_id: str) -> Clip | None:
        return next((c for c in self.clips if c.clip_id == clip_id), None)

    def segment(self, segment_id: str) -> Segment | None:
        return next((s for s in self.segments if s.segment_id == segment_id), None)

    def enabled_clips(self, segment_id: str | None = None) -> list[Clip]:
        clips = [c for c in self.clips if c.enabled]
        if segment_id is not None:
            clips = [c for c in clips if c.segment_id == segment_id]
        return clips

    def camera_groups(self) -> list[str]:
        """Distinct camera groups, in stable first-seen order.

        One group becomes one folder under ``frames/`` and therefore one COLMAP
        camera via ``--ImageReader.single_camera_per_folder``.
        """
        seen: list[str] = []
        for clip in self.clips:
            if clip.enabled and clip.camera_group not in seen:
                seen.append(clip.camera_group)
        return seen


# -- atomic IO ---------------------------------------------------------------


class ManifestIO:
    """Reads and writes ``project.json`` atomically.

    Only the event loop calls ``save``. Worker threads emit events instead: a job
    that wrote the manifest directly would race the request handlers, and progress
    ticks would rewrite the document many times a second for no benefit.
    """

    FILENAME = "project.json"

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.path = run_dir / self.FILENAME

    def exists(self) -> bool:
        return self.path.exists()

    def load(self) -> RunManifest:
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        raw = migrate(raw)
        return RunManifest.model_validate(raw)

    def save(self, manifest: RunManifest) -> None:
        manifest.updated_at = utcnow()
        payload = manifest.model_dump(mode="json")
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Write to a sibling temp file, then os.replace -- atomic on NTFS, so a
        # crash mid-write can never leave a truncated manifest behind.
        fd, tmp_name = tempfile.mkstemp(
            dir=self.run_dir, prefix=".project-", suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2, sort_keys=False)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp_name, self.path)
        except BaseException:
            Path(tmp_name).unlink(missing_ok=True)
            raise


def migrate(raw: dict[str, Any]) -> dict[str, Any]:
    """Bring an older manifest up to the current schema version."""
    version = raw.get("schema_version", 0)
    if version > SCHEMA_VERSION:
        raise ValueError(
            f"manifest schema v{version} is newer than this build understands "
            f"(v{SCHEMA_VERSION}); upgrade the harness"
        )
    # No migrations yet -- v1 is the first schema. Each future bump appends a
    # block here and leaves the earlier ones intact.
    raw["schema_version"] = SCHEMA_VERSION
    return raw
