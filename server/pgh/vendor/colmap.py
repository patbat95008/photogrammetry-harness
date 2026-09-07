"""COLMAP: argv builders and sparse-model parsing.

Everything that knows COLMAP's command-line grammar lives here.

Two things this module exists to get right, both of which cost an afternoon to
rediscover:

**Logging.** COLMAP logs through glog, whose ``--log_target`` defaults to
``stderr_and_file``. Every builder therefore appends ``--log_target stdout
--log_color 0`` so output arrives on the merged pipe as plain text. Never do this at
a call site -- one builder that forgets is a stage that looks hung.

**Two namespaces, both live.** COLMAP 4.2 moved pipeline and GPU options to
``FeatureMatching.*`` / ``FeatureExtraction.*`` while leaving algorithm tuning on
``SiftMatching.*`` / ``SiftExtraction.*``. Verified against the binary in ``tools/``
(4.2.0, commit be5e291): ``--FeatureExtraction.max_image_size`` and
``--SiftExtraction.max_num_features`` are *both* correct, and
``filter_stationary_matches`` sits on ``TwoViewGeometry``. A blanket rename in either
direction breaks. Ask ``registry.colmap_dialect()``; do not guess.

Model reading goes through ``model_converter --output_type TXT`` and is parsed here,
so the harness needs no pycolmap and no binary-format reader of its own.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config import TOOL_SPECS, resolve_tool
from ..proc import capture

#: glog would otherwise split output across stderr and a file, and colour it.
LOG_FLAGS: list[str] = ["--log_target", "stdout", "--log_color", "0"]


def _tool(key: str) -> Path:
    spec = next(s for s in TOOL_SPECS if s.key == key)
    path = resolve_tool(spec)
    if path is None:
        raise RuntimeError(f"{spec.label} not found; check the doctor page")
    return path


def colmap_path() -> Path:
    return _tool("colmap")


def version() -> str:
    result = capture([colmap_path(), "-h"], cwd=Path.cwd(), timeout=30)
    match = re.search(r"COLMAP\s+([0-9]+\.[0-9]+\.[0-9]+[^\n)]*)", result.combined)
    return match.group(1).strip() if match else "unknown"


def _flag(argv: list[str | Path], name: str, value: Any) -> None:
    """COLMAP takes booleans as 1/0, not as bare presence flags."""
    if isinstance(value, bool):
        value = 1 if value else 0
    argv += [name, str(value)]


# -- argv builders -----------------------------------------------------------


def feature_extractor(
    *,
    database_path: Path,
    image_path: Path,
    dialect: dict[str, Any],
    camera_model: str = "OPENCV",
    single_camera_per_folder: bool = True,
    max_image_size: int = 3200,
    max_num_features: int = 8192,
    use_gpu: bool = True,
    mask_path: Path | None = None,
) -> list[str | Path]:
    argv: list[str | Path] = [colmap_path(), "feature_extractor"]
    _flag(argv, "--database_path", database_path)
    _flag(argv, "--image_path", image_path)
    _flag(argv, "--ImageReader.camera_model", camera_model)

    if single_camera_per_folder and dialect.get("single_camera_per_folder"):
        # One intrinsic per camera folder. This is why the frame layout keeps each
        # camera in its own directory all the way from extraction.
        _flag(argv, "--ImageReader.single_camera_per_folder", True)
    if mask_path is not None and dialect.get("mask_path"):
        _flag(argv, "--ImageReader.mask_path", mask_path)

    # max_image_size moved to the new namespace; max_num_features did not.
    if dialect.get("feature_extraction_ns"):
        _flag(argv, "--FeatureExtraction.max_image_size", max_image_size)
        _flag(argv, "--FeatureExtraction.use_gpu", use_gpu)
    else:
        _flag(argv, "--SiftExtraction.max_image_size", max_image_size)
        _flag(argv, "--SiftExtraction.use_gpu", use_gpu)
    if dialect.get("sift_extraction_ns"):
        _flag(argv, "--SiftExtraction.max_num_features", max_num_features)

    return argv + LOG_FLAGS


def _matcher_common(
    argv: list[str | Path],
    dialect: dict[str, Any],
    *,
    use_gpu: bool,
    filter_stationary_matches: bool,
) -> list[str | Path]:
    if dialect.get("feature_matching_ns"):
        _flag(argv, "--FeatureMatching.use_gpu", use_gpu)
    else:
        _flag(argv, "--SiftMatching.use_gpu", use_gpu)
    if filter_stationary_matches and dialect.get("filter_stationary_matches"):
        # Frames where nothing moved contribute matches with no baseline, which drag
        # bundle adjustment toward a degenerate solution.
        _flag(argv, "--TwoViewGeometry.filter_stationary_matches", True)
    return argv + LOG_FLAGS


def exhaustive_matcher(
    *,
    database_path: Path,
    dialect: dict[str, Any],
    use_gpu: bool = True,
    filter_stationary_matches: bool = True,
) -> list[str | Path]:
    argv: list[str | Path] = [colmap_path(), "exhaustive_matcher"]
    _flag(argv, "--database_path", database_path)
    return _matcher_common(
        argv,
        dialect,
        use_gpu=use_gpu,
        filter_stationary_matches=filter_stationary_matches,
    )


def sequential_matcher(
    *,
    database_path: Path,
    dialect: dict[str, Any],
    overlap: int = 10,
    quadratic_overlap: bool = True,
    loop_detection: bool = True,
    vocab_tree_path: Path | None = None,
    use_gpu: bool = True,
    filter_stationary_matches: bool = True,
) -> list[str | Path]:
    """Match each frame against its neighbours in time, plus loop-closure candidates.

    Loop detection is what lets the far side of an orbit recognise the near side:
    frames half a revolution apart see the same surface but sit hundreds of indices
    apart, so sequential overlap alone never compares them. It needs a vocabulary
    tree, and COLMAP quietly does nothing useful without one -- hence the guard.
    """
    argv: list[str | Path] = [colmap_path(), "sequential_matcher"]
    _flag(argv, "--database_path", database_path)
    _flag(argv, "--SequentialMatching.overlap", overlap)
    _flag(argv, "--SequentialMatching.quadratic_overlap", quadratic_overlap)
    if loop_detection and vocab_tree_path is not None:
        _flag(argv, "--SequentialMatching.loop_detection", True)
        _flag(argv, "--SequentialMatching.vocab_tree_path", vocab_tree_path)
    return _matcher_common(
        argv,
        dialect,
        use_gpu=use_gpu,
        filter_stationary_matches=filter_stationary_matches,
    )


def mapper(
    *,
    database_path: Path,
    image_path: Path,
    output_path: Path,
    min_model_size: int = 10,
    ba_refine_principal_point: bool = False,
) -> list[str | Path]:
    argv: list[str | Path] = [colmap_path(), "mapper"]
    _flag(argv, "--database_path", database_path)
    _flag(argv, "--image_path", image_path)
    _flag(argv, "--output_path", output_path)
    _flag(argv, "--Mapper.min_model_size", min_model_size)
    _flag(argv, "--Mapper.ba_refine_principal_point", ba_refine_principal_point)
    return argv + LOG_FLAGS


def model_converter(
    *, input_path: Path, output_path: Path, output_type: str
) -> list[str | Path]:
    argv: list[str | Path] = [colmap_path(), "model_converter"]
    _flag(argv, "--input_path", input_path)
    _flag(argv, "--output_path", output_path)
    _flag(argv, "--output_type", output_type)
    return argv + LOG_FLAGS


def model_analyzer(*, path: Path) -> list[str | Path]:
    argv: list[str | Path] = [colmap_path(), "model_analyzer"]
    _flag(argv, "--path", path)
    return argv + LOG_FLAGS


def image_undistorter(
    *,
    image_path: Path,
    input_path: Path,
    output_path: Path,
    max_image_size: int = -1,
    output_type: str = "COLMAP",
) -> list[str | Path]:
    """Rectify images so OpenMVS sees a pinhole camera.

    ``max_image_size`` and the ROI flags must be recorded and reused verbatim if
    masks are ever undistorted in a second pass (HANDOVER 6.8) -- drifting flags
    between the two produce masks that are subtly wrong exactly where ears and hair
    live.
    """
    argv: list[str | Path] = [colmap_path(), "image_undistorter"]
    _flag(argv, "--image_path", image_path)
    _flag(argv, "--input_path", input_path)
    _flag(argv, "--output_path", output_path)
    _flag(argv, "--output_type", output_type)
    _flag(argv, "--max_image_size", max_image_size)
    return argv + LOG_FLAGS


# -- reading a sparse model --------------------------------------------------


@dataclass(slots=True)
class CameraIntrinsics:
    camera_id: int
    model: str
    width: int
    height: int
    params: list[float]

    @property
    def focal_px(self) -> float | None:
        return self.params[0] if self.params else None


@dataclass(slots=True)
class RegisteredImage:
    image_id: int
    name: str
    camera_id: int
    #: Camera-from-world rotation as a quaternion (w, x, y, z), COLMAP's convention.
    qvec: tuple[float, float, float, float]
    tvec: tuple[float, float, float]
    num_points: int = 0

    def center(self) -> tuple[float, float, float]:
        """Camera position in world coordinates: ``-R^T t``."""
        rot = _quat_to_matrix(*self.qvec)
        return (
            -sum(rot[i][0] * self.tvec[i] for i in range(3)),
            -sum(rot[i][1] * self.tvec[i] for i in range(3)),
            -sum(rot[i][2] * self.tvec[i] for i in range(3)),
        )


@dataclass(slots=True)
class SparseModel:
    cameras: dict[int, CameraIntrinsics] = field(default_factory=dict)
    images: list[RegisteredImage] = field(default_factory=list)
    num_points3D: int = 0
    mean_track_length: float | None = None
    mean_reprojection_error_px: float | None = None


def _quat_to_matrix(w: float, x: float, y: float, z: float) -> list[list[float]]:
    norm = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]


def read_text_model(model_dir: Path) -> SparseModel:
    """Parse ``cameras.txt`` / ``images.txt`` / ``points3D.txt``.

    The format is stable and line-oriented: comments start with ``#``, and each
    image occupies two lines -- a pose line followed by its 2D observations, which is
    why the image loop steps in pairs.
    """
    model = SparseModel()

    cameras_txt = model_dir / "cameras.txt"
    if cameras_txt.exists():
        for line in cameras_txt.read_text(encoding="utf-8").splitlines():
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            model.cameras[int(parts[0])] = CameraIntrinsics(
                camera_id=int(parts[0]),
                model=parts[1],
                width=int(parts[2]),
                height=int(parts[3]),
                params=[float(p) for p in parts[4:]],
            )

    images_txt = model_dir / "images.txt"
    if images_txt.exists():
        lines = [
            ln
            for ln in images_txt.read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.lstrip().startswith("#")
        ]
        for i in range(0, len(lines) - 1, 2):
            parts = lines[i].split()
            if len(parts) < 10:
                continue
            observations = lines[i + 1].split()
            # Each observation is (x, y, point3D_id); -1 means "not triangulated".
            num_points = sum(
                1 for j in range(2, len(observations), 3) if observations[j] != "-1"
            )
            model.images.append(
                RegisteredImage(
                    image_id=int(parts[0]),
                    qvec=(
                        float(parts[1]),
                        float(parts[2]),
                        float(parts[3]),
                        float(parts[4]),
                    ),
                    tvec=(float(parts[5]), float(parts[6]), float(parts[7])),
                    camera_id=int(parts[8]),
                    name=" ".join(parts[9:]).replace("\\", "/"),
                    num_points=num_points,
                )
            )

    points_txt = model_dir / "points3D.txt"
    if points_txt.exists():
        track_lengths: list[int] = []
        errors: list[float] = []
        for line in points_txt.read_text(encoding="utf-8").splitlines():
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            errors.append(float(parts[7]))
            track_lengths.append(max(0, (len(parts) - 8) // 2))
        model.num_points3D = len(track_lengths)
        if track_lengths:
            model.mean_track_length = sum(track_lengths) / len(track_lengths)
        if errors:
            model.mean_reprojection_error_px = sum(errors) / len(errors)

    return model


#: model_analyzer emits "Key: value" lines; these are the ones worth keeping.
_ANALYZER_KEYS = {
    "cameras": "num_cameras",
    "images": "num_images",
    "registered images": "registered_images",
    "points": "num_points",
    "observations": "num_observations",
    "mean track length": "mean_track_length",
    "mean observations per image": "mean_observations_per_image",
    "mean reprojection error": "mean_reprojection_error_px",
}


def parse_analyzer(text: str) -> dict[str, float]:
    """Pull the summary statistics out of ``model_analyzer`` output."""
    stats: dict[str, float] = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        raw_key, _, raw_value = line.partition(":")
        key = _ANALYZER_KEYS.get(raw_key.strip().lower())
        if key is None:
            continue
        match = re.search(r"-?\d+(?:\.\d+)?", raw_value)
        if match:
            stats[key] = float(match.group(0))
    return stats
