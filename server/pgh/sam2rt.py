"""The only module in this codebase that knows SAM 2's grammar.

Same role as ``vendor/colmap.py``: everything version-specific, and everything that
can go silently wrong about loading a model, lives here -- so the stage above it is
about masking rather than about SAM 2.

**Nothing here is imported at module load.** ``torch`` and ``sam2`` are pulled in
inside the functions that need them, so ``stages/mask.py`` -- which imports this
module and never imports ``sam2`` -- can be tested against a fake predictor on a
machine with no GPU and no checkpoints. That boundary is the whole reason this
module exists; do not add a top-level ``import torch``.

Two traps, both of which produce a working-looking model rather than an error:

**The checkout must not be named ``sam2``.** The server runs with the project root
as its working directory, so a directory of that name there shadows the installed
package: ``import sam2`` binds to the directory as a namespace package, succeeds,
and every submodule import then fails. ``import_sam2`` checks for exactly that and
says so, because the raw failure is SAM 2's own guard message a long way from its
cause. See HANDOVER 6.28.

**A checkpoint and its config are a pair.** A mismatched pair loads *without
raising* and produces poor masks. The pairing is pinned once in ``config.py`` and
read from there; nothing here derives a config from a checkpoint filename.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Any, Iterator, Protocol, runtime_checkable

from . import config

#: Cached predictors, keyed (kind, model, device). Loading the large checkpoint is
#: about five seconds, which is fine once per run and not fine per preview click.
_CACHE: dict[tuple[str, str, str], Any] = {}


@runtime_checkable
class VideoPredictor(Protocol):
    """The slice of SAM 2's video predictor this harness uses.

    Declared as a Protocol so ``tests/test_mask.py`` can substitute a fake and
    assert on how the stage seeds each chunk -- which is where the errors that
    produce a plausible-but-wrong mask live.
    """

    def init_state(self, video_path: str, **kwargs: Any) -> Any: ...

    def add_new_points_or_box(
        self, inference_state: Any, frame_idx: int, obj_id: int, **kwargs: Any
    ) -> Any: ...

    def add_new_mask(
        self, inference_state: Any, frame_idx: int, obj_id: int, mask: Any
    ) -> Any: ...

    def propagate_in_video(self, inference_state: Any, **kwargs: Any) -> Iterator[Any]: ...

    def reset_state(self, inference_state: Any) -> None: ...


def import_sam2() -> ModuleType:
    """Import ``sam2``, or explain precisely why it could not be imported."""
    try:
        import sam2
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            f"SAM 2 could not be imported ({type(exc).__name__}: {exc}). It installs "
            f"from the checkout as an editable package; see the SAM 2 block in "
            f".gitignore for the clone and install commands."
        ) from exc

    if getattr(sam2, "__file__", None) is None:
        raise RuntimeError(
            "'import sam2' resolved to a namespace package rather than a real module, "
            "which means a directory named 'sam2' is shadowing the installed package. "
            "The server's working directory is the project root, so the checkout must "
            "not be named 'sam2' there -- it is expected at "
            f"{config.SAM2_DIR}. See HANDOVER 6.28."
        )
    return sam2


def resolve_model(key: str) -> tuple[Path, str]:
    """Return ``(checkpoint path, hydra config name)`` for a model key.

    Both come from ``config.SAM2_MODELS``, never from each other. The config name is
    resolved by Hydra against the ``sam2`` config module that ``sam2/__init__.py``
    registers, so it stays a module-relative string like
    ``configs/sam2.1/sam2.1_hiera_l.yaml`` rather than becoming a filesystem path.
    """
    spec = config.SAM2_MODELS.get(key)
    if spec is None:
        known = ", ".join(sorted(config.SAM2_MODELS))
        raise RuntimeError(f"unknown SAM 2 model {key!r}; expected one of {known}")

    checkpoint = config.get_settings().sam2_dir / spec["checkpoint"]
    if not checkpoint.is_file():
        raise RuntimeError(
            f"the {spec['label']} checkpoint is not at {checkpoint}. Download it with "
            f"checkpoints/download_ckpts.sh in the SAM 2 checkout, or pick a model "
            f"the doctor page reports as present."
        )
    return checkpoint, spec["config"]


def _device(device: str) -> str:
    if device != "cuda":
        return device
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "torch cannot see the GPU, and SAM 2 on the processor is minutes per "
            "frame rather than a fraction of a second. Check the doctor page."
        )
    return device


def load_video_predictor(model: str = "", device: str = "cuda") -> VideoPredictor:
    """Build the video predictor used to propagate a mask through a chunk."""
    model = model or config.DEFAULT_SAM2_MODEL
    key = ("video", model, device)
    if key not in _CACHE:
        import_sam2()
        from sam2.build_sam import build_sam2_video_predictor

        checkpoint, cfg = resolve_model(model)
        _CACHE[key] = build_sam2_video_predictor(
            cfg, str(checkpoint), device=_device(device)
        )
    return _CACHE[key]


def load_image_predictor(model: str = "", device: str = "cuda") -> Any:
    """Build the single-image predictor behind the mask preview endpoint.

    A separate predictor from the video one on purpose: the preview answers "did
    that click land where I meant it to" for one frame, and building a video
    inference state to answer that would cost seconds and a great deal of memory.
    """
    model = model or config.DEFAULT_SAM2_MODEL
    key = ("image", model, device)
    if key not in _CACHE:
        import_sam2()
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        checkpoint, cfg = resolve_model(model)
        _CACHE[key] = SAM2ImagePredictor(
            build_sam2(cfg, str(checkpoint), device=_device(device))
        )
    return _CACHE[key]


def release() -> None:
    """Drop every cached predictor and hand the VRAM back.

    Called before a stage runs, so a predictor the preview endpoint left resident is
    not competing for the card with the run the user just started.
    """
    _CACHE.clear()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # pragma: no cover - environment dependent
        pass


@lru_cache(maxsize=1)
def sam2_version() -> str:
    """The installed SAM 2 distribution version, for stage fingerprints.

    Deliberately coarse, and deliberately not volatile: which checkpoint is used is
    already a parameter, and anything that changed per run -- free VRAM, a file
    mtime -- would make the stage look stale on every page load.
    """
    try:
        from importlib.metadata import version

        return version("SAM-2")
    except Exception:  # pragma: no cover - environment dependent
        return "unknown"
