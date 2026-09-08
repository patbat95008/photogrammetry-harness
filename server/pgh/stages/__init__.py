"""Pipeline stages.

Importing this package registers every implemented stage with the global registry.
Unimplemented stages still exist in the DAG (see manifest.STAGE_ORDER) and render
as stubs; they simply have no Stage object yet.
"""

from .extract import ExtractStage
from .registry import registry
from .dense import DenseStage
from .mesh import MeshStage
from .select import SelectStage
from .sparse import SparseStage

registry.register(ExtractStage())
registry.register(SelectStage())
registry.register(SparseStage())
registry.register(DenseStage())
registry.register(MeshStage())

__all__ = [
    "DenseStage",
    "ExtractStage",
    "MeshStage",
    "SelectStage",
    "SparseStage",
    "registry",
]
