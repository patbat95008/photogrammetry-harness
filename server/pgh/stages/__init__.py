"""Pipeline stages.

Importing this package registers every implemented stage with the global registry.
Unimplemented stages still exist in the DAG (see manifest.STAGE_ORDER) and render
as stubs; they simply have no Stage object yet.
"""

from .extract import ExtractStage
from .registry import registry

registry.register(ExtractStage())

__all__ = ["ExtractStage", "registry"]
