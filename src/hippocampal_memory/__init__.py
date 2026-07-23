"""Provenance-aware, evidence-driven memory consolidation."""

from .replay import HippocampusEngine, ReplayConfig, ReplayPreempted
from .retrieval import FactRetriever
from .store import MemoryStore

__all__ = [
    "FactRetriever",
    "HippocampusEngine",
    "MemoryStore",
    "ReplayConfig",
    "ReplayPreempted",
]

__version__ = "0.1.0"
