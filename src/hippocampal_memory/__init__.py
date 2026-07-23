"""Provenance-aware, evidence-driven memory consolidation."""

from .replay import HippocampusEngine, ReplayConfig, ReplayPreempted
from .retrieval import FactRetriever
from .coordination import MemoryCoordinator, MemoryEvent, RevisionConflict
from .circuit import CircuitConfig, TrisynapticCircuit
from .store import MemoryStore
from .vault import VaultProjector, VaultSynchronizer

__all__ = [
    "FactRetriever",
    "CircuitConfig",
    "HippocampusEngine",
    "MemoryCoordinator",
    "MemoryEvent",
    "MemoryStore",
    "ReplayConfig",
    "ReplayPreempted",
    "RevisionConflict",
    "VaultProjector",
    "VaultSynchronizer",
    "TrisynapticCircuit",
]

__version__ = "0.2.0"
