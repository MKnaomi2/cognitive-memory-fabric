"""Provenance-aware, evidence-driven memory consolidation."""

from .replay import HippocampusEngine, ReplayConfig, ReplayPreempted
from .retrieval import FactRetriever
from .coordination import MemoryCoordinator, MemoryEvent, RevisionConflict
from .circuit import CircuitConfig, TrisynapticCircuit
from .cognition import CognitiveMemorySystem
from .store import MemoryStore
from .vault import VaultProjector, VaultSynchronizer

__all__ = [
    "FactRetriever",
    "CircuitConfig",
    "CognitiveMemorySystem",
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

__version__ = "0.4.0"
