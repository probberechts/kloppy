"""Approaches for synchronizing data from different sources."""
from . import masking, scoring
from .synchronizer import EventTrackingSynchronizer
from .strategy import (
    SynchronizationStrategy,
    SynchronizationStrategyBuilder,
    create_synchronization_strategy,
)

# register all of them
from . import strategies as _strategies

__all__ = [
    "EventTrackingSynchronizer",
    "SynchronizationStrategy",
    "SynchronizationStrategyBuilder",
    "create_synchronization_strategy",
    "scoring",
    "masking",
]
