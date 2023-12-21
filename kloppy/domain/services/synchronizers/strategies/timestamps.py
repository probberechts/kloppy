from .. import alignment
from ..strategy import SynchronizationStrategyBuilder

strategy = (
    SynchronizationStrategyBuilder()
    .with_alignment_fn(alignment.timestamp_alignment)
    .build()
)
strategy.register("timestamps")
