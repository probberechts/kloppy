"""
.. [1] Allan Clark, and Marek Kwiatkowski. "The right way to synchronise
   event and tracking data"(2020). https://kwiatkowski.io/sync.soccer
"""
from .. import scoring, alignment
from ..strategy import SynchronizationStrategyBuilder

scale_clock = 1  # Clock mismatch leading to unit penalty
scale_location = 1  # Location mismatch leading to unit penalty
scale_player = 1  # Player-ball gap leading to unit penalty
scale_ball = 1  # Penalty for syncing to dead-ball frame

score_fn = scoring.combine(
    scoring.mis_clock,
    scoring.mis_ball_location,
    scoring.mis_possession,
    scoring.mis_ball_state,
)
score_fn.weights = [scale_clock, scale_location, scale_player, scale_ball]


def mask_fn(event, frames):
    # sync.soccer does not use masks
    return [True] * len(frames)


def window_fn(event):
    # in fact, sync.soccer does not use a window
    # but we use a window for efficiency reasons.
    return 10


def set_scale_clock(strategy):
    def scale_clock(value):
        strategy.score_fn.weights[0] = value

    return scale_clock


strategy = (
    SynchronizationStrategyBuilder()
    .with_window_fn(window_fn)
    .with_mask_fn(mask_fn)
    .with_score_fn(score_fn)
    .with_alignment_fn(alignment.optimal_alignment)
    .build()
)
strategy.register("sync.soccer")
