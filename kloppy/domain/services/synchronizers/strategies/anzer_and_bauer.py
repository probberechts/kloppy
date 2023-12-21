"""
.. [2] Gabriel Anzer and Pascal Bauer. "A goal scoring probability model
       for shots based on synchronized positional and event data in
       football (soccer)." Frontiers in Sports and Active Living (2021):
       53.
.. [3] Gabriel Anzer, and Pascal Bauer. "Expected passes: Determining the
       difficulty of a pass in football (soccer) using spatio-temporal
       data." Data mining and knowledge discovery 36.1 (2022): 295-317.
"""
from functools import partial

from kloppy.domain import EventType
from .. import masking, scoring, alignment, features, config
from ..strategy import SynchronizationStrategyBuilder


def window_fn(event):
    return 10


def reduce_fn(event, frame):
    acceleration = features.acceleration_player_ball(
        frame, event.player, max_speed=config.max_speed_ball, window=None
    )
    return -acceleration


mask_fn_shots = masking.combine_and_reduce(
    reduce_fn,
    partial(masking.mask_ball_possession, max_dist=2.0),
)
mask_fn_passes = masking.combine_and_reduce(
    reduce_fn,
    partial(masking.mask_ball_possession, max_dist=2.0),
    partial(
        masking.mask_ball_possession_receiver, max_dist=2.0, cutoff_time=5.0
    ),
)

score_fn = scoring.combine(
    scoring.mis_clock,
    scoring.mis_possession,
    scoring.mis_player_location,
    scoring.mis_ball_location,
)
score_fn.weights = [1.0, 1.0, 1.0, 1.0]


strategy_passes = (
    SynchronizationStrategyBuilder()
    .with_window_fn(window_fn)
    .with_filter_fn(lambda event: event.event_type == EventType.PASS)
    .with_mask_fn(mask_fn_passes)
    .with_score_fn(score_fn)
    .with_alignment_fn(alignment.local_alignment)
    .build()
)
strategy_passes.register("Anzer and Bauer (passes)")

strategy_shots = (
    SynchronizationStrategyBuilder()
    .with_window_fn(window_fn)
    .with_filter_fn(lambda event: event.event_type == EventType.SHOT)
    .with_mask_fn(mask_fn_shots)
    .with_score_fn(score_fn)
    .with_alignment_fn(alignment.local_alignment)
    .build()
)
strategy_shots.register("Anzer and Bauer (shots)")
