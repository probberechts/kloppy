from functools import partial

from .. import masking, scoring, alignment, config
from ..strategy import SynchronizationStrategyBuilder

window_fn = lambda event: 10

mask_fn = masking.combine(
    masking.apply_if(
        config.is_on_ball_action,
        partial(masking.mask_ball_possession, max_dist=2.0),
    ),
    masking.apply_if(
        config.is_on_ball_action,
        partial(
            masking.mask_ball_height,
            max_height=3.0,
            max_height_foot=1.5,
            min_height_head=1.0,
        ),
    ),
    masking.apply_if(
        config.is_on_ball_action,
        partial(masking.mask_ball_location, max_dist=5.0),
    ),
)

score_fn = scoring.combine(
    scoring.mis_clock,
    scoring.mis_ball_state,
    scoring.apply_if(config.is_on_ball_action, scoring.mis_ball_location),
    scoring.apply_if(config.is_on_ball_action, scoring.mis_possession),
    scoring.apply_if(
        config.is_on_ball_action,
        scoring.mis_player_ball_acceleration,
    ),
)
score_fn.weights = [
    1.0,
    1.0,
    1.0,
    1.0,
    0.20,
]

strategy = (
    SynchronizationStrategyBuilder()
    .with_window_fn(window_fn)
    .with_mask_fn(mask_fn)
    .with_score_fn(score_fn)
    .with_alignment_fn(alignment.optimal_alignment)
    .build()
)
strategy.register("default")
