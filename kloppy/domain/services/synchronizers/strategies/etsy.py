"""
.. [4] Maaike Van Roy, Lorenzo Cascioli, and Jesse Davis. "ETSY:
       A rule-based approach to Event and Tracking data SYnchronization".
       Machine Learning and Data Mining for Sports Analytics ECML/PKDD
       2023 Workshop (2023).
"""
import math
from functools import partial

from kloppy.domain import (
    EventType,
    SetPieceQualifier,
    GoalkeeperQualifier,
    GoalkeeperActionType,
    DuelQualifier,
    DuelType,
)

from .. import masking, scoring, alignment
from ..strategy import SynchronizationStrategyBuilder


def _is_set_piece(event):
    if event.get_qualifier_value(SetPieceQualifier) is not None:
        return True
    return False


def _is_pass_like(event):
    if event.event_type in [
        EventType.PASS,
        EventType.SHOT,
        EventType.TAKE_ON,
        EventType.CLEARANCE,
    ]:
        return True
    if event.event_type == EventType.GOALKEEPER:
        actiontype = event.get_qualifier_value(GoalkeeperQualifier)
        if actiontype in [GoalkeeperActionType.PUNCH]:
            return True
    return False


def _is_incoming(event):
    if event.event_type in [EventType.INTERCEPTION, EventType.RECOVERY]:
        return True
    if event.event_type == EventType.GOALKEEPER:
        actiontype = event.get_qualifier_value(GoalkeeperQualifier)
        if actiontype in [
            GoalkeeperActionType.SAVE,
            GoalkeeperActionType.CLAIM,
            GoalkeeperActionType.PICK_UP,
        ]:
            return True
    return False


def _is_bad_touch(event):
    return event.event_type == EventType.MISCONTROL


def _is_fault_like(event):
    if event.event_type == EventType.FOUL_COMMITTED:
        return True
    if event.event_type == EventType.DUEL:
        dueltype = event.get_qualifier_values(DuelQualifier)
        if DuelType.TACKLE in dueltype:
            return True
    return False


def window_fn(event):
    """Returns ETSY's action-specific window size."""
    # Pass-like in set-piece
    if _is_set_piece(event):
        return 20.0
    # Pass-like in open play
    elif _is_pass_like(event):
        return 10.0
    # Incoming
    elif _is_incoming(event):
        return 10.0
    # Bad touch
    elif _is_bad_touch(event):
        return 10.0
    # Fault-like
    elif _is_fault_like(event):
        return 10.0
    return 0.0


def mask_fn(event, frames):
    """Implementation of the ETSY's action-specific frame filters."""
    # Pass-like in set-piece
    if _is_set_piece(event):
        return masking.combine(
            partial(masking.mask_ball_possession, max_dist=2.5),
            partial(
                masking.mask_ball_height,
                max_height=3.0,
                max_height_foot=1.5,
                min_height_head=None,
            ),
            partial(masking.mask_ball_acceleration, min_acceleration=0.0),
        )(event, frames)
    # Pass-like in open play
    elif _is_pass_like(event):
        return masking.combine(
            partial(masking.mask_ball_possession, max_dist=2.5),
            partial(
                masking.mask_ball_height,
                max_height=3.0,
                max_height_foot=1.5,
                min_height_head=1.0,
            ),
            partial(masking.mask_ball_acceleration, min_acceleration=0.0),
        )(event, frames)
    # Incoming
    elif _is_incoming(event):
        return masking.combine(
            partial(masking.mask_ball_possession, max_dist=2.0),
            partial(
                masking.mask_ball_height,
                max_height=3.0,
                max_height_foot=None,
                min_height_head=None,
            ),
            partial(masking.mask_ball_deceleration, min_deceleration=0.0),
        )(event, frames)
    # Bad touch
    elif _is_bad_touch(event):
        return masking.combine(
            partial(masking.mask_ball_possession, max_dist=3.0),
            partial(
                masking.mask_ball_height,
                max_height=3.0,
                max_height_foot=1.5,
                min_height_head=1.0,
            ),
        )(event, frames)
    # Fault-like
    elif _is_fault_like(event):
        return masking.combine(
            partial(masking.mask_ball_possession, max_dist=3.0),
            partial(
                masking.mask_ball_height,
                max_height=4.0,
                max_height_foot=None,
                min_height_head=None,
            ),
        )(event, frames)
    return [True] * len(frames)


score_fn = scoring.combine(
    scoring.scale_upw_lin(
        scoring.mis_possession, 0.0, math.sqrt(105**2 + 68**2), 0, 100 / 3
    ),
    scoring.scale_upw_lin(
        scoring.mis_player_location,
        0.0,
        math.sqrt(105**2 + 68**2),
        0,
        100 / 3,
    ),
    scoring.scale_upw_lin(
        scoring.mis_ball_location,
        0.0,
        math.sqrt(105**2 + 68**2),
        0,
        100 / 3,
    ),
)


strategy = (
    SynchronizationStrategyBuilder()
    .with_window_fn(window_fn)
    .with_mask_fn(mask_fn)
    .with_score_fn(score_fn)
    .with_alignment_fn(alignment.greedy_alignment)
    .build()
)
strategy.register("ETSY")
