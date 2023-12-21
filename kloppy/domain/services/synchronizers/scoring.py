"""Functions for scoring a match between an event and a tracking frame.

A scoring function takes an event and a tracking frame and returns a
non-negative number. A score of 0 means that the event and the tracking frame
are perfectly aligned. The higher the score, the worse the alignment is.
"""
import math
from typing import Callable, List, Optional

import numpy as np

from kloppy.domain import BallState, Event, Frame

from . import config
from .utils import eucl
from .features import (
    ball_acceleration,
    ball_state,
    dist_player_ball,
    acceleration_player_ball,
)


def mis_clock(event: Event, frame: Frame) -> float:
    """Mismatch between clocks.

    Parameters
    ----------
    event : Event
    frame : Frame

    Returns
    -------
    float
        Absolute difference between the event timestamp and the frame
        timestamp.
    """
    ts_frame = frame.timestamp
    ts_event = event.timestamp
    return abs(ts_frame - ts_event)


def mis_ball_location(event: Event, frame: Frame) -> float:
    """Mismatch between event location and ball location in tracking frame.

    Parameters
    ----------
    event : Event
    frame : Frame

    Returns
    -------
    float
        Euclidean distance between the event location and the ball location in
        the tracking frame.
    """
    event_coords = event.coordinates
    ball_coords = frame.ball_coordinates
    if event_coords is None or ball_coords is None:
        return float("nan")
    return eucl(event_coords, ball_coords)


def mis_player_location(
    event: Event,
    frame: Frame,
) -> float:
    """Mismatch between event location and player position in tracking frame.

    Parameters
    ----------
    event : Event
    frame : Frame

    Returns
    -------
    float
        Euclidean distance between the event location and the player location
        in the tracking frame.
    """
    if event.player is None:
        return float("nan")
    player_side = event.player.team.ground
    player_number = event.player.jersey_no
    try:
        player_tracking_data = next(
            data
            for player, data in frame.players_data.items()
            if player.jersey_no == player_number
            and player.team.ground == player_side
        )
    except StopIteration:
        return float("inf")

    player_coords = player_tracking_data.coordinates
    event_coords = event.coordinates
    if player_coords is None or event_coords is None:
        return float("nan")
    return eucl(event_coords, player_coords)


def mis_possession(
    event: Event,
    frame: Frame,
) -> float:
    """Distance between the acting player and the ball in tracking frame.

    Parameters
    ----------
    event : Event
    frame : Frame

    Returns
    -------
    float
        Euclidean distance between the player and the ball in the tracking
        frame.
    """
    if event.player is None:
        return float("nan")
    return dist_player_ball(frame, event.player)


def mis_ball_state(
    event,
    frame,
    is_open_play: Callable = config.is_open_play,
    is_dead_ball: Callable = config.is_dead_ball,
    is_dead_ball_start: Callable = config.is_dead_ball_start,
    is_dead_ball_end: Callable = config.is_set_piece,
    infer_ball_state: bool = config.infer_ball_state,
) -> float:
    """Disagreement on ball state.

    If an event marks the start of a dead ball state (e.g. a foul), the ball
    state should be DEAD in the next frame. If an event marks the end of a dead
    ball state (e.g. a free kick), the ball state should be ALIVE in the next
    frame. For open play events (e.g., a dribble), the ball state should be
    ALIVE. For dead ball events (e.g., a card), the ball state should be DEAD.

    Parameters
    ----------
    event : Event
    frame : Frame
    is_open_play : (Event) -> bool, optional
        Function to determine whether the event is an open play event, by
        default config.is_open_play
    is_dead_ball : (Event) -> bool, optional
        Function to determine whether the event is a dead ball event, by
        default config.is_dead_ball
    is_dead_ball_start : (Event) -> bool, optional
        Function to determine whether the event is the start of a dead ball,
        by default config.is_dead_ball_start
    is_dead_ball_end : (Event) -> bool, optional
        Function to determine whether the event is the end of a dead ball, by
        default config.is_set_piece
    infer_ball_state : bool, optional
        Whether to infer the ball state if it is not available in the tracking
        data, by default config.infer_ball_state

    Returns
    -------
    float
        1.0 if the event and the frame disagree on the ball status, 0.0
        otherwise.
    """
    status = ball_state(frame, infer=infer_ball_state)
    if status is None:
        return float("nan")
    if is_dead_ball_start(event):
        if (
            ball_state(frame, infer=infer_ball_state) == BallState.ALIVE
            and frame.next_record is not None
            and ball_state(frame.next_record, infer=infer_ball_state)
            == BallState.DEAD
        ):
            return 0.0
        return 1.0
    if is_dead_ball_end(event):
        if (
            ball_state(frame, infer=infer_ball_state) == BallState.ALIVE
            and frame.prev_record is not None
            and ball_state(frame.prev_record, infer=infer_ball_state)
            == BallState.DEAD
        ):
            return 0.0
        return 1.0
    if is_open_play(event):
        if status == BallState.ALIVE:
            return 0.0
        return 1.0
    if is_dead_ball(event):
        if status == BallState.DEAD:
            return 0.0
        return 1.0
    return 0.0


def mis_ball_acceleration(
    event: Event,
    frame: Frame,
    is_first_touch: Callable = config.is_first_touch,
    is_run_with_ball: Callable = config.is_run_with_ball,
    is_pass_like: Callable = config.is_pass_like,
    max_ball_speed: float = config.max_speed_ball,
    smoothing_window: int = config.smoothing_window_speed,
) -> float:
    """Mismatch between event type and ball acceleration.

    The acceleration of the ball is computed as the second derivative of the
    distance betweeen the ball's location in the current frame and the ball's
    location in the previous frame.

    For some event types, the ball should be accelerating (e.g., a pass),
    while for others, the ball should be decelerating (e.g., an interception).
    The mismatch is computed by squashing the acceleration to a range between 0
    and 1.

    Parameters
    ----------
    event : Event
    frame : Frame
    is_first_touch : (Event) -> bool, optional
        Function to determine whether the event is a first touch, by default
        config.is_first_touch
    is_run_with_ball : (Event) -> bool, optional
        Function to determine whether the event is a run with the ball, by
        default config.is_run_with_ball
    is_pass_like : (Event) -> bool, optional
        Function to determine whether the event is a pass-like event, by
        default config.is_pass_like
    max_ball_speed : float
        The maximum speed that the ball can realisitically achieve (in
        meters/second). Speed measures that exceed `max_speed` are tagged as
        outliers and set to NaN.
    smoothing_window : int
        The window size for smoothing the ball speed.

    Returns
    -------
    float
        A value between 0 and 1, where 0 means that the event type and the ball
        acceleration agree, and 1 means that they disagree.
    """
    acceleration = ball_acceleration(
        frame, max_speed=max_ball_speed, window=smoothing_window
    )
    a = 2.0 if smoothing_window is None else 2.0 / smoothing_window
    r = 1.0 if smoothing_window is None else 1.0 / smoothing_window
    if is_pass_like(event):
        # the ball should be accelerating
        return 1 / (1 + np.exp(acceleration - a) / r)
    elif is_first_touch(event):
        # the ball should be decelerating
        return 1 / (1 + np.exp(-acceleration - a) / r)
    elif is_run_with_ball(event):
        # the ball should be accelerating
        return 1 / (1 + np.exp(acceleration - a) / r)
    else:
        return 0.0


def mis_player_ball_acceleration(
    event: Event,
    frame: Frame,
    is_first_touch: Callable = config.is_first_touch,
    is_run_with_ball: Callable = config.is_run_with_ball,
    is_pass_like: Callable = config.is_pass_like,
    max_ball_speed: float = config.max_speed_ball,
    smoothing_window: int = config.smoothing_window_speed,
) -> float:
    """Mismatch between event type and player-ball acceleration.

    The acceleration of the ball is computed as the second derivative of the
    distance betweeen the ball's location and the acting player's location.

    For some event types, the ball should be accelerating (e.g., a pass),
    while for others, the ball should be decelerating (e.g., an interception).
    The mismatch is computed by squashing the acceleration to a range between 0
    and 1.

    Parameters
    ----------
    event : Event
    frame : Frame
    is_first_touch : (Event) -> bool, optional
        Function to determine whether the event is a first touch, by default
        config.is_first_touch
    is_run_with_ball : (Event) -> bool, optional
        Function to determine whether the event is a run with the ball, by
        default config.is_run_with_ball
    is_pass_like : (Event) -> bool, optional
        Function to determine whether the event is a pass-like event, by
        default config.is_pass_like
    max_ball_speed : float
        The maximum speed that the ball can realisitically achieve (in
        meters/second). Speed measures that exceed `max_speed` are tagged as
        outliers and set to NaN.
    smoothing_window : int
        The window size for smoothing the ball speed.

    Returns
    -------
    float
        A value between 0 and 1, where 0 means that the event type and the ball
        acceleration agree, and 1 means that they disagree.
    """
    if event.player is None:
        return float("nan")
    acceleration = acceleration_player_ball(
        frame, event.player, max_speed=max_ball_speed, window=smoothing_window
    )
    a = 2.0 if smoothing_window is None else 2.0 / smoothing_window
    r = 1.0 if smoothing_window is None else 1.0 / smoothing_window
    if is_pass_like(event):
        # the ball should be accelerating away from the player
        return 1 / (1 + np.exp(acceleration / 100 - a) / r)
    elif is_first_touch(event):
        # the ball should be decelerating towards the player
        return 1 / (1 + np.exp(-acceleration / 100 - a) / r)
    elif is_run_with_ball(event):
        # the ball and player should be accelerating together
        return abs(acceleration)
    else:
        return 0.0


def scale_down_lin(func, mini, maxi, minval, maxval):
    def inner(event, frame):
        dist = func(event, frame)
        if maxi == mini:
            return maxval

        a = (minval - maxval) / (maxi - mini)
        b = (maxi * maxval - mini * minval) / (maxi - mini)

        return a * dist + b

    return inner


def scale_upw_lin(func, mini, maxi, minval, maxval):
    def inner(event, frame):
        dist = func(event, frame)
        if maxi == mini:
            return maxval

        a = (maxval - minval) / (maxi - mini)
        b = (maxi * minval - mini * maxval) / (maxi - mini)

        return a * dist + b

    return inner


def scale_down_exp(func, mini, maxi, minval, maxval):
    def inner(event, frame):
        dist = func(event, frame)
        if maxi == mini:
            return maxval

        a = (math.log(minval) - math.log(maxval)) / (maxi - mini)
        b = math.log(maxval) - a * mini

        return math.exp(a * dist + b)

    return inner


def scale_upw_exp(func, mini, maxi, minval, maxval):
    def inner(event, frame):
        dist = func(event, frame)
        if maxi == mini:
            return maxval

        a = (math.log(maxval) - math.log(minval)) / (maxi - mini)
        b = math.log(minval) - a * mini

        return math.exp(a * dist + b)

    return inner


def scale(func, scale=1.0):
    """Wrap a distance function to scale the distance by a given factor."""

    def inner(event, frame):
        dist = func(event, frame)
        return dist / scale

    return inner


def apply_if(
    condition: Callable[[Event], bool],
    mis_func: Callable[[Event, Frame], float],
):
    """Apply a scoring function if a condition is met.

    Parameters
    ----------
    condition : Callable[[Event], float]
        Condition function.
    mis_func : Callable[[Event, Frame], bool]
        Scoring function.

    Returns
    -------
    Callable[[Event, Frame], float]
        Scoring function.
    """

    def inner(event: Event, frame: Frame) -> float:
        if condition(event):
            return mis_func(event, frame)
        return 0.0

    inner.__name__ = mis_func.__name__
    return inner


class combine:
    """Combine an arbitrary number of scoring functions to a single one.

    Takes the length of the vector of individual mismatch scores.

    Parameters
    ----------
    mis_funcs : Callable[[Event, Frame], float]
        Scoring functions.

    Returns
    -------
    (Event, list(Frame), list(bool)) -> list(float)
        Scoring function.
    """

    def __init__(self, *mis_funcs):
        self.mis_funcs = mis_funcs
        self.weights = [1.0] * len(mis_funcs)

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, value):
        if len(value) != len(self.mis_funcs):
            raise ValueError(
                "The number of weights must match the number of scoring functions."
            )
        self._weights = value

    def __call__(
        self,
        event: Event,
        frames: List[Frame],
        mask: Optional[List[bool]] = None,
    ) -> List[float]:
        if mask is None:
            mask = [True] * len(frames)
        return [
            sum(
                (f(event, frame) / w) ** 2
                for f, w in zip(self.mis_funcs, self.weights)
            )
            ** 0.5
            if mask
            else float("inf")
            for mask, frame in zip(mask, frames)
        ]
