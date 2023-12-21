"""Masking functions for event matching.

Each masking function takes an event and a frame and returns a boolean value
indicating whether the frame should be considered for matching with the event
or not. If a function does not have enough information to make a decision, it
should return True.
"""
import math
from typing import List, Optional, Callable

import numpy as np

from kloppy.domain import (
    BodyPart,
    BodyPartQualifier,
    Event,
    Frame,
)

from . import config
from . import features as fs
from .utils import eucl


def mask_ball_location(event: Event, frame: Frame, max_dist: float = 2.0):
    """The ball must be within a certain distance of the event location.

    Parameters
    ----------
    event : Event
    frame : Frame
    max_dist : float, optional
        Maximum distance between the ball and the event location, by default 2.0

    Returns
    -------
    bool
        True if the ball is within `max_dist` meter of the event location.
    """
    event_coords = event.coordinates
    ball_coords = frame.ball_coordinates
    if event_coords is None or ball_coords is None:
        return True
    return eucl(event_coords, ball_coords) <= max_dist


def mask_ball_possession(
    event: Event,
    frame: Frame,
    max_dist: float = 2.0,
):
    """The acting player must be within a certain distance of the ball.

    Parameters
    ----------
    event : Event
    frame : Frame
    max_dist : float, optional
        Maximum distance between the acting player and the ball, by default 2.0

    Returns
    -------
    bool
        True if the acting player is within `max_dist` meter of the ball.
    """
    dist = fs.dist_player_ball(frame, event.player)
    if math.isnan(dist):
        return True
    return dist <= max_dist


def mask_ball_control(
    event: Event,
    frame: Frame,
    own_team_only: bool = False,
):
    """The acting player must be closer to the ball than any other player.

    Parameters
    ----------
    event : Event
    frame : Frame
    own_team_only : bool, optional
        Only consider players from the same team, by default False

    Returns
    -------
    bool
        True if the acting player is closer to the ball than any other player.
    """
    dist = fs.dist_player_ball(frame, event.player)
    if math.isnan(dist):
        return True
    for player in frame.players_data:
        if player != event.player and (
            not own_team_only or player.team == event.player.team
        ):
            if fs.dist_player_ball(frame, player) < dist:
                return False
    return True


def mask_ball_possession_receiver(
    event: Event, frame: Frame, max_dist: float = 2.0, cutoff_time: float = 5.0
):
    """The receiving player must be within a certain distance of the ball.

    The actual receiver of the pass (if there is one), must be within
    `cutoff_dist` meter of the ball within `cutoff_time` seconds after the
    pass moment.

    Parameters
    ----------
    event : Event
    frame : Frame
    max_dist : float, optional
        Maximum distance between the receiving player and the ball, by default 2.0
    cutoff_time : float, optional
        Maximum time between the pass moment and the receiving moment, by default 5.0

    Returns
    -------
    bool
        True if the receiving player is within `max_dist` meter of the ball
        within `cutoff_time` seconds after the pass moment. If there is no
        receiver, True is returned.
    """
    if not hasattr(event, "receiver_player") or event.receiver_player is None:
        return True  # there is no receiver

    receiver = event.receiver_player
    receiver_frame = frame
    while (
        receiver_frame.next_record is not None
        and receiver_frame.next_record.timestamp - frame.timestamp
        < cutoff_time
    ):
        receiver_frame = receiver_frame.next_record
        if fs.dist_player_ball(receiver_frame, receiver) <= max_dist:
            return True
    return False


def mask_ball_height(
    event: Event,
    frame: Frame,
    max_height: float = 3.0,
    max_height_foot: Optional[float] = 1.5,
    min_height_head: Optional[float] = 1.0,
):
    """The ball must be within a reachable height.

    Parameters
    ----------
    event : Event
    frame : Frame
    max_height : float, optional
        Maximum height of the ball at the time of an event, by default 3.0
    max_height_foot : float, optional
        Maximum height of the ball at the time of an event with bodypart foot,
        by default 1.5. If None, this constraint is ignored.
    min_height_head : float, optional
        Minimum height of the ball at the time of an event with bodypart head,
        by default 1.0. If None, this constraint is ignored.

    Returns
    -------
    bool
        True if the ball is within the specified height range or if the ball
        height is unknown.
    """
    ball_height = fs.ball_height(frame)
    if math.isnan(ball_height):
        return True  # we don't know the ball height

    bodypart = event.get_qualifier_value(BodyPartQualifier)
    if (
        bodypart in [BodyPart.LEFT_FOOT, BodyPart.RIGHT_FOOT]
        and max_height_foot is not None
    ):
        return ball_height <= max_height_foot
    elif bodypart == BodyPart.HEAD and min_height_head is not None:
        return ball_height >= min_height_head and ball_height <= max_height

    return ball_height <= max_height  # we don't know the bodypart


def mask_player_ball_acceleration(
    event: Event,
    frame: Frame,
    min_acceleration: float = 0.0,
    max_ball_speed: float = config.max_speed_ball,
    smoothing_window: int = config.smoothing_window_speed,
):
    """The ball must be accelerating away from the player.

    Computes the second derivative of the Euclidean distance between the player
    and the ball.

    Parameters
    ----------
    event : Event
    frame : Frame
    min_acceleration : float, optional
        Minimum acceleration of the ball away from the player, by default 0.0
    max_ball_speed : float
        The maximum speed that the ball can realisitically achieve (in
        meters/second). Speed measures that exceed `max_speed` are tagged as
        outliers and set to NaN.
    smoothing_window : int
        The window size for smoothing the ball speed.

    Returns
    -------
    bool
        True if the ball is accelerating away from the player with at least
        `min_acceleration` m/s^2.
    """
    player_ball_acceleration = fs.acceleration_player_ball(
        frame, event.player, max_speed=max_ball_speed, window=smoothing_window
    )
    if math.isnan(player_ball_acceleration):
        return True
    return player_ball_acceleration >= min_acceleration


def mask_player_ball_deceleration(
    event: Event,
    frame: Frame,
    min_deceleration: float = 0.0,
    max_ball_speed: float = config.max_speed_ball,
    smoothing_window: int = config.smoothing_window_speed,
):
    """The ball must be decelerating towards the player.

    Computes the second derivative of the Euclidean distance between the player
    and the ball.

    Parameters
    ----------
    event : Event
    frame : Frame
    min_deceleration : float, optional
        Minimum deceleration of the ball away from the player, by default 0.0
    max_ball_speed : float
        The maximum speed that the ball can realisitically achieve (in
        meters/second). Speed measures that exceed `max_speed` are tagged as
        outliers and set to NaN.
    smoothing_window : int
        The window size for smoothing the ball speed.

    Returns
    -------
    bool
        True if the ball is decelerating towards the acting player with at least
        `min_deceleration` m/s^2.
    """
    player_ball_acceleration = fs.acceleration_player_ball(
        frame, event.player, max_speed=max_ball_speed, window=smoothing_window
    )
    if math.isnan(player_ball_acceleration):
        return True
    return -player_ball_acceleration >= min_deceleration


def mask_ball_acceleration(
    event: Event,
    frame: Frame,
    min_acceleration: float = 0.0,
    max_ball_speed: float = config.max_speed_ball,
    smoothing_window: int = config.smoothing_window_speed,
):
    """The ball must be accelerating.

    Computes the second derivative of the Euclidean distance between the current
    and previous ball coordinates.

    Parameters
    ----------
    event : Event
    frame : Frame
    min_acceleration : float, optional
        Minimum acceleration of the ball, by default 0.0
    max_ball_speed : float
        The maximum speed that the ball can realisitically achieve (in
        meters/second). Speed measures that exceed `max_speed` are tagged as
        outliers and set to NaN.
    smoothing_window : int
        The window size for smoothing the ball speed.

    Returns
    -------
    bool
        True if the ball is accelerating with at least `min_acceleration` m/s^2.

    """
    ball_acceleration = fs.ball_acceleration(
        frame, max_speed=max_ball_speed, window=smoothing_window
    )
    if math.isnan(ball_acceleration):
        return True
    return ball_acceleration >= min_acceleration


def mask_ball_deceleration(
    event: Event,
    frame: Frame,
    min_deceleration: float = 0.0,
    max_ball_speed: float = config.max_speed_ball,
    smoothing_window: int = config.smoothing_window_speed,
):
    """The ball must be decelerating.

    Computes the second derivative of the Euclidean distance between the current
    and previous ball coordinates.

    Parameters
    ----------
    event : Event
    frame : Frame
    min_deceleration : float, optional
        Minimum deceleration of the ball, by default 0.0
    max_ball_speed : float
        The maximum speed that the ball can realisitically achieve (in
        meters/second). Speed measures that exceed `max_speed` are tagged as
        outliers and set to NaN.
    smoothing_window : int
        The window size for smoothing the ball speed.

    Returns
    -------
    bool
        True if the ball is decelerating with at least `min_deceleration` m/s^2.

    """
    ball_acceleration = fs.ball_acceleration(
        frame, max_speed=max_ball_speed, window=smoothing_window
    )
    if math.isnan(ball_acceleration):
        return True
    return -ball_acceleration >= min_deceleration


def apply_if(
    condition: Callable[[Event], bool],
    mask_func: Callable[[Event, Frame], bool],
):
    """Apply a masking function if a condition is met.

    Parameters
    ----------
    condition : Callable[[Event], bool]
        Condition function.
    mask_func : Callable[[Event, Frame], bool]
        Masking function.

    Returns
    -------
    Callable[[Event, Frame], bool]
        Masking function.
    """

    def inner(event: Event, frame: Frame) -> bool:
        if condition(event):
            return mask_func(event, frame)
        return True

    return inner


def combine(*mask_funcs) -> Callable[[Event, List[Frame]], List[bool]]:
    """Combine an arbitrary number of masking functions to a single one.

    Takes the conjunction of all masking functions.

    Parameters
    ----------
    *mask_funcs : Callable[[Event, Frame], bool]
        Masking functions.

    Returns
    -------
    (Event, list(Frame)) -> list(bool)
        Combined masking function.
    """

    def inner(event: Event, frames: List[Frame]) -> List[bool]:
        return [all(f(event, frame) for f in mask_funcs) for frame in frames]

    return inner


def combine_and_reduce(
    score_func, *mask_funcs
) -> Callable[[Event, List[Frame]], List[bool]]:
    """Combine an arbitrary number of masking functions to a single one.

    Takes the conjunction of all masking functions. If a window of consecutive
    frames is masked, the frame with the highest score is selected. This can be
    used to select the frame with the highest ball acceleration in a possession
    window, for example.

    Parameters
    ----------
    score_func : Callable[[Event, Frame], float]
        Score function to select the best frame in a masked window. The frame
        with the lowest score is selected.
    *mask_funcs : Callable[[Event, Frame], bool]
        Masking functions.

    Returns
    -------
    (Event, list(Frame)) -> list(bool)
        Combined masking function.
    """

    def find_runs(a):
        # Create an array that is 1 where a is `value`, and pad each end with an extra 0.
        isvalue = np.concatenate(([0], np.equal(a, True).view(np.int8), [0]))
        absdiff = np.abs(np.diff(isvalue))
        # Runs start and end where absdiff is 1.
        ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
        return ranges

    def inner(event: Event, frames: List[Frame]) -> List[bool]:
        masked_frames = [
            all(f(event, frame) for f in mask_funcs) for frame in frames
        ]
        masked_windows = [False] * len(masked_frames)
        for s, e in find_runs(np.array(masked_frames)):
            scores = [score_func(event, frames[i]) for i in range(s, e)]
            try:
                best_idx = np.nanargmin(scores)
                masked_windows[s + best_idx] = True
            except ValueError:
                pass
        return masked_windows

    return inner
