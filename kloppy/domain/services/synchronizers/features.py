"""Feature functions for tracking / event data."""
import math
from functools import partial
from typing import Callable, Optional

from kloppy.domain import BallState, Frame, Player, PlayerData, Point3D

from .utils import eucl


def _get_player_data(frame: Frame, player: Player) -> Optional[PlayerData]:
    player_side = player.team.ground
    player_number = player.jersey_no

    try:
        player_tracking_data = next(
            data
            for player, data in frame.players_data.items()
            if player.jersey_no == player_number
            and player.team.ground == player_side
        )
    except StopIteration:
        return None

    return player_tracking_data


def _smooth_ma(func: Callable[[Frame, Optional[int]], float], window: int):
    """Apply moving average smoothing."""

    def inner(frame):
        raw_values = []
        prev_frame = frame
        for _ in range(window // 2):
            if prev_frame.prev_record is None:
                return float("nan")
            prev_frame = prev_frame.prev_record
            raw_values.append(func(prev_frame, window=None))
        raw_values.append(func(frame, window=None))
        next_frame = frame
        for _ in range(window // 2):
            if next_frame.next_record is None:
                return float("nan")
            next_frame = next_frame.next_record
            raw_values.append(func(next_frame, window=None))
        return sum(raw_values) / len(raw_values)

    return inner


def ball_state(frame: Frame, infer: bool) -> Optional[BallState]:
    """State of the ball.

    Parameters
    ----------
    frame : Frame
        The frame for which the ball state should be computed.
    infer : bool
        Whether to infer the ball state from the ball coordinates if the
        `ball_state` attribute is not defined.

    Returns
    -------
    BallState
        If defined, the value of the frame's `ball_state` attribute.
        Otherwise, it will check the `ball_coordinates` attribute and return
        `BallState.DEAD` if the coordinates are not defined or missing.
    """
    # return frame.ball_state
    if frame.ball_state is not None:
        return frame.ball_state

    # infer ball state from ball coordinates
    if infer:
        if frame.ball_coordinates is None:
            return BallState.DEAD
        if math.isnan(frame.ball_coordinates.x):
            return BallState.DEAD
        return BallState.ALIVE
    else:
        return None


def ball_height(frame: Frame) -> float:
    """Height of the ball.

    Parameters
    ----------
    frame : Frame
        The frame for which the ball height should be computed.

    Returns
    -------
    float
        The z-coordinate of the ball, or `nan` if the height is not defined.
    """
    ball_coords = frame.ball_coordinates
    if isinstance(ball_coords, Point3D) and ball_coords.z is not None:
        return ball_coords.z
    return float("nan")


def ball_speed(
    frame: Frame,
    max_speed: float,
    window: int,
) -> float:
    """Speed of the ball in m/s.

    Parameters
    ----------
    frame : Frame
        The frame for which the ball speed should be computed.
    max_speed : float
        The maximum speed that the ball can realisitically achieve (in
        meters/second). Speed measures that exceed `max_speed` are tagged as
        outliers and set to NaN.
    window : int
        The window size for smoothing the ball speed.

    Returns
    -------
    float
        If defined, the value of the frame's `ball_speed` attribute.
        Otherwise, it will compute the speed from the current and previous
        frame.
    """
    # return frame.ball_speed
    if frame.ball_speed is not None:
        return frame.ball_speed

    # compute moving average over window
    if window is not None:
        func = partial(ball_speed, max_speed=max_speed)
        return _smooth_ma(func, window)(frame)

    # compute speed for current frame
    next_frame = frame.next_record
    if next_frame is None:
        return float("nan")
    ball_coords = frame.ball_coordinates
    next_ball_coords = next_frame.ball_coordinates
    timestamp = frame.timestamp
    next_timestamp = next_frame.timestamp
    if ball_coords is None or next_ball_coords is None:
        return float("nan")

    dist = eucl(ball_coords, next_ball_coords)
    dt = next_timestamp - timestamp
    speed = dist / dt if dt > 0 else float("nan")

    if max_speed is not None and speed > max_speed:
        return float("nan")

    return max(abs(speed), 1e-10)  # avoid zero


def ball_acceleration(
    frame: Frame,
    max_speed: float,
    window: int,
) -> float:
    """Acceleration of the ball.

    Parameters
    ----------
    frame : Frame
        The frame for which the ball acceleration should be computed.
    max_speed : float
        The maximum speed that the ball can realisitically achieve (in
        meters/second). Speed measures that exceed `max_speed` are tagged as
        outliers and set to NaN.
    window : int
        The window size for smoothing the ball acceleration.

    Returns
    -------
    float
        The acceleration of the ball, compute from the current and two
        previous frames.
    """
    prev_frame = frame.prev_record
    if prev_frame is None:
        return float("nan")

    speed = ball_speed(frame, max_speed=max_speed, window=window)
    prev_speed = ball_speed(prev_frame, max_speed=max_speed, window=window)
    timestamp = frame.timestamp
    prev_timestamp = prev_frame.timestamp
    dspeed = speed - prev_speed
    dt = timestamp - prev_timestamp
    acceleration = dspeed / dt if dt > 0 else float("nan")
    return acceleration


def player_speed(
    frame: Frame,
    player: Player,
    max_speed: float,
    window: int,
) -> float:
    """Speed of a player in m/s.

    Parameters
    ----------
    frame : Frame
        The frame for which the player's speed should be computed.
    player : Player
        The player for which the speed should be computed.
    max_speed : float
        The maximum speed that the player can realisitically achieve (in
        meters/second). Speed measures that exceed maxspeed are tagged as
        outliers and set to NaN.
    window : int
        The window size for smoothing the player's speed, by default `None`.

    Returns
    -------
    float
        If defined, the value of the player's `speed` attribute.
        Otherwise, it will compute the speed from the current and previous
        frame.
    """
    player_data = _get_player_data(frame, player)
    if player_data is None:
        return float("nan")

    # return frame.ball_speed
    if player_data.speed is not None:
        return player_data.speed

    # compute moving average over window
    if window is not None:
        func = partial(player_speed, player=player, max_speed=max_speed)
        return _smooth_ma(func, window)(frame)

    # compute speed for current frame
    next_frame = frame.next_record
    if next_frame is None:
        return float("nan")
    next_player_data = _get_player_data(next_frame, player)
    if next_player_data is None:
        return float("nan")
    player_coords = player_data.coordinates
    next_player_coords = next_player_data.coordinates
    timestamp = frame.timestamp
    next_timestamp = next_frame.timestamp
    if player_coords is None or next_player_coords is None:
        return float("nan")

    dist = eucl(player_coords, next_player_coords)
    dt = next_timestamp - timestamp
    speed = dist / dt if dt > 0 else float("nan")

    if max_speed is not None and speed > max_speed:
        return float("nan")

    return max(abs(speed), 1e-10)  # avoid zero


def player_acceleration(
    frame: Frame,
    player: Player,
    max_speed: float,
    window: int,
) -> float:
    """Acceleration of the ball.

    Parameters
    ----------
    frame : Frame
        The frame for which the ball acceleration should be computed.
    player : Player
        The player for which the acceleration should be computed.
    max_speed : float
        The maximum speed that the player can realisitically achieve (in
        meters/second). Speed measures that exceed maxspeed are tagged as
        outliers and set to NaN.
    window : int
        The window size for smoothing the player's speed.

    Returns
    -------
    float
        The acceleration of the ball, compute from the current and two
        previous frames.
    """
    prev_frame = frame.prev_record
    if prev_frame is None:
        return float("nan")

    speed = player_speed(frame, player, max_speed=max_speed, window=window)
    prev_speed = player_speed(
        prev_frame, player, max_speed=max_speed, window=window
    )
    timestamp = frame.timestamp
    prev_timestamp = prev_frame.timestamp
    dspeed = speed - prev_speed
    dt = timestamp - prev_timestamp
    acceleration = dspeed / dt if dt > 0 else float("nan")
    return acceleration


def dist_player_ball(frame: Frame, player: Player) -> float:
    """Euclidean distance between a player and the ball.

    Parameters
    ----------
    frame : Frame
        The frame for which the distance should be computed.
    player : Player
        The player for which the distance to the ball should be computed.

    Returns
    -------
    float
        The Euclidean distance between the player and the ball, or `nan` if
        the distance cannot be computed. This can happen if the player or
        ball coordinates are not defined.
    """
    player_side = player.team.ground
    player_number = player.jersey_no
    try:
        player_tracking_data = next(
            data
            for player, data in frame.players_data.items()
            if player.jersey_no == player_number
            and player.team.ground == player_side
        )
    except StopIteration:
        return float("nan")

    player_coords = player_tracking_data.coordinates
    ball_coords = frame.ball_coordinates
    if player_coords is None or ball_coords is None:
        return float("nan")
    return eucl(player_coords, ball_coords)


def speed_player_ball(
    frame: Frame,
    player: Player,
    max_speed: float,
    window: int,
) -> float:
    """Speed of the ball relative to a player.

    Parameters
    ----------
    frame : Frame
        The frame for which the acceleration should be computed.
    player : Player
        The player for which the acceleration should be computed.
    max_speed : float
        The maximum speed that the ball can realisitically achieve (in
        meters/second). Speed measures that exceed maxspeed are tagged as
        outliers and set to NaN.
    window : int
        The window size for smoothing the speed.

    Returns
    -------
    float
        The acceleration of the ball relative to the player, or `nan` if the
        acceleration cannot be computed. This can happen if the player or
        ball coordinates are not defined.
    """
    if window is not None:
        func = partial(speed_player_ball, player=player, max_speed=max_speed)
        return _smooth_ma(func, window)(frame)

    next_frame = frame.next_record
    if next_frame is None:
        return float("nan")
    dist = dist_player_ball(frame, player)
    next_dist = dist_player_ball(next_frame, player)
    timestamp = frame.timestamp
    next_timestamp = next_frame.timestamp
    ddist = next_dist - dist
    dt = next_timestamp - timestamp
    speed = ddist / dt if dt > 0 else float("nan")
    return max(abs(speed), 1e-10)  # avoid zero


def acceleration_player_ball(
    frame: Frame,
    player: Player,
    max_speed: float,
    window: int,
) -> float:
    """Acceleration of the ball relative to a player.

    Parameters
    ----------
    frame : Frame
        The frame for which the acceleration should be computed.
    player : Player
        The player for which the acceleration should be computed.
    max_speed : float
        The maximum speed that the ball can realisitically achieve (in
        meters/second). Speed measures that exceed maxspeed are tagged as
        outliers and set to NaN.
    window : int
        The window size for smoothing the speed.

    Returns
    -------
    float
        The acceleration of the ball relative to the player, or `nan` if the
        acceleration cannot be computed. This can happen if the player or
        ball coordinates are not defined.
    """
    prev_frame = frame.prev_record
    if prev_frame is None:
        return float("nan")
    speed = speed_player_ball(
        frame, player, max_speed=max_speed, window=window
    )
    prev_speed = speed_player_ball(
        prev_frame, player, max_speed=max_speed, window=window
    )
    timestamp = frame.timestamp
    prev_timestamp = prev_frame.timestamp
    dspeed = speed - prev_speed
    dt = timestamp - prev_timestamp
    acceleration = dspeed / dt if dt > 0 else float("nan")
    return acceleration


def dist_receiver_ball(
    frame: Frame, receiver: Player, duration: float
) -> float:
    """Euclidean distance between a receiving player and the ball.

    The distance is computed as the minimum distance between the player and
    the ball within a certain time window after the pass moment.

    Parameters
    ----------
    frame : Frame
        The frame for which the distance should be computed.
    receiver : Player
        The player for which the distance to the ball should be computed.
    duration : float
        The maximum time (in seconds) after the pass moment for which the
        distance should be computed.

    Returns
    -------
    float
        The Euclidean distance between the player and the ball, or `nan` if
        the distance cannot be computed. This can happen if the player or
        ball coordinates are not defined.
    """
    best_dist = dist_player_ball(frame, receiver)

    next_frame = frame
    while (
        next_frame.next_record is not None
        and next_frame.next_record.timestamp - frame.timestamp < duration
    ):
        next_frame = next_frame.next_record
        dist = dist_player_ball(next_frame, receiver)
        if dist < best_dist or math.isnan(best_dist):
            best_dist = dist

    return best_dist
