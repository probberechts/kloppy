"""Functions for sequence alignment."""
from enum import Enum
from typing import Callable, List, Optional

import numpy as np

from kloppy.domain import Event, Frame


class Direction(Enum):
    """Define Unicode arrows we'll use in the traceback matrix."""

    UP = "\u2191"
    RIGHT = "\u2192"
    DOWN = "\u2193"
    LEFT = "\u2190"
    DOWN_RIGHT = "\u2198"
    UP_LEFT = "\u2196"


def _get_frames_to_check(
    event: Event,
    idx_start: int,
    idx_end: int,
    fps: float,
    offset: float,
    window: float,
) -> slice:
    return slice(
        max(
            idx_start,
            int(fps * event.timestamp - window / 2 * fps - offset * fps),
        ),
        min(
            int(fps * event.timestamp + window / 2 * fps - offset * fps),
            idx_end,
        ),
    )


def timestamp_alignment(
    s1: List[Event],
    s2: List[Frame],
    fps: float,
    score_fn: Callable[
        [Event, List[Frame], Optional[List[bool]]], List[float]
    ],
    mask_fn: Callable[[Event, List[Frame]], List[bool]],
    window_fn: Callable[[Event], float],
    offset: float = 0,
):
    """Alignment based on timestamps."""
    alignment = [
        (
            event_idx,
            round((event.timestamp - offset) * fps),
        )
        for event_idx, event in enumerate(s1)
    ]
    return alignment


def local_alignment(
    s1: List[Event],
    s2: List[Frame],
    fps: float,
    score_fn: Callable[
        [Event, List[Frame], Optional[List[bool]]], List[float]
    ],
    mask_fn: Callable[[Event, List[Frame]], List[bool]],
    window_fn: Callable[[Event], float],
    offset: float = 0,
):
    """Local sequence alignment."""
    c = len(s2)

    p = []
    for i, event in enumerate(s1):
        frames_to_check = _get_frames_to_check(
            event, 0, c, fps, offset, window_fn(event)
        )
        mask = mask_fn(event, s2[frames_to_check])
        scores = score_fn(event, s2[frames_to_check], mask)
        try:
            best_idx = np.nanargmin(scores)
            if scores[best_idx] == np.inf:
                raise ValueError(
                    "Could not match a tracking frame to the kickoff event."
                )
        except ValueError:
            p.append((i, None))
            continue
        best_frame_idx = frames_to_check.start + best_idx
        p.append((i, best_frame_idx))
    return p


def greedy_alignment(
    s1: List[Event],
    s2: List[Frame],
    fps: float,
    score_fn: Callable[
        [Event, List[Frame], Optional[List[bool]]], List[float]
    ],
    mask_fn: Callable[[Event, List[Frame]], List[bool]],
    window_fn: Callable[[Event], float],
    offset: float = np.inf,
):
    """Greedy sequence alignment."""
    c = len(s2)

    p = []
    best_frame_idx = -1
    for i, event in enumerate(s1):
        frames_to_check = _get_frames_to_check(
            event, best_frame_idx + 1, c, fps, offset, window_fn(event)
        )
        mask = mask_fn(event, s2[frames_to_check])
        scores = score_fn(event, s2[frames_to_check], mask)
        try:
            best_idx = np.nanargmin(scores)
            if scores[best_idx] == np.inf:
                raise ValueError(
                    "Could not match a tracking frame to the kickoff event."
                )
        except ValueError:
            p.append((i, None))
            continue
        best_frame_idx = frames_to_check.start + best_idx
        p.append((i, best_frame_idx))
    return p


def optimal_alignment(
    s1: List[Event],
    s2: List[Frame],
    fps: float,
    score_fn: Callable[
        [Event, List[Frame], Optional[List[bool]]], List[float]
    ],
    mask_fn: Callable[[Event, List[Frame]], List[bool]],
    window_fn: Callable[[Event], float],
    offset: float = 0,
    max_score: float = 10,
):
    """Optimal sequence alignment using dynamic programming."""
    r, c = len(s1), len(s2)
    # Init scoring matrix
    scores = np.full((r + 1, c + 1), np.inf)
    scores[0, :] = 0
    scores[1:, 0] = np.cumsum([max_score] * r)

    # Init traceback matrix
    paths = np.full([r + 1, c + 1], Direction.LEFT.value, dtype="<U4")

    # Fill the scoring and traceback matrices
    i1, j1 = 0, 0
    for i0, event in enumerate(s1):
        i1 = i0 + 1
        frames_to_check = _get_frames_to_check(
            event, 0, c, fps, offset, window_fn(event)
        )
        d_mask = mask_fn(event, s2[frames_to_check])
        d_scores = score_fn(event, s2[frames_to_check], d_mask)
        i = i0
        while i > 0 and np.isinf(scores[i, :]).all():
            i -= 1
        for j0 in range(frames_to_check.start, frames_to_check.stop):
            j1 = j0 + 1
            from_left_score = (
                0 + scores[i1, j0]
            )  # representing an insertion in the event sequence
            from_diag_score = (
                d_scores[j0 - frames_to_check.start] + scores[i, j1]
            )  # representing a match
            from_up_score = (
                max_score + scores[i, j1]
            )  # representing a deletion in the event sequence
            scores[i1, j1] = min(
                from_left_score, from_diag_score, from_up_score
            )
            # make note of which cell was best in the traceback array
            if scores[i1, j1] == from_left_score:
                paths[i1, j1] = Direction.LEFT.value
            elif scores[i1, j1] == from_diag_score:
                paths[i1, j1] = Direction.UP_LEFT.value
            elif scores[i1, j1] == from_up_score:
                paths[i1, j1] = Direction.UP.value
        scores[i1, j1:] = scores[i1, j1]

    # Traceback to find the optimal path
    p = []
    ops = {
        Direction.UP_LEFT.value: (-1, -0, True),
        Direction.LEFT.value: (-0, -1, False),
        Direction.UP.value: (-1, -0, False),
    }
    i, j = int(paths.shape[0] - 1), int(paths.shape[1] - 1)
    while i > 0 and (paths[i] == Direction.LEFT.value).all():
        i -= 1
        p.append((i, None))
    while i > 0 and j > 0:
        opi, opj, match = ops[paths[i, j]]
        i, j = i + opi, j + opj
        if opi < 0:
            if match:
                p.append((i, j - 1))
            while (paths[i] == Direction.LEFT.value).all():
                i -= 1
                p.append((i, None))

    p.reverse()
    return p
