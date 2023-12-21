import logging
import warnings
from functools import partial
from typing import Callable, List, Literal, Optional, Tuple

import numpy as np

from kloppy.domain import (
    Event,
    EventType,
    EventDataset,
    Frame,
    Point,
    SetPieceQualifier,
    SetPieceType,
    TrackingDataset,
)
from kloppy.exceptions import SynchronizationError

from . import config, masking, scoring
from .strategy import SynchronizationStrategy
from .utils import normalize_datasets

logger = logging.getLogger(__name__)

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


class EventTrackingSynchronizer:
    """Synchronizes an event dataset with a tracking dataset.

    Parameters
    ----------
    strategy : SynchronizationStrategy
        The synchronization strategy to use.
    is_handled : (Event) -> bool | list(EventType), default: config.is_handled
        A function that returns True if the event should be considered for
        alignment. If a list of event types is given, only events of these
        types are considered for alignment.
    """

    def __init__(
        self,
        strategy: SynchronizationStrategy,
        is_handled: Callable | List[EventType] = config.is_handled,
    ):
        self.strategy = strategy

        # Parse handled events
        if callable(is_handled):
            self.is_handled = is_handled
        else:
            self.is_handled = lambda event: event.event_type in is_handled

    def _find_kickoff(
        self,
        events: List[Event],
        frames: List[Frame],
        fps: float,
        score_fn: Optional[Callable] = None,
        mask_fn: Optional[Callable] = None,
    ) -> Tuple[Frame, float]:
        """Searches for the kickoff frame.

        Parameters
        ----------
        events : list(Event)
            The events in a period.
        frames : list(Frame)
            The tracking frames in a period.
        fps : float
            The number of frames per second in the tracking sequence.
        score_fn: (Event, Frame) -> float, optional
            The scoring function to use to score the match between each
            event-frame pair. Lower values indicate a better match. If None,
            a default distance function is used which scores frames based on
            the acceleration of the ball relative to the player who kicked off
            and the ball state.
        mask_fn : (Event, Frame) -> bool, optional
            A function that returns True if the event-frame pair should be
            considered for alignment. If None, a default masking function is
            used which masks event-frame pairs where other players are closer
            to the ball than the acting player or where the ball is futher than
            2 meters away from the center of the field.

        Returns
        -------
        Frame
            The found kickoff frame.
        float
            The offset between the kickoff event and the kickoff frame in
            seconds. A negative offset means that the kickoff event is before
            the kickoff frame.
        """
        # Parse default parameters
        if score_fn is None:
            score_fn = scoring.combine(
                scoring.mis_player_ball_acceleration,
                scoring.mis_ball_state,
            )
        if mask_fn is None:
            mask_fn = masking.combine(
                partial(masking.mask_ball_location, max_dist=2.0),
                partial(masking.mask_ball_possession, max_dist=2.0),
                partial(masking.mask_ball_control, own_team_only=False),
            )

        # Select the kickoff event
        try:
            kickoff_event = next(
                event
                for event in events
                if event.get_qualifier_value(SetPieceQualifier)
                == SetPieceType.KICK_OFF
            )
            # Some deserializers do not set the coordinates of the kickoff event
            kickoff_event.coordinates = Point(52.5, 34)
            logger.debug(
                f"Found kickoff event at ts={kickoff_event.timestamp}"
            )
        except StopIteration:
            raise SynchronizationError("Could not find a kickoff event.")

        # Find the frame idx that matches the kickoff event based on the timestamp
        frame_idx = int(fps * kickoff_event.timestamp)

        # Check the frames with a timestamp before the kickoff event
        # and within 60 seconds after the kickoff event
        nb_frames = len(frames)
        frames_to_check = slice(
            0,
            min(nb_frames, int(frame_idx + fps * 60)),
        )
        mask = mask_fn(kickoff_event, frames[frames_to_check])
        scores = score_fn(kickoff_event, frames[frames_to_check], mask)

        best_idx = np.nanargmin(scores)
        if scores[best_idx] == np.inf:
            raise SynchronizationError(
                "Could not match a tracking frame to the kickoff event."
            )
        kickoff_frame = frames[best_idx]
        logger.debug(f"Found kickoff frame at ts={kickoff_frame.timestamp}")

        return kickoff_frame, kickoff_event.timestamp - kickoff_frame.timestamp

    def sync(
        self,
        events: EventDataset,
        frames: TrackingDataset,
        offset: Optional[Literal["auto"] | List[float]] = "auto",
        show_progress: bool = False,
    ) -> EventDataset:
        """Synchronizes the event dataset with the tracking dataset.


        Parameters
        ----------
        events : EventDataset
            The event dataset to sync.
        frames : TrackingDataset
            The tracking dataset to sync to.
        offset : list(float) | 'auto'
            The offset between the kickoff event and the kickoff frame in
            seconds for each period. A negative offset means that the kickoff
            event is before the kickoff frame. If 'auto', the kickoff event is
            automatically detected. If `None`, no offset is applied.
        show_progress : bool, default: False
            If True, show a progress bar. This requires the tqdm package.

        Returns
        -------
        EventDataset
            The synchronized event dataset. The tracking frames are stored in
            the `freeze_frame` attribute of each event.
        """
        fps = frames.metadata.frame_rate
        periods = events.metadata.periods

        if fps is None:
            raise ValueError("The tracking dataset must have a frame rate.")

        if isinstance(offset, (list, tuple)) and len(offset) != len(periods):
            raise ValueError("You must provide an offset for each period.")

        # Transform the events and frames to the same coordinate system
        # and orientation with metric pitch dimensions.
        # This also makes a copy of the event dataset such that we do not
        # modify the original dataset.
        # TODO: this is inefficient, but we need this to compute distances
        # and speeds correctly.
        events_norm, frames_norm = normalize_datasets(events, frames)

        # Sync each period separately
        # TODO: this can be parallelized
        for i, period in enumerate(periods):
            period_events = [
                e
                for e in events_norm
                if e.period == period and self.strategy.filter_fn(e)
            ]
            period_frames = [f for f in frames_norm if f.period == period]

            if show_progress:
                if tqdm is None:
                    raise ImportError(
                        "You must install tqdm to show a progress bar."
                    )
                period_events_it = tqdm(
                    period_events, desc=f"Synchronizing period {period.id}"
                )
            else:
                logger.info(f"Synchronizing period {period.id}")
                period_events_it = period_events

            # Parse offset
            if offset == "auto":
                try:
                    _, period_offset = self._find_kickoff(
                        period_events, period_frames, fps
                    )
                except SynchronizationError:
                    warnings.warn(
                        f"Could not find kickoff for period {period.id}. Using offset 0."
                    )
                    period_offset = 0
            elif isinstance(offset, (list, tuple)):
                period_offset = offset[i]
            elif offset is None:
                period_offset = 0
            else:
                raise ValueError("Invalid offset parameter.")

            period_offset += period.start_timestamp

            # Align the events and frames
            alignment = self.strategy(
                period_events_it, period_frames, fps, period_offset
            )

            # Add the aligned frames to the events
            for event_idx, frame_idx in alignment:
                if frame_idx is not None and frame_idx > len(period_frames):
                    break
                event = period_events[event_idx]
                frame = (
                    period_frames[frame_idx] if frame_idx is not None else None
                )
                if frame is not None:
                    # TODO: should we also update the event's coordinates and timestamp?
                    # event.coordinates = frame.ball_coordinates
                    # event.timestamp = frame.timestamp
                    event.freeze_frame = frame

        # Transform the events back to the original coordinate system
        return events_norm.transform(
            to_coordinate_system=events.metadata.coordinate_system,
            to_orientation=events.metadata.orientation,
        )
