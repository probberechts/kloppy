import math
import random
from dataclasses import replace

from kloppy.domain import (
    Orientation,
    Point,
    EventDataset,
    TrackingDataset,
    SkillCornerCoordinateSystem,
    SportecTrackingDataCoordinateSystem,
    SportecEventDataCoordinateSystem,
    SecondSpectrumCoordinateSystem,
    UEFACoordinateSystem,
)


def normalize_datasets(events: EventDataset, frames: TrackingDataset):
    """Transform the events and frames to the same metric coordinate system.

    To be able to compute distances, we need to transform the events and frames
    to the same coordinate system with metric pitch dimensions and orientation.

    If the coordinate system of the events and frames are already compatible,
    this function will return a copy of the events.
    """
    metric_coordinate_systems = (
        SkillCornerCoordinateSystem,
        SportecTrackingDataCoordinateSystem,
        SportecEventDataCoordinateSystem,
        SecondSpectrumCoordinateSystem,
        UEFACoordinateSystem,
    )
    if (
        events.metadata.coordinate_system != frames.metadata.coordinate_system
        or events.metadata.orientation != frames.metadata.orientation
        or not isinstance(
            events.metadata.coordinate_system, metric_coordinate_systems
        )
        or events.metadata.orientation
        in (Orientation.ACTION_EXECUTING_TEAM, Orientation.BALL_OWNING_TEAM)
    ):
        # determine the cheapest transformation to a common coordinate system
        # and orientation. Since transforming frames is more expensive than
        # transforming events, we try to keep the coordinate system and
        # orientation of the frames.
        if not isinstance(
            frames.metadata.coordinate_system, metric_coordinate_systems
        ):
            to_coordinate_system = UEFACoordinateSystem(normalized=False)
        else:
            to_coordinate_system = frames.metadata.coordinate_system
        if frames.metadata.orientation in (
            Orientation.ACTION_EXECUTING_TEAM,
            Orientation.BALL_OWNING_TEAM,
        ):
            to_orientation = Orientation.FIXED_HOME_AWAY
        else:
            to_orientation = frames.metadata.orientation

        events_norm = events.transform(
            to_coordinate_system=to_coordinate_system,
            to_orientation=to_orientation,
        )
        frames_norm = frames.transform(
            to_coordinate_system=to_coordinate_system,
            to_orientation=to_orientation,
        )
        return events_norm, frames_norm

    return (
        EventDataset(
            metadata=events.metadata,
            records=[replace(e) for e in events.records],
        ),
        frames,
    )


def eucl(a: Point, b: Point) -> float:
    """Euclidean distance between a pair of points."""
    dist = ((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5
    if math.isnan(dist):
        return math.inf
    return dist


def unsync(event_dataset, offset=2, sigma_ts=1, sigma_loc=0.01, seed=1234):
    """Add noise to the timestamp and location of each event."""
    random.seed(seed)
    unsync_events = []
    for i, event in enumerate(event_dataset):
        attempts = 1
        noise_ts = random.gauss(mu=0, sigma=sigma_ts)
        noise_x = random.gauss(mu=0, sigma=sigma_loc)
        noise_y = random.gauss(mu=0, sigma=sigma_loc)
        # make sure the order of the events does not change
        while (
            (
                (
                    event.prev_record
                    and event.timestamp + noise_ts
                    < event.prev_record.timestamp
                )
                or (
                    event.next_record
                    and event.timestamp + noise_ts
                    > event.next_record.timestamp
                )
            )
            and attempts
            < 20  # we need this because some events are not correctly ordered
        ):
            noise_ts = random.gauss(mu=0, sigma=sigma_ts)
            attempts += 1
        unsync_event = replace(
            event,
            timestamp=event.timestamp + noise_ts + offset
            if i > 0
            else event.timestamp + offset,
            coordinates=Point(
                x=event.coordinates.x + noise_x,
                y=event.coordinates.y + noise_y,
            )
            if event.coordinates
            else None,
        )
        unsync_events.append(unsync_event)
    return replace(event_dataset, records=unsync_events)
