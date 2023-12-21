import math
import random

from dataclasses import dataclass, replace

from kloppy.domain import (
    CoordinateSystem,
    Dimension,
    Orientation,
    Origin,
    PitchDimensions,
    Point,
    Provider,
    VerticalOrientation,
)


@dataclass
class UEFACoordinateSystem(CoordinateSystem):
    """UEFA coordinate system.

    This coordinate system has the origin on the bottom left of the pitch, and
    a uniform field of 105m x 68m.
    """

    @property
    def provider(self) -> Provider:
        return Provider.OTHER

    @property
    def origin(self) -> Origin:
        return Origin.BOTTOM_LEFT

    @property
    def vertical_orientation(self) -> VerticalOrientation:
        return VerticalOrientation.BOTTOM_TO_TOP

    @property
    def pitch_dimensions(self) -> PitchDimensions:
        return PitchDimensions(
            x_dim=Dimension(0, 105),
            y_dim=Dimension(0, 68),
        )


def normalize_datasets(events, frames):
    # Transform the events and frames to the same coordinate system
    # and orientation with true pitch dimensions. We use the
    # UEFACoordinateSystem such that we can compute correct distances.
    # This also makes a copy of the event dataset such that we do not
    # modify the original dataset.
    events_norm = events.transform(
        to_coordinate_system=UEFACoordinateSystem(normalized=False),
        to_orientation=Orientation.FIXED_HOME_AWAY,
    )
    frames_norm = frames.transform(
        to_coordinate_system=UEFACoordinateSystem(normalized=False),
        to_orientation=Orientation.FIXED_HOME_AWAY,
    )
    return events_norm, frames_norm


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
