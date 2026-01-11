from dataclasses import fields, replace
from typing import Optional, Union
import warnings

from kloppy.domain import (
    DEFAULT_PITCH_LENGTH,
    DEFAULT_PITCH_WIDTH,
    AttackingDirection,
    CoordinateSystem,
    CustomCoordinateSystem,
    Dataset,
    DatasetFlag,
    DatasetType,
    EventDataset,
    Frame,
    Orientation,
    Period,
    PitchDimensions,
    Point,
    Point3D,
    Provider,
    ProviderCoordinateSystem,
    Team,
    TrackingDataset,
    build_coordinate_system,
)
from kloppy.domain.models.event import Event
from kloppy.domain.models.tracking import PlayerData
from kloppy.exceptions import (
    KloppyError,
    MissingPitchSizeWarning,
    warn_missing_pitch_dimensions,
)


class DatasetTransformer:
    def __init__(
        self,
        from_coordinate_system: Optional[CoordinateSystem] = None,
        from_pitch_dimensions: Optional[PitchDimensions] = None,
        from_orientation: Optional[Orientation] = None,
        to_coordinate_system: Optional[CoordinateSystem] = None,
        to_pitch_dimensions: Optional[PitchDimensions] = None,
        to_orientation: Optional[Orientation] = None,
    ):
        if (
            from_pitch_dimensions
            and from_coordinate_system
            or to_pitch_dimensions
            and to_coordinate_system
        ):
            raise ValueError(
                "You can't specify both a PitchDimension and CoordinateSystem transformation on the same Transformer instance"
            )

        self._from_coordinate_system = from_coordinate_system
        if from_pitch_dimensions:
            self._from_pitch_dimensions = from_pitch_dimensions
        elif from_coordinate_system:
            self._from_pitch_dimensions = (
                from_coordinate_system.pitch_dimensions
            )
        else:
            raise ValueError(
                "You must either specify the source PitchDimension or CoordinateSystem"
            )

        self._to_coordinate_system = to_coordinate_system
        if to_pitch_dimensions:
            if from_pitch_dimensions is None:
                raise ValueError(
                    "You must specify the source PitchDimension when specifying the target PitchDimension"
                )
            self._to_pitch_dimensions = to_pitch_dimensions
        elif to_coordinate_system:
            if from_coordinate_system is None:
                raise ValueError(
                    "You must specify the source CoordinateSystem when specifying the target CoordinateSystem"
                )
            self._to_pitch_dimensions = to_coordinate_system.pitch_dimensions

        self._from_orientation = from_orientation
        self._to_orientation = to_orientation
        if (
            from_orientation
            and not to_orientation
            or not from_orientation
            and to_orientation
        ):
            raise ValueError(
                "You must specify both the source and target Orientation"
            )

    @property
    def _needs_coordinate_system_change(self):
        return self._from_coordinate_system != self._to_coordinate_system

    @property
    def _needs_pitch_dimensions_change(self):
        return self._from_pitch_dimensions != self._to_pitch_dimensions

    @property
    def _needs_orientation_change(self):
        return self._from_orientation != self._to_orientation

    def change_point_dimensions(
        self, point: Union[Point, Point3D, None]
    ) -> Union[Point, Point3D, None]:
        if point is None:
            return None

        base_pitch_length = (
            self._from_pitch_dimensions.pitch_length or DEFAULT_PITCH_LENGTH
        )
        base_pitch_width = (
            self._from_pitch_dimensions.pitch_width or DEFAULT_PITCH_WIDTH
        )

        point_base = self._from_pitch_dimensions.to_metric_base(
            point, pitch_length=base_pitch_length, pitch_width=base_pitch_width
        )
        point_to = self._to_pitch_dimensions.from_metric_base(
            point=point_base,
            pitch_length=base_pitch_length,
            pitch_width=base_pitch_width,
        )

        return point_to

    def flip_point(
        self, point: Union[Point, Point3D, None]
    ) -> Union[Point, Point3D, None]:
        if not point:
            return None

        x_base = self._to_pitch_dimensions.x_dim.to_base(point.x)
        y_base = self._to_pitch_dimensions.y_dim.to_base(point.y)

        x_base = 1 - x_base
        y_base = 1 - y_base

        x = self._to_pitch_dimensions.x_dim.from_base(x_base)
        y = self._to_pitch_dimensions.y_dim.from_base(y_base)

        if isinstance(point, Point3D):
            return Point3D(x=x, y=y, z=point.z)
        else:
            return Point(x=x, y=y)

    def __needs_flip(
        self,
        period: Period,
        ball_owning_team: Optional[Team] = None,
        action_executing_team: Optional[Team] = None,
    ) -> bool:
        if (
            self._from_orientation is None
            or self._to_orientation is None
            or self._from_orientation == self._to_orientation
        ):
            flip = False
        else:
            if action_executing_team is None:
                action_executing_team = ball_owning_team

            attacking_direction_from = AttackingDirection.from_orientation(
                self._from_orientation,
                period=period,
                ball_owning_team=ball_owning_team,
                action_executing_team=action_executing_team,
            )
            attacking_direction_to = AttackingDirection.from_orientation(
                self._to_orientation,
                period=period,
                ball_owning_team=ball_owning_team,
                action_executing_team=action_executing_team,
            )
            flip = (
                attacking_direction_from != attacking_direction_to
                and attacking_direction_to != AttackingDirection.NOT_SET
            )
        return flip

    def transform_frame(self, frame: Frame) -> Frame:
        # Change coordinate system
        if self._needs_coordinate_system_change:
            frame = self.__change_frame_coordinate_system(frame)

        # Change dimensions
        elif self._needs_pitch_dimensions_change:
            frame = self.__change_frame_dimensions(frame)

        # Flip frame based on orientation
        if self._needs_orientation_change:
            if self.__needs_flip(
                period=frame.period,
                ball_owning_team=frame.ball_owning_team,
            ):
                frame = self.__flip_frame(frame)

        return frame

    def __change_frame_coordinate_system(self, frame: Frame):
        return Frame(
            # doesn't change
            timestamp=frame.timestamp,
            frame_id=frame.frame_id,
            ball_owning_team=frame.ball_owning_team,
            ball_state=frame.ball_state,
            period=frame.period,
            # changes
            ball_coordinates=self.__change_point_coordinate_system(
                frame.ball_coordinates
            ),
            ball_speed=frame.ball_speed,
            players_data={
                key: PlayerData(
                    coordinates=self.__change_point_coordinate_system(
                        player_data.coordinates
                    ),
                    distance=player_data.distance,
                    speed=player_data.speed,
                    other_data=player_data.other_data,
                )
                for key, player_data in frame.players_data.items()
            },
            other_data=frame.other_data,
            statistics=frame.statistics,
        )

    def __change_frame_dimensions(self, frame: Frame):
        return Frame(
            # doesn't change
            timestamp=frame.timestamp,
            frame_id=frame.frame_id,
            ball_owning_team=frame.ball_owning_team,
            ball_state=frame.ball_state,
            period=frame.period,
            # changes
            ball_coordinates=self.change_point_dimensions(
                frame.ball_coordinates
            ),
            players_data={
                key: PlayerData(
                    coordinates=self.change_point_dimensions(
                        player_data.coordinates
                    ),
                    distance=player_data.distance,
                    speed=player_data.speed,
                    other_data=player_data.other_data,
                )
                for key, player_data in frame.players_data.items()
            },
            other_data=frame.other_data,
            statistics=frame.statistics,
        )

    def __change_point_coordinate_system(
        self, point: Union[Point, Point3D, None]
    ) -> Union[Point, Point3D, None]:
        if not point:
            return None

        base_pitch_length = (
            self._from_pitch_dimensions.pitch_length or DEFAULT_PITCH_LENGTH
        )
        base_pitch_width = (
            self._from_pitch_dimensions.pitch_width or DEFAULT_PITCH_WIDTH
        )

        point_base = self._from_pitch_dimensions.to_metric_base(
            point, pitch_length=base_pitch_length, pitch_width=base_pitch_width
        )

        if (
            self._from_coordinate_system.vertical_orientation
            != self._to_coordinate_system.vertical_orientation
        ):
            point_base = replace(
                point_base,
                y=base_pitch_width - point_base.y,
            )

        point_to = self._to_pitch_dimensions.from_metric_base(
            point_base,
            pitch_length=base_pitch_length,
            pitch_width=base_pitch_width,
        )

        return point_to

    def __flip_frame(self, frame: Frame):
        players_data = {}
        for player, data in frame.players_data.items():
            players_data[player] = PlayerData(
                coordinates=self.flip_point(data.coordinates),
                distance=data.distance,
                speed=data.speed,
                other_data=data.other_data,
            )

        return Frame(
            # doesn't change
            timestamp=frame.timestamp,
            frame_id=frame.frame_id,
            ball_owning_team=frame.ball_owning_team,
            ball_state=frame.ball_state,
            period=frame.period,
            # changes
            ball_coordinates=self.flip_point(frame.ball_coordinates),
            players_data=players_data,
            other_data=frame.other_data,
            statistics=frame.statistics,
        )

    def transform_event(self, event: Event) -> Event:
        # Change coordinate system
        if self._needs_coordinate_system_change:
            event = self.__change_event_coordinate_system(event)

        # Change dimensions
        elif self._needs_pitch_dimensions_change:
            event = self.__change_event_dimensions(event)

        # Flip event based on orientation
        if self._needs_orientation_change:
            if self.__needs_flip(
                period=event.period,
                ball_owning_team=event.ball_owning_team,
                action_executing_team=event.team,
            ):
                event = self.__flip_event(event)

        if event.freeze_frame:
            event.freeze_frame = self.transform_frame(event.freeze_frame)

        return event

    def __change_event_coordinate_system(self, event: Event):
        position_changes = {
            field.name: self.__change_point_coordinate_system(
                getattr(event, field.name)
            )
            for field in fields(event)
            if field.name.endswith("coordinates") and getattr(event, field.name)
        }

        return replace(event, **position_changes)

    def __change_event_dimensions(self, event: Event):
        position_changes = {
            field.name: self.change_point_dimensions(getattr(event, field.name))
            for field in fields(event)
            if field.name.endswith("coordinates") and getattr(event, field.name)
        }

        return replace(event, **position_changes)

    def __flip_event(self, event: Event):
        position_changes = {
            field.name: self.flip_point(getattr(event, field.name))
            for field in fields(event)
            if field.name.endswith("coordinates") and getattr(event, field.name)
        }

        return replace(event, **position_changes)

    def get_to_coordinate_system(self) -> Optional[CoordinateSystem]:
        return self._to_coordinate_system

    @classmethod
    def transform_dataset(
        cls,
        dataset: Dataset,
        to_pitch_dimensions: Optional[PitchDimensions] = None,
        to_orientation: Optional[Orientation] = None,
        to_coordinate_system: Optional[CoordinateSystem] = None,
    ) -> Dataset:
        # Early exit for no-op
        if (
            to_pitch_dimensions is None
            and to_orientation is None
            and to_coordinate_system is None
        ):
            return dataset

        # Define transformation parameters
        transform = {}

        # Resolve origin and target orientation
        transform["from_orientation"] = dataset.metadata.orientation
        transform["to_orientation"] = (
            to_orientation or transform["from_orientation"]
        )
        if (
            transform["to_orientation"] == Orientation.BALL_OWNING_TEAM
            and not dataset.metadata.flags & DatasetFlag.BALL_OWNING_TEAM
        ):
            raise ValueError(
                "Cannot transform to BALL_OWNING_TEAM orientation since "
                "the source dataset does not define the ball-owning team"
            )

        # Resolve origin and target spatial reference system
        if to_pitch_dimensions is not None:
            if dataset.metadata.coordinate_system:
                # If source has a CS, we must convert the new dimensions into a CS
                transform["from_coordinate_system"] = (
                    dataset.metadata.coordinate_system
                )
                transform["to_coordinate_system"] = CustomCoordinateSystem(
                    origin=dataset.metadata.coordinate_system.origin,
                    pitch_dimensions=to_pitch_dimensions,
                    vertical_orientation=dataset.metadata.coordinate_system.vertical_orientation,
                )
            else:
                # Otherwise, we can just convert the pitch dimensions
                transform["from_pitch_dimensions"] = (
                    dataset.metadata.pitch_dimensions
                )
                transform["to_pitch_dimensions"] = to_pitch_dimensions

        elif to_coordinate_system is not None:
            transform["from_coordinate_system"] = (
                dataset.metadata.coordinate_system
            )
            transform["to_coordinate_system"] = to_coordinate_system

        else:
            # If only transforming orientation, we need to check what spatial
            # reference system is defined in the source dataset and respect that
            if dataset.metadata.coordinate_system:
                transform["from_coordinate_system"] = (
                    dataset.metadata.coordinate_system
                )
                transform["to_coordinate_system"] = (
                    dataset.metadata.coordinate_system
                )
            elif dataset.metadata.pitch_dimensions:
                transform["from_pitch_dimensions"] = (
                    dataset.metadata.pitch_dimensions
                )
                transform["to_pitch_dimensions"] = (
                    dataset.metadata.pitch_dimensions
                )
            else:
                raise ValueError(
                    "Cannot transform orientation when the dataset does not "
                    "contain the pitch dimensions or a coordinate system"
                )

        # Execution
        transformer = cls(**transform)
        metadata_changes = {"orientation": transform["to_orientation"]}
        if "to_coordinate_system" in transform:
            metadata_changes["coordinate_system"] = transform[
                "to_coordinate_system"
            ]
        elif "to_pitch_dimensions" in transform:
            metadata_changes["pitch_dimensions"] = transform[
                "to_pitch_dimensions"
            ]
        metadata = replace(dataset.metadata, **metadata_changes)

        if isinstance(dataset, TrackingDataset):
            records = [
                transformer.transform_frame(record)
                for record in dataset.records
            ]
            return TrackingDataset(metadata=metadata, records=records)
        elif isinstance(dataset, EventDataset):
            records = [
                transformer.transform_event(record)
                for record in dataset.records
            ]
            return EventDataset(metadata=metadata, records=records)
        else:
            raise KloppyError(f"Unsupported dataset type: {type(dataset)}")


class DatasetTransformerBuilder:
    def __init__(
        self, to_coordinate_system: Optional[Union[str, Provider]] = None
    ):
        from kloppy.config import get_config

        if not to_coordinate_system:
            to_coordinate_system = get_config("coordinate_system")

        if not to_coordinate_system:
            to_coordinate_system = Provider.KLOPPY

        to_dataset_type = None
        if isinstance(to_coordinate_system, str):
            if ":" in to_coordinate_system:
                provider_name, dataset_type_name = to_coordinate_system.split(
                    ":"
                )
                to_coordinate_system = Provider[provider_name.upper()]
                to_dataset_type = DatasetType[dataset_type_name.upper()]
            else:
                to_coordinate_system = Provider[to_coordinate_system.upper()]

        self.to_coordinate_system = to_coordinate_system
        self.to_dataset_type = to_dataset_type

    def build(
        self,
        provider: Provider,
        dataset_type: DatasetType,
        pitch_length: Optional[float] = None,
        pitch_width: Optional[float] = None,
    ):
        with warnings.catch_warnings():
            # Tell Python to ignore this specific warning class within this block
            warnings.simplefilter("ignore", category=MissingPitchSizeWarning)
            from_coordinate_system = build_coordinate_system(
                provider,
                dataset_type=dataset_type,
                pitch_length=pitch_length,
                pitch_width=pitch_width,
            )

            to_coordinate_system = build_coordinate_system(
                self.to_coordinate_system,
                dataset_type=self.to_dataset_type or dataset_type,
                pitch_length=pitch_length,
                pitch_width=pitch_width,
            )

        needs_pitch_dimensions_change = (
            from_coordinate_system.pitch_dimensions
            != to_coordinate_system.pitch_dimensions
        )
        not_standardized = (
            not from_coordinate_system.pitch_dimensions.standardized
            or not to_coordinate_system.pitch_dimensions.standardized
        )
        missing_dimensions = pitch_length is None or pitch_width is None
        if (
            needs_pitch_dimensions_change
            and not_standardized
            and missing_dimensions
        ):
            from_coordinate_system_name = (
                from_coordinate_system.provider
                if isinstance(from_coordinate_system, ProviderCoordinateSystem)
                else "custom"
            )
            to_coordinate_system_name = (
                to_coordinate_system.provider
                if isinstance(to_coordinate_system, ProviderCoordinateSystem)
                else "custom"
            )
            warn_missing_pitch_dimensions(
                context=f"transforming coordinates from {from_coordinate_system_name} to {to_coordinate_system_name}",
                stacklevel=2,
            )
            return self.build(
                provider,
                dataset_type,
                pitch_length=DEFAULT_PITCH_LENGTH,
                pitch_width=DEFAULT_PITCH_WIDTH,
            )

        return DatasetTransformer(
            from_coordinate_system=from_coordinate_system,
            to_coordinate_system=to_coordinate_system,
        )
