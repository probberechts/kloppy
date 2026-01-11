from dataclasses import replace
from typing import Optional, Union
import warnings

from kloppy.domain import (
    DEFAULT_PITCH_LENGTH,
    DEFAULT_PITCH_WIDTH,
    CoordinateSystem,
    CustomCoordinateSystem,
    Dataset,
    DatasetTransformer,
    Orientation,
    PitchDimensions,
    Provider,
    ProviderCoordinateSystem,
    build_coordinate_system,
)
from kloppy.exceptions import (
    MissingPitchSizeWarning,
    warn_missing_pitch_dimensions,
)


def transform(
    dataset: Dataset,
    to_orientation: Optional[Union[Orientation, str]] = None,
    to_pitch_dimensions: Optional[PitchDimensions] = None,
    to_coordinate_system: Optional[
        Union[CoordinateSystem, Provider, str]
    ] = None,
) -> Dataset:
    """
    Transform the dataset to a new orientation, coordinate system, or pitch dimensions.
    """
    # Input validation
    if to_pitch_dimensions is not None and to_coordinate_system is not None:
        warnings.warn(
            "Both a pitch dimension and coordinate system transformation were requested. "
            "The pitch dimensions will be ignored in favor of the coordinate system's transformation."
        )

    #  Normalize inputs
    if to_orientation is not None:
        if isinstance(to_orientation, str):
            to_orientation = Orientation[to_orientation.upper()]

    if to_coordinate_system is not None:
        # Resolve Provider/String to a CoordinateSystem object
        if isinstance(to_coordinate_system, str):
            provider = Provider[to_coordinate_system.upper()]
        elif isinstance(to_coordinate_system, Provider):
            provider = to_coordinate_system
        else:
            provider = None

        if provider:
            with warnings.catch_warnings():
                # Tell Python to ignore this specific warning class within this block
                warnings.simplefilter(
                    "ignore", category=MissingPitchSizeWarning
                )
                to_coordinate_system = build_coordinate_system(
                    provider=provider,
                    pitch_length=dataset.metadata.pitch_dimensions.pitch_length,
                    pitch_width=dataset.metadata.pitch_dimensions.pitch_width,
                )

    # Validate requested transformation
    if to_coordinate_system is not None:
        dataset, to_coordinate_system = _ensure_valid_dimensions(
            dataset,
            target_object=to_coordinate_system,
            is_coordinate_system=True,
        )
    elif to_pitch_dimensions is not None:
        dataset, to_pitch_dimensions = _ensure_valid_dimensions(
            dataset,
            target_object=to_pitch_dimensions,
            is_coordinate_system=False,
        )

    # Execute transformation
    return DatasetTransformer.transform_dataset(
        dataset=dataset,
        to_orientation=to_orientation,
        to_coordinate_system=to_coordinate_system,
        to_pitch_dimensions=to_pitch_dimensions,
    )


def _ensure_valid_dimensions(
    dataset: Dataset,
    target_object: Union[CoordinateSystem, PitchDimensions],
    is_coordinate_system: bool,
) -> Dataset:
    """
    Checks if the transformation requires dimensions that are missing.
    If so, warns the user, applies defaults to the target, and backfills the dataset.
    """

    current_dims = dataset.metadata.pitch_dimensions
    target_dims = (
        target_object.pitch_dimensions
        if is_coordinate_system
        else target_object
    )

    # Check conditions
    needs_change = current_dims != target_dims
    not_standardized = (
        not current_dims.standardized or not target_dims.standardized
    )
    missing_source_dims = (
        current_dims.pitch_length is None or current_dims.pitch_width is None
    )

    if needs_change and not_standardized and missing_source_dims:
        # Generate context-aware warning
        if is_coordinate_system:
            provider = getattr(target_object, "provider", "custom")
            if provider == "custom":
                fix = "CustomCoordinateSystem(..., pitch_dimensions=PitchDimensions(pitch_length=105, pitch_width=68))"
            else:
                fix = (
                    f"build_coordinate_system(provider='{provider}', "
                    f"pitch_length=105, pitch_width=68)"
                )
            param_name = "to_coordinate_system"
        else:
            cls_name = type(target_object).__name__
            fix = f"{cls_name}(..., pitch_length=105, pitch_width=68)"
            param_name = "to_pitch_dimensions"

        warn_missing_pitch_dimensions(
            context="transforming between standardized and non-standardized pitch dimensions",
            fix_suggestion=f"`dataset.transform({param_name}={fix})`.",
        )

        # Apply defaults to Target
        new_dims = replace(
            target_dims,
            pitch_length=DEFAULT_PITCH_LENGTH,
            pitch_width=DEFAULT_PITCH_WIDTH,
        )

        # Apply defaults to Source Dataset (Backfill)
        dataset = _set_true_pitch_dimensions(
            dataset, new_dims.pitch_length, new_dims.pitch_width
        )

    else:
        new_dims = target_dims

    if is_coordinate_system:
        return dataset, _update_coordinate_system_dims(target_object, new_dims)
    else:
        return dataset, new_dims


def _update_coordinate_system_dims(
    cs: CoordinateSystem, new_dims: PitchDimensions
) -> CoordinateSystem:
    """Helper to safely update dimensions on either Provider or Custom CS."""
    if isinstance(cs, ProviderCoordinateSystem):
        return build_coordinate_system(
            provider=cs.provider,
            pitch_length=new_dims.pitch_length,
            pitch_width=new_dims.pitch_width,
        )
    elif isinstance(cs, CustomCoordinateSystem):
        return replace(cs, pitch_dimensions=new_dims)
    return cs


def _set_true_pitch_dimensions(
    dataset: Dataset,
    pitch_length: Optional[float],
    pitch_width: Optional[float],
) -> Dataset:
    """
    Updates the dataset with the provided pitch length and width.

    If the dataset already has pitch dimensions set, and they conflict
    with the provided ones, an error is raised.
    """
    from_dims = dataset.metadata.pitch_dimensions

    # Check if update is actually needed
    if (
        from_dims.pitch_length == pitch_length
        and from_dims.pitch_width == pitch_width
    ):
        return dataset

    # Check for conflicts
    if from_dims.pitch_length is not None and from_dims.pitch_width is not None:
        raise ValueError(
            f"The dataset already has pitch dimensions {from_dims.pitch_length}x{from_dims.pitch_width}. "
            f"These are incompatible with the required transform dimensions of {pitch_length}x{pitch_width}."
        )

    # Apply update
    new_metadata = dataset.metadata
    new_dims = replace(
        new_metadata.pitch_dimensions,
        pitch_length=pitch_length,
        pitch_width=pitch_width,
    )

    if new_metadata.coordinate_system is None:
        # Just update dimensions directly
        new_metadata = replace(new_metadata, pitch_dimensions=new_dims)

    else:
        # Update the existing coordinate system
        new_cs = _update_coordinate_system_dims(
            new_metadata.coordinate_system, new_dims
        )
        new_metadata = replace(new_metadata, coordinate_system=new_cs)

    return replace(dataset, metadata=new_metadata)
