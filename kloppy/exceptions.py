from typing import Optional
import warnings


class KloppyError(Exception):
    pass


class SerializationError(KloppyError):
    pass


class DeserializationError(KloppyError):
    pass


class OrientationError(KloppyError):
    pass


class MissingDimensionError(KloppyError):
    """
    Raised when the coordinate boundaries (x_dim, y_dim) are missing or invalid.
    """


class MissingPitchSizeError(KloppyError):
    """
    Raised when the physical pitch length or width (in meters) is not defined.
    """


class MissingPitchSizeWarning(Warning):
    """
    Warning raised when the physical pitch length or width (in meters),
    triggering a fallback to default values.
    """


class OrphanedRecordError(KloppyError):
    pass


class InvalidFilterError(KloppyError):
    pass


class AdapterError(KloppyError):
    pass


class InputNotFoundError(KloppyError):
    pass


class UnknownEncoderError(KloppyError):
    pass


class KloppyParameterError(KloppyError):
    pass


class DeserializationWarning(Warning):
    pass


def warn_missing_pitch_dimensions(
    context: str,
    fix_suggestion: Optional[str] = None,
    stacklevel: int = 2,
) -> None:
    """
    Centralized logic for issuing pitch dimension warnings.

    Args:
        context: Description of what was happening (e.g. "transforming from X to Y").
        fix_suggestion: Optional code snippet showing the user how to fix it.
        stacklevel: Controls which line of code is pointed to in the trace.
                    Default is 2 (the caller of this function).
    """
    from kloppy.domain import DEFAULT_PITCH_LENGTH, DEFAULT_PITCH_WIDTH

    message = (
        f"The physical pitch length and width are missing but required for {context}. "
        f"Defaulting to standard {DEFAULT_PITCH_LENGTH}x{DEFAULT_PITCH_WIDTH}. "
        "This may cause inaccuracies."
    )

    if fix_suggestion:
        message += (
            f" To fix this, specify dimensions explicitly: `{fix_suggestion}`."
        )

    warnings.warn(
        message, category=MissingPitchSizeWarning, stacklevel=stacklevel
    )
