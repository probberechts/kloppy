from math import sqrt

import pytest

from kloppy.domain import (
    Dimension,
    ImperialPitchDimensions,
    MetricPitchDimensions,
    NormalizedPitchDimensions,
    OptaPitchDimensions,
    Point,
    Point3D,
    Unit,
)
from kloppy.exceptions import (
    MissingDimensionError,
    MissingPitchSizeError,
    MissingPitchSizeWarning,
)


def assert_point_approx_equal(p1: Point, p2: Point, tolerance=0.001):
    assert p1.x == pytest.approx(p2.x, abs=tolerance)
    assert p1.y == pytest.approx(p2.y, abs=tolerance)
    if isinstance(p1, Point3D) and isinstance(p2, Point3D):
        if p1.z is not None and p2.z is not None:
            assert p1.z == pytest.approx(p2.z, abs=tolerance)
        else:
            assert p1.z == p2.z


class TestUnits:
    def test_unit_conversion(self):
        """It should convert between units correctly."""
        # Test known constants
        assert Unit.METERS.convert(Unit.CENTIMETERS, 1) == 100
        assert Unit.METERS.convert(Unit.YARDS, 1) == pytest.approx(1.09361)

        # Test identity
        assert Unit.METERS.convert(Unit.METERS, 10) == 10

        # Test reverse (Yard -> Meter)
        # 1 meter = 1.09361 yards -> 1 yard = 1/1.09361 meters
        assert Unit.YARDS.convert(Unit.METERS, 1.09361) == pytest.approx(1.0)

    def test_normed_unit_error(self):
        """It should raise an error when trying to convert between NORMED and absolute units."""
        with pytest.raises(ValueError):
            Unit.NORMED.convert(Unit.METERS, 0.5)

        with pytest.raises(ValueError):
            Unit.METERS.convert(Unit.NORMED, 10)


class TestPitchdimensions:
    @pytest.fixture
    def metric_dims(self) -> MetricPitchDimensions:
        """Returns a standard 105x68 metric pitch."""
        return MetricPitchDimensions(
            x_dim=Dimension(0, 105),
            y_dim=Dimension(0, 68),
            standardized=False,
            pitch_length=105,
            pitch_width=68,
        )

    @pytest.fixture
    def statsbomb_dims(self) -> ImperialPitchDimensions:
        """Returns standard 120x80 Imperial pitch dimensions."""
        return ImperialPitchDimensions(
            x_dim=Dimension(0, 120),
            y_dim=Dimension(0, 80),
            standardized=True,
            pitch_length=None,
            pitch_width=None,
        )

    def test_create_non_standardized(self):
        """Test creating non-standardized pitch dimensions."""
        # fully specified dimensions
        dims = MetricPitchDimensions(
            x_dim=Dimension(0, 105),
            y_dim=Dimension(0, 68),
            standardized=False,
            pitch_length=105,
            pitch_width=68,
        )
        assert dims.pitch_length == 105
        assert dims.pitch_width == 68

        # inferred pitch size
        dims = MetricPitchDimensions(
            x_dim=Dimension(0, 105),
            y_dim=Dimension(0, 68),
            standardized=False,
        )
        assert dims.pitch_length == 105
        assert dims.pitch_width == 68

        # unkown pitch size
        dims = MetricPitchDimensions(
            x_dim=Dimension(0, None),
            y_dim=Dimension(0, None),
            standardized=False,
        )
        assert dims.pitch_length is None
        assert dims.pitch_width is None

        # contradicting pitch size
        with pytest.raises(ValueError):
            dims = MetricPitchDimensions(
                x_dim=Dimension(0, 105),
                y_dim=Dimension(0, 68),
                standardized=False,
                pitch_length=68,
                pitch_width=105,
            )

    def test_create_standardized(self):
        """Test creating standardized pitch dimensions."""
        # fully specified dimensions
        dims = ImperialPitchDimensions(
            x_dim=Dimension(0, 120),
            y_dim=Dimension(0, 80),
            standardized=True,
            pitch_length=105,
            pitch_width=68,
        )
        assert dims.pitch_length == 105
        assert dims.pitch_width == 68

        # missing true pitch size
        dims = ImperialPitchDimensions(
            x_dim=Dimension(0, 120),
            y_dim=Dimension(0, 80),
            standardized=True,
        )
        assert dims.pitch_length is None
        assert dims.pitch_width is None

        # missing dimensions
        with pytest.raises(MissingDimensionError):
            dims = ImperialPitchDimensions(
                x_dim=Dimension(0, None),
                y_dim=Dimension(0, None),
                standardized=True,
            )

    def test_create_normalized_from_non_standardized(
        self, metric_dims: MetricPitchDimensions
    ):
        """Test creating normalized pitch dimensions from a non-standardized pitch."""
        pitch_with_scale = NormalizedPitchDimensions.scale_from(
            metric_dims,
            x_dim=Dimension(-100, 100),
            y_dim=Dimension(-50, 50),
        )

        assert pitch_with_scale.x_dim == Dimension(-100, 100)
        assert pitch_with_scale.y_dim == Dimension(-50, 50)
        assert pitch_with_scale.pitch_length == 105
        assert pitch_with_scale.pitch_width == 68
        assert pitch_with_scale.standardized is False

        # Penalty area length check
        assert pitch_with_scale.penalty_area_length == pytest.approx(
            (16.5 / 105) * 200
        )

        # Should raise error when pitch size is not specified
        with pytest.raises(MissingPitchSizeError):
            NormalizedPitchDimensions.scale_from(
                MetricPitchDimensions(
                    x_dim=Dimension(0, None),
                    y_dim=Dimension(0, None),
                    standardized=False,
                ),
                x_dim=Dimension(-100, 100),
                y_dim=Dimension(-50, 50),
            )

    def test_create_normalized_from_standardized(
        self, statsbomb_dims: ImperialPitchDimensions
    ):
        """Test creating normalized pitch dimensions from a standardized pitch."""
        pitch_with_scale = NormalizedPitchDimensions.scale_from(
            statsbomb_dims,
            x_dim=Dimension(-100, 100),
            y_dim=Dimension(-50, 50),
        )

        assert pitch_with_scale.x_dim == Dimension(-100, 100)
        assert pitch_with_scale.y_dim == Dimension(-50, 50)
        assert pitch_with_scale.pitch_length is None
        assert pitch_with_scale.pitch_width is None
        assert pitch_with_scale.standardized is True

        # Penalty area length check
        assert pitch_with_scale.penalty_area_length == pytest.approx(
            (18 / 120) * 200
        )

    def test_create_normalized_direct_initialization_error(self):
        """It should raise a RuntimeError when instantiated without factory method or arguments."""
        with pytest.raises(
            RuntimeError, match="cannot be instantiated directly"
        ):
            NormalizedPitchDimensions(
                x_dim=Dimension(0, 1), y_dim=Dimension(0, 1)
            )

    def test_to_metric_base_dimensions(self):
        """Test converting FROM Opta TO Standard Metric (IFAB)."""
        pitch = OptaPitchDimensions()

        # Opta 11.5 is the defined penalty spot distance.
        # IFAB Metric penalty spot is 11m.
        # Due to zone-based transformation, Opta 11.5 should map EXACTLY to Metric 11.0.
        # Opta Y=50 (Center) -> Metric Y=34 (Center of 68).
        ifab_point = pitch.to_metric_base(Point(11.5, 50))
        assert_point_approx_equal(ifab_point, Point(11, 34))

        # Test Z-axis (Goal height)
        # Opta Goal Height = 38. IFAB Goal Height = 2.44.
        # Point at top of Opta goal (38) should match top of IFAB goal (2.44).
        ifab_point_3d = pitch.to_metric_base(Point3D(0, 50, 38))
        assert_point_approx_equal(ifab_point_3d, Point3D(0, 34, 2.44))

        # Test specific pitch size overrides
        ifab_point_custom = pitch.to_metric_base(
            Point(60, 61), pitch_length=105, pitch_width=68
        )
        assert ifab_point_custom.x == pytest.approx(62.78, abs=0.01)
        assert ifab_point_custom.y == pytest.approx(41.72, abs=0.01)

    def test_to_metric_base_dimensions_out_of_bounds(self, metric_dims):
        """
        Test behavior when points fall outside the defined zones.
        Note: to_metric_base uses default 105x68 IFAB pitch if not specified.
        """
        pitch = NormalizedPitchDimensions.scale_from(
            metric_dims, x_dim=Dimension(-100, 100), y_dim=Dimension(-50, 50)
        )

        # Point -100 is the goal line (min) -> Maps to 0 in Metric
        ifab_point = pitch.to_metric_base(Point(-100, 0))
        assert_point_approx_equal(ifab_point, Point(0, 34))

        # Point -105 is 5 units "outside" the 200 unit long pitch.
        # The fallback logic scales global length:
        # Metric Length (105) / Source Length (200) = 0.525 scale factor.
        # -5 units * 0.525 = -2.625 meters.
        ifab_point_outside = pitch.to_metric_base(Point(-105, 0))
        assert_point_approx_equal(ifab_point_outside, Point(-2.625, 34))

        # Positive bound
        # 105 is 5 units outside positive side.
        # 105m (metric length) + 2.625m = 107.625m
        ifab_point_outside_pos = pitch.to_metric_base(Point(105, 0))
        assert_point_approx_equal(ifab_point_outside_pos, Point(107.625, 34))

    def test_from_metric_base_dimensions(self):
        """Test converting FROM Standard Metric TO Opta."""
        pitch = OptaPitchDimensions()

        # Metric 11m (Penalty Spot) -> Opta 11.5
        opta_point = pitch.from_metric_base(Point(11, 34))
        assert_point_approx_equal(opta_point, Point(11.5, 50))

        # Metric Z=2.44 -> Opta Z=38
        opta_point_3d = pitch.from_metric_base(Point3D(0, 34, 2.44))
        assert_point_approx_equal(opta_point_3d, Point3D(0, 50, 38))

    def test_from_metric_base_dimensions_out_of_bounds(self, metric_dims):
        pitch = NormalizedPitchDimensions.scale_from(
            metric_dims, x_dim=Dimension(-100, 100), y_dim=Dimension(-50, 50)
        )

        # Metric 0 -> Normalized -100
        point = pitch.from_metric_base(Point(0, 34))
        assert_point_approx_equal(point, Point(-100, 0))

        # Metric -2.625.
        # Scale factor (Source 200 / Metric 105) = 1.9047...
        # -2.625 * 1.9047 = -5.
        # -100 - 5 = -105.
        point = pitch.from_metric_base(Point(-2.625, 34))
        assert_point_approx_equal(point, Point(-105, 0))

        # Metric 107.625 (105 + 2.625)
        # 100 + 5 = 105.
        point = pitch.from_metric_base(Point(107.625, 34))
        assert_point_approx_equal(point, Point(105, 0))

    def test_distance_between(self):
        """Test distance calculations logic."""
        pitch = OptaPitchDimensions(pitch_length=105, pitch_width=68)

        # Corner to Corner diagonal
        # Opta (0,0) -> Metric (0,0)
        # Opta (100,100) -> Metric (105, 68)
        distance = pitch.distance_between(
            Point(0, 0),
            Point(100, 100),
        )
        assert distance == pytest.approx(sqrt(105**2 + 68**2))

        # Penalty spot distance
        # Opta (0,50) -> Metric (0, 34)
        # Opta (11.5, 50) -> Metric (11, 34)
        # Distance = 11 meters
        distance = pitch.distance_between(Point(0, 50), Point(11.5, 50))
        assert distance == pytest.approx(11.0)

        # Mirrored Penalty spot (Right side)
        # Opta (100, 50) -> Metric (105, 34)
        # Opta (100-11.5, 50) -> Metric (105-11, 34)
        # Distance = 11 meters
        distance = pitch.distance_between(Point(100, 50), Point(100 - 11.5, 50))
        assert distance == pytest.approx(11.0)

        # Unit conversion check (Meters to CM)
        distance_cm = pitch.distance_between(
            Point(0, 50), Point(11.5, 50), unit=Unit.CENTIMETERS
        )
        assert distance_cm == pytest.approx(1100.0)

        # Warn if pitch size is missing
        pitch_no_size = OptaPitchDimensions()
        with pytest.warns(MissingPitchSizeWarning):
            distance = pitch_no_size.distance_between(
                Point(0, 0),
                Point(100, 100),
            )
        assert distance == pytest.approx(sqrt(105**2 + 68**2))

    def test_coordinate_transformation_logic(self):
        """
        Simulates the logic of DatasetTransformer using only PitchDimensions methods.
        Tests converting Opta -> Metric -> Custom Metric.
        """
        source_pitch = OptaPitchDimensions()
        target_pitch = MetricPitchDimensions(
            x_dim=Dimension(0, 105),
            y_dim=Dimension(0, 68),
            standardized=False,
        )

        # Point: Opta (17, 78.9).
        # Opta X=17 is the penalty Area Length (Opta defines it as 17.0).
        # So this point is exactly on the penalty box line.
        original_point = Point(17, 78.9)

        # 1. Transform Opta -> IFAB Metric Base
        # Opta 17.0 (Box Line) -> IFAB 16.5 (Box Line)
        ifab_point = source_pitch.to_metric_base(original_point)

        # 2. Transform IFAB Metric Base -> Target Metric
        # Since Target is standard 105x68, the points should remain 16.5.
        target_point = target_pitch.from_metric_base(ifab_point)

        # Check X: Should be exactly 16.5 (Penalty area length in meters)
        assert target_point.x == pytest.approx(16.5)

        # Check Y:
        assert target_point.y == pytest.approx(54.16, abs=0.01)
