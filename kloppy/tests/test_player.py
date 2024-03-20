import pytest

from datetime import timedelta
from kloppy import statsbomb
from kloppy.domain import EventDataset, Position

API_URL = "https://raw.githubusercontent.com/statsbomb/open-data/master/data/"


class TestPlayer:
    """"""

    @pytest.fixture(scope="module")
    def dataset(self) -> EventDataset:
        """Load StatsBomb data for Belgium - Portugal at Euro 2020"""
        dataset = statsbomb.load(
            event_data=f"{API_URL}/events/3794687.json",
            lineup_data=f"{API_URL}/lineups/3794687.json",
            three_sixty_data=f"{API_URL}/three-sixty/3794687.json",
            coordinates="statsbomb",
        )
        return dataset

    def test_positions(self, dataset: EventDataset):
        """
        Test positions of players
        """
        home_team = dataset.metadata.teams[0]
        away_team = dataset.metadata.teams[1]
        # Full time player
        full_time_player = home_team.get_player_by_id("2954")
        assert full_time_player.positions == [
            {
                "end": {
                    "period_id": 2,
                    "timestamp": timedelta(seconds=5887, microseconds=760000),
                },
                "position": Position(
                    position_id="9",
                    name="Right Defensive Midfield",
                    coordinates=None,
                ),
                "start": {"period_id": 1, "timestamp": timedelta(0)},
            }
        ]
        # Substituted off player
        sub_off_player = dataset.metadata.teams[0].get_player_by_id("3089")
        assert sub_off_player.positions == [
            {
                "end": {"period_id": 2, "timestamp": timedelta(seconds=2856)},
                "position": Position(
                    position_id="18",
                    name="Right Attacking Midfield",
                    coordinates=None,
                ),
                "start": {"period_id": 1, "timestamp": timedelta(0)},
            }
        ]
        # Substituted on player
        sub_on_player = dataset.metadata.teams[0].get_player_by_id("5630")
        assert sub_on_player.positions == [
            {
                "start": {
                    "period_id": 2,
                    "timestamp": timedelta(seconds=2856),
                },
                "end": {
                    "period_id": 2,
                    "timestamp": timedelta(seconds=5887, microseconds=760000),
                },
                "position": Position(
                    position_id="18",
                    name="Right Attacking Midfield",
                    coordinates=None,
                ),
            }
        ]
        # Player with multiple positions
        multi_positions_player = away_team.get_player_by_id("5204")
        assert multi_positions_player.positions == [
            {
                "start": {
                    "period_id": 2,
                    "timestamp": timedelta(seconds=3313),
                },
                "end": {"period_id": 2, "timestamp": timedelta(seconds=3539)},
                "position": Position(
                    position_id="17", name="Right Wing", coordinates=None
                ),
            },
            {
                "start": {
                    "period_id": 2,
                    "timestamp": timedelta(seconds=3539),
                },
                "end": {
                    "period_id": 2,
                    "timestamp": timedelta(seconds=5887, microseconds=760000),
                },
                "position": Position(
                    position_id="13",
                    name="Right Center Midfield",
                    coordinates=None,
                ),
            },
        ]
        assert multi_positions_player.starting_position == Position(
            position_id="17", name="Right Wing", coordinates=None
        )
