import logging
import math
from datetime import datetime, timedelta
from typing import IO, Dict, List, NamedTuple, Optional

import pytz

from kloppy.domain import (
    BallState,
    BodyPart,
    BodyPartQualifier,
    CardQualifier,
    CardType,
    CounterAttackQualifier,
    DatasetFlag,
    DuelQualifier,
    DuelResult,
    DuelType,
    EventDataset,
    ExpectedGoals,
    GoalkeeperActionType,
    GoalkeeperQualifier,
    InterceptionResult,
    Metadata,
    Orientation,
    PassQualifier,
    PassResult,
    PassType,
    Period,
    Point,
    Point3D,
    PositionType,
    PostShotExpectedGoals,
    Provider,
    Qualifier,
    SetPieceQualifier,
    SetPieceType,
    ShotResult,
    TakeOnResult,
    Team,
)
from kloppy.exceptions import DeserializationError
from kloppy.infra.serializers.event.deserializer import EventDataDeserializer
from kloppy.utils import performance_logging

from .formation_mapping import formation_id_mapping, formation_position_mapping
from .parsers import OptaEvent, get_parser

logger = logging.getLogger(__name__)

EVENT_TYPE_START_PERIOD = 32
EVENT_TYPE_END_PERIOD = 30
EVENT_TYPE_DELETED_EVENT = 43
EVENT_TYPE_START_DELAY = 27
EVENT_TYPE_END_DELAY = 28
EVENT_TYPE_OFFSIDE_PROVOKED = 55

EVENT_TYPE_PASS = 1
EVENT_TYPE_OFFSIDE_PASS = 2
EVENT_TYPE_TAKE_ON = 3
EVENT_TYPE_TACKLE = 7
EVENT_TYPE_AERIAL = 44
EVENT_TYPE_50_50 = 67
EVENT_TYPE_INTERCEPTION = 8
EVENT_TYPE_CLEARANCE = 12
EVENT_TYPE_SHOT_MISS = 13
EVENT_TYPE_SHOT_POST = 14
EVENT_TYPE_SHOT_SAVED = 15
EVENT_TYPE_SHOT_GOAL = 16
EVENT_TYPE_BALL_OUT = 5
EVENT_TYPE_CORNER_AWARDED = 6
EVENT_TYPE_FOUL_COMMITTED = 4
EVENT_TYPE_CARD = 17
EVENT_TYPE_RECOVERY = 49
EVENT_TYPE_TEAM_SET_UP = 34
EVENT_TYPE_FORMATION_CHANGE = 40
EVENT_TYPE_BALL_TOUCH = 61
EVENT_TYPE_BLOCKED_PASS = 74

EVENT_TYPE_SAVE = 10
EVENT_TYPE_CLAIM = 11
EVENT_TYPE_PUNCH = 41
EVENT_TYPE_KEEPER_PICK_UP = 52
EVENT_TYPE_SMOTHER = 54
KEEPER_EVENTS = [
    EVENT_TYPE_SAVE,
    EVENT_TYPE_CLAIM,
    EVENT_TYPE_PUNCH,
    EVENT_TYPE_KEEPER_PICK_UP,
    EVENT_TYPE_SMOTHER,
]

BALL_OUT_EVENTS = [EVENT_TYPE_BALL_OUT, EVENT_TYPE_CORNER_AWARDED]
DUEL_EVENTS = [
    EVENT_TYPE_TACKLE,
    EVENT_TYPE_AERIAL,
    EVENT_TYPE_50_50,
]

BALL_OWNING_EVENTS = (
    EVENT_TYPE_PASS,
    EVENT_TYPE_OFFSIDE_PASS,
    EVENT_TYPE_TAKE_ON,
    EVENT_TYPE_SHOT_MISS,
    EVENT_TYPE_SHOT_POST,
    EVENT_TYPE_SHOT_SAVED,
    EVENT_TYPE_SHOT_GOAL,
    EVENT_TYPE_RECOVERY,
    EVENT_TYPE_BALL_TOUCH,
)
DEAD_BALL_EVENTS = [
    EVENT_TYPE_BALL_OUT,
    EVENT_TYPE_CORNER_AWARDED,
    EVENT_TYPE_SHOT_GOAL,
    EVENT_TYPE_FOUL_COMMITTED,
    EVENT_TYPE_CARD,
    EVENT_TYPE_START_DELAY,
    EVENT_TYPE_END_DELAY,
    EVENT_TYPE_OFFSIDE_PROVOKED,
]

EVENT_QUALIFIER_GOAL_KICK = 124
EVENT_QUALIFIER_FREE_KICK = 5
EVENT_QUALIFIER_THROW_IN = 107
EVENT_QUALIFIER_CORNER_KICK = 6
EVENT_QUALIFIER_PENALTY = 9
EVENT_QUALIFIER_KICK_OFF = 279
EVENT_QUALIFIER_FREE_KICK_SHOT = 26

EVENT_QUALIFIER_HEAD_PASS = 3
EVENT_QUALIFIER_HEAD = 15
EVENT_QUALIFIER_LEFT_FOOT = 72
EVENT_QUALIFIER_RIGHT_FOOT = 20
EVENT_QUALIFIER_OTHER_BODYPART = 21

EVENT_QUALIFIER_LONG_BALL = 1
EVENT_QUALIFIER_CROSS = 2
EVENT_QUALIFIER_THROUGH_BALL = 4
EVENT_QUALIFIER_CHIPPED_BALL = 155
EVENT_QUALIFIER_LAUNCH = 157
EVENT_QUALIFIER_FLICK_ON = 168
EVENT_QUALIFIER_SWITCH_OF_PLAY = 196
EVENT_QUALIFIER_SHOT_ASSIST = 210
EVENT_QUALIFIER_ASSIST_2ND = 218

EVENT_QUALIFIER_FIRST_YELLOW_CARD = 31
EVENT_QUALIFIER_SECOND_YELLOW_CARD = 32
EVENT_QUALIFIER_RED_CARD = 33

EVENT_QUALIFIER_COUNTER_ATTACK = 23

EVENT_TYPE_PLAYER_OFF = 18
EVENT_TYPE_PLAYER_ON = 19
EVENT_QUALIFIER_TEAM_FORMATION = 130
EVENT_QUALIFIER_FORMATION_PLAYER_IDS = 30
EVENT_QUALIFIER_FORMATION_PLAYER_POSITIONS = 131

EVENT_QUALIFIER_XG = 321
EVENT_QUALIFIER_POST_SHOT_XG = 322

EVENT_TYPE_CHALLENGE = 45

event_type_names = {
    1: "pass",
    2: "offside pass",
    3: "take on",
    4: "foul",
    5: "out",
    6: "corner awarded",
    7: "tackle",
    8: "interception",
    9: "turnover",
    10: "save",
    11: "claim",
    12: "clearance",
    13: "miss",
    14: "post",
    15: "attempt saved",
    16: "goal",
    17: "card",
    18: "player off",
    19: "player on",
    20: "player retired",
    21: "player returns",
    22: "player becomes goalkeeper",
    23: "goalkeeper becomes player",
    24: "condition change",
    25: "official change",
    26: "unknown26",
    27: "start delay",
    28: "end delay",
    29: "unknown29",
    30: "end",
    31: "unknown31",
    32: "start",
    33: "unknown33",
    34: "team set up",
    35: "player changed position",
    36: "player changed jersey number",
    37: "collection end",
    38: "temp_goal",
    39: "temp_attempt",
    40: "formation change",
    41: "punch",
    42: "good skill",
    43: "deleted event",
    44: "aerial",
    45: "challenge",
    46: "unknown46",
    47: "rescinded card",
    48: "unknown46",
    49: "ball recovery",
    50: "dispossessed",
    51: "error",
    52: "keeper pick-up",
    53: "cross not claimed",
    54: "smother",
    55: "offside provoked",
    56: "shield ball opp",
    57: "foul throw in",
    58: "penalty faced",
    59: "keeper sweeper",
    60: "chance missed",
    61: "ball touch",
    62: "unknown62",
    63: "temp_save",
    64: "resume",
    65: "contentious referee decision",
    66: "possession data",
    67: "50/50",
    68: "referee drop ball",
    69: "failed to block",
    70: "injury time announcement",
    71: "coach setup",
    72: "caught offside",
    73: "other ball contact",
    74: "blocked pass",
    75: "delayed start",
    76: "early end",
    77: "player off pitch",
}

position_line_mapping = {
    "Goalkeeper": PositionType.Goalkeeper,
    "Defender": PositionType.Defender,
    "Midfielder": PositionType.Midfielder,
    "Forward": PositionType.Striker,
}


def _parse_pass(
    raw_event: OptaEvent,
    next_event: OptaEvent,
    next_next_event: OptaEvent,
    team: Team,
    period: Period,
) -> Dict:
    result = (
        PassResult.COMPLETE if raw_event.outcome else PassResult.INCOMPLETE
    )
    receiver_coordinates = _get_end_coordinates(raw_event.qualifiers)
    qualifiers = _get_pass_qualifiers(
        raw_event.qualifiers
    ) + _get_event_qualifiers(raw_event.qualifiers)

    # Look for a ball receipt event based on the result and sequence of events
    ball_receipt_event = None
    if result == PassResult.COMPLETE and next_event is not None:
        if (
            next_event.contestant_id == team.team_id
            and next_event.type_id in BALL_OWNING_EVENTS
        ):
            ball_receipt_event = next_event
        elif (
            next_event.type_id == EVENT_TYPE_CHALLENGE
            and next_next_event is not None
            and next_next_event.contestant_id == team.team_id
            and next_next_event.type_id in BALL_OWNING_EVENTS
        ):
            ball_receipt_event = next_next_event
    # Assign the receiver_player and receive_timestamp
    # if a ball_receipt_event was found
    if (
        ball_receipt_event is not None
        and ball_receipt_event.player_id is not None
    ):
        receiver_player = team.get_player_by_id(ball_receipt_event.player_id)
        receive_timestamp = (
            ball_receipt_event.timestamp - period.start_timestamp
        )
    else:
        receiver_player = None
        receive_timestamp = None

    # Set the end location of a deflected pass to the start location
    # of the next action and the outcome to "success" if the deflected
    # pass reached a teammate
    if (
        next_event is not None
        and next_next_event is not None
        and next_event.type_id == EVENT_TYPE_BALL_TOUCH
        and next_event.outcome == 1
        and next_next_event.contestant_id == raw_event.contestant_id
    ):
        result = PassResult.COMPLETE
        receiver_coordinates = Point(
            x=next_next_event.x,
            y=next_next_event.y,
        )
        receive_timestamp = next_next_event.timestamp - period.start_timestamp

    return dict(
        result=result,
        receiver_coordinates=receiver_coordinates,
        receiver_player=receiver_player,
        receive_timestamp=receive_timestamp,
        qualifiers=qualifiers,
    )


def _parse_offside_pass(raw_event: OptaEvent) -> Dict:
    pass_qualifiers = _get_pass_qualifiers(raw_event.qualifiers)
    overall_qualifiers = _get_event_qualifiers(raw_event.qualifiers)

    qualifiers = pass_qualifiers + overall_qualifiers

    return dict(
        result=PassResult.OFFSIDE,
        receiver_coordinates=_get_end_coordinates(raw_event.qualifiers),
        receiver_player=None,
        receive_timestamp=None,
        qualifiers=qualifiers,
    )


def _parse_take_on(raw_event: OptaEvent) -> Dict:
    if raw_event.outcome == 1:
        result = TakeOnResult.COMPLETE
    else:
        result = TakeOnResult.INCOMPLETE
    return dict(result=result)


def _parse_clearance(raw_event: OptaEvent) -> Dict:
    return dict(qualifiers=_get_event_qualifiers(raw_event.qualifiers))


def _parse_card(raw_event: OptaEvent) -> Dict:
    qualifiers = _get_event_qualifiers(raw_event.qualifiers)

    card_type = None
    for q in qualifiers:
        if isinstance(q, CardQualifier):
            card_type = q.value
            break

    return dict(result=None, qualifiers=None, card_type=card_type)


def _parse_lineup_qualifiers(raw_event: OptaEvent):
    formation_id = int(raw_event.qualifiers[EVENT_QUALIFIER_TEAM_FORMATION])
    formation = formation_id_mapping[formation_id]

    player_ids = raw_event.qualifiers[
        EVENT_QUALIFIER_FORMATION_PLAYER_IDS
    ].split(", ")
    position_ids = raw_event.qualifiers[
        EVENT_QUALIFIER_FORMATION_PLAYER_POSITIONS
    ].split(", ")

    positions_mapping = formation_position_mapping[formation]

    return formation, player_ids, position_ids, positions_mapping


def _parse_formation_change(raw_event: OptaEvent, team: Team) -> Dict:
    (
        formation,
        player_ids,
        position_ids,
        positions_mapping,
    ) = _parse_lineup_qualifiers(raw_event)

    player_positions = {}
    for player_id, position_id in zip(player_ids, position_ids):
        player = team.get_player_by_id(player_id)
        position = positions_mapping[int(position_id)]
        player_positions[player] = position

    return dict(formation_type=formation, player_positions=player_positions)


def _parse_substitution(next_event: OptaEvent, team: Team) -> Dict:
    replacement_player = None
    position = None
    if next_event.type_id == EVENT_TYPE_PLAYER_ON:
        replacement_player = team.get_player_by_id(next_event.player_id)

    raw_position_line = next_event.qualifiers.get(44)
    if raw_position_line:
        position = position_line_mapping.get(
            raw_position_line, PositionType.Unknown
        )

    return dict(replacement_player=replacement_player, position=position)


def _parse_shot(raw_event: OptaEvent) -> Dict:
    coordinates = Point(x=raw_event.x, y=raw_event.y)
    if raw_event.type_id == EVENT_TYPE_SHOT_GOAL:
        if 28 in raw_event.qualifiers:
            coordinates = Point(x=100 - raw_event.x, y=100 - raw_event.y)
            result = ShotResult.OWN_GOAL
            # ball_owning_team =
            # timestamp =
        else:
            result = ShotResult.GOAL
    elif 82 in raw_event.qualifiers:
        result = ShotResult.BLOCKED
    elif raw_event.type_id == EVENT_TYPE_SHOT_MISS:
        result = ShotResult.OFF_TARGET
    elif raw_event.type_id == EVENT_TYPE_SHOT_POST:
        result = ShotResult.OFF_TARGET
    elif raw_event.type_id == EVENT_TYPE_SHOT_SAVED:
        result = ShotResult.SAVED
    else:
        result = None

    qualifiers = _get_event_qualifiers(raw_event.qualifiers)
    result_coordinates = _get_end_coordinates(
        raw_event.qualifiers, start_coordinates=coordinates
    )
    if result == ShotResult.OWN_GOAL:
        if isinstance(result_coordinates, Point3D):
            result_coordinates = Point3D(
                x=100 - result_coordinates.x,
                y=100 - result_coordinates.y,
                z=result_coordinates.z,
            )
        elif isinstance(result_coordinates, Point):
            result_coordinates = Point(
                x=100 - result_coordinates.x,
                y=100 - result_coordinates.y,
            )

    statistics = []
    if 321 in raw_event.qualifiers:
        xg_value = float(raw_event.qualifiers[321])
        statistics.append(ExpectedGoals(value=xg_value))

    event_info = dict(
        coordinates=coordinates,
        result=result,
        result_coordinates=result_coordinates,
        qualifiers=qualifiers,
        statistics=statistics,
    )

    statistics = []
    for event_qualifier, statistic in zip(
        [EVENT_QUALIFIER_XG, EVENT_QUALIFIER_POST_SHOT_XG],
        [ExpectedGoals, PostShotExpectedGoals],
    ):
        xg_value = raw_event.qualifiers.get(event_qualifier)
        if xg_value:
            statistics.append(statistic(value=float(xg_value)))

    if statistics:
        event_info["statistics"] = statistics

    return event_info


def _parse_goalkeeper_events(raw_event: OptaEvent) -> Dict:
    qualifiers = _get_event_qualifiers(raw_event.qualifiers)
    goalkeeper_qualifiers = _get_goalkeeper_qualifiers(raw_event.type_id)
    qualifiers.extend(goalkeeper_qualifiers)

    return dict(result=None, qualifiers=qualifiers)


def _parse_duel(raw_event: OptaEvent) -> Dict:
    qualifiers = _get_event_qualifiers(raw_event.qualifiers)
    if raw_event.type_id == EVENT_TYPE_TACKLE:
        qualifiers.extend([DuelQualifier(value=DuelType.GROUND)])
    elif raw_event.type_id == EVENT_TYPE_AERIAL:
        qualifiers.extend(
            [
                DuelQualifier(value=DuelType.LOOSE_BALL),
                DuelQualifier(value=DuelType.AERIAL),
            ]
        )
    elif raw_event.type_id == EVENT_TYPE_50_50:
        qualifiers.extend(
            [
                DuelQualifier(value=DuelType.LOOSE_BALL),
                DuelQualifier(value=DuelType.GROUND),
            ]
        )

    result = DuelResult.WON if raw_event.outcome == 1 else DuelResult.LOST

    return dict(
        result=result,
        qualifiers=qualifiers,
    )


def _parse_interception(
    raw_event: OptaEvent, team: Team, next_event: OptaEvent
) -> Dict:
    qualifiers = _get_event_qualifiers(raw_event.qualifiers)
    result = InterceptionResult.SUCCESS

    if next_event is not None:
        next_event_type_id = int(next_event.type_id)
        if next_event_type_id in BALL_OUT_EVENTS:
            result = InterceptionResult.OUT
        elif (next_event_type_id in BALL_OWNING_EVENTS) and (
            next_event.contestant_id != team.team_id
        ):
            result = InterceptionResult.LOST

    return dict(
        result=result,
        qualifiers=qualifiers,
    )


def _get_end_coordinates(
    raw_qualifiers: Dict[int, str], start_coordinates: Optional[Point] = None
) -> Optional[Point]:
    x, y, z = None, None, None

    # pass
    if 140 in raw_qualifiers and 141 in raw_qualifiers:
        x = float(raw_qualifiers[140])
        y = float(raw_qualifiers[141])

    # blocked shot
    elif 146 in raw_qualifiers and 147 in raw_qualifiers:
        x = float(raw_qualifiers[146])
        y = float(raw_qualifiers[147])
        if 102 in raw_qualifiers and 103 in raw_qualifiers:
            # the goal mouth z-coordinate is projected back to the location
            # where the shot was blocked
            assert start_coordinates is not None
            x0, y0 = start_coordinates.x, start_coordinates.y
            x_proj = float(100)
            y_proj = float(raw_qualifiers[102])
            z_proj = float(raw_qualifiers[103])
            adj_proj = math.sqrt((x_proj - x0) ** 2 + (y_proj - y0) ** 2)
            adj_block = math.sqrt((x - x0) ** 2 + (y - y0) ** 2)
            z = z_proj / adj_proj * adj_block

    # passed the goal line
    elif 102 in raw_qualifiers:
        x = float(100)
        y = float(raw_qualifiers[102])
        if 103 in raw_qualifiers:
            z = float(raw_qualifiers[103])

    if x is not None and y is not None and z is not None:
        return Point3D(x=x, y=y, z=z)
    if x is not None and y is not None:
        return Point(x=x, y=y)

    return None


def _get_event_qualifiers(raw_qualifiers: Dict[int, str]) -> List[Qualifier]:
    qualifiers = []
    qualifiers.extend(_get_event_setpiece_qualifiers(raw_qualifiers))
    qualifiers.extend(_get_event_bodypart_qualifiers(raw_qualifiers))
    qualifiers.extend(_get_event_card_qualifiers(raw_qualifiers))
    qualifiers.extend(_get_event_counter_attack_qualifiers(raw_qualifiers))
    return qualifiers


def _get_pass_qualifiers(raw_qualifiers: Dict[int, str]) -> List[Qualifier]:
    qualifiers = []
    pass_qualifier_mapping = {
        EVENT_QUALIFIER_CROSS: PassType.CROSS,
        EVENT_QUALIFIER_LONG_BALL: PassType.LONG_BALL,
        EVENT_QUALIFIER_CHIPPED_BALL: PassType.CHIPPED_PASS,
        EVENT_QUALIFIER_THROUGH_BALL: PassType.THROUGH_BALL,
        EVENT_QUALIFIER_LAUNCH: PassType.LAUNCH,
        EVENT_QUALIFIER_FLICK_ON: PassType.FLICK_ON,
        EVENT_QUALIFIER_ASSIST_2ND: PassType.ASSIST_2ND,
    }
    for (
        sp_pass_qualifier,
        pass_qualifier_value,
    ) in pass_qualifier_mapping.items():
        if sp_pass_qualifier in raw_qualifiers:
            qualifiers.append(PassQualifier(value=pass_qualifier_value))

    if EVENT_QUALIFIER_SHOT_ASSIST in raw_qualifiers:
        # Qualifier '210' defines that the pass was an assist for a shot
        qualifiers.append(PassQualifier(value=PassType.SHOT_ASSIST))
        # The qualifier value then dictates whether it was a goal assist
        # or just key pass (note: values only available since May 2021).
        shot_result_qualifier = raw_qualifiers[EVENT_QUALIFIER_SHOT_ASSIST]
        if shot_result_qualifier is not None:
            if int(shot_result_qualifier) == EVENT_TYPE_SHOT_GOAL:
                qualifiers.append(PassQualifier(value=PassType.ASSIST))

    return qualifiers


def _get_event_setpiece_qualifiers(
    raw_qualifiers: Dict[int, str]
) -> List[Qualifier]:
    qualifiers = []
    if EVENT_QUALIFIER_CORNER_KICK in raw_qualifiers:
        qualifiers.append(SetPieceQualifier(value=SetPieceType.CORNER_KICK))
    elif (
        EVENT_QUALIFIER_FREE_KICK in raw_qualifiers
        or EVENT_QUALIFIER_FREE_KICK_SHOT in raw_qualifiers
    ):
        qualifiers.append(SetPieceQualifier(value=SetPieceType.FREE_KICK))
    elif EVENT_QUALIFIER_PENALTY in raw_qualifiers:
        qualifiers.append(SetPieceQualifier(value=SetPieceType.PENALTY))
    elif EVENT_QUALIFIER_THROW_IN in raw_qualifiers:
        qualifiers.append(SetPieceQualifier(value=SetPieceType.THROW_IN))
    elif EVENT_QUALIFIER_KICK_OFF in raw_qualifiers:
        qualifiers.append(SetPieceQualifier(value=SetPieceType.KICK_OFF))
    elif EVENT_QUALIFIER_GOAL_KICK in raw_qualifiers:
        qualifiers.append(SetPieceQualifier(value=SetPieceType.GOAL_KICK))
    return qualifiers


def _get_event_bodypart_qualifiers(
    raw_qualifiers: Dict[int, str]
) -> List[Qualifier]:
    qualifiers = []
    if EVENT_QUALIFIER_HEAD_PASS in raw_qualifiers:
        qualifiers.append(BodyPartQualifier(value=BodyPart.HEAD))
    elif EVENT_QUALIFIER_HEAD in raw_qualifiers:
        qualifiers.append(BodyPartQualifier(value=BodyPart.HEAD))
    elif EVENT_QUALIFIER_FLICK_ON in raw_qualifiers:
        qualifiers.append(BodyPartQualifier(value=BodyPart.HEAD))
    elif EVENT_QUALIFIER_LEFT_FOOT in raw_qualifiers:
        qualifiers.append(BodyPartQualifier(value=BodyPart.LEFT_FOOT))
    elif EVENT_QUALIFIER_RIGHT_FOOT in raw_qualifiers:
        qualifiers.append(BodyPartQualifier(value=BodyPart.RIGHT_FOOT))
    elif EVENT_QUALIFIER_OTHER_BODYPART in raw_qualifiers:
        qualifiers.append(BodyPartQualifier(value=BodyPart.OTHER))
    return qualifiers


def _get_event_card_qualifiers(
    raw_qualifiers: Dict[int, str]
) -> List[Qualifier]:
    qualifiers = []
    if EVENT_QUALIFIER_RED_CARD in raw_qualifiers:
        qualifiers.append(CardQualifier(value=CardType.RED))
    elif EVENT_QUALIFIER_FIRST_YELLOW_CARD in raw_qualifiers:
        qualifiers.append(CardQualifier(value=CardType.FIRST_YELLOW))
    elif EVENT_QUALIFIER_SECOND_YELLOW_CARD in raw_qualifiers:
        qualifiers.append(CardQualifier(value=CardType.SECOND_YELLOW))

    return qualifiers


def _get_goalkeeper_qualifiers(type_id: int) -> List[Qualifier]:
    qualifiers = []
    goalkeeper_qualifier = None
    if type_id == EVENT_TYPE_SAVE:
        goalkeeper_qualifier = GoalkeeperActionType.SAVE
    elif type_id == EVENT_TYPE_CLAIM:
        goalkeeper_qualifier = GoalkeeperActionType.CLAIM
    elif type_id == EVENT_TYPE_PUNCH:
        goalkeeper_qualifier = GoalkeeperActionType.PUNCH
    elif type_id == EVENT_TYPE_KEEPER_PICK_UP:
        goalkeeper_qualifier = GoalkeeperActionType.PICK_UP
    elif type_id == EVENT_TYPE_SMOTHER:
        goalkeeper_qualifier = GoalkeeperActionType.SMOTHER

    if goalkeeper_qualifier:
        qualifiers.append(GoalkeeperQualifier(value=goalkeeper_qualifier))

    return qualifiers


def _get_event_counter_attack_qualifiers(
    raw_qualifiers: Dict[int, str],
) -> List[Qualifier]:
    qualifiers = []
    if EVENT_QUALIFIER_COUNTER_ATTACK in raw_qualifiers:
        qualifiers.append(CounterAttackQualifier(True))

    return qualifiers


def _get_event_type_name(type_id: int) -> str:
    return event_type_names.get(type_id, "unknown")


class StatsPerformInputs(NamedTuple):
    meta_data: IO[bytes]
    meta_feed: str
    event_data: IO[bytes]
    event_feed: str
    meta_datatype: Optional[str] = None
    event_datatype: Optional[str] = None
    pitch_length: Optional[float] = None
    pitch_width: Optional[float] = None


class StatsPerformDeserializer(EventDataDeserializer[StatsPerformInputs]):
    @property
    def provider(self) -> Provider:
        return Provider.OPTA

    def deserialize(self, inputs: StatsPerformInputs) -> EventDataset:
        transformer = self.get_transformer(
            pitch_length=inputs.pitch_length, pitch_width=inputs.pitch_width
        )

        with performance_logging("load data", logger=logger):
            metadata_parser = get_parser(
                inputs.meta_data, inputs.meta_feed, inputs.event_datatype
            )
            events_parser = get_parser(
                inputs.event_data, inputs.event_feed, inputs.event_datatype
            )

        with performance_logging("parse data", logger=logger):
            periods = metadata_parser.extract_periods()
            score = metadata_parser.extract_score()
            teams = metadata_parser.extract_lineups()
            date = metadata_parser.extract_date()
            game_week = metadata_parser.extract_game_week()
            game_id = metadata_parser.extract_game_id()
            raw_events = [
                event
                for event in events_parser.extract_events()
                if event.type_id != EVENT_TYPE_DELETED_EVENT
            ]

            possession_team = None
            events = []
            for idx, raw_event in enumerate(raw_events):
                if raw_event.contestant_id == teams[0].team_id:
                    team = teams[0]
                elif raw_event.contestant_id == teams[1].team_id:
                    team = teams[1]
                else:
                    raise DeserializationError(
                        f"Unknown team_id {raw_event.contestant_id}"
                    )

                next_event = (
                    raw_events[idx + 1]
                    if (idx + 1) < len(raw_events)
                    else None
                )
                next_next_event = (
                    raw_events[idx + 2]
                    if (idx + 2) < len(raw_events)
                    else None
                )
                period = next(
                    (
                        period
                        for period in periods
                        if period.id == raw_event.period_id
                    ),
                    None,
                )
                if period is None:
                    logger.debug(
                        f"Skipping event {raw_event.id} because period doesn't match {raw_event.period_id}"
                    )
                    continue

                if raw_event.type_id == EVENT_TYPE_START_PERIOD:
                    logger.debug(
                        f"Set start of period {period.id} to {raw_event.timestamp}"
                    )
                    period.start_timestamp = raw_event.timestamp
                elif raw_event.type_id == EVENT_TYPE_END_PERIOD:
                    logger.debug(
                        f"Set end of period {period.id} to {raw_event.timestamp}"
                    )
                    period.end_timestamp = raw_event.timestamp
                elif raw_event.type_id == EVENT_TYPE_PLAYER_ON:
                    continue
                else:
                    if not period.start_timestamp:
                        # not started yet
                        continue

                    if raw_event.contestant_id == teams[0].team_id:
                        team = teams[0]
                    elif raw_event.contestant_id == teams[1].team_id:
                        team = teams[1]
                    else:
                        raise DeserializationError(
                            f"Unknown team_id {raw_event.contestant_id}"
                        )

                    player = None
                    if raw_event.player_id is not None:
                        player = team.get_player_by_id(raw_event.player_id)

                    if raw_event.type_id in BALL_OWNING_EVENTS:
                        possession_team = team

                    if raw_event.type_id in DEAD_BALL_EVENTS:
                        ball_state = BallState.DEAD
                    else:
                        ball_state = BallState.ALIVE

                    generic_event_kwargs = dict(
                        # from DataRecord
                        period=period,
                        timestamp=raw_event.timestamp - period.start_timestamp,
                        ball_owning_team=possession_team,
                        ball_state=ball_state,
                        # from Event
                        event_id=str(raw_event.id),
                        team=team,
                        player=player,
                        coordinates=Point(x=raw_event.x, y=raw_event.y),
                        raw_event=raw_event,
                    )

                    if raw_event.type_id == EVENT_TYPE_PASS:
                        pass_event_kwargs = _parse_pass(
                            raw_event,
                            next_event,
                            next_next_event,
                            team,
                            period,
                        )
                        event = self.event_factory.build_pass(
                            **pass_event_kwargs,
                            **generic_event_kwargs,
                        )
                    elif raw_event.type_id == EVENT_TYPE_OFFSIDE_PASS:
                        pass_event_kwargs = _parse_offside_pass(raw_event)
                        event = self.event_factory.build_pass(
                            **pass_event_kwargs,
                            **generic_event_kwargs,
                        )
                    elif raw_event.type_id == EVENT_TYPE_TAKE_ON:
                        take_on_event_kwargs = _parse_take_on(raw_event)
                        event = self.event_factory.build_take_on(
                            **take_on_event_kwargs,
                            **generic_event_kwargs,
                            qualifiers=None,
                        )
                    elif raw_event.type_id in (
                        EVENT_TYPE_SHOT_MISS,
                        EVENT_TYPE_SHOT_POST,
                        EVENT_TYPE_SHOT_SAVED,
                        EVENT_TYPE_SHOT_GOAL,
                    ):
                        if raw_event.type_id == EVENT_TYPE_SHOT_GOAL:
                            if 374 in raw_event.qualifiers:
                                # Qualifier 374 specifies the actual time of the shot for all goal events
                                # It uses London timezone for both MA3 and F24 feeds
                                naive_datetime = datetime.strptime(
                                    raw_event.qualifiers[374],
                                    "%Y-%m-%d %H:%M:%S.%f",
                                )
                                timezone = pytz.timezone("Europe/London")
                                aware_datetime = timezone.localize(
                                    naive_datetime
                                )
                                generic_event_kwargs["timestamp"] = (
                                    aware_datetime.astimezone(pytz.utc)
                                    - period.start_timestamp
                                )
                        shot_event_kwargs = _parse_shot(raw_event)
                        kwargs = {}
                        kwargs.update(generic_event_kwargs)
                        kwargs.update(shot_event_kwargs)
                        event = self.event_factory.build_shot(**kwargs)
                    elif raw_event.type_id == EVENT_TYPE_RECOVERY:
                        event = self.event_factory.build_recovery(
                            result=None,
                            qualifiers=None,
                            **generic_event_kwargs,
                        )
                    elif raw_event.type_id == EVENT_TYPE_CLEARANCE:
                        clearance_event_kwargs = _parse_clearance(raw_event)
                        event = self.event_factory.build_clearance(
                            result=None,
                            **clearance_event_kwargs,
                            **generic_event_kwargs,
                        )
                    elif raw_event.type_id in DUEL_EVENTS:
                        duel_event_kwargs = _parse_duel(raw_event)
                        event = self.event_factory.build_duel(
                            **duel_event_kwargs,
                            **generic_event_kwargs,
                        )
                    elif raw_event.type_id in (
                        EVENT_TYPE_INTERCEPTION,
                        EVENT_TYPE_BLOCKED_PASS,
                    ):
                        interception_event_kwargs = _parse_interception(
                            raw_event, team, next_event
                        )
                        event = self.event_factory.build_interception(
                            **interception_event_kwargs,
                            **generic_event_kwargs,
                        )
                    elif raw_event.type_id in KEEPER_EVENTS:
                        # Qualifier 94 means the "save" event is a shot block by a defender
                        if 94 in raw_event.qualifiers:
                            event = self.event_factory.build_generic(
                                **generic_event_kwargs,
                                result=None,
                                qualifiers=None,
                                event_name="block",
                            )
                        else:
                            goalkeeper_event_kwargs = _parse_goalkeeper_events(
                                raw_event
                            )
                            event = self.event_factory.build_goalkeeper_event(
                                **goalkeeper_event_kwargs,
                                **generic_event_kwargs,
                            )
                    elif (raw_event.type_id == EVENT_TYPE_BALL_TOUCH) & (
                        raw_event.outcome == 0
                    ):
                        event = self.event_factory.build_miscontrol(
                            result=None,
                            qualifiers=None,
                            **generic_event_kwargs,
                        )
                    elif (raw_event.type_id == EVENT_TYPE_FOUL_COMMITTED) and (
                        raw_event.outcome == 0
                    ):
                        event = self.event_factory.build_foul_committed(
                            result=None,
                            qualifiers=None,
                            **generic_event_kwargs,
                        )
                    elif raw_event.type_id in BALL_OUT_EVENTS:
                        event = self.event_factory.build_ball_out(
                            result=None,
                            qualifiers=None,
                            **generic_event_kwargs,
                        )
                    elif raw_event.type_id == EVENT_TYPE_FORMATION_CHANGE:
                        generic_event_kwargs["timestamp"] = max(
                            timedelta(0), generic_event_kwargs["timestamp"]
                        )
                        formation_change_event_kwargs = (
                            _parse_formation_change(raw_event, team)
                        )
                        event = self.event_factory.build_formation_change(
                            result=None,
                            qualifiers=None,
                            **formation_change_event_kwargs,
                            **generic_event_kwargs,
                        )
                    elif raw_event.type_id == EVENT_TYPE_PLAYER_OFF:
                        generic_event_kwargs["timestamp"] = max(
                            timedelta(0), generic_event_kwargs["timestamp"]
                        )
                        substitution_event_kwargs = _parse_substitution(
                            next_event, team
                        )
                        event = self.event_factory.build_substitution(
                            result=None,
                            qualifiers=None,
                            **substitution_event_kwargs,
                            **generic_event_kwargs,
                        )

                    elif raw_event.type_id == EVENT_TYPE_CARD:
                        card_event_kwargs = _parse_card(raw_event)

                        event = self.event_factory.build_card(
                            **card_event_kwargs,
                            **generic_event_kwargs,
                        )
                    else:
                        event = self.event_factory.build_generic(
                            **generic_event_kwargs,
                            result=None,
                            qualifiers=None,
                            event_name=_get_event_type_name(raw_event.type_id),
                        )

                    if self.should_include_event(event):
                        events.append(transformer.transform_event(event))

        metadata = Metadata(
            teams=list(teams),
            periods=periods,
            pitch_dimensions=transformer.get_to_coordinate_system().pitch_dimensions,
            score=score,
            frame_rate=None,
            orientation=Orientation.ACTION_EXECUTING_TEAM,
            flags=DatasetFlag.BALL_OWNING_TEAM | DatasetFlag.BALL_STATE,
            provider=(
                Provider.OPTA
                if inputs.event_feed.upper() == "F24"
                else Provider.STATSPERFORM
            ),
            coordinate_system=transformer.get_to_coordinate_system(),
            date=date,
            game_week=game_week,
            game_id=game_id,
        )

        return EventDataset(
            metadata=metadata,
            records=events,
        )
