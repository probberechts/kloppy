from dataclasses import dataclass, replace
from typing import Optional

from kloppy.domain import (
    BallOutEvent,
    CarryEvent,
    DuelEvent,
    Event,
    EventDataset,
    FoulCommittedEvent,
    GoalkeeperActionType,
    GoalkeeperEvent,
    GoalkeeperQualifier,
    InterceptionEvent,
    PassEvent,
    PassResult,
    RecoveryEvent,
    SetPieceQualifier,
    ShotEvent,
    TakeOnEvent,
    TakeOnResult,
    Team,
)

from ..builder import StateBuilder


@dataclass
class Possession:
    possession_id: int
    team: Optional[Team]


class PossessionStateBuilder(StateBuilder):
    def __init__(self, strict: bool = True):
        self.strict = strict
        print(f"strict: {strict}")

    def _open_possession(
        self, state: Possession, event: Optional[Event]
    ) -> bool:
        if event is None:
            return False
        # a recovery by the other team
        if (
            isinstance(event, RecoveryEvent)
            and (state.team != event.team)
            and (
                not self.strict
                or self._open_possession(state, event.next_record)
            )
        ):
            return True
        # a shot attempt by the other team
        if isinstance(event, ShotEvent) and (state.team != event.team):
            return True
        # a pass attempt by the other team
        if (
            isinstance(event, PassEvent)
            and (state.team != event.team)
            and (not self.strict or (event.result == PassResult.COMPLETE))
        ):
            return True
        # a take-on attempt by the other team
        if (
            isinstance(event, TakeOnEvent)
            and (state.team != event.team)
            and (not self.strict or (event.result == TakeOnResult.COMPLETE))
        ):
            return True
        # a carry by the other team, followed by a possesion-starting action
        if (
            isinstance(event, CarryEvent)
            and (state.team != event.team)
            and self._open_possession(state, event.next_record)
        ):
            return True
        # an interception by the other team, followed by a possesion-starting action
        if (
            isinstance(event, InterceptionEvent)
            and (state.team != event.team)
            and self._open_possession(state, event.next_record)
        ):
            return True
        # a duel won by the other team, followed by a possesion-starting action
        if (
            isinstance(event, DuelEvent)
            and (state.team != event.team)
            and self._open_possession(state, event.next_record)
        ):
            return True
        # a goalkeeper picking up the ball
        if (
            isinstance(event, GoalkeeperEvent)
            and (state.team != event.team)
            and (
                GoalkeeperActionType.CLAIM
                in event.get_qualifier_values(GoalkeeperQualifier)
                or GoalkeeperActionType.PICK_UP
                in event.get_qualifier_values(GoalkeeperQualifier)
            )
        ):
            return True
        # a set piece
        if event.get_qualifier_value(SetPieceQualifier):
            return True
        return False

    def _close_possession(self, state: Possession, event: Event) -> bool:
        # a ball out event
        if isinstance(event, BallOutEvent):
            return True
        # a foul
        if isinstance(event, FoulCommittedEvent):
            return True
        return False

    def initial_state(self, dataset: EventDataset) -> Possession:
        first_event = next(iter(dataset.events), None)
        if self._open_possession(
            Possession(possession_id=0, team=None), first_event
        ):
            return Possession(possession_id=0, team=dataset.events[0].team)
        return Possession(possession_id=0, team=None)

    def reduce_before(self, state: Possession, event: Event) -> Possession:
        if self._open_possession(state, event):
            state = replace(
                state, possession_id=state.possession_id + 1, team=event.team
            )

        return state

    def reduce_after(self, state: Possession, event: Event) -> Possession:
        if self._close_possession(state, event):
            state = replace(
                state, possession_id=state.possession_id + 1, team=None
            )

        return state
