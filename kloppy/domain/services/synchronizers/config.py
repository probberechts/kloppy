from kloppy.domain import (
    DuelQualifier,
    DuelType,
    EventType,
    GoalkeeperActionType,
    GoalkeeperQualifier,
    SetPieceQualifier,
)


def is_handled(event):
    """Returns True if the event is considered for alignment."""
    if event.event_type in [
        EventType.PASS,
        EventType.SHOT,
        EventType.TAKE_ON,
        EventType.CARRY,
        EventType.CLEARANCE,
        EventType.INTERCEPTION,
        EventType.DUEL,
        EventType.RECOVERY,
        EventType.MISCONTROL,
        EventType.FOUL_COMMITTED,
        EventType.GOALKEEPER,
    ]:
        return True
    return False


def is_not_handled(event):
    """Returns True if the event is not considered for alignment."""
    if event.event_type in [
        EventType.GENERIC,
        EventType.FORMATION_CHANGE,
        EventType.SUBSTITUTION,
        EventType.CARD,
        EventType.PLAYER_ON,
        EventType.PLAYER_OFF,
        EventType.BALL_OUT,
    ]:
        return True
    return False


def is_on_ball_action(event):
    """Returns True if the event is considered to be an action on the ball."""
    if event.event_type in [
        EventType.PASS,
        EventType.SHOT,
        EventType.TAKE_ON,
        EventType.CARRY,
        EventType.CLEARANCE,
        EventType.INTERCEPTION,
        EventType.RECOVERY,
        EventType.MISCONTROL,
        EventType.FOUL_COMMITTED,
        EventType.GOALKEEPER,
    ]:
        return True
    return False


def is_open_play(event):
    """Returns True if the event can only happen during open play."""
    if event.event_type in [
        EventType.PASS,
        EventType.SHOT,
        EventType.TAKE_ON,
        EventType.CARRY,
        EventType.CLEARANCE,
        EventType.INTERCEPTION,
        EventType.DUEL,
        EventType.RECOVERY,
        EventType.MISCONTROL,
        EventType.GOALKEEPER,
    ] and not is_set_piece(event):
        return True
    return False


def is_dead_ball(event):
    """Returns True if the event can only happen during a dead ball state."""
    if event.event_type in [
        EventType.SUBSTITUTION,
        EventType.CARD,
    ]:
        return True
    return False


def is_set_piece(event):
    """Returns True if the event is considered to be a set piece."""
    if event.get_qualifier_value(SetPieceQualifier) is not None:
        return True
    return False


def is_dead_ball_start(event):
    """Returns True if the event is considered to be the start of a dead ball."""
    if event.event_type in [EventType.BALL_OUT, EventType.FOUL_COMMITTED]:
        return True
    # if event.event_type == EventType.SHOT and event.result in [
    #     ShotResult.GOAL,
    #     ShotResult.OWN_GOAL,
    # ]:
    #     return True  # the end of the shot is a dead ball start
    return False


def is_pass_like(event):
    """Returns True if the event ends the player's possession."""
    if event.event_type in [
        EventType.PASS,
        EventType.SHOT,
        # EventType.TAKE_ON,
        EventType.CLEARANCE,
    ]:
        return True
    if event.event_type == EventType.GOALKEEPER:
        actiontype = event.get_qualifier_value(GoalkeeperQualifier)
        if actiontype in [GoalkeeperActionType.PUNCH]:
            return True
    return False


def is_first_touch(event):
    """Returns True if the event starts the player's possession."""
    if event.event_type in [
        EventType.INTERCEPTION,
        EventType.RECOVERY,
    ]:
        return True
    elif event.event_type == EventType.CARRY:
        return True
        # TODO: in the Metrica data, a carry is alayws at the start of
        # a individual possession but maybe be should blacklist some previous
        # events for other datasets
        # prev_event = event.prev_record
        # if prev_event is None or event.player != prev_event.player:
        # return True
    elif event.event_type == EventType.GOALKEEPER:
        actiontype = event.get_qualifier_value(GoalkeeperQualifier)
        if actiontype in [
            GoalkeeperActionType.SAVE,
            GoalkeeperActionType.CLAIM,
            GoalkeeperActionType.PICK_UP,
        ]:
            return True
    return False


def is_run_with_ball(event):
    """Returns True if the event is considered to be a run with the ball."""
    if event.event_type in [
        EventType.TAKE_ON,
        EventType.CARRY,
    ]:
        return True
    return False


def is_bad_touch(event):
    """Returns True if the event is considered to be a bad touch."""
    return event.event_type == EventType.MISCONTROL


def is_fault_like(event):
    """Returns True if the event is a fault or tackle."""
    if event.event_type == EventType.FOUL_COMMITTED:
        return True
    if event.event_type == EventType.DUEL:
        dueltype = event.get_qualifier_values(DuelQualifier)
        if DuelType.TACKLE in dueltype:
            return True
    return False


infer_ball_state = True
max_speed_ball = 55.0
max_speed_player = 12.0
smoothing_window_speed = None
