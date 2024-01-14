"""Code to visualise the alignment between event and tracking data."""
import math
from collections import defaultdict

from kloppy.domain import (
    DatasetTransformer,
    TrackingDataset,
    UEFACoordinateSystem,
)

from . import utils

try:
    import matplotlib.pyplot as plt
    from matplotlib import animation
    from mplsoccer import Pitch
except ImportError:
    raise ImportError(
        "To use the visualisation module, you need to install the "
        "mplsoccer package. You can do so by running: "
        "pip install mplsoccer"
    )
    plt = None
    animation = None
    Pitch = None


def _get_dot_style(
    name,
    home_color,
    away_color,
    ball_color,
    player_size,
    ball_size,
    event_size,
):
    dot_styles = {
        "simple": {
            "home": {
                "marker": "o",
                "markerfacecolor": home_color,
                "markeredgecolor": "black",
                "markersize": player_size,
                "linestyle": "None",
                "zorder": 2,
            },
            "away": {
                "marker": "o",
                "markerfacecolor": away_color,
                "markeredgecolor": "black",
                "markersize": player_size,
                "linestyle": "None",
                "zorder": 2,
            },
            "ball": {
                "marker": "o",
                "markerfacecolor": ball_color,
                "markeredgecolor": "black",
                "markersize": ball_size,
                "linestyle": "None",
                "zorder": 3,
            },
            "event": {
                "marker": "x",
                "markerfacecolor": "red",
                "markeredgecolor": "red",
                "markersize": event_size,
                "linestyle": "None",
                "zorder": 4,
            },
        }
    }

    return dot_styles[name]


def animate_alignment(
    events,
    frames,
    pitch_type="metricasports",
    framerate=25,
    figsize=(8, 5.2),
    home_color="#7f63b8",
    away_color="#b94b75",
    ball_color="yellow",
    dot_style="simple",
    player_size=10,
    ball_size=6,
    event_size=14,
    clock_size=11,
    pause_on_event=1,
):
    """Animate the alignment between the event and tracking data."""
    # set up the figure
    pitch = Pitch(
        pitch_type=pitch_type,
        goal_type="line",
        pitch_width=68,
        pitch_length=105,
    )
    fig, ax = pitch.draw(figsize=figsize)

    # then setup the pitch plot markers we want to animate
    cosmetics = _get_dot_style(
        dot_style,
        home_color,
        away_color,
        ball_color,
        player_size,
        ball_size,
        event_size,
    )
    (ball,) = ax.plot([], [], **cosmetics["ball"])
    (home,) = ax.plot([], [], **cosmetics["home"])
    (away,) = ax.plot([], [], **cosmetics["away"])
    (event,) = ax.plot([], [], **cosmetics["event"])
    frame_clock = ax.text(
        0.02,
        0.03,
        "Frame clock - 00:00.000",
        fontsize=clock_size,
        color="black",
        horizontalalignment="left",
        verticalalignment="top",
    )
    event_clock = ax.text(
        0.98,
        0.03,
        "Event - 00:00",
        fontsize=clock_size,
        color="red",
        horizontalalignment="right",
        verticalalignment="top",
    )

    def gen_data():
        frame_to_event = {
            event.freeze_frame.frame_id: event
            for event in events
            if event.freeze_frame is not None
        }
        for frame in frames:
            active_event = frame_to_event.get(frame.frame_id)
            if active_event:
                for _ in range(framerate * pause_on_event):
                    # repeat the frame `framerate` times to simulate a pause
                    yield frame, active_event
            else:
                yield frame, None

    def animate(data):
        frame, active_event = data

        # display the clock
        ts = frame.timestamp
        frame_clock.set_text(
            "Frame clock - {min:02d}:{sec:02d}.{ms:03d}".format(
                min=int(ts // 60), sec=int(ts % 60), ms=int(ts * 1000 % 1000)
            )
        )

        # set the ball data with the x and y positions for the ith frame
        ball_coo = frame.ball_coordinates
        ball.set_data([ball_coo.x], [ball_coo.y])

        # set the player data
        coordinates_per_team = defaultdict(list)
        for (
            player,
            coordinates,
        ) in frame.players_coordinates.items():
            if coordinates is not None:
                coordinates_per_team[f"{player.team.ground}_x"].append(
                    coordinates.x
                )
                coordinates_per_team[f"{player.team.ground}_y"].append(
                    coordinates.y
                )
        home.set_data(
            coordinates_per_team["home_x"], coordinates_per_team["home_y"]
        )
        away.set_data(
            coordinates_per_team["away_x"], coordinates_per_team["away_y"]
        )

        # display the active event
        if active_event:
            if active_event.coordinates:
                event.set_data(
                    [active_event.coordinates.x], [active_event.coordinates.y]
                )
            ts = active_event.timestamp
            etype = (
                active_event.event_name.title()
                if active_event.event_name
                else "Generic"
            )
            event_clock.set_text(
                "{etype} - {min:02d}:{sec:02d}.{ms:03d}".format(
                    etype=etype,
                    min=int(ts // 60),
                    sec=int(ts % 60),
                    ms=int(ts * 1000 % 1000),
                )
            )
        else:
            event.set_data([], [])
            event_clock.set_text("")

        return frame_clock, event_clock, ball, home, away, event

    # call the animator
    anim = animation.FuncAnimation(
        fig, animate, frames=gen_data, interval=framerate, blit=True
    )
    plt.close()
    return anim


def plot_alignment(
    events,
    pitch_type="metricasports",
    figsize=(8, 5.2),
    home_color="#7f63b8",
    away_color="#b94b75",
    ball_color="yellow",
    dot_style="simple",
    player_size=10,
    ball_size=6,
    event_size=14,
    clock_size=11,
):
    """Plot each event in the event dataset with the corresponding frame."""
    # set up the figure
    pitch = Pitch(
        pitch_type=pitch_type,
        goal_type="line",
        pitch_width=68,
        pitch_length=105,
    )
    fig, axs = pitch.grid(
        nrows=math.ceil(len(events) / 3),
        ncols=3,
        figheight=figsize[0] * math.ceil(len(events) / 3),
        axis=False,
        endnote_height=0,
        title_height=0,
    )

    # then setup the pitch plot markers we want to plot
    cosmetics = _get_dot_style(
        dot_style,
        home_color,
        away_color,
        ball_color,
        player_size,
        ball_size,
        event_size,
    )

    # cycle through the grid axes and plot the events with their corresponding frame
    for event, ax in zip(events, axs.flat):
        # display the event
        if event.coordinates:
            ax.plot(
                [event.coordinates.x],
                [event.coordinates.y],
                **cosmetics["event"],
            )

        # display the event clock
        ts = event.timestamp
        etype = event.event_name.title() if event.event_name else "Generic"
        event_clock_str = "{etype} - {min:02d}:{sec:02d}.{ms:03d}".format(
            etype=etype,
            min=int(ts // 60),
            sec=int(ts % 60),
            ms=int(ts * 1000 % 1000),
        )
        ax.text(
            0.98,
            0.03,
            event_clock_str,
            fontsize=clock_size,
            color="red",
            horizontalalignment="right",
            verticalalignment="top",
        )

        # display the frame clock
        frame = event.freeze_frame
        if frame is None:
            ax.text(
                0.02,
                0.03,
                "No matching frame",
                fontsize=clock_size,
                color="black",
                horizontalalignment="left",
                verticalalignment="top",
            )
            continue

        ts = frame.timestamp
        frame_clock_str = "Frame clock - {min:02d}:{sec:02d}.{ms:03d}".format(
            min=int(ts // 60), sec=int(ts % 60), ms=int(ts * 1000 % 1000)
        )
        ax.text(
            0.02,
            0.03,
            frame_clock_str,
            fontsize=clock_size,
            color="black",
            horizontalalignment="left",
            verticalalignment="top",
        )

        # display the ball data
        ball_coo = frame.ball_coordinates
        ax.plot(
            [ball_coo.x],
            [ball_coo.y],
            **cosmetics["ball"],
        )

        # display the player data
        coordinates_per_team = defaultdict(list)
        for (
            player,
            coordinates,
        ) in frame.players_coordinates.items():
            if coordinates is not None:
                coordinates_per_team[f"{player.team.ground}_x"].append(
                    coordinates.x
                )
                coordinates_per_team[f"{player.team.ground}_y"].append(
                    coordinates.y
                )
        ax.plot(
            coordinates_per_team["home_x"],
            coordinates_per_team["home_y"],
            **cosmetics["home"],
        )
        ax.plot(
            coordinates_per_team["away_x"],
            coordinates_per_team["away_y"],
            **cosmetics["away"],
        )

    plt.close()
    return fig


def animate_score(
    event,
    frames,
    coordinate_system,
    score_fn,
    mask_fn=None,
    framerate=25,
    figsize=(6, 5.2),
    home_color="#7f63b8",
    away_color="#b94b75",
    ball_color="yellow",
    dot_style="simple",
    player_size=10,
    ball_size=6,
    event_size=14,
    clock_size=11,
):
    if mask_fn is None:
        mask_fn = lambda event, frames: [True] * len(frames)

    # FIXME: this gets the coordinates in the right coordinate system
    # to compute the score, but it is not very clean
    transformer = DatasetTransformer(
        from_coordinate_system=coordinate_system,
        to_coordinate_system=UEFACoordinateSystem(normalized=False),
    )
    norm_event = transformer.transform_event(event)
    norm_event.prev_record = event.prev_record
    norm_event.next_record = event.next_record
    norm_frames = TrackingDataset(
        records=[transformer.transform_frame(f) for f in frames], metadata=None
    )

    # set up the figure
    pitch = Pitch(
        pitch_type="uefa",
        goal_type="line",
        pitch_width=68,
        pitch_length=105,
    )
    fig, axs = pitch.grid(
        figheight=figsize[0],
        grid_height=0.6,
        grid_width=0.5,
        title_height=0,
        endnote_height=0.3,
    )

    # then setup the pitch plot markers we want to animate
    cosmetics = _get_dot_style(
        dot_style,
        home_color,
        away_color,
        ball_color,
        player_size,
        ball_size,
        event_size,
    )
    (ball,) = axs["pitch"].plot([], [], **cosmetics["ball"])
    (home,) = axs["pitch"].plot([], [], **cosmetics["home"])
    (away,) = axs["pitch"].plot([], [], **cosmetics["away"])
    frame_clock = axs["pitch"].text(
        105 * 0.02,
        68 * 0.97,
        "Frame clock - 00:00.000",
        fontsize=clock_size,
        color="black",
        horizontalalignment="left",
        verticalalignment="top",
    )
    # plot the score
    axs["endnote"].plot(
        [frame.frame_id for frame in norm_frames],
        score_fn(norm_event, norm_frames),
        linewidth=1,
        color="grey",
    )
    axs["endnote"].scatter(
        [frame.frame_id for frame in norm_frames],
        score_fn(norm_event, norm_frames, mask_fn(norm_event, norm_frames)),
        linewidth=1,
        color="grey",
    )
    (score,) = axs["endnote"].plot(
        [],
        [],
        ms=6,
        markerfacecolor="red",
        markeredgecolor="red",
        marker="o",
        zorder=3,
    )
    axs["endnote"].set_xlim(norm_frames[0].frame_id, norm_frames[-1].frame_id)

    # display the event
    if norm_event.coordinates:
        axs["pitch"].plot(
            [norm_event.coordinates.x],
            [norm_event.coordinates.y],
            **cosmetics["event"],
        )
    ts = norm_event.timestamp
    etype = (
        norm_event.event_name.title() if norm_event.event_name else "Generic"
    )
    event_clock_str = "{etype} - {min:02d}:{sec:02d}.{ms:03d}".format(
        etype=etype,
        min=int(ts // 60),
        sec=int(ts % 60),
        ms=int(ts * 1000 % 1000),
    )
    axs["pitch"].text(
        105 * 0.98,
        68 * 0.97,
        event_clock_str,
        fontsize=clock_size,
        color="red",
        horizontalalignment="right",
        verticalalignment="top",
    )

    def gen_data():
        for frame in norm_frames:
            yield frame

    def animate(data):
        frame = data

        # display the clock
        ts = frame.timestamp
        frame_clock.set_text(
            "Frame clock - {min:02d}:{sec:02d}.{ms:03d}".format(
                min=int(ts // 60), sec=int(ts % 60), ms=int(ts * 1000 % 1000)
            )
        )

        # set the ball data with the x and y positions for the ith frame
        ball_coo = frame.ball_coordinates
        ball.set_data([ball_coo.x], [ball_coo.y])

        # set the player data
        coordinates_per_team = defaultdict(list)
        for (
            player,
            coordinates,
        ) in frame.players_coordinates.items():
            if coordinates is not None:
                coordinates_per_team[f"{player.team.ground}_x"].append(
                    coordinates.x
                )
                coordinates_per_team[f"{player.team.ground}_y"].append(
                    coordinates.y
                )
        home.set_data(
            coordinates_per_team["home_x"], coordinates_per_team["home_y"]
        )
        away.set_data(
            coordinates_per_team["away_x"], coordinates_per_team["away_y"]
        )

        score.set_data([frame.frame_id], score_fn(norm_event, [frame]))

        return frame_clock, ball, home, away, score

    # call the animator
    anim = animation.FuncAnimation(
        fig, animate, frames=gen_data, interval=framerate, blit=True
    )
    plt.close()
    return anim
