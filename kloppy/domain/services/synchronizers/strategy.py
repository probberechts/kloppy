from typing import Callable, Dict, List, Tuple

from kloppy.domain import Event, Frame
from . import config

_SYNCHRONIZATION_STRATEGY_REGISTRY: Dict[str, "SynchronizationStrategy"] = {}


class SynchronizationStrategy:
    """A synhronization strategy.

    A synchronization strategy consists of a scoring function that assigns a
    score to each event-frame pair, a masking function that masks invalid
    event-frame pairs, and an alignment algorithm that finds the best
    alignment between the event and tracking sequences.

    Four alignment methods are implemented:

    - time: Each event is matched with the tracking frame that is closest in
      time to the event's timestamp. This method should be used if the
      tracking data is already synchronized with the event data.
    - local: Defines a window around each event and matches the event with the
      best-scoring tracking frame within this window. The windows of events
      can overlap. This method works well for aligning a single category of
      events that are unique within a time window, such as shots [2] or passes
      [3].
    - greedy: Defines a window around each event and matches the event with
      the best-scoring tracking frame within this window. A window can only
      start after the last matched frame. This is the approach used by ETSY
      [4]. It is computationally efficient but can lead to suboptimal
      alignments when similar events occur in close proximity (e.g., a one-two
      pass).
    - dp: Uses the Needleman-Wunsch dynamic programming algorithm to find the
      best global alignment between the event and tracking data. This approach
      is used by sync.soccer [1]. It has the highest computational costs but
      guarantees the most optimal global alignment.

    Parameters
    ----------
    window : float | (Event) -> int, default: 5
        For each event, score a maximum of `window`/2 seconds of tracking
        data before and after the event timestamp. If None, consider the
        entire tracking sequence. If a function is given, the window size
        is determined dynamically for each event.
    mask_fn : (Event, Frame) -> bool, optional
        A function that returns True if the event-frame pair should be
        considered for alignment. If None, a default masking function is
        used which masks event-frame pairs where the acting player is not
        in possession of the ball.
    score_fn : (Event, list(Frame), list(bool)) -> list(float)
        A function that assigns a score to each event-frame pair. Lower
        values indicate a better match. If None, a default score
        function is used which scores frames based on the distance between
        the acting player and the ball and the acceleration of the ball.
    alignment_fn : str, default: "dp"
        The alignment method to use for synchronization. One of
        "time", "local", "greedy" or "dp".

    References
    ----------
    .. [1] Allan Clark, and Marek Kwiatkowski. "The right way to synchronise
           event and tracking data"(2020). https://kwiatkowski.io/sync.soccer
    .. [2] Gabriel Anzer and Pascal Bauer. "A goal scoring probability model
           for shots based on synchronized positional and event data in
           football (soccer)." Frontiers in Sports and Active Living (2021):
           53.
    .. [3] Gabriel Anzer, and Pascal Bauer. "Expected passes: Determining the
           difficulty of a pass in football (soccer) using spatio-temporal
           data." Data mining and knowledge discovery 36.1 (2022): 295-317.
    .. [4] Maaike Van Roy, Lorenzo Cascioli, and Jesse Davis. "ETSY:
           A rule-based approach to Event and Tracking data SYnchronization".
           Machine Learning and Data Mining for Sports Analytics ECML/PKDD
           2023 Workshop (2023).
    """

    def __init__(self, window_fn, filter_fn, mask_fn, score_fn, alignment_fn):
        self.window_fn = window_fn
        self.filter_fn = filter_fn
        self.mask_fn = mask_fn
        self.score_fn = score_fn
        self.alignment_fn = alignment_fn

    def configure(self, **kwargs) -> "SynchronizationStrategy":
        """Configure the synchronization strategy."""
        scale_params = (
            [
                f"scale_{mis_func.__name__}"
                for mis_func in self.score_fn.mis_funcs
            ]
            if self.score_fn
            else []
        )
        for param_name, param_value in kwargs.items():
            if param_name == "window":
                self.window_fn = lambda event: param_value
            elif param_name.startswith("scale_"):
                idx = scale_params.index(param_name)
                if idx is None:
                    raise ValueError(
                        f'Unknown scale parameter "{param_name}". '
                        "Known parameters: {}".format(
                            ", ".join(f'"{w}"' for w in scale_params),
                        )
                    )
                self.score_fn.weights[idx] = param_value
            else:
                raise ValueError(
                    f'Unknown strategy parameter "{param_name}". '
                    "Known parameters: {}".format(
                        ", ".join(f'"{w}"' for w in scale_params + ["window"]),
                    )
                )
        return self

    def register(self, name: str) -> None:
        """Register the strategy under the given name."""
        _SYNCHRONIZATION_STRATEGY_REGISTRY[name] = self

    def __call__(
        self,
        events: List[Event],
        frames: List[Frame],
        fps: float,
        offset: float,
    ) -> List[Tuple[int, int]]:
        """Returns the alignment between the events and the frames."""
        alignment = self.alignment_fn(
            events,
            frames,
            fps=fps,
            offset=offset,
            window_fn=self.window_fn,
            mask_fn=self.mask_fn,
            score_fn=self.score_fn,
        )
        return alignment


class SynchronizationStrategyBuilder:
    def __init__(self):
        self.window_fn = None
        self.filter_fn = config.is_handled
        self.mask_fn = None
        self.score_fn = None
        self.alignment_fn = None

    def with_window_fn(
        self, window_fn: Callable
    ) -> "SynchronizationStrategyBuilder":
        self.window_fn = window_fn
        return self

    def with_filter_fn(
        self, filter_fn: Callable
    ) -> "SynchronizationStrategyBuilder":
        self.filter_fn = filter_fn
        return self

    def with_mask_fn(
        self, mask_fn: Callable
    ) -> "SynchronizationStrategyBuilder":
        self.mask_fn = mask_fn
        return self

    def with_score_fn(
        self, score_fn: Callable
    ) -> "SynchronizationStrategyBuilder":
        self.score_fn = score_fn
        return self

    def with_alignment_fn(
        self, alignment_fn: Callable
    ) -> "SynchronizationStrategyBuilder":
        self.alignment_fn = alignment_fn
        return self

    def build(self) -> "SynchronizationStrategy":
        return SynchronizationStrategy(
            window_fn=self.window_fn,
            filter_fn=self.filter_fn,
            mask_fn=self.mask_fn,
            score_fn=self.score_fn,
            alignment_fn=self.alignment_fn,
        )


def create_synchronization_strategy(
    strategy_key: str,
) -> SynchronizationStrategy:
    if strategy_key not in _SYNCHRONIZATION_STRATEGY_REGISTRY:
        raise ValueError(
            f'Synchronization strategy "{strategy_key}" not found. '
            "Known strategies: {}".format(
                ", ".join(
                    [
                        f'"{w}"'
                        for w in _SYNCHRONIZATION_STRATEGY_REGISTRY.keys()
                    ]
                ),
            )
        )
    return _SYNCHRONIZATION_STRATEGY_REGISTRY[strategy_key]
