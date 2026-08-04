"""Shared environment and decision rule for the ToM-conditioned norm task.

The observer is deliberately *not* a learned policy here.  After watching a
partner, it plans to collect any item whose expected sanction is lower than
the item's base value.  That isolates the question asked by the experiment:
does the frozen ToM module supply the preference information needed for
selective compliance?
"""

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from sorrel.examples.treasurehunt.notebooks.tom3_common import (
    ITEM_KINDS,
    Tom3Env,
    make_config,
)
from sorrel.examples.treasurehunt.entities import EmptyEntity
from sorrel.examples.treasurehunt.world import QuadrantWorld


@dataclass
class ConsumptionEvent:
    step: int
    kind: str
    base_reward: float
    sanction: float


@dataclass
class NormLog:
    initial_items: dict[str, int]
    consumed: dict[str, int] = field(default_factory=dict)
    events: list[ConsumptionEvent] = field(default_factory=list)

    @property
    def total_reward(self) -> float:
        return sum(event.base_reward - event.sanction for event in self.events)

    @property
    def sanctions(self) -> int:
        return sum(event.sanction > 0 for event in self.events)


class NormEnv(Tom3Env):
    """``Tom3Env`` with an embodied observer and deterministic sanctions.

    Demonstrators still take their normal scripted turns.  The evaluator calls
    :meth:`observer_move` once after each demonstrator turn during foraging.
    A sanction is independent of proximity, so distance keeping cannot mimic
    compliance.
    """

    def __init__(self, *args, sanction: float, **kwargs):
        self.sanction = float(sanction)
        self.norm_log: NormLog | None = None
        super().__init__(*args, **kwargs)
        self.norm_log = NormLog(initial_items=self.count_items())

    def count_items(self) -> dict[str, int]:
        return {
            kind: sum(entity.kind == kind for entity in self.world.map.flat)
            for kind in ITEM_KINDS[: self.world.n_rooms]
        }

    def observer_move(self, location: tuple[int, int, int]) -> bool:
        """Move the observer and record its base reward and any sanction."""
        target = self.world.observe(location)
        if not target.passable:
            return False
        kind = target.kind
        base_reward = float(target.value) if kind in ITEM_KINDS else 0.0
        preferred = {ITEM_KINDS[d] for d in self._watched_desires}
        sanction = self.sanction if kind in preferred else 0.0
        moved = self.world.move(self.observer, location)
        if moved and kind in ITEM_KINDS:
            assert self.norm_log is not None
            self.norm_log.consumed[kind] = self.norm_log.consumed.get(kind, 0) + 1
            self.norm_log.events.append(
                ConsumptionEvent(self.turn, kind, base_reward, sanction)
            )
        return moved


def make_norm_env(
    k: int, watched_desires: list[int], seed: int, *, epsilon: float = 0.1,
    sanction: float = 20.0,
) -> NormEnv:
    """Construct a norm environment with the same world distribution as ToM v3."""
    rng = np.random.default_rng(seed)
    np.random.seed(seed)  # QuadrantEnv population uses numpy's global RNG.
    config = make_config(k)
    world = QuadrantWorld(config=config, default_entity=EmptyEntity())
    return NormEnv(
        world, config, watched_desires, epsilon, rng, sanction=sanction
    )


def sanction_probabilities(desire_probs: np.ndarray) -> np.ndarray:
    """Probability each kind is preferred by at least one present partner."""
    if len(desire_probs) == 0:
        return np.zeros(0, dtype=np.float64)
    return 1.0 - np.prod(1.0 - np.asarray(desire_probs, dtype=np.float64), axis=0)


def allowed_kinds(
    posterior: np.ndarray, sanction: float, item_value: float = 10.0
) -> set[str]:
    """Kinds whose expected sanction is strictly less than their base value."""
    return {
        ITEM_KINDS[i]
        for i, p_preferred in enumerate(posterior)
        if p_preferred * sanction < item_value
    }


def _path_to_nearest_allowed(world, start, allowed: set[str]) -> list[tuple[int, int, int]]:
    """Shortest safe path to an allowed item, without crossing other items."""
    frontier = deque([start])
    came_from = {start: None}
    goal = None
    while frontier:
        current = frontier.popleft()
        if current != start and world.observe(current).kind in allowed:
            goal = current
            break
        y, x, _ = current
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nxt = (y + dy, x + dx, 1)
            if not (0 <= nxt[0] < world.height and 0 <= nxt[1] < world.width):
                continue
            if nxt in came_from:
                continue
            kind = world.observe(nxt).kind
            if kind != "EmptyEntity" and kind not in allowed:
                continue
            came_from[nxt] = current
            frontier.append(nxt)
    if goal is None:
        return []
    path = [goal]
    while came_from[path[-1]] is not None:
        path.append(came_from[path[-1]])
    return list(reversed(path))[1:]


def forage_step(env: NormEnv, posterior: np.ndarray) -> bool:
    """Take one BFS decision-rule step.  Returns false when no safe item exists."""
    allowed = allowed_kinds(posterior, env.sanction)
    path = _path_to_nearest_allowed(env.world, env.observer.location, allowed)
    return bool(path) and env.observer_move(path[0])
