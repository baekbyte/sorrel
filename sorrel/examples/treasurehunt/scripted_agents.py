"""Scripted goal-directed demonstrators for the ToM v2 pipeline.

The watched agents in a Theory-of-Mind experiment are *stimuli*: their job is
to produce behavior that reveals a preference. Learning them with RL made the
stimulus the most fragile link in the pipeline (habits instead of perception,
hour-long retrains per world tweak). Following the classic ToM setups (Baker &
Tenenbaum's inverse planning; Rabinowitz's ToMnet), demonstrators here are
scripted: BFS-shortest-path foragers with full world knowledge and a
preference parameter. Desire-revealing by construction, ground truth for free.
"""

from collections import deque

import numpy as np

from sorrel.action.action_spec import ActionSpec
from sorrel.agents import MovingAgent
from sorrel.examples.treasurehunt.world import TwoRoomsWorld
from sorrel.observation.observation_spec import OneHotObservationSpec

ENTITY_LIST = ["EmptyEntity", "Wall", "Gem", "Bone", "Food", "TreasurehuntAgent"]
# Positive item kinds a forager can prefer, in tie-break priority order.
ITEM_KINDS = ("Gem", "Food", "Coin", "Berry")
ACTION_NAMES = ["up", "down", "left", "right"]
# delta (dy, dx) -> action index, matching ACTION_NAMES order.
_DELTA_TO_ACTION = {(-1, 0): 0, (1, 0): 1, (0, -1): 2, (0, 1): 3}


class _NullMemory:
    """Memory stub: scripted agents record nothing."""

    extra_data: dict = {}

    def add(self, *args, **kwargs):
        pass


class _NullModel:
    """Model stub satisfying the Agent/Environment hooks."""

    def __init__(self):
        self.memory = _NullMemory()
        self.epsilon = 0.0

    def reset(self):
        pass

    def start_epoch_action(self, **kwargs):
        pass

    def end_epoch_action(self, **kwargs):
        pass

    def train_step(self):
        return 0.0

    def save(self, **kwargs):
        pass


class ScriptedForager(MovingAgent[TwoRoomsWorld]):
    """Noisy-greedy goal-directed forager with full world knowledge.

    Each step: BFS shortest path (walls, bones, and other agents are
    obstacles; non-preferred items are also avoided so they are never
    accidentally eaten) to the NEAREST item of the preferred kind, then take
    the first step of that path. With probability ``epsilon`` take a uniform
    random step instead. If no preferred item is reachable, take a random
    step.

    The resulting behavior is desire-revealing by construction: the agent
    picks its preferred type out of the corridor preview items and then
    routes through the door to whichever room holds its preferred type.
    """

    def __init__(
        self,
        preferences: dict[str, float],
        epsilon: float = 0.1,
        rng: np.random.Generator | None = None,
    ):
        obs_spec = OneHotObservationSpec(ENTITY_LIST, full_view=False, vision_radius=1)
        act_spec = ActionSpec(ACTION_NAMES)
        super().__init__(obs_spec, act_spec, _NullModel())
        self.kind = "TreasurehuntAgent"  # render as an agent in one-hot views
        self.preferences: dict[str, float] = dict(preferences)
        # The preferred (targeted) kind: argmax preference over the item
        # kinds present in the dict; ITEM_KINDS order breaks ties.
        self.target_kind = max(
            (k for k in ITEM_KINDS if k in self.preferences),
            key=lambda k: self.preferences[k],
        )
        self.epsilon = float(epsilon)
        self.rng = rng if rng is not None else np.random.default_rng()
        self._world: TwoRoomsWorld | None = None

    # --- Agent interface -----------------------------------------------------

    def reset(self) -> None:
        pass

    def pov(self, world: TwoRoomsWorld) -> np.ndarray:
        # Scripted policy plans on the world directly; stash the reference for
        # get_action (which only receives the state array).
        self._world = world
        return np.zeros((1, 1), dtype=np.float32)

    def get_action(self, state: np.ndarray) -> int:
        assert self._world is not None
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(4))
        action = self._bfs_action(self._world)
        return action if action is not None else int(self.rng.integers(4))

    def act(self, world: TwoRoomsWorld, action: int) -> float:
        new_location = self.movement(action)
        target = world.observe(new_location)
        reward = self.preferences.get(target.kind, 1.0) * target.value
        world.move(self, new_location)
        return reward

    def is_done(self, world: TwoRoomsWorld) -> bool:
        return world.is_done

    # --- planning -------------------------------------------------------------

    def _traversable(self, world: TwoRoomsWorld, loc: tuple[int, int, int]) -> bool:
        kind = world.observe(loc).kind
        # Empty cells and the preferred item kind only: walls/agents block,
        # bones are avoided, and the non-preferred item kind is routed AROUND
        # (walking over it would consume it and muddy the desire signal).
        return kind == "EmptyEntity" or kind == self.target_kind

    def _bfs_action(self, world: TwoRoomsWorld) -> int | None:
        """First step of the shortest path to the nearest preferred item."""
        start = (self.location[0], self.location[1])
        frontier = deque([start])
        came_from: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
        goal = None
        while frontier:
            y, x = frontier.popleft()
            if (y, x) != start and world.observe((y, x, 1)).kind == self.target_kind:
                goal = (y, x)
                break
            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                ny, nx = y + dy, x + dx
                if not (0 <= ny < world.height and 0 <= nx < world.width):
                    continue
                if (ny, nx) in came_from:
                    continue
                if not self._traversable(world, (ny, nx, 1)):
                    continue
                came_from[(ny, nx)] = (y, x)
                frontier.append((ny, nx))
        if goal is None:
            return None
        # Walk back to the first step out of `start`.
        node = goal
        while came_from[node] != start:
            node = came_from[node]
            assert node is not None
        dy, dx = node[0] - start[0], node[1] - start[1]
        return _DELTA_TO_ACTION[(dy, dx)]
