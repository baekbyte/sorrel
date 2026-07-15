"""Shared pieces of the ToM v2 pipeline (scripted demonstrators, learned observer).

The v2 experiment in one paragraph: a two-room world with a per-episode hidden
latent (which room is gem-rich). A scripted watched agent reveals its DESIRE
by picking its preferred kind out of the corridor preview items, then reveals
the LATENT (to anyone who knows its desire) by walking through a door to its
preferred room. The observer is frozen at corridor center during the watch
phase, sees only its own 9x9 FOV (third-person: watched agents' bodies and
displacements, never their POVs), and must output (a) the watched agent's
desire and (b) a posterior over the latent. Action = a decision rule on the
posterior: walk through the door with the higher expected value under the
observer's own (gem) preference.
"""

from pathlib import Path

import numpy as np

from sorrel.entities import Entity
from sorrel.examples.treasurehunt.entities import EmptyEntity
from sorrel.examples.treasurehunt.env import TwoRoomsEnv
from sorrel.examples.treasurehunt.scripted_agents import ScriptedForager
from sorrel.examples.treasurehunt.world import TwoRoomsWorld
from sorrel.observation.observation_spec import OneHotObservationSpec

DATA_DIR = Path(__file__).parent / "../data"
DATASET_PATH = DATA_DIR / "memories/tom2_dataset.npz"
OBSERVER_CKPT = DATA_DIR / "checkpoints/tom2_observer.pt"

ENTITY_LIST = ["EmptyEntity", "Wall", "Gem", "Bone", "Food", "TreasurehuntAgent"]
GEM_PREF = {"Gem": 1.0, "Food": 0.2, "Bone": 1.0}
FOOD_PREF = {"Gem": 0.2, "Food": 1.0, "Bone": 1.0}

OBS_RADIUS = 4          # observer's real FOV: 9x9
T_WATCH = 20            # watch-phase length (steps recorded per episode)
N_CHANNELS = 6
# Encoder input adds a 7th "tracked agent" marker channel: the position of the
# watched agent this window is about. Without it, multi-agent episodes are an
# unsolvable binding problem — the shared frames show two bodies eating items,
# and nothing says which one the displacement tokens describe. Marking which
# body is being attended to is indexing, not privileged information.
N_INPUT_CHANNELS = N_CHANNELS + 1
H = W = 2 * OBS_RADIUS + 1  # 9

# Displacement tokens (observed watched-agent movement between turns).
DISP_STAY = 4           # up/down/left/right = 0..3 (matching ACTION_NAMES), stay = 4
N_DISP_TOKENS = 5

WATCHED_CONFIGS = ["gem_only", "food_only", "both"]

CONFIG = {
    "experiment": {"epochs": 1, "max_turns": T_WATCH + 40, "record_period": 1},
    "model": {"agent_vision_radius": OBS_RADIUS, "num_agents": 0, "preferences": []},
    "world": {
        "height": 20,
        "width": 20,
        "gem_value": 10,
        "food_value": 10,
        "bone_value": -10,
        "spawn_prob": 0.0,
        "n_gems": 8,
        "n_food": 8,
        "n_bones_per_room": 2,
        "n_preview_gems": 2,
        "n_preview_food": 2,
    },
}


class ObserverBody(Entity[TwoRoomsWorld]):
    """Inert observer body: occupies a cell and renders as an agent in
    one-hot views, but has no policy — the eval script moves it directly."""

    def __init__(self):
        super().__init__()
        self.kind = "TreasurehuntAgent"
        self.passable = False
        self.sprite = Path(__file__).parent.parent / "assets/hero.png"


class Tom2Env(TwoRoomsEnv):
    """TwoRoomsEnv with scripted demonstrators and a frozen observer body.

    The observer is NOT in self.agents (it takes no turns); it is placed as an
    entity so watched agents route around it and it appears in observations.
    """

    def __init__(self, world, config, watched_config: str, epsilon: float, rng):
        self._watched_config = watched_config
        self._epsilon = epsilon
        self._rng = rng
        self.observer = ObserverBody()
        super().__init__(world, config)

    def setup_agents(self):
        prefs = {
            "gem_only": [GEM_PREF],
            "food_only": [FOOD_PREF],
            "both": [GEM_PREF, FOOD_PREF],
            "none": [],
        }[self._watched_config]
        self.agents = [
            ScriptedForager(p, epsilon=self._epsilon, rng=self._rng) for p in prefs
        ]

    def populate_environment(self):
        super().populate_environment()  # walls, rooms, items, previews + corridor agent spawns
        world = self.world
        # Re-place agents deliberately: observer at a center cell, watched
        # agents in the preview block (inside the observer's FOV).
        for agent in self.agents:
            world.remove(agent.location)
        spawn_cells = world.observer_spawn_cells()
        obs_loc = spawn_cells[self._rng.integers(len(spawn_cells))]
        world.add(obs_loc, self.observer)
        free = [
            c
            for c in world.preview_cells()
            if world.observe(c).kind == "EmptyEntity"
        ]
        chosen = self._rng.choice(len(free), size=len(self.agents), replace=False)
        for i, agent in zip(chosen, self.agents):
            world.add(free[i], agent)


def make_env(
    watched_config: str,
    seed: int,
    epsilon: float = 0.1,
    forced_gem_side: str | None = None,
) -> Tom2Env:
    rng = np.random.default_rng(seed)
    np.random.seed(seed)  # populate_environment uses global np.random
    world = TwoRoomsWorld(config=CONFIG, default_entity=EmptyEntity())
    world.forced_gem_side = forced_gem_side
    return Tom2Env(world, CONFIG, watched_config, epsilon, rng)


def make_obs_spec() -> OneHotObservationSpec:
    return OneHotObservationSpec(ENTITY_LIST, full_view=False, vision_radius=OBS_RADIUS)


def displacement_token(prev_loc, cur_loc) -> int:
    dy, dx = cur_loc[0] - prev_loc[0], cur_loc[1] - prev_loc[1]
    return {(-1, 0): 0, (1, 0): 1, (0, -1): 2, (0, 1): 3}.get((dy, dx), DISP_STAY)


def room_side(world: TwoRoomsWorld, loc) -> str:
    x = loc[1]
    if x < world.left_wall_x:
        return "left"
    if x > world.right_wall_x:
        return "right"
    return "corridor"


def run_watch_phase(env: Tom2Env, obs_spec: OneHotObservationSpec, on_turn=None):
    """Run T_WATCH turns with the observer frozen; record per watched agent.

    Returns:
        frames: (T, C, H, W) float32 — observer POV before each turn.
        disp:   dict agent_index -> (T,) int64 displacement tokens.
        marks:  dict agent_index -> (T, H, W) float32 — the tracked agent's
                position within the observer's FOV (all-zero when outside).
        door_side: dict agent_index -> "left"/"right"/None (first room entered).
        door_step: dict agent_index -> step of first room entry (or None).
    """
    world = env.world
    obs_loc = env.observer.location
    n = len(env.agents)
    frames = np.zeros((T_WATCH, N_CHANNELS, H, W), dtype=np.float32)
    disp = {i: np.full(T_WATCH, DISP_STAY, dtype=np.int64) for i in range(n)}
    marks = {i: np.zeros((T_WATCH, H, W), dtype=np.float32) for i in range(n)}
    door_side: dict[int, str | None] = {i: None for i in range(n)}
    door_step: dict[int, int | None] = {i: None for i in range(n)}
    for t in range(T_WATCH):
        frames[t] = obs_spec.observe(world, obs_loc)
        for i, agent in enumerate(env.agents):
            ry = agent.location[0] - obs_loc[0] + OBS_RADIUS
            rx = agent.location[1] - obs_loc[1] + OBS_RADIUS
            if 0 <= ry < H and 0 <= rx < W:
                marks[i][t, ry, rx] = 1.0
        prev = {i: a.location for i, a in enumerate(env.agents)}
        env.take_turn()
        for i, agent in enumerate(env.agents):
            # Strict third person: a displacement is only observed while the
            # agent is inside the observer's FOV (marker nonzero this frame).
            if marks[i][t].any():
                disp[i][t] = displacement_token(prev[i], agent.location)
            side = room_side(world, agent.location)
            if door_side[i] is None and side != "corridor":
                door_side[i] = side
                door_step[i] = t
        if on_turn is not None:
            on_turn(world)
    return frames, disp, marks, door_side, door_step


def agent_window(frames: np.ndarray, mark: np.ndarray) -> np.ndarray:
    """Compose the encoder input for one watched agent: shared observer frames
    (T, 6, H, W) + that agent's marker channel (T, H, W) -> (T, 7, H, W)."""
    return np.concatenate([frames, mark[:, None]], axis=1)
