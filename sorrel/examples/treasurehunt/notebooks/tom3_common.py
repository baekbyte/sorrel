"""Shared pieces of the ToM v3 pipeline (K rooms, K preferences, N agents).

Generalizes ToM v2 from the binary two-room latent to a K-way latent in the
quadrant world (K in {2, 3, 4}): a per-episode random permutation assigns K
item kinds to K rooms, and the observer must infer (a) each watched agent's
K-way desire and (b) the K-way latent "which room holds the gems." Beyond
sharper stats, K >= 3 unlocks inference by EXCLUSION: watching only
non-gem-lovers pins the gem room even though nobody ever moved toward it —
behavior no mimicry account can produce.

The entity list (and thus the encoder architecture) is fixed at 8 kinds for
ALL K; with K < 4 the unused item channels are simply always zero. One
architecture covers the whole ablation sweep.
"""

from pathlib import Path

import numpy as np

from sorrel.examples.treasurehunt.entities import EmptyEntity
from sorrel.examples.treasurehunt.env import QuadrantEnv
from sorrel.examples.treasurehunt.notebooks.tom2_common import (
    ObserverBody,
    agent_window,
    displacement_token,
)
from sorrel.examples.treasurehunt.scripted_agents import ScriptedForager
from sorrel.examples.treasurehunt.world import QuadrantWorld
from sorrel.observation.observation_spec import OneHotObservationSpec

DATA_DIR = Path(__file__).parent / "../data"

ENTITY_LIST = [
    "EmptyEntity",
    "Wall",
    "Gem",
    "Bone",
    "Food",
    "Coin",
    "Berry",
    "TreasurehuntAgent",
]
ITEM_KINDS = ["Gem", "Food", "Coin", "Berry"]  # desire label = index here
GEM_DESIRE = 0

OBS_RADIUS = 4  # observer's real FOV: 9x9
T_WATCH = 20  # watch-phase length (steps recorded per episode)
N_CHANNELS = len(ENTITY_LIST)
N_INPUT_CHANNELS = N_CHANNELS + 1  # + tracked-agent marker channel
H = W = 2 * OBS_RADIUS + 1  # 9

# Displacement tokens (observed watched-agent movement between turns).
DISP_STAY = 4
N_DISP_TOKENS = 5


def dataset_path(k: int) -> Path:
    return DATA_DIR / f"memories/tom3_dataset_k{k}.npz"


def observer_ckpt(k: int) -> Path:
    return DATA_DIR / f"checkpoints/tom3_observer_k{k}.pt"


def pref_for(desire: int) -> dict[str, float]:
    """Preference dict for a desire index: 1.0 on the preferred kind, 0.2 on
    every other item kind, 1.0 on Bone (so its -10 value stays a penalty)."""
    prefs = {kind: 0.2 for kind in ITEM_KINDS}
    prefs[ITEM_KINDS[desire]] = 1.0
    prefs["Bone"] = 1.0
    return prefs


def make_config(k: int) -> dict:
    return {
        "experiment": {"epochs": 1, "max_turns": T_WATCH + 40, "record_period": 1},
        "model": {
            "agent_vision_radius": OBS_RADIUS,
            "num_agents": 0,
            "preferences": [],
        },
        "world": {
            "height": 20,
            "width": 20,
            "gem_value": 10,
            "food_value": 10,
            "bone_value": -10,
            "spawn_prob": 0.0,
            "n_rooms": k,
            "n_main_items_per_room": 6,
            "n_bones_per_room": 2,
            "n_preview_per_kind": 2,
        },
    }


class Tom3Env(QuadrantEnv):
    """QuadrantEnv with scripted demonstrators and a frozen observer body.

    ``watched_desires`` is a list of desire indices (into ITEM_KINDS), one
    scripted forager per entry. The observer is NOT in self.agents (it takes
    no turns); it is placed as an entity so watched agents route around it
    and it appears in observations.
    """

    def __init__(self, world, config, watched_desires: list[int], epsilon, rng):
        self._watched_desires = list(watched_desires)
        self._epsilon = epsilon
        self._rng = rng
        self.observer = ObserverBody()
        super().__init__(world, config)

    def setup_agents(self):
        self.agents = [
            ScriptedForager(pref_for(d), epsilon=self._epsilon, rng=self._rng)
            for d in self._watched_desires
        ]

    def populate_environment(self):
        super().populate_environment()  # walls, rooms, items, previews + chamber spawns
        world = self.world
        # Re-place agents deliberately: observer at a center cell, watched
        # agents in the chamber (every chamber cell is inside the FOV of
        # every observer spawn cell).
        for agent in self.agents:
            world.remove(agent.location)
        spawn_cells = world.observer_spawn_cells()
        obs_loc = spawn_cells[self._rng.integers(len(spawn_cells))]
        world.add(obs_loc, self.observer)
        free = [
            c for c in world.preview_cells() if world.observe(c).kind == "EmptyEntity"
        ]
        chosen = self._rng.choice(len(free), size=len(self.agents), replace=False)
        for i, agent in zip(chosen, self.agents):
            world.add(free[i], agent)


def make_env(
    k: int,
    watched_desires: list[int],
    seed: int,
    epsilon: float = 0.1,
    forced_gem_room: int | None = None,
) -> Tom3Env:
    rng = np.random.default_rng(seed)
    np.random.seed(seed)  # populate_environment uses global np.random
    config = make_config(k)
    world = QuadrantWorld(config=config, default_entity=EmptyEntity())
    world.forced_gem_room = forced_gem_room
    return Tom3Env(world, config, watched_desires, epsilon, rng)


def make_obs_spec() -> OneHotObservationSpec:
    return OneHotObservationSpec(ENTITY_LIST, full_view=False, vision_radius=OBS_RADIUS)


def run_watch_phase(env: Tom3Env, obs_spec: OneHotObservationSpec, on_turn=None):
    """Run T_WATCH turns with the observer frozen; record per watched agent.

    Returns:
        frames: (T, C, H, W) float32 — observer POV before each turn.
        disp:   dict agent_index -> (T,) int64 displacement tokens.
        marks:  dict agent_index -> (T, H, W) float32 — the tracked agent's
                position within the observer's FOV (all-zero when outside).
        commit_room: dict agent_index -> room index in used_rooms (or None).
        commit_step: dict agent_index -> step of first room entry (or None).
    """
    world = env.world
    rooms = world.used_rooms()
    obs_loc = env.observer.location
    n = len(env.agents)
    frames = np.zeros((T_WATCH, N_CHANNELS, H, W), dtype=np.float32)
    disp = {i: np.full(T_WATCH, DISP_STAY, dtype=np.int64) for i in range(n)}
    marks = {i: np.zeros((T_WATCH, H, W), dtype=np.float32) for i in range(n)}
    commit_room: dict[int, int | None] = {i: None for i in range(n)}
    commit_step: dict[int, int | None] = {i: None for i in range(n)}
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
            region = world.room_of(agent.location)
            if commit_room[i] is None and region != "chamber":
                commit_room[i] = rooms.index(region)
                commit_step[i] = t
        if on_turn is not None:
            on_turn(world)
    return frames, disp, marks, commit_room, commit_step
