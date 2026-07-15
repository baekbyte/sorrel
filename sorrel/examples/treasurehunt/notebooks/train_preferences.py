"""
Trains two agents that share the same world but have **different reward
functions** (a gem-lover and a food-lover), each with its own IQN policy. Their
differing preferences make each agent's goal inferable from behavior — the
signal the later belief-desire-inference phases depend on.

World: TWO ROOMS + corridor (see `TwoRoomsWorld`). Each episode a hidden
latent picks which room is gem-rich; items are persistent (no respawn). The
agents spawn in the corridor and must head to their preferred room, so their
direction of travel through a door reveals both their preference AND the
latent. The post-training diagnostic below (correct-room rate) is the Phase-0
success criterion: if the trained agents don't reliably end up in their
preferred room, later belief phases have no signal to recover.
"""

import shutil
from datetime import datetime
from pathlib import Path

import numpy as np

from sorrel.buffers import TransformerBuffer
from sorrel.examples.treasurehunt.entities import EmptyEntity
from sorrel.examples.treasurehunt.env import TwoRoomsEnv
from sorrel.examples.treasurehunt.world import TwoRoomsWorld
from sorrel.utils.logging import TensorboardLogger

# ==========================================
# Configuration
# ==========================================

STATIC_RUNTIME = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
DATA_DIR = Path(__file__).parent / "../data"

# Preference multipliers over entity kinds (entity.kind == class name).
# A multiplier scales that entity's value when the agent collects it.
GEM_PREF = {"Gem": 1.0, "Food": 0.2, "Bone": 1.0}
FOOD_PREF = {"Gem": 0.2, "Food": 1.0, "Bone": 1.0}
# Bone keeps multiplier 1.0 so its negative value still penalizes both agents
# (avoidance behavior is shared; only the positive-reward target differs).

PREFERENCE_NAMES = ["gemlover", "foodlover"]

config = {
    "experiment": {
        "epochs": 5000,
        "max_turns": 100,
        "record_period": 50,
        "log_dir": DATA_DIR / "logs/preferences" / STATIC_RUNTIME,
    },
    "model": {
        "agent_vision_radius": 7,  # 15x15 obs (was 4 = 9x9). Inner 9x9 = real FOV
        "epsilon_decay": 0.0005,
        "num_agents": 2,
        "preferences": [GEM_PREF, FOOD_PREF],
        "save_weights": True,
    },
    "world": {
        "height": 20,
        "width": 20,
        "gem_value": 10,
        "food_value": 10,
        "bone_value": -10,
        # Persistent items: placed once per episode, no respawn churn. The
        # per-episode gem_side latent (which room is gem-rich) is the ONLY
        # thing that varies, so it cannot be memorized — it must be observed
        # (by entering a room) or inferred (by watching who walks where).
        "spawn_prob": 0.0,
        "n_gems": 8,
        "n_food": 8,
        "n_bones_per_room": 2,
    },
}

NUM_MEMORY_GAMES = 1024

# ==========================================
# Step 1: Co-train two preference policies
# ==========================================

print("=" * 50)
print("STEP 1: Training gem-lover + food-lover IQNs")
print("=" * 50)

world = TwoRoomsWorld(config=config, default_entity=EmptyEntity())
env = TwoRoomsEnv(world, config)

# Sanity: confirm preferences were applied per agent.
for name, agent in zip(PREFERENCE_NAMES, env.agents):
    print(f"  {name}: preferences={agent.preferences}")

env.run_experiment(
    output_dir=DATA_DIR,
    logger=TensorboardLogger.from_config(config),
)

# ==========================================
# Step 1b: Corridor curriculum fine-tune
# ==========================================
# Stage 1 ("anywhere" spawns) teaches item values but lets the policies get
# away with directional HABITS: enter any door, grab what's near. (Observed:
# greedy room choice at chance, heavily asymmetric left/right action stats.)
# From a corridor spawn, the only way to beat a coin flip is to READ the
# visible room contents and pick the correct door — so this stage retrains
# under the deployment spawn distribution with a partial epsilon reset.

print("\n" + "=" * 50)
print("STEP 1b: Corridor-spawn fine-tune")
print("=" * 50)

FINE_TUNE_EPOCHS = 5000
FINE_TUNE_EPSILON = 0.3

world.spawn_region = "corridor"
config["experiment"]["epochs"] = FINE_TUNE_EPOCHS
config["experiment"]["log_dir"] = DATA_DIR / "logs/preferences_ft" / STATIC_RUNTIME
env.config.experiment.epochs = FINE_TUNE_EPOCHS
for agent in env.agents:
    agent.model.epsilon = FINE_TUNE_EPSILON

env.run_experiment(
    output_dir=DATA_DIR,
    logger=TensorboardLogger.from_config(config),
)

# Save each agent's trained IQN under a preference-named checkpoint.
ckpt_dir = DATA_DIR / "checkpoints"
ckpt_dir.mkdir(parents=True, exist_ok=True)
for name, agent in zip(PREFERENCE_NAMES, env.agents):
    ckpt_path = ckpt_dir / f"treasurehunt_model_{name}.pkl"
    agent.model.save(file_path=str(ckpt_path))
    print(f"  saved {name} -> {ckpt_path}")

# ==========================================
# Step 2: Generate per-preference memories
# ==========================================

print("\n" + "=" * 50)
print("STEP 2: Generating memories")
print("=" * 50)

# Memories feed Phases 1-2, so record them under the DEPLOYMENT spawn
# distribution: everyone starts in the corridor and must walk to their room.
world.spawn_region = "corridor"
env.generate_memories(num_games=NUM_MEMORY_GAMES, animate=False, output_dir=DATA_DIR)

mem_dir = DATA_DIR / "memories"
for i, name in enumerate(PREFERENCE_NAMES):
    src = mem_dir / f"agent{i}.npz"
    dst = mem_dir / f"{name}.npz"
    shutil.copyfile(src, dst)
    print(f"  {name}: {dst}")

# ==========================================
# Step 3: Behavior diagnostics
# ==========================================

print("\n" + "=" * 50)
print("STEP 3: Per-preference behavior diagnostics")
print("=" * 50)

for name in PREFERENCE_NAMES:
    buf = TransformerBuffer.load(mem_dir / f"{name}.npz")
    actions = buf.actions[: buf.size].flatten()
    rewards = buf.rewards[: buf.size]
    print(f"\n{name}:")
    print("  action distribution:")
    for j, label in enumerate(["up", "down", "left", "right"]):
        pct = np.mean(actions == j) * 100
        print(f"    {label:>5s}: {pct:.1f}%")
    print(
        f"  reward: mean_step={np.mean(rewards):.4f} ± {np.std(rewards):.4f} "
        f"(total over buffer={np.sum(rewards):.1f})"
    )

# ==========================================
# Step 4: Correct-room rate (Phase-0 success criterion)
# ==========================================
# The whole ToM pipeline rests on the watched agents' behavior REVEALING their
# preference: the gem-lover must reliably head to whichever room is gem-rich
# this episode (and the food-lover to the other). If these rates are near
# chance (0.5), the belief encoder has nothing to infer — fix Phase 0 before
# training Phases 1-2.

print("\n" + "=" * 50)
print("STEP 4: Correct-room rate (greedy policy)")
print("=" * 50)

N_EVAL_EPISODES = 40
PREFERRED_ENTITY = {"gemlover": "Gem", "foodlover": "Food"}

# Deployment condition: greedy policy, corridor spawns (training uses
# "anywhere" spawns so item rewards are experienced without door discovery).
world.spawn_region = "corridor"
for agent in env.agents:
    agent.model.epsilon = 0.0


def room_of(agent) -> str:
    x = agent.location[1]
    if x < world.left_wall_x:
        return "left"
    if x > world.right_wall_x:
        return "right"
    return "corridor"


stats = {
    name: {"first_correct": 0, "first_total": 0, "final_correct": 0, "entered": 0}
    for name in PREFERENCE_NAMES
}
for episode in range(N_EVAL_EPISODES):
    env.reset()
    first_room = {name: None for name in PREFERENCE_NAMES}
    for _ in range(config["experiment"]["max_turns"]):
        env.take_turn()
        for name, agent in zip(PREFERENCE_NAMES, env.agents):
            r = room_of(agent)
            if first_room[name] is None and r != "corridor":
                first_room[name] = r
    for name, agent in zip(PREFERENCE_NAMES, env.agents):
        preferred_room = (
            world.gem_side
            if PREFERRED_ENTITY[name] == "Gem"
            else ("right" if world.gem_side == "left" else "left")
        )
        if first_room[name] is not None:
            stats[name]["first_total"] += 1
            stats[name]["first_correct"] += int(first_room[name] == preferred_room)
        stats[name]["entered"] += int(room_of(agent) != "corridor")
        stats[name]["final_correct"] += int(room_of(agent) == preferred_room)

for name in PREFERENCE_NAMES:
    s = stats[name]
    first_rate = s["first_correct"] / max(1, s["first_total"])
    print(
        f"  {name}: entered a room in {s['first_total']}/{N_EVAL_EPISODES} episodes; "
        f"first room correct: {first_rate:.2f}; "
        f"final position correct: {s['final_correct'] / N_EVAL_EPISODES:.2f}"
    )
print("  (chance = 0.50; the belief pipeline needs these well above chance)")

print("\nPhase 0 complete.")
