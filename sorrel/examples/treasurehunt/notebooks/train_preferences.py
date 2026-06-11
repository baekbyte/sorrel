"""
Trains two agents that share the same world but have **different reward
functions** (a gem-lover and a food-lover), each with its own IQN policy. Their
differing preferences make each agent's goal inferable from behavior — the
signal the later belief-desire-inference phases depend on.
"""

import shutil
from datetime import datetime
from pathlib import Path

import numpy as np

from sorrel.buffers import TransformerBuffer
from sorrel.examples.treasurehunt.entities import EmptyEntity
from sorrel.examples.treasurehunt.env import TreasurehuntEnv
from sorrel.examples.treasurehunt.world import TreasurehuntWorld
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
        "spawn_prob": 0.01,
        # Non-segregated: gems and food spawn uniformly everywhere. The base
        # therefore has NO spatial prior about where each entity lives, so the
        # observer can only know "gems are over there" by watching the gem-lover.
        # This stresses the belief module -- behavior should diverge with vs
        # without belief.
        "segregate": False,
        "bone_fraction": 0.2,
    },
}

NUM_MEMORY_GAMES = 1024

# ==========================================
# Step 1: Co-train two preference policies
# ==========================================

print("=" * 50)
print("STEP 1: Training gem-lover + food-lover IQNs")
print("=" * 50)

world = TreasurehuntWorld(config=config, default_entity=EmptyEntity())
env = TreasurehuntEnv(world, config)

# Sanity: confirm preferences were applied per agent.
for name, agent in zip(PREFERENCE_NAMES, env.agents):
    print(f"  {name}: preferences={agent.preferences}")

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

print("\nPhase 0 complete.")
