"""Phase 2 hardening: multi-seed + leakage control for the belief module.

For each seed, trains a fresh BeliefEncoder on the SAME frozen self-model in two
modes and reports the full-observation action lift (baseline - belief) on the
base's own agent (gem-lover) vs the unseen agent (food-lover):

  coupled    : belief encoder reads the same trajectory it predicts (original).
  decoupled  : belief encoder reads an INDEPENDENT same-agent window (leakage
               control) — g must summarize the goal, not peek at the answer.

A large, stable food-lover lift under `decoupled` across seeds is the rigorous
form of the Theory-of-Mind claim.

Usage:
    python -m sorrel.examples.treasurehunt.notebooks.harden_belief
"""

from pathlib import Path

import numpy as np
import torch

from sorrel.buffers import TransformerBuffer
from sorrel.models.pytorch.transformer import BeliefModel, ViTOneHot

DATA_DIR = Path(__file__).parent / "../data"
SELF_MODEL_PATH = DATA_DIR / "checkpoints/self_model.pkl"

SEEDS = [0, 1, 2]
EPOCHS = 8000
EVAL_STEPS = 300
TRAIN_MASK = "random"
EVAL_MASK = "full"  # isolates goal inference (no masking confound)

ARCH = dict(
    state_size=(6, 9, 9),
    action_space=4,
    layer_size=192,
    patch_size=3,
    num_frames=5,
    num_heads=3,
    batch_size=64,
    num_layers=2,
)

gem_buf = TransformerBuffer.load(DATA_DIR / "memories/gemlover.npz")
food_buf = TransformerBuffer.load(DATA_DIR / "memories/foodlover.npz")
combined = TransformerBuffer.combine([gem_buf, food_buf])

# One frozen base, reused across all runs (its weights never change).
base = ViTOneHot(
    memory=combined, LR=1e-3, device="cpu", seed=0, reward_loss_weight=1.0, **ARCH
)
base.load(str(SELF_MODEL_PATH))


def eval_lift(bm, buf, decouple):
    base.memory = buf
    belief_losses, base_losses = [], []
    for _ in range(EVAL_STEPS):
        _, ab = bm.evaluate_belief(
            mask_type=EVAL_MASK, use_belief=True, decouple=decouple
        )
        _, an = bm.evaluate_belief(mask_type=EVAL_MASK, use_belief=False)
        belief_losses.append(ab)
        base_losses.append(an)
    b, n = float(np.mean(belief_losses)), float(np.mean(base_losses))
    return n, b, n - b  # baseline, belief, lift


def run(seed, mode):
    decouple = mode == "decoupled"
    torch.manual_seed(seed)
    np.random.seed(seed)
    bm = BeliefModel(
        self_model=base,
        layer_size=ARCH["layer_size"],
        patch_size=ARCH["patch_size"],
        num_heads=ARCH["num_heads"],
        num_layers=ARCH["num_layers"],
        device="cpu",
        LR=1e-3,
    )
    for epoch in range(EPOCHS):
        if decouple:
            # single-agent per step so context & query are the same agent
            base.memory = gem_buf if epoch % 2 == 0 else food_buf
        else:
            base.memory = combined
        bm.train_belief(mask_type=TRAIN_MASK, decouple=decouple)
    gem = eval_lift(bm, gem_buf, decouple)
    food = eval_lift(bm, food_buf, decouple)
    return gem, food


print("=" * 78)
print(f"Phase 2 hardening — {EPOCHS} epochs/run, eval mask={EVAL_MASK!r}")
print("action loss: baseline (no belief) / belief / lift")
print("=" * 78)
rows = []
for mode in ["coupled", "decoupled"]:
    for seed in SEEDS:
        gem, food = run(seed, mode)
        rows.append((mode, seed, gem, food))
        print(
            f"[{mode:>9s} seed={seed}] "
            f"gem-lover {gem[0]:.3f}/{gem[1]:.3f}/{gem[2]:+.3f}  |  "
            f"food-lover {food[0]:.3f}/{food[1]:.3f}/{food[2]:+.3f}"
        )

print("\n--- summary (food-lover lift = the ToM signal) ---")
for mode in ["coupled", "decoupled"]:
    lifts = [f[2] for m, s, g, f in rows if m == mode]
    print(f"  {mode:>9s}: food-lover lift {np.mean(lifts):+.3f} ± {np.std(lifts):.3f}")
print("\nHardening complete.")
