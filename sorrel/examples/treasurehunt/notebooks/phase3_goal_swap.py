"""Phase 3: observer generalization diagnostic — goal decoding + causal goal-swap.

Two tests on the frozen self-model + trained belief module:

1. GOAL DECODABILITY. Encode g from gem-lover vs food-lover windows and train a
   linear probe to recover the agent type. High held-out accuracy => g encodes
   the agent's desire (desire attribution).

2. CAUSAL GOAL-SWAP (headline). Take a FIXED region-masked observation context.
   Condition the frozen base on a gem-goal prototype g_gem vs a food-goal
   prototype g_food. Because the input is identical and the base is frozen, any
   change in what entity is reconstructed in the HIDDEN region is caused purely
   by the injected goal. If g_gem -> "gem here" and g_food -> "food here", the
   inferred desire causally drives the inferred belief about hidden world state
   — the literal Theory-of-Mind scenario.

Usage:
    python -m sorrel.examples.treasurehunt.notebooks.phase3_goal_swap
"""

import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from sorrel.buffers import TransformerBuffer
from sorrel.models.pytorch.transformer import BeliefModel, ViTOneHot

DATA_DIR = Path(__file__).parent / "../data"
SELF_MODEL_PATH = DATA_DIR / "checkpoints/self_model.pkl"
# Belief checkpoint to evaluate. Override via BELIEF_CKPT env var to test the
# decoupled (leakage-controlled) module against the original coupled one.
BELIEF_PATH = Path(
    os.environ.get("BELIEF_CKPT", str(DATA_DIR / "checkpoints/belief_model.pkl"))
)
print(f"Evaluating belief checkpoint: {BELIEF_PATH.name}")

HIDDEN_REGION = "right"  # hide the right half of the 9x9 FOV
GEM_CH, FOOD_CH = 2, 4
N_PROTO = 200  # windows averaged into each goal prototype
N_QUERY = 64  # fixed query batch size for the swap test
N_PROBE = 600  # windows per class for the decodability probe

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

torch.manual_seed(0)
np.random.seed(0)

gem_buf = TransformerBuffer.load(DATA_DIR / "memories/gemlover.npz")
food_buf = TransformerBuffer.load(DATA_DIR / "memories/foodlover.npz")
combined = TransformerBuffer.combine([gem_buf, food_buf])

base = ViTOneHot(
    memory=combined, LR=1e-3, device="cpu", seed=0, reward_loss_weight=1.0, **ARCH
)
base.load(str(SELF_MODEL_PATH))
bm = BeliefModel(
    self_model=base,
    layer_size=ARCH["layer_size"],
    patch_size=ARCH["patch_size"],
    num_heads=ARCH["num_heads"],
    num_layers=ARCH["num_layers"],
    device="cpu",
    LR=1e-3,
)
bm.load(str(BELIEF_PATH))
bm.belief_encoder.eval()


def encode_g(buf, n):
    """Encode goal embeddings g from `n` windows of `buf` (full obs)."""
    base.memory = buf
    gs = []
    with torch.no_grad():
        for _ in range(n // ARCH["batch_size"] + 1):
            s, a, _, _, _, _ = base.get_batch()
            gs.append(bm.belief_encoder(s, a))
    return torch.cat(gs, dim=0)[:n]


# ============================================================
# Test 1: goal decodability (linear probe)
# ============================================================
print("=" * 70)
print("TEST 1: goal decodability from g (linear probe)")
print("=" * 70)
g_gem_all = encode_g(gem_buf, N_PROBE)
g_food_all = encode_g(food_buf, N_PROBE)
X = torch.cat([g_gem_all, g_food_all], dim=0).detach()
y = torch.cat([torch.zeros(N_PROBE), torch.ones(N_PROBE)]).long()
perm = torch.randperm(X.size(0))
X, y = X[perm], y[perm]
split = int(0.8 * X.size(0))
Xtr, ytr, Xte, yte = X[:split], y[:split], X[split:], y[split:]

probe = torch.nn.Linear(ARCH["layer_size"], 2)
opt = torch.optim.Adam(probe.parameters(), lr=1e-2)
for _ in range(500):
    opt.zero_grad()
    loss = F.cross_entropy(probe(Xtr), ytr)
    loss.backward()
    opt.step()
acc = (probe(Xte).argmax(1) == yte).float().mean().item()
print(f"  held-out agent-type accuracy from g: {acc * 100:.1f}%  (chance = 50%)")


# ============================================================
# Test 2: causal goal-swap on a fixed region-masked query
# ============================================================
print("\n" + "=" * 70)
print(f"TEST 2: causal goal-swap (hidden region = {HIDDEN_REGION!r} half of FOV)")
print("=" * 70)

# Goal prototypes (mean g per agent type).
g_gem = encode_g(gem_buf, N_PROTO).mean(0, keepdim=True)  # (1, D)
g_food = encode_g(food_buf, N_PROTO).mean(0, keepdim=True)

# Fixed neutral query batch (region-masked), shared across all conditions.
base.memory = combined
base.batch_size = N_QUERY
q_states, q_actions, _, _, _, _ = base.get_batch()
region = base.region_mask(q_states, HIDDEN_REGION)  # True = visible
hidden = ~region  # (B, T, C, H, W) True where hidden
masked_query = q_states * region.float()
# Spatial hidden mask for one channel (B, T, H, W).
hidden_cells = hidden[:, :, 0, :, :]


def hidden_region_probs(belief):
    """Mean predicted gem / food probability in the hidden region."""
    with torch.no_grad():
        b = None if belief is None else belief.expand(N_QUERY, -1)
        preds, _ = base.forward(masked_query, q_actions, belief_embedding=b)
        # preds: (B, T, 2, H, W, C) -> positive-class prob via softmax over dim 2
        pos = F.softmax(preds, dim=2)[:, :, 1]  # (B, T, H, W, C)
        gem = pos[..., GEM_CH][hidden_cells].mean().item()
        food = pos[..., FOOD_CH][hidden_cells].mean().item()
    return gem, food


print("  predicted entity probability in HIDDEN region:")
print(f"  {'condition':<18}{'gem_prob':>10}{'food_prob':>11}{'gem-food':>11}")
results = {}
for name, g in [("no belief", None), ("g_gem", g_gem), ("g_food", g_food)]:
    gem_p, food_p = hidden_region_probs(g)
    results[name] = (gem_p, food_p)
    print(f"  {name:<18}{gem_p:>10.4f}{food_p:>11.4f}{gem_p - food_p:>+11.4f}")

# Causal effect: does swapping the goal shift the gem-vs-food balance?
swing = (results["g_gem"][0] - results["g_gem"][1]) - (
    results["g_food"][0] - results["g_food"][1]
)
print(
    f"\n  causal goal-swap effect (gem-food balance shift, g_gem - g_food): {swing:+.4f}"
)
print("  positive => inferred desire causally drives belief about hidden region.")
print("\nPhase 3 complete.")
