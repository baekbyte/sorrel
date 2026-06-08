"""Link C: observer self-action rollout.

The observer = the frozen self-model (= gem-lover's "brain"). Given a region-
masked observation, we ask: does watching another agent's behavior change what
action the observer takes?

Two-pass scheme:
  PASS 1 (belief formation): frozen base + injected goal embedding g ->
    reconstruct the full observation. Substitute the hidden cells with the
    argmax reconstruction. This is the observer's BELIEF about the hidden world.
  PASS 2 (observer action): frozen base on the belief-completed observation,
    with NO g injected. The action head now reflects the observer's OWN
    preference (gem-lover) acting on its inferred world.

The Theory-of-Mind action claim:
  - g_gem -> belief fills hidden region with GEMS -> observer (gem-lover)
    chooses an action toward that region.
  - g_food -> belief fills hidden region with FOOD -> observer doesn't want
    food -> action shouldn't shift toward that region.
  - no belief -> baseline action distribution given the masked input.

Usage:
    python -m sorrel.examples.treasurehunt.notebooks.link_c_observer_action
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
BELIEF_PATH = Path(
    os.environ.get(
        "BELIEF_CKPT", str(DATA_DIR / "checkpoints/belief_model_decoupled.pkl")
    )
)
print(f"Evaluating belief checkpoint: {BELIEF_PATH.name}")

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
ACTION_NAMES = ["up", "down", "left", "right"]
GEM_CH, FOOD_CH = 2, 4
N_PROTO = 200
N_QUERY = 256

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
    base.memory = buf
    gs = []
    with torch.no_grad():
        for _ in range(n // ARCH["batch_size"] + 1):
            s, a, _, _, _, _ = base.get_batch()
            gs.append(bm.belief_encoder(s, a))
    return torch.cat(gs, dim=0)[:n]


g_gem = encode_g(gem_buf, N_PROTO).mean(0, keepdim=True)
g_food = encode_g(food_buf, N_PROTO).mean(0, keepdim=True)


def belief_completed_obs(masked_states, actions, hidden_mask, belief):
    """Pass 1: reconstruct hidden cells via frozen base + belief; return a fully
    visible (belief-completed) observation. Visible cells = original obs;
    hidden cells = argmax reconstruction."""
    visible_mask = (~hidden_mask).float()
    with torch.no_grad():
        b = None if belief is None else belief.expand(masked_states.size(0), -1)
        preds, _ = base.forward(masked_states, actions, belief_embedding=b)
        # preds: (B, T, 2, H, W, C); softmax over neg/pos dim -> positive probability
        pos = F.softmax(preds, dim=2)[:, :, 1]  # (B, T, H, W, C)
        ch = pos.argmax(dim=-1)  # (B, T, H, W) -> winning channel per cell
        recon_onehot = (
            F.one_hot(ch, num_classes=ARCH["state_size"][0])
            .permute(0, 1, 4, 2, 3)
            .float()
        )
    return masked_states * visible_mask + recon_onehot * (1 - visible_mask)


def hidden_region_entity_counts(masked_states, actions, hidden_mask, belief):
    """How many hidden cells does the belief fill with gems vs food?"""
    completed = belief_completed_obs(masked_states, actions, hidden_mask, belief)
    h = hidden_mask[:, :, 0, :, :]
    gem_cells = (completed[:, :, GEM_CH] * h).sum().item()
    food_cells = (completed[:, :, FOOD_CH] * h).sum().item()
    return gem_cells, food_cells


def observer_action_dist(masked_states, actions, hidden_mask, belief):
    """Pass 2: run the frozen base on the belief-completed obs WITHOUT injecting
    belief, return mean action distribution at the last timestep."""
    completed_obs = belief_completed_obs(masked_states, actions, hidden_mask, belief)
    with torch.no_grad():
        _, action_logits = base.forward(
            completed_obs, actions, belief_embedding=None
        )
    return F.softmax(action_logits[:, -1], dim=-1).mean(dim=0)  # (4,)


# Fixed neutral query batch from the combined buffer (same input across conditions).
base.memory = combined
base.batch_size = N_QUERY
q_states, q_actions, _, _, _, _ = base.get_batch()

print("\n" + "=" * 78)
print("LINK C — observer self-action under inferred beliefs (frozen base = gem-lover)")
print("=" * 78)

for region in ["right", "left"]:
    region_visible_mask = base.region_mask(q_states, region)
    hidden_mask = ~region_visible_mask
    masked_q = q_states * region_visible_mask.float()
    print(f"\n--- HIDDEN region = {region.upper()} half ---")

    # Belief composition in the hidden region (sanity: does g actually shift it?)
    print("  belief-completed hidden region (cell counts):")
    print(
        f"    {'condition':<14}{'gem_cells':>11}{'food_cells':>11}{'gem-food':>11}"
    )
    for name, g in [("no belief", None), ("g_gem", g_gem), ("g_food", g_food)]:
        gem_n, food_n = hidden_region_entity_counts(masked_q, q_actions, hidden_mask, g)
        print(
            f"    {name:<14}{gem_n:>11.0f}{food_n:>11.0f}{gem_n - food_n:>+11.0f}"
        )

    # Observer's predicted next action under each belief condition.
    print("  observer action distribution (last-timestep):")
    print(f"    {'condition':<14}" + "".join(f"{n:>9s}" for n in ACTION_NAMES))
    rows = {}
    for name, g in [("no belief", None), ("g_gem", g_gem), ("g_food", g_food)]:
        probs = observer_action_dist(masked_q, q_actions, hidden_mask, g)
        rows[name] = probs
        print(f"    {name:<14}" + "".join(f"{p.item():>9.3f}" for p in probs))

    toward = "right" if region == "right" else "left"
    idx = ACTION_NAMES.index(toward)
    shift_gem = (rows["g_gem"][idx] - rows["no belief"][idx]).item()
    shift_food = (rows["g_food"][idx] - rows["no belief"][idx]).item()
    print(
        f"  shift in P({toward}) vs no-belief: g_gem={shift_gem:+.3f}  g_food={shift_food:+.3f}"
    )
    print(
        f"  Link C signal: g_gem - g_food shift toward {toward}: "
        f"{(rows['g_gem'][idx] - rows['g_food'][idx]).item():+.3f}"
    )
    print(
        "  positive => observer (gem-lover) heads toward region when belief"
        " says gems are there, but not when it says food."
    )

print("\nLink C complete.")
