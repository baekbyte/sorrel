"""Phase 2: train the feudal belief module on top of the frozen self-model.

The frozen self/world-model (Phase 1) was trained on the OBSERVER's own
experience (gem-lover). Here a trainable BeliefEncoder reads another agent's
(masked) trajectory, infers a goal embedding ``g``, and injects it into the
frozen base to modulate its predictions toward that agent. Only the belief
encoder trains.

The Theory-of-Mind readout: belief conditioning should help most when
predicting the OTHER agent (food-lover) — whom the frozen base never saw — and
little when predicting the base's own agent (gem-lover). The no-belief baseline
(frozen base alone) is the projection/prior control.

Usage:
    python -m sorrel.examples.treasurehunt.notebooks.train_belief
"""

from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from sorrel.buffers import TransformerBuffer
from sorrel.models.pytorch.transformer import BeliefModel, ViTOneHot
from sorrel.utils.logging import TensorboardLogger

# ==========================================
# Configuration
# ==========================================

STATIC_RUNTIME = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
DATA_DIR = Path(__file__).parent / "../data"

SELF_MODEL_PATH = DATA_DIR / "checkpoints/self_model.pkl"
OBSERVER_BUFFER = DATA_DIR / "memories/gemlover.npz"  # base was trained on this
OTHER_BUFFER = DATA_DIR / "memories/foodlover.npz"  # the agent to infer

TRAINING_EPOCHS = 20000
EVAL_STEPS = 1000
TRAIN_MASK_TYPE = "random"  # behavior is the cue for the masked-out content

# Architecture — must match the frozen self-model (Phase 1).
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

BELIEF_CKPT = DATA_DIR / "checkpoints/belief_model.pkl"

# ==========================================
# Load frozen self-model + data
# ==========================================

print("=" * 50)
print("Phase 2: Training feudal belief module")
print("=" * 50)

assert SELF_MODEL_PATH.exists(), f"{SELF_MODEL_PATH} not found. Run Phase 1 first."

gem_buf = TransformerBuffer.load(OBSERVER_BUFFER)
food_buf = TransformerBuffer.load(OTHER_BUFFER)
combined = TransformerBuffer.combine([gem_buf, food_buf])
print(f"combined buffer size={combined.size} (gem + food, agent-tagged)")

base = ViTOneHot(
    memory=combined,
    LR=0.001,
    device="cpu",
    seed=torch.random.seed(),
    reward_loss_weight=1.0,
    **ARCH,
)
base.load(str(SELF_MODEL_PATH))
print(f"Loaded frozen self-model from {SELF_MODEL_PATH}")

belief = BeliefModel(
    self_model=base,
    layer_size=ARCH["layer_size"],
    patch_size=ARCH["patch_size"],
    num_heads=ARCH["num_heads"],
    num_layers=ARCH["num_layers"],
    device="cpu",
    LR=1e-3,
)

logger = TensorboardLogger(
    TRAINING_EPOCHS,
    DATA_DIR / "logs/belief" / STATIC_RUNTIME,
    "state_loss",
    "action_loss",
)

# ==========================================
# Train belief encoder (base frozen)
# ==========================================

for epoch in range(TRAINING_EPOCHS):
    state_loss, action_loss = belief.train_belief(mask_type=TRAIN_MASK_TYPE)
    logger.record_turn(
        epoch=epoch,
        loss=state_loss + action_loss,
        reward=0.0,
        state_loss=state_loss,
        action_loss=action_loss,
    )
    if epoch % 100 == 0 or epoch == TRAINING_EPOCHS - 1:
        print(
            f"  Epoch {epoch:>5d}/{TRAINING_EPOCHS}: "
            f"state={state_loss:.4f} action={action_loss:.4f}"
        )

BELIEF_CKPT.parent.mkdir(parents=True, exist_ok=True)
belief.save(str(BELIEF_CKPT))
print(f"Belief module saved to: {BELIEF_CKPT}")


# ==========================================
# Evaluation: belief vs no-belief baseline, per agent
# ==========================================
def eval_on(buf, label):
    base.memory = buf
    bs, ba, ns, na = [], [], [], []
    for _ in range(EVAL_STEPS):
        s, a = belief.evaluate_belief(mask_type=TRAIN_MASK_TYPE, use_belief=True)
        s0, a0 = belief.evaluate_belief(mask_type=TRAIN_MASK_TYPE, use_belief=False)
        bs.append(s)
        ba.append(a)
        ns.append(s0)
        na.append(a0)
    print(f"\n{label}:")
    print(f"  with belief : state={np.mean(bs):.4f}  action={np.mean(ba):.4f}")
    print(f"  no belief   : state={np.mean(ns):.4f}  action={np.mean(na):.4f}")
    print(
        f"  action lift : {(np.mean(na) - np.mean(ba)):+.4f} "
        f"({(np.mean(na) - np.mean(ba)) / max(np.mean(na), 1e-8) * 100:+.1f}%)"
    )


print("\n" + "=" * 50)
print("EVALUATION (action lift = baseline - belief; positive = belief helps)")
print("=" * 50)
eval_on(gem_buf, "gem-lover (base's OWN agent — expect small lift)")
eval_on(food_buf, "food-lover (OTHER agent — expect large lift = ToM signal)")
print("\nPhase 2 complete.")
