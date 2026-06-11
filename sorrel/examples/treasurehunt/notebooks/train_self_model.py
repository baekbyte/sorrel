"""
Trains a ViTOneHot on the OBSERVER's own experience (the gem-lover buffer) under
the self/world-model objective: given (state, action) predict
  - the next state (per-channel cross-entropy reconstruction),
  - the next action (cross-entropy),
  - the reward of the next transition (MSE, via the new reward head).
"""

from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from sorrel.buffers import TransformerBuffer
from sorrel.models.pytorch.transformer import ViTOneHot
from sorrel.utils.logging import TensorboardLogger

# ==========================================
# Configuration
# ==========================================

STATIC_RUNTIME = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
DATA_DIR = Path(__file__).parent / "../data"

# The observer's own preference buffer (gem-lover, per the motivating scenario).
OBSERVER_BUFFER = DATA_DIR / "memories/gemlover.npz"

TRAINING_EPOCHS = 20000
EVAL_STEPS = 1000

DROPOUT = 0.0
WEIGHT_DECAY = 0.0
ACTION_LOSS_WEIGHT = 1.0
REWARD_LOSS_WEIGHT = 1.0  # enables the self/world-model reward head

# Self-model trains on the agent's own FULL observation (no masking) — it models
# "what happens when I act", not partial-observation inference.
TRAIN_MASK_TYPE = "full"

CHECKPOINT_PATH = DATA_DIR / "checkpoints/self_model.pkl"

# ==========================================
# Train
# ==========================================

print("=" * 50)
print("Phase 1: Training observer self/world-model")
print("=" * 50)

assert (
    OBSERVER_BUFFER.exists()
), f"{OBSERVER_BUFFER} not found. Run train_preferences.py (Phase 0) first."

buffer = TransformerBuffer.load(OBSERVER_BUFFER)
print(f"Loaded observer buffer: size={buffer.size}")

model = ViTOneHot(
    state_size=(6, 15, 15),
    action_space=4,
    layer_size=192,
    patch_size=3,
    num_frames=5,
    num_heads=3,
    batch_size=64,
    num_layers=2,
    memory=buffer,
    LR=0.001,
    device="cpu",
    seed=torch.random.seed(),
    dropout=DROPOUT,
    weight_decay=WEIGHT_DECAY,
    action_loss_weight=ACTION_LOSS_WEIGHT,
    reward_loss_weight=REWARD_LOSS_WEIGHT,
)

logger = TensorboardLogger(
    TRAINING_EPOCHS,
    DATA_DIR / "logs/self_model" / STATIC_RUNTIME,
    "state_loss",
    "action_loss",
    "reward_loss",
)

for epoch in range(TRAINING_EPOCHS):
    state_loss, action_loss, reward_loss = model.train_self_model(
        mask_type=TRAIN_MASK_TYPE
    )
    logger.record_turn(
        epoch=epoch,
        loss=state_loss + action_loss + reward_loss,
        reward=0.0,
        state_loss=state_loss,
        action_loss=action_loss,
        reward_loss=reward_loss,
    )
    if epoch % 100 == 0 or epoch == TRAINING_EPOCHS - 1:
        print(
            f"  Epoch {epoch:>5d}/{TRAINING_EPOCHS}: "
            f"state={state_loss:.4f} action={action_loss:.4f} reward={reward_loss:.4f}"
        )

CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
model.save(str(CHECKPOINT_PATH))
print(f"Self-model saved to: {CHECKPOINT_PATH}")

# ==========================================
# Held-out evaluation
# ==========================================

print("\nHeld-out evaluation:")
state_losses, action_losses, reward_losses = [], [], []
for _ in range(EVAL_STEPS):
    s, a, r = model.evaluate_self_model(mask_type=TRAIN_MASK_TYPE)
    state_losses.append(s)
    action_losses.append(a)
    reward_losses.append(r)
print(
    f"  state={np.mean(state_losses):.4f}  "
    f"action={np.mean(action_losses):.4f}  "
    f"reward_MSE={np.mean(reward_losses):.4f} ± {np.std(reward_losses):.4f}"
)
print("\nPhase 1 complete.")
