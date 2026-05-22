"""Train ONE decoupled (leakage-controlled) belief module on the frozen self-model.

The belief encoder reads an INDEPENDENT same-agent window (decouple=True) rather
than the window being predicted, so g cannot peek at the answer and must
summarize the agent's goal. Each step alternates the per-agent memory so context
and query are always the same agent.

Saves to checkpoints/belief_model_decoupled.pkl for the Phase 3 re-test.

Usage:
    python -m sorrel.examples.treasurehunt.notebooks.train_belief_decoupled
"""

from datetime import datetime
from pathlib import Path

import torch

from sorrel.buffers import TransformerBuffer
from sorrel.models.pytorch.transformer import BeliefModel, ViTOneHot
from sorrel.utils.logging import TensorboardLogger

DATA_DIR = Path(__file__).parent / "../data"
SELF_MODEL_PATH = DATA_DIR / "checkpoints/self_model.pkl"
OUT_CKPT = DATA_DIR / "checkpoints/belief_model_decoupled.pkl"

EPOCHS = 8000
TRAIN_MASK = "random"
SEED = 0

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

torch.manual_seed(SEED)

gem_buf = TransformerBuffer.load(DATA_DIR / "memories/gemlover.npz")
food_buf = TransformerBuffer.load(DATA_DIR / "memories/foodlover.npz")

base = ViTOneHot(
    memory=gem_buf, LR=1e-3, device="cpu", seed=SEED, reward_loss_weight=1.0, **ARCH
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

logger = TensorboardLogger(
    EPOCHS,
    DATA_DIR / "logs/belief_decoupled" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
    "state_loss",
    "action_loss",
)

print(f"Training decoupled belief module ({EPOCHS} epochs)...")
for epoch in range(EPOCHS):
    # single-agent per step so context & query windows are the same agent
    base.memory = gem_buf if epoch % 2 == 0 else food_buf
    state_loss, action_loss = bm.train_belief(mask_type=TRAIN_MASK, decouple=True)
    logger.record_turn(
        epoch=epoch,
        loss=state_loss + action_loss,
        reward=0.0,
        state_loss=state_loss,
        action_loss=action_loss,
    )
    if epoch % 200 == 0 or epoch == EPOCHS - 1:
        print(f"  Epoch {epoch:>5d}/{EPOCHS}: state={state_loss:.4f} action={action_loss:.4f}")

OUT_CKPT.parent.mkdir(parents=True, exist_ok=True)
bm.save(str(OUT_CKPT))
print(f"Decoupled belief module saved to: {OUT_CKPT}")
