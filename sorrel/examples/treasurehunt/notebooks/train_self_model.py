"""Phase 1: train the observer's frozen self/world-model with mask channel.

Loads the gem-lover's 6-channel buffer, samples 5-frame windows, and trains
ViTOneHot on a 7-channel input where channel 6 is a 'mask' indicator (1 where
the observer cannot see this cell, 0 elsewhere). With probability 0.5 each
training step we apply the observer's outer-ring mask (chebyshev > 4 from
center) to the input -- the model learns BOTH:

  - mask channel = 0 (full unmasked input, like a watched agent's view)
  - mask channel = 1 in outer ring (observer's deployment-time input)

Target is always the FULL next state (6 entity channels + mask channel = 0).
The model trivially learns to output 0 on the mask channel; the meaningful
loss comes from per-channel state reconstruction over entity channels 0..5
plus action and reward heads.

This matches the deployment-time distribution: the observer feeds in 7-channel
masked input (Pass 1 of get_action), and Pass 2 sees a belief-completed
observation with mask channel = 0 -- both within-distribution.
"""

from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from sorrel.buffers import TransformerBuffer
from sorrel.models.pytorch.transformer import ViTOneHot
from sorrel.utils.logging import TensorboardLogger

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

STATIC_RUNTIME = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
DATA_DIR = Path(__file__).parent / "../data"
OBSERVER_BUFFER = DATA_DIR / "memories/gemlover.npz"
CHECKPOINT_PATH = DATA_DIR / "checkpoints/self_model.pkl"

TRAINING_EPOCHS = 20000
EVAL_STEPS = 1000
BATCH_SIZE = 64
LR = 1e-3
SEED = 0

DROPOUT = 0.0
WEIGHT_DECAY = 0.0
ACTION_LOSS_WEIGHT = 1.0
REWARD_LOSS_WEIGHT = 1.0
MASK_PROB = 0.5  # probability of applying outer-ring mask per training sample

NUM_ENTITY_CHANNELS = 6   # underlying buffers are 6-channel
MASK_CHANNEL_IDX = 6      # mask channel index in 7-channel model input
ACTUAL_FOV_RADIUS = 4

ARCH = dict(
    state_size=(7, 15, 15),  # 6 entity + 1 mask channel
    action_space=4,
    layer_size=192,
    patch_size=3,
    num_frames=5,
    num_heads=3,
    batch_size=BATCH_SIZE,
    num_layers=2,
)
C7, H, W = ARCH["state_size"]
T = ARCH["num_frames"]
FLAT_6CH = NUM_ENTITY_CHANNELS * H * W

# Precomputed outer-ring mask (H, W) — 1 outside inner FOV, 0 inside.
_cy, _cx = H // 2, W // 2
_yy, _xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
OUTER_RING_2D = torch.from_numpy(
    (np.maximum(np.abs(_yy - _cy), np.abs(_xx - _cx)) > ACTUAL_FOV_RADIUS).astype(np.float32)
)  # (H, W)

torch.manual_seed(SEED)
np.random.seed(SEED)

# --------------------------------------------------------------------------
# Setup
# --------------------------------------------------------------------------

print("=" * 60)
print("Phase 1: Training self/world-model with explicit mask channel")
print("=" * 60)

assert OBSERVER_BUFFER.exists(), f"{OBSERVER_BUFFER} not found."
buffer = TransformerBuffer.load(OBSERVER_BUFFER)
print(f"Loaded observer buffer: idx={buffer.idx}")

model = ViTOneHot(
    memory=buffer,
    LR=LR,
    device="cpu",
    seed=SEED,
    dropout=DROPOUT,
    weight_decay=WEIGHT_DECAY,
    action_loss_weight=ACTION_LOSS_WEIGHT,
    reward_loss_weight=REWARD_LOSS_WEIGHT,
    **ARCH,
)


# --------------------------------------------------------------------------
# Sampling + masking
# --------------------------------------------------------------------------


def sample_batch():
    """Sample BATCH_SIZE 5-frame windows from the 6-channel buffer."""
    real_size = max(1, int(buffer.idx) - T - 1)
    base_idx = np.random.choice(real_size, BATCH_SIZE, replace=False)
    indices = base_idx[:, None] + np.arange(T)
    states_6 = torch.tensor(
        buffer.states[indices].reshape(BATCH_SIZE, T, NUM_ENTITY_CHANNELS, H, W),
        dtype=torch.float32,
    )
    actions = torch.tensor(
        buffer.actions[indices].reshape(BATCH_SIZE, T, 1), dtype=torch.long
    )
    next_states_6 = torch.tensor(
        buffer.states[indices + 1].reshape(BATCH_SIZE, T, NUM_ENTITY_CHANNELS, H, W),
        dtype=torch.float32,
    )
    next_actions = torch.tensor(
        buffer.actions[indices + 1].reshape(BATCH_SIZE, T, 1), dtype=torch.long
    )
    next_rewards = torch.tensor(
        buffer.rewards[indices + 1].reshape(BATCH_SIZE, T, 1), dtype=torch.float32
    )
    return states_6, actions, next_states_6, next_actions, next_rewards


def to_7ch_input(states_6: torch.Tensor) -> torch.Tensor:
    """Build the 7-channel model input. Per (batch, frame) sample, with
    probability MASK_PROB apply outer-ring mask + mask channel = 1; otherwise
    leave entity channels untouched + mask channel = 0.

    states_6: (B, T, 6, H, W) -> returns (B, T, 7, H, W).
    """
    B, T_in = states_6.shape[:2]
    apply = (torch.rand(B, T_in) < MASK_PROB).float()  # (B, T)
    # Per-sample mask: apply * OUTER_RING_2D
    sample_mask = apply.view(B, T_in, 1, 1) * OUTER_RING_2D.view(1, 1, H, W)  # (B, T, H, W)
    # Zero entity channels where mask=1
    visible = 1.0 - sample_mask  # (B, T, H, W)
    masked_entities = states_6 * visible.unsqueeze(2)  # (B, T, 6, H, W)
    mask_ch = sample_mask.unsqueeze(2)  # (B, T, 1, H, W)
    return torch.cat([masked_entities, mask_ch], dim=2)


def to_7ch_target(next_states_6: torch.Tensor) -> torch.Tensor:
    """Pad next-state target with zero mask channel (target is the FULL
    next state -- the model trivially learns to predict 0 in the mask channel)."""
    B, T_in = next_states_6.shape[:2]
    zero_ch = torch.zeros(B, T_in, 1, H, W, dtype=next_states_6.dtype)
    return torch.cat([next_states_6, zero_ch], dim=2)


# --------------------------------------------------------------------------
# Training step
# --------------------------------------------------------------------------


def step():
    states_6, actions, next_states_6, next_actions, next_rewards = sample_batch()
    inputs = to_7ch_input(states_6)
    targets = to_7ch_target(next_states_6)

    # Forward + reward head
    x = model._run_transformer(inputs, actions, agent_id=None)
    B_, T_ = x.size(0), x.size(1)
    state_preds, action_preds = model._apply_heads(x, B_, T_)
    reward_preds = model.reward_head(x)

    state_loss = model.state_loss(state_preds, targets, mask=None)
    action_loss = model.action_loss(action_preds, next_actions)
    reward_loss = model.reward_loss(reward_preds, next_rewards)

    loss = (
        state_loss
        + ACTION_LOSS_WEIGHT * action_loss
        + REWARD_LOSS_WEIGHT * reward_loss
    )
    model.optimizer.zero_grad()
    loss.backward()
    model.optimizer.step()
    return float(state_loss.detach()), float(action_loss.detach()), float(reward_loss.detach())


def eval_step():
    """No-grad evaluation step at the same input distribution."""
    with torch.no_grad():
        states_6, actions, next_states_6, next_actions, next_rewards = sample_batch()
        inputs = to_7ch_input(states_6)
        targets = to_7ch_target(next_states_6)
        x = model._run_transformer(inputs, actions, agent_id=None)
        B_, T_ = x.size(0), x.size(1)
        state_preds, action_preds = model._apply_heads(x, B_, T_)
        reward_preds = model.reward_head(x)
        state_loss = model.state_loss(state_preds, targets, mask=None)
        action_loss = model.action_loss(action_preds, next_actions)
        reward_loss = model.reward_loss(reward_preds, next_rewards)
    return float(state_loss), float(action_loss), float(reward_loss)


# --------------------------------------------------------------------------
# Train
# --------------------------------------------------------------------------

logger = TensorboardLogger(
    TRAINING_EPOCHS,
    DATA_DIR / "logs/self_model" / STATIC_RUNTIME,
    "state_loss",
    "action_loss",
    "reward_loss",
)

print(f"Training {TRAINING_EPOCHS} epochs (mask_prob={MASK_PROB}, batch={BATCH_SIZE})")
for epoch in range(TRAINING_EPOCHS):
    sl, al, rl = step()
    logger.record_turn(
        epoch=epoch,
        loss=sl + al + rl,
        reward=0.0,
        state_loss=sl,
        action_loss=al,
        reward_loss=rl,
    )
    if epoch % 100 == 0 or epoch == TRAINING_EPOCHS - 1:
        print(f"  Epoch {epoch:>5d}/{TRAINING_EPOCHS}: state={sl:.4f} action={al:.4f} reward={rl:.4f}")

CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
model.save(str(CHECKPOINT_PATH))
print(f"Self-model saved to: {CHECKPOINT_PATH}")

# --------------------------------------------------------------------------
# Held-out evaluation
# --------------------------------------------------------------------------

print("\nHeld-out evaluation:")
sls, als, rls = [], [], []
for _ in range(EVAL_STEPS):
    sl, al, rl = eval_step()
    sls.append(sl); als.append(al); rls.append(rl)
print(
    f"  state={np.mean(sls):.4f}  action={np.mean(als):.4f}  reward_MSE={np.mean(rls):.4f} ± {np.std(rls):.4f}"
)
print("\nPhase 1 complete.")
