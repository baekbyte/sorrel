"""
Train the belief encoder for FOV-gated first-person ToM.

At deployment the encoder feeds g into the frozen base for the OBSERVER's
two-pass action selection (see tom_rollout.py): pass 1 uses g to reconstruct
the observer's masked outer ring; pass 2 runs the base WITHOUT g on the
belief-completed observation, so the action reflects the observer's own
preference.
"""

from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from sorrel.buffers import TransformerBuffer
from sorrel.models.pytorch.transformer import BeliefEncoder, ViTOneHot
from sorrel.utils.logging import TensorboardLogger

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

DATA_DIR = Path(__file__).parent / "../data"
SELF_MODEL_PATH = DATA_DIR / "checkpoints/self_model.pkl"
GEM_BUFFER = DATA_DIR / "memories/gemlover.npz"
FOOD_BUFFER = DATA_DIR / "memories/foodlover.npz"
OUT_CKPT = DATA_DIR / "checkpoints/belief.pkl"

EPOCHS = 8000
SEED = 0
BATCH_SIZE = 64
LR = 1e-3
PREF_LOSS_WEIGHT = 1.0       # weight on the preference-classification cross-entropy
RECON_LOSS_WEIGHT = 1.0      # weight on state-reconstruction under random masking
NUM_PREFERENCES = 2           # {gem-lover, food-lover}

ARCH = dict(
    state_size=(6, 15, 15),
    action_space=4,
    layer_size=192,
    patch_size=3,
    num_frames=5,
    num_heads=3,
    batch_size=BATCH_SIZE,
    num_layers=2,
)
C, H, W = ARCH["state_size"]
FLAT_OBS = C * H * W

torch.manual_seed(SEED)
np.random.seed(SEED)


# --------------------------------------------------------------------------
# Per-buffer sampler that uses idx (real write count) rather than capacity
# --------------------------------------------------------------------------


def sample_window(buf: TransformerBuffer, batch_size: int, num_frames: int):
    """Sample a batch of windows from one preference's buffer. Returns a dict
    of arrays for the window's (states, actions, next_states, next_actions)."""
    real_size = max(1, int(buf.idx) - num_frames - 1)
    base_idx = np.random.choice(real_size, batch_size, replace=False)
    indices = base_idx[:, None] + np.arange(num_frames)
    return dict(
        states=buf.states[indices].reshape(batch_size, -1),
        actions=buf.actions[indices].reshape(batch_size, -1),
        next_states=buf.states[indices + 1].reshape(batch_size, -1),
        next_actions=buf.actions[indices + 1].reshape(batch_size, -1).astype(np.int64),
    )


# --------------------------------------------------------------------------
# Build frozen base + trainable encoder + classifier head
# --------------------------------------------------------------------------

# Frozen base -- its memory is unused; we pass batches in directly.
dummy_mem = TransformerBuffer(capacity=10, obs_shape=(FLAT_OBS,), n_frames=ARCH["num_frames"])
base = ViTOneHot(
    memory=dummy_mem,
    LR=LR,
    device="cpu",
    seed=SEED,
    reward_loss_weight=0.0,
    **ARCH,
)
base.load(str(SELF_MODEL_PATH))
base.eval()
for p in base.parameters():
    p.requires_grad_(False)

encoder = BeliefEncoder(
    state_size=ARCH["state_size"],
    action_space=ARCH["action_space"],
    layer_size=ARCH["layer_size"],
    patch_size=ARCH["patch_size"],
    num_frames=ARCH["num_frames"],
    num_heads=ARCH["num_heads"],
    num_layers=ARCH["num_layers"],
    device="cpu",
)
# Tiny classifier head on top of g.
pref_classifier = torch.nn.Linear(ARCH["layer_size"], NUM_PREFERENCES)

opt = torch.optim.Adam(
    list(encoder.parameters()) + list(pref_classifier.parameters()), lr=LR
)

gem_buf = TransformerBuffer.load(GEM_BUFFER)
food_buf = TransformerBuffer.load(FOOD_BUFFER)
print(f"gem-lover buffer: idx={gem_buf.idx}, food-lover buffer: idx={food_buf.idx}")

logger = TensorboardLogger(
    EPOCHS,
    DATA_DIR / "logs/belief" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
    "state_loss",
    "pref_loss",
    "pref_acc",
)


def step():
    """One training step. Alternates which preference's buffer the batch
    comes from -- but we mix both per step so the gradient sees both
    preferences. Returns (state_loss, pref_loss, pref_acc) as floats."""
    # Half batch from each preference. Concatenate, label, shuffle.
    half = BATCH_SIZE // 2
    gem_batch = sample_window(gem_buf, half, ARCH["num_frames"])
    food_batch = sample_window(food_buf, half, ARCH["num_frames"])
    states = np.concatenate([gem_batch["states"], food_batch["states"]], axis=0)
    actions = np.concatenate([gem_batch["actions"], food_batch["actions"]], axis=0)
    next_states = np.concatenate([gem_batch["next_states"], food_batch["next_states"]], axis=0)
    labels = np.concatenate([np.zeros(half, dtype=np.int64), np.ones(half, dtype=np.int64)])
    # Shuffle so the batch is mixed.
    perm = np.random.permutation(BATCH_SIZE)
    states, actions, next_states, labels = states[perm], actions[perm], next_states[perm], labels[perm]

    s = torch.tensor(states, dtype=torch.float32).view(BATCH_SIZE, ARCH["num_frames"], C, H, W)
    a = torch.tensor(actions, dtype=torch.long).view(BATCH_SIZE, ARCH["num_frames"], 1)
    s_target = torch.tensor(next_states, dtype=torch.float32).view(
        BATCH_SIZE, ARCH["num_frames"], C, H, W
    )
    y = torch.tensor(labels, dtype=torch.long)

    # Encoder produces g from the unmasked first-person window.
    g = encoder(s, a)  # (B, layer_size)
    # Preference classification.
    pref_logits = pref_classifier(g)
    pref_loss = F.cross_entropy(pref_logits, y)
    pref_acc = (pref_logits.argmax(1) == y).float().mean()

    # State reconstruction support: random-mask the base's state input, see if
    # g helps the frozen base predict the next (unmasked) state.
    state_mask = base.random_mask(s)
    s_masked = s * state_mask.float()
    state_preds, _ = base.forward(s_masked, a, belief_embedding=g)
    state_loss = base.state_loss(state_preds, s_target, mask=None)

    loss = RECON_LOSS_WEIGHT * state_loss + PREF_LOSS_WEIGHT * pref_loss
    opt.zero_grad()
    loss.backward()
    opt.step()
    return float(state_loss.detach()), float(pref_loss.detach()), float(pref_acc.detach())


# --------------------------------------------------------------------------
# Train
# --------------------------------------------------------------------------

print(f"Training belief encoder ({EPOCHS} epochs)")
for epoch in range(EPOCHS):
    sl, pl, pa = step()
    logger.record_turn(
        epoch=epoch, loss=sl + pl, reward=0.0, state_loss=sl, pref_loss=pl, pref_acc=pa
    )
    if epoch % 200 == 0 or epoch == EPOCHS - 1:
        print(
            f"  Epoch {epoch:>5d}/{EPOCHS}: "
            f"state={sl:.4f} pref_loss={pl:.4f} pref_acc={pa * 100:.1f}%"
        )

OUT_CKPT.parent.mkdir(parents=True, exist_ok=True)
torch.save(
    {
        "belief_encoder": encoder.state_dict(),
        "pref_classifier": pref_classifier.state_dict(),
        "optim": opt.state_dict(),
    },
    OUT_CKPT,
)
print(f"Belief encoder + classifier saved to: {OUT_CKPT}")
