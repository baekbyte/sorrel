"""Train the ToM v2 observer: encoder + desire head + latent (belief) head.

The encoder (reused `BeliefEncoder`) reads the observer's own POV frames plus
the watched agent's observed displacement tokens — strictly third person —
and produces an embedding with two supervised heads:

    desire: {gem-lover, food-lover}         (who is this agent?)
    latent: {gems-left, gems-right}         (what does their behavior imply
                                             about the hidden world?)

Truncation augmentation (freeze the window at a random t: repeat the last
seen frame, pad displacements with STAY) keeps partially-watched windows
in-distribution, so accuracy-vs-watch-length curves are meaningful and the
deployed observer can act before the window is full.

Usage:
    python -m sorrel.examples.treasurehunt.notebooks.tom2_train_observer
"""

import numpy as np
import torch
import torch.nn.functional as F

from sorrel.examples.treasurehunt.notebooks.tom2_common import (
    DATASET_PATH,
    DISP_STAY,
    N_DISP_TOKENS,
    OBSERVER_CKPT,
    T_WATCH,
)
from sorrel.models.pytorch.transformer import BeliefEncoder

SEED = 0
BATCH_SIZE = 256      # small random batches left the encoder stuck in its
STEPS = 3000          # collapsed init (g-std -> 0); big batches + warmup escape it
LR = 1e-3
WARMUP_STEPS = 300    # linear LR warmup (post-LN transformer cold-start fix)
TRUNC_START = 500     # no truncation augmentation during the fragile phase
TRUNC_PROB = 0.5      # fraction of truncated windows after TRUNC_START

ARCH = dict(
    state_size=(7, 9, 9),  # 6 entity channels + tracked-agent marker
    action_space=N_DISP_TOKENS,
    layer_size=192,
    patch_size=3,
    num_frames=T_WATCH,
    num_heads=3,
    num_layers=2,
)

torch.manual_seed(SEED)
np.random.seed(SEED)


def load_split():
    d = np.load(DATASET_PATH, allow_pickle=False)
    frames, disp = d["frames"], d["disp"]
    desire, latent, episode = d["desire"], d["latent"], d["episode"]
    train_mask = (episode % 10) < 8  # split by episode, not window
    tr = dict(
        frames=frames[train_mask], disp=disp[train_mask],
        desire=desire[train_mask], latent=latent[train_mask],
    )
    va = dict(
        frames=frames[~train_mask], disp=disp[~train_mask],
        desire=desire[~train_mask], latent=latent[~train_mask],
    )
    print(f"train windows: {len(tr['desire'])}, val windows: {len(va['desire'])}")
    return tr, va


def truncate(frames: torch.Tensor, disp: torch.Tensor, t: int):
    """Freeze the window at watch-length t: repeat frame t-1, STAY after."""
    frames = frames.clone()
    disp = disp.clone()
    frames[:, t:] = frames[:, t - 1 : t]
    disp[:, t:] = DISP_STAY
    return frames, disp


def batch_tensors(data, idx):
    frames = torch.tensor(data["frames"][idx], dtype=torch.float32)
    disp = torch.tensor(data["disp"][idx], dtype=torch.long).unsqueeze(-1)
    desire = torch.tensor(data["desire"][idx], dtype=torch.long)
    latent = torch.tensor(data["latent"][idx], dtype=torch.long)
    return frames, disp, desire, latent


def evaluate(encoder, desire_head, latent_head, data, t: int | None = None, bs=256):
    n = len(data["desire"])
    d_correct = l_correct = 0
    with torch.no_grad():
        for i in range(0, n, bs):
            idx = np.arange(i, min(i + bs, n))
            frames, disp, desire, latent = batch_tensors(data, idx)
            if t is not None:
                frames, disp = truncate(frames, disp, t)
            g = encoder(frames, disp)
            d_correct += (desire_head(g).argmax(1) == desire).sum().item()
            l_correct += (latent_head(g).argmax(1) == latent).sum().item()
    return d_correct / n, l_correct / n


def main():
    tr, va = load_split()
    encoder = BeliefEncoder(**ARCH, device="cpu")
    desire_head = torch.nn.Linear(ARCH["layer_size"], 2)
    latent_head = torch.nn.Linear(ARCH["layer_size"], 2)
    params = (
        list(encoder.parameters())
        + list(desire_head.parameters())
        + list(latent_head.parameters())
    )
    opt = torch.optim.Adam(params, lr=LR)

    def lr_lambda(s: int) -> float:
        # Linear warmup, then cosine decay to 5% — the constant-LR run
        # oscillated (g-std swings) through the second half.
        if s < WARMUP_STEPS:
            return (s + 1) / WARMUP_STEPS
        progress = (s - WARMUP_STEPS) / max(1, STEPS - WARMUP_STEPS)
        return 0.05 + 0.95 * 0.5 * (1 + np.cos(np.pi * progress))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    n_train = len(tr["desire"])
    for step in range(STEPS):
        idx = np.random.choice(n_train, BATCH_SIZE, replace=False)
        frames, disp, desire, latent = batch_tensors(tr, idx)
        if step >= TRUNC_START and np.random.random() < TRUNC_PROB:
            t = int(np.random.randint(2, T_WATCH + 1))
            if t < T_WATCH:
                frames, disp = truncate(frames, disp, t)
        g = encoder(frames, disp)
        loss = F.cross_entropy(desire_head(g), desire) + F.cross_entropy(
            latent_head(g), latent
        )
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % 250 == 0 or step == STEPS - 1:
            d_acc, l_acc = evaluate(encoder, desire_head, latent_head, va)
            g_std = float(g.std(0).mean().detach())
            print(
                f"step {step:>5d}: loss={float(loss.detach()):.4f}  g-std={g_std:.4f}  "
                f"val desire={d_acc * 100:.1f}%  val latent={l_acc * 100:.1f}%"
            )

    print("\nAccuracy vs watch length (val):")
    print(f"  {'t':>4}{'desire':>10}{'latent':>10}")
    for t in range(2, T_WATCH + 1, 2):
        d_acc, l_acc = evaluate(
            encoder, desire_head, latent_head, va, t=t if t < T_WATCH else None
        )
        print(f"  {t:>4}{d_acc * 100:>9.1f}%{l_acc * 100:>9.1f}%")

    OBSERVER_CKPT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "encoder": encoder.state_dict(),
            "desire_head": desire_head.state_dict(),
            "latent_head": latent_head.state_dict(),
            "arch": ARCH,
        },
        OBSERVER_CKPT,
    )
    print(f"\nsaved -> {OBSERVER_CKPT}")


if __name__ == "__main__":
    main()
