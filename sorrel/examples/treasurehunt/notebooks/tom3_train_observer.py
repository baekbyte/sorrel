"""Train the ToM v3 observer(s): desire, gem-desire, and latent heads.

Same recipe as tom2_train_observer (BeliefEncoder, LR warmup + cosine decay,
big batches, deferred truncation augmentation), with K-way desire/latent
heads plus a binary gem-desire head for the factorized behavioral readout.
The fixed 9-channel input (8 entity kinds + tracked-agent marker) covers
every K with one architecture. One checkpoint per K.

Usage:
    python -m sorrel.examples.treasurehunt.notebooks.tom3_train_observer [K ...]
    (default: 2 3 4)
"""

import sys
import os
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from sorrel.examples.treasurehunt.notebooks.tom3_common import (
    DATA_DIR,
    DISP_STAY,
    N_DISP_TOKENS,
    N_INPUT_CHANNELS,
    T_WATCH,
    dataset_path,
    observer_ckpt,
    visibility_observer_ckpt,
)
from sorrel.models.pytorch.transformer import BeliefEncoder

SEED = 0
BATCH_SIZE = 384  # small random batches left the encoder stuck in its
# collapsed init (g-std -> 0); big batches + warmup escape it (bumped
# 256 -> 384 on MPS for a stronger escape gradient signal).
# Per-K step budgets: 3000 converges for the binary task; the K-way tasks
# were still descending at 3000 (K=3 latent +7 pts over the last 750 steps).
STEPS = {2: 3000, 3: 9000, 4: 9000}
LR = 1e-3
WARMUP_STEPS = 300  # linear LR warmup (post-LN transformer cold-start fix)
TRUNC_START = 500  # no truncation augmentation during the fragile phase
TRUNC_PROB = 0.5  # fraction of desire windows truncated after TRUNC_START
# Variant-only augmentation: unlike ``truncate``, which freezes the whole
# third-person scene, this removes only the tracked partner's marker/action
# evidence after a sampled point. The shared frames continue to show the
# other agent and world evolving, matching the partial FOV-gated evidence that
# drove multi-agent errors in the norm spatial audit.
VISIBILITY_PROB = 0.75
VISIBILITY_MIN_STEPS = 4
# MPS is ~3x faster than CPU for this encoder on Apple silicon (verified
# numerically equivalent to CPU: forward ~1e-7, grads ~1e-8).
# A launchd background job cannot always obtain a usable MPS context even
# though an interactive terminal can.  ``SORREL_TOM_DEVICE=cpu`` provides a
# deterministic fallback for persistent training jobs; ordinary interactive
# runs retain the automatic MPS preference.
DEVICE = os.environ.get("SORREL_TOM_DEVICE") or (
    "mps" if torch.backends.mps.is_available() else "cpu"
)
# Cold-start collapse escape checks: g-std must clear each bar or training
# restarts with the next seed. Healthy runs sit at 0.24-0.36 by step 750 and
# 0.4+ by ~2000; a fully collapsed run sat at 0.02-0.05, and a MARGINAL
# escape (0.10 at 750, wobbling 0.06-0.22 after) kept the latent head at
# chance for 4750 steps — hence the second, sustained-escape bar.
ESCAPE_CHECKS = {750: 0.15, 2250: 0.25}  # step -> min g-std
MAX_INIT_TRIES = 5
# TensorBoard event files land here, one run per (K, attempt seed):
#   poetry run tensorboard --logdir sorrel/examples/treasurehunt/data/tensorboard
TB_DIR = DATA_DIR / "tensorboard"
TB_EVERY = 25  # steps between train-loss/lr points

ARCH = dict(
    state_size=(N_INPUT_CHANNELS, 9, 9),
    action_space=N_DISP_TOKENS,
    layer_size=192,
    patch_size=3,
    num_frames=T_WATCH,
    num_heads=3,
    num_layers=2,
)


def load_split(k: int):
    d = np.load(dataset_path(k), allow_pickle=False)
    frames, disp = d["frames"], d["disp"]
    desire, latent, episode = d["desire"], d["latent"], d["episode"]
    train_mask = (episode % 10) < 8  # split by episode, not window
    tr = dict(
        frames=frames[train_mask],
        disp=disp[train_mask],
        desire=desire[train_mask],
        latent=latent[train_mask],
    )
    va = dict(
        frames=frames[~train_mask],
        disp=disp[~train_mask],
        desire=desire[~train_mask],
        latent=latent[~train_mask],
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


def mask_tracked_suffix(frames: torch.Tensor, disp: torch.Tensor, t: int):
    """Hide the tracked partner after ``t`` while retaining scene evolution.

    The tracked-agent marker is the final input channel. Later displacements
    are changed to STAY, precisely the representation used when an agent is
    outside the observer's FOV. Unlike :func:`truncate`, all non-marker image
    channels are retained, including other agents' behavior.
    """
    frames = frames.clone()
    disp = disp.clone()
    frames[:, t:, -1, :, :] = 0.0
    disp[:, t:] = DISP_STAY
    return frames, disp


def batch_tensors(data, idx):
    frames = torch.tensor(data["frames"][idx], dtype=torch.float32, device=DEVICE)
    disp = torch.tensor(data["disp"][idx], dtype=torch.long, device=DEVICE).unsqueeze(
        -1
    )
    desire = torch.tensor(data["desire"][idx], dtype=torch.long, device=DEVICE)
    latent = torch.tensor(data["latent"][idx], dtype=torch.long, device=DEVICE)
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


def train(k: int, *, variant: str = "baseline", steps_override: int | None = None) -> None:
    if variant not in {"baseline", "visibility"}:
        raise ValueError(f"Unknown ToM training variant: {variant}")
    steps = steps_override or STEPS[k]
    print(f"===== K={k} ({steps} steps, device={DEVICE}, variant={variant}) =====")
    tr, va = load_split(k)
    n_train = len(tr["desire"])

    def lr_lambda(s: int) -> float:
        if s < WARMUP_STEPS:
            return (s + 1) / WARMUP_STEPS
        progress = (s - WARMUP_STEPS) / max(1, steps - WARMUP_STEPS)
        return 0.05 + 0.95 * 0.5 * (1 + np.cos(np.pi * progress))

    # Escaping the post-LN cold-start collapse (g-std -> 0) is stochastic:
    # most init/data-order draws escape by ~step 500, but one 8500-step run
    # never did. Check g-std shortly after warmup and reseed if collapsed.
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    for attempt in range(MAX_INIT_TRIES):
        seed = SEED + attempt
        torch.manual_seed(seed)
        np.random.seed(seed)
        writer = SummaryWriter(log_dir=str(TB_DIR / f"tom3_k{k}_{stamp}_seed{seed}"))
        encoder = BeliefEncoder(**ARCH, device=DEVICE).to(DEVICE)
        desire_head = torch.nn.Linear(ARCH["layer_size"], k).to(DEVICE)
        # The analytic room-inference rule only needs to know whether the
        # demonstrator wants gems.  Keep the full K-way head for the ToM
        # desire report, but train this easier, directly relevant auxiliary
        # classifier so K=4 does not bottleneck on distinguishing every pair
        # of non-gem preferences.
        gem_head = torch.nn.Linear(ARCH["layer_size"], 2).to(DEVICE)
        latent_head = torch.nn.Linear(ARCH["layer_size"], k).to(DEVICE)
        params = (
            list(encoder.parameters())
            + list(desire_head.parameters())
            + list(gem_head.parameters())
            + list(latent_head.parameters())
        )
        opt = torch.optim.Adam(params, lr=LR)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

        collapsed = False
        for step in range(steps):
            idx = np.random.choice(n_train, BATCH_SIZE, replace=False)
            frames, disp, desire, latent = batch_tensors(tr, idx)
            # A truncated window is useful augmentation for recognizing a
            # demonstrator's desire, but it can cut off the *door commit*.
            # The latent label is then genuinely ambiguous (particularly for
            # K > 2), so never train the latent head on that corrupted view.
            # Run the full trajectory for the latent loss and, when selected,
            # a second truncated pass for the desire loss.  The shared encoder
            # still learns from both objectives without being asked to infer a
            # room from evidence it was deliberately denied.
            desire_frames, desire_disp = frames, disp
            if step >= TRUNC_START and np.random.random() < TRUNC_PROB:
                t = int(np.random.randint(2, T_WATCH + 1))
                if t < T_WATCH:
                    desire_frames, desire_disp = truncate(frames, disp, t)
            if step >= TRUNC_START and variant == "visibility" and np.random.random() < VISIBILITY_PROB:
                t = int(np.random.randint(VISIBILITY_MIN_STEPS, T_WATCH + 1))
                if t < T_WATCH:
                    desire_frames, desire_disp = mask_tracked_suffix(
                        desire_frames, desire_disp, t
                    )
            g = encoder(frames, disp)
            g_desire = (
                g
                if desire_frames is frames
                else encoder(desire_frames, desire_disp)
            )
            is_gem = (desire == 0).long()
            loss = (
                F.cross_entropy(desire_head(g_desire), desire)
                + F.cross_entropy(gem_head(g_desire), is_gem)
                + F.cross_entropy(latent_head(g), latent)
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            if step % TB_EVERY == 0 or step == steps - 1:
                writer.add_scalar("train/loss", float(loss.detach()), step)
                writer.add_scalar("train/lr", sched.get_last_lr()[0], step)
                writer.add_scalar("train/g_std", float(g.std(0).mean().detach()), step)
            if step in ESCAPE_CHECKS:
                g_std = float(g.std(0).mean().detach())
                if g_std < ESCAPE_CHECKS[step]:
                    collapsed = True
                    print(
                        f"g-std collapsed/marginal at step {step} ({g_std:.4f} < "
                        f"{ESCAPE_CHECKS[step]}); retrying with seed {seed + 1}"
                    )
                    break
            if step % 250 == 0 or step == steps - 1:
                g_std = float(g.std(0).mean().detach())
                d_acc, l_acc = evaluate(encoder, desire_head, latent_head, va)
                writer.add_scalar("val/desire_acc", d_acc, step)
                writer.add_scalar("val/latent_acc", l_acc, step)
                print(
                    f"step {step:>5d}: loss={float(loss.detach()):.4f}  g-std={g_std:.4f}  "
                    f"val desire={d_acc * 100:.1f}%  val latent={l_acc * 100:.1f}%"
                )
        if not collapsed:
            break
        writer.close()
    else:
        raise RuntimeError(f"K={k}: encoder collapsed in all {MAX_INIT_TRIES} tries")

    print("\nAccuracy vs watch length (val):")
    print(f"  {'t':>4}{'desire':>10}{'latent':>10}")
    for t in range(2, T_WATCH + 1, 2):
        d_acc, l_acc = evaluate(
            encoder, desire_head, latent_head, va, t=t if t < T_WATCH else None
        )
        writer.add_scalar("watch_len/desire_acc", d_acc, t)
        writer.add_scalar("watch_len/latent_acc", l_acc, t)
        print(f"  {t:>4}{d_acc * 100:>9.1f}%{l_acc * 100:>9.1f}%")
    writer.close()

    ckpt = observer_ckpt(k) if variant == "baseline" else visibility_observer_ckpt(k)
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "encoder": encoder.state_dict(),
            "desire_head": desire_head.state_dict(),
            "gem_head": gem_head.state_dict(),
            "latent_head": latent_head.state_dict(),
            "arch": ARCH,
            "k": k,
            "variant": variant,
        },
        ckpt,
    )
    print(f"\nsaved -> {ckpt}\n")


def main():
    ks = [int(a) for a in sys.argv[1:]] or [2, 3, 4]
    for k in ks:
        train(k)


if __name__ == "__main__":
    main()
