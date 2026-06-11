"""Diagnostic: how behaviorally distinguishable are the two preference agents?

Trains a small MLP to classify agent type (gem-lover vs food-lover) from raw
trajectory windows. This is the ceiling on any behavior-based goal inference: if
the two agents look the same over a window, there is no goal signal for the
belief module to recover. Used to decide whether an environment change (e.g.
spatial segregation) actually made goals legible before retraining the full
pipeline.

Usage:
    python -m sorrel.examples.treasurehunt.notebooks.check_separability
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from sorrel.buffers import TransformerBuffer
from sorrel.models.pytorch.transformer import ViTOneHot

DATA_DIR = Path(__file__).parent / "../data"
GEM = DATA_DIR / "memories/gemlover.npz"
FOOD = DATA_DIR / "memories/foodlover.npz"
N = 1500

ARCH = dict(
    state_size=(6, 15, 15),
    action_space=4,
    layer_size=192,
    patch_size=3,
    num_frames=5,
    num_heads=3,
    batch_size=256,
    num_layers=2,
)

torch.manual_seed(0)
np.random.seed(0)


def grab(buf, n):
    base = ViTOneHot(memory=buf, LR=1e-3, device="cpu", seed=0, **ARCH)
    S, A = [], []
    for _ in range(n // 256 + 1):
        s, a, _, _, _, _ = base.get_batch()
        S.append(s)
        A.append(a)
    return torch.cat(S)[:n], torch.cat(A)[:n]


gem = TransformerBuffer.load(GEM)
food = TransformerBuffer.load(FOOD)
sg, ag = grab(gem, N)
sf, af = grab(food, N)


def actfeat(a):
    a = a.squeeze(-1).long()
    return torch.stack([(a == k).float().mean(1) for k in range(4)], 1)


Xa = torch.cat([actfeat(ag), actfeat(af)])
Xs = torch.cat(
    [
        torch.cat([sg.flatten(1), ag.squeeze(-1).float()], 1),
        torch.cat([sf.flatten(1), af.squeeze(-1).float()], 1),
    ]
)
y = torch.cat([torch.zeros(N), torch.ones(N)]).long()


def probe(X, y, name):
    p = torch.randperm(X.size(0))
    X, y = X[p], y[p]
    sp = int(0.8 * X.size(0))
    clf = torch.nn.Sequential(
        torch.nn.Linear(X.size(1), 64), torch.nn.ReLU(), torch.nn.Linear(64, 2)
    )
    o = torch.optim.Adam(clf.parameters(), 1e-3)
    for _ in range(800):
        o.zero_grad()
        F.cross_entropy(clf(X[:sp]), y[:sp]).backward()
        o.step()
    acc = (clf(X[sp:]).argmax(1) == y[sp:]).float().mean().item()
    print(f"  {name}: {acc * 100:.1f}%")
    return acc


print("Agent-type classification from RAW trajectory (ceiling on goal inference):")
probe(Xa, y, "action-distribution only ")
probe(Xs, y, "full state+action window")
print("\n(Symmetric-env baseline was 57.5% / 67.2%. Higher = goals more legible.)")
