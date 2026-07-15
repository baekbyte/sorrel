"""Oracle ablations: localize WHERE the belief pipeline fails, without retraining.

Three diagnostics, run in the current (non-segregated) world:

  A. PROTOTYPE g's. Encode many windows per preference class from the training
     buffers; the class-mean g is a "ground-truth preference" embedding. Also
     reports the cosine between class means (how separated preference-space is).

  B. DECODE-PATH PROBE. Take identical observer-style masked contexts and run
     the frozen base with g_gem vs g_food prototypes (and no g). If the
     outer-ring reconstruction composition barely moves between the two g's,
     the g -> spatial-reconstruction path is the bottleneck regardless of how
     good online inference is.

  C. ROLLOUT CELLS.
       Ceiling (watched=none):    oracle_world vs masked vs imagine.
         oracle_world - masked = the information value of the outer ring; the
         hard ceiling on what ANY belief mechanism could add in this world.
       Localization (gem_only / food_only): belief vs oracle_g vs masked.
         oracle_g >> belief  -> online inference (FOV gating, partial windows)
                                is the weak link.
         oracle_g ~= belief  -> the decode path (or the world itself) is.

Usage:
    python -m sorrel.examples.treasurehunt.notebooks.tom_oracles
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from sorrel.buffers import TransformerBuffer
from sorrel.examples.treasurehunt.notebooks.tom_experiment import run_trial
from sorrel.examples.treasurehunt.notebooks.tom_rollout import (
    ARCH,
    BELIEF_PATH,
    ENTITY_LIST,
    NUM_ENTITY_CHANNELS,
    OUTER_RING_2D,
    SELF_MODEL_PATH,
    add_mask_channel,
    add_zero_mask_channel,
)
from sorrel.models.pytorch.transformer import BeliefEncoder, ViTOneHot

DATA_DIR = Path(__file__).parent / "../data"
GEM_BUFFER = DATA_DIR / "memories/gemlover.npz"
FOOD_BUFFER = DATA_DIR / "memories/foodlover.npz"
PROTO_G_PATH = DATA_DIR / "checkpoints/proto_g.pt"

SEED = 0
N_PROTO_WINDOWS = 512   # windows per class for the prototype
N_PROBE_CONTEXTS = 256  # masked contexts for the decode-path probe
N_TRIALS = 10           # rollout trials per cell

C, H, W = ARCH["state_size"]
T = ARCH["num_frames"]

torch.manual_seed(SEED)
np.random.seed(SEED)


# --------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------


def load_models() -> tuple[ViTOneHot, BeliefEncoder]:
    dummy_mem = TransformerBuffer(
        capacity=10, obs_shape=(int(np.prod(ARCH["state_size"])),), n_frames=T
    )
    base = ViTOneHot(
        memory=dummy_mem, LR=1e-3, device="cpu", seed=SEED, reward_loss_weight=0.0, **ARCH
    )
    base.load(str(SELF_MODEL_PATH))
    base.eval()
    encoder = BeliefEncoder(
        state_size=ARCH["state_size"],
        action_space=ARCH["action_space"],
        layer_size=ARCH["layer_size"],
        patch_size=ARCH["patch_size"],
        num_frames=T,
        num_heads=ARCH["num_heads"],
        num_layers=ARCH["num_layers"],
        device="cpu",
    )
    ckpt = torch.load(BELIEF_PATH, map_location="cpu")
    encoder.load_state_dict(ckpt["belief_encoder"])
    encoder.eval()
    return base, encoder


def sample_windows(buf: TransformerBuffer, n: int) -> tuple[torch.Tensor, torch.Tensor]:
    """(n, T, 6, H, W) states + (n, T, 1) actions from one preference buffer.
    Mirrors train_belief.sample_window (idx-based real size)."""
    real_size = max(1, int(buf.idx) - T - 1)
    base_idx = np.random.choice(real_size, n, replace=False)
    indices = base_idx[:, None] + np.arange(T)
    s = torch.tensor(
        buf.states[indices], dtype=torch.float32
    ).view(n, T, NUM_ENTITY_CHANNELS, H, W)
    a = torch.tensor(buf.actions[indices], dtype=torch.long).view(n, T, 1)
    return s, a


def encode_all(encoder: BeliefEncoder, s6: torch.Tensor, a: torch.Tensor, bs: int = 64) -> torch.Tensor:
    """Encoder input is the watched agent's full view + zero mask channel,
    exactly as in train_belief.py."""
    gs = []
    with torch.no_grad():
        for i in range(0, s6.size(0), bs):
            gs.append(encoder(add_zero_mask_channel(s6[i : i + bs]), a[i : i + bs]))
    return torch.cat(gs)


# --------------------------------------------------------------------------
# A. Prototype g's
# --------------------------------------------------------------------------


def compute_prototypes(encoder: BeliefEncoder, gem_buf, food_buf) -> dict[str, torch.Tensor]:
    print("=" * 80)
    print("A. PROTOTYPE g's (class-mean belief embeddings from training buffers)")
    print("=" * 80)
    protos = {}
    per_class_g = {}
    for label, buf in [("gemlover", gem_buf), ("foodlover", food_buf)]:
        s6, a = sample_windows(buf, N_PROTO_WINDOWS)
        g = encode_all(encoder, s6, a)  # (N, layer_size)
        mean_g = g.mean(dim=0)
        # Rescale the mean back to the typical single-window norm: the decode
        # path has only ever seen g's of that magnitude, and averaging shrinks
        # the norm when directions vary.
        proto = mean_g / (mean_g.norm() + 1e-8) * g.norm(dim=1).mean()
        protos[label] = proto.unsqueeze(0)  # (1, layer_size) like online g
        per_class_g[label] = g
        print(
            f"  {label}: mean-g norm={mean_g.norm():.3f}, "
            f"typical single-g norm={g.norm(dim=1).mean():.3f} "
            f"(prototype rescaled to the latter)"
        )
    cos = F.cosine_similarity(protos["gemlover"], protos["foodlover"]).item()
    print(f"  cosine(gem prototype, food prototype) = {cos:+.4f}  (near +1 = classes barely separated)")
    # Within- vs between-class dispersion: how much of g-space is preference?
    for label, g in per_class_g.items():
        within = F.cosine_similarity(g, per_class_g[label].mean(0, keepdim=True)).mean()
        print(f"  {label}: mean cosine(window g, class mean) = {within:.4f}")
    PROTO_G_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(protos, PROTO_G_PATH)
    print(f"  saved -> {PROTO_G_PATH}")
    return protos


# --------------------------------------------------------------------------
# B. Decode-path probe
# --------------------------------------------------------------------------


def ring_composition(pos_last: torch.Tensor, ring_bool: torch.Tensor) -> dict[str, float]:
    """pos_last: (B, H, W, C) positive-class probs. Returns argmax-fraction per
    entity channel over outer-ring cells, aggregated across the batch."""
    entity_pos = pos_last[..., :NUM_ENTITY_CHANNELS]  # (B, H, W, 6)
    ch = entity_pos.argmax(dim=-1)                    # (B, H, W)
    ring_ch = ch[:, ring_bool]                        # (B, n_ring)
    total = ring_ch.numel()
    return {name: float((ring_ch == k).sum()) / total for k, name in enumerate(ENTITY_LIST)}


def decode_path_probe(base: ViTOneHot, protos: dict[str, torch.Tensor], gem_buf, food_buf):
    print("\n" + "=" * 80)
    print("B. DECODE-PATH PROBE (same masked contexts, swap the injected g)")
    print("=" * 80)
    # Contexts: half from each buffer, outer-ring-masked like the observer's input.
    half = N_PROBE_CONTEXTS // 2
    s_gem, a_gem = sample_windows(gem_buf, half)
    s_food, a_food = sample_windows(food_buf, half)
    s6 = torch.cat([s_gem, s_food])
    a = torch.cat([a_gem, a_food])
    base_input = add_mask_channel(s6, OUTER_RING_2D)  # (B, T, 7, H, W)
    ring_bool = OUTER_RING_2D.bool()

    results = {}
    gem_probs, food_probs = {}, {}
    for gname, g in [("g_gem", protos["gemlover"]), ("g_food", protos["foodlover"]), ("no_g", None)]:
        pos_chunks = []
        with torch.no_grad():
            for i in range(0, s6.size(0), 64):
                bi, ai = base_input[i : i + 64], a[i : i + 64]
                gb = g.expand(bi.size(0), -1) if g is not None else None
                preds, _ = base.forward(bi, ai, belief_embedding=gb)
                pos_chunks.append(F.softmax(preds, dim=2)[:, :, 1][:, -1])  # (b, H, W, C)
        pos_last = torch.cat(pos_chunks)
        results[gname] = ring_composition(pos_last, ring_bool)
        gem_probs[gname] = float(pos_last[..., 2][:, ring_bool].mean())   # Gem channel
        food_probs[gname] = float(pos_last[..., 4][:, ring_bool].mean())  # Food channel

    hdr = "".join(f"{n:>14}" for n in ENTITY_LIST)
    print(f"  outer-ring argmax composition{'':<6}{hdr}")
    for gname, comp in results.items():
        row = "".join(f"{comp[n] * 100:>13.1f}%" for n in ENTITY_LIST)
        print(f"    {gname:<32}{row}")
    print("\n  mean positive-prob in outer ring (soft, pre-argmax):")
    print(f"    {'':<12}{'P(Gem)':>10}{'P(Food)':>10}")
    for gname in results:
        print(f"    {gname:<12}{gem_probs[gname]:>10.4f}{food_probs[gname]:>10.4f}")
    d_gem = gem_probs["g_gem"] - gem_probs["g_food"]
    d_food = food_probs["g_food"] - food_probs["g_gem"]
    print(
        f"\n  SELECTIVITY: dP(Gem | g_gem - g_food) = {d_gem:+.4f}, "
        f"dP(Food | g_food - g_gem) = {d_food:+.4f}"
    )
    print("  (both ~0 -> the g -> reconstruction path is preference-blind: decode bottleneck)")


# --------------------------------------------------------------------------
# C. Rollout cells
# --------------------------------------------------------------------------


def run_cell(watched: str, mode: str, g_override=None) -> dict[str, float]:
    wr, gems, foods, bones = [], [], [], []
    for trial in range(N_TRIALS):
        res = run_trial(
            watched, belief_on=True, seed=100 + trial, save_gif=False,
            mode=mode, g_override=g_override,
        )
        wr.append(res["weighted_reward"])
        gems.append(res["collected"]["Gem"])
        foods.append(res["collected"]["Food"])
        bones.append(res["collected"]["Bone"])
    return {
        "wr": float(np.mean(wr)), "wr_std": float(np.std(wr)),
        "gems": float(np.mean(gems)), "food": float(np.mean(foods)),
        "bones": float(np.mean(bones)),
    }


def rollout_cells(protos: dict[str, torch.Tensor]):
    print("\n" + "=" * 80)
    print(f"C. ROLLOUT CELLS ({N_TRIALS} trials each, gem-pref weighted reward)")
    print("=" * 80)

    def show(tag: str, r: dict[str, float]):
        print(
            f"    {tag:<28} wR={r['wr']:+7.2f} ± {r['wr_std']:<6.2f} "
            f"gems={r['gems']:.1f} food={r['food']:.1f} bones={r['bones']:.1f}"
        )

    print("\n  CEILING (watched=none): how much is the outer ring even worth?")
    ceiling = {m: run_cell("none", m) for m in ["oracle_world", "masked", "imagine"]}
    for m, r in ceiling.items():
        show(m, r)
    print(
        f"    -> ceiling (oracle_world - masked) = "
        f"{ceiling['oracle_world']['wr'] - ceiling['masked']['wr']:+.2f} "
        f"(max any belief could add); "
        f"imagination cost (imagine - masked) = "
        f"{ceiling['imagine']['wr'] - ceiling['masked']['wr']:+.2f}"
    )

    print("\n  LOCALIZATION (inference vs decode):")
    for watched in ["gem_only", "food_only"]:
        print(f"  watched={watched}:")
        cells = {
            "belief (encoded g)": run_cell(watched, "belief"),
            "oracle_g (prototype g)": run_cell(watched, "oracle_g", g_override=protos),
            "masked (no completion)": run_cell(watched, "masked"),
        }
        for tag, r in cells.items():
            show(tag, r)
        print(
            f"    -> oracle_g - belief = "
            f"{cells['oracle_g (prototype g)']['wr'] - cells['belief (encoded g)']['wr']:+.2f} "
            f"(large -> inference path weak; ~0 -> decode path/world is the limit)"
        )


def main():
    base, encoder = load_models()
    gem_buf = TransformerBuffer.load(GEM_BUFFER)
    food_buf = TransformerBuffer.load(FOOD_BUFFER)
    protos = compute_prototypes(encoder, gem_buf, food_buf)
    decode_path_probe(base, protos, gem_buf, food_buf)
    rollout_cells(protos)


if __name__ == "__main__":
    main()
