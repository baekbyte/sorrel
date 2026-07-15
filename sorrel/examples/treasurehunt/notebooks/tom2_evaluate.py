"""ToM v2 behavioral evaluation: door choice from inferred belief.

The observer watches for T_WATCH steps, the trained encoder turns each watched
agent's (frames, displacements) into a posterior over the latent (multi-agent:
sum of per-agent log-odds), and a decision rule on the observer's OWN gem
preference picks a door: choose left iff P(gems left) > 0.5 (EV-equivalent for
gem 10 vs food 2). No trained policy — the learned component is the inference;
the act is a readout of it.

Experiments:
  1. Door-choice table: watched {gem_only, food_only, both, none}
     x belief {inferred, oracle, uniform}. Prediction: inferred ~ oracle >>
     uniform = 0.5; none = 0.5 by construction.
  2. Same-direction contrast (headline): among episodes where the watched
     agent went LEFT, the observer must FOLLOW a gem-lover but MIRROR a
     food-lover. Impossible for a mimic.
  3. Belief-head calibration.

Usage:
    python -m sorrel.examples.treasurehunt.notebooks.tom2_evaluate
"""

import numpy as np
import torch

from sorrel.examples.treasurehunt.notebooks.tom2_common import (
    OBSERVER_CKPT,
    T_WATCH,
    agent_window,
    make_env,
    make_obs_spec,
    run_watch_phase,
)
from sorrel.models.pytorch.transformer import BeliefEncoder

N_EPISODES = 300
EPSILON = 0.1
SEED0 = 100_000  # disjoint from training seeds

rng = np.random.default_rng(0)


def load_observer():
    ckpt = torch.load(OBSERVER_CKPT, map_location="cpu")
    encoder = BeliefEncoder(**ckpt["arch"], device="cpu")
    encoder.load_state_dict(ckpt["encoder"])
    encoder.eval()
    desire_head = torch.nn.Linear(ckpt["arch"]["layer_size"], 2)
    desire_head.load_state_dict(ckpt["desire_head"])
    latent_head = torch.nn.Linear(ckpt["arch"]["layer_size"], 2)
    latent_head.load_state_dict(ckpt["latent_head"])
    return encoder, desire_head, latent_head


def infer(encoder, desire_head, latent_head, frames, disp_by_agent, marks):
    """Per-agent desire predictions + combined P(gems left) via log-odds sum."""
    log_odds_left = 0.0
    desires = []
    with torch.no_grad():
        for i, disp in disp_by_agent.items():
            window = agent_window(frames, marks[i])  # (T, 7, H, W)
            f = torch.tensor(window, dtype=torch.float32).unsqueeze(0)
            a = torch.tensor(disp, dtype=torch.long).view(1, -1, 1)
            g = encoder(f, a)
            desires.append(int(desire_head(g).argmax(1)))
            logits = latent_head(g)[0]
            log_odds_left += float(logits[0] - logits[1])
    p_left = 1.0 / (1.0 + np.exp(-log_odds_left)) if disp_by_agent else 0.5
    return p_left, desires


def choose_door(p_left: float) -> str:
    if p_left == 0.5:
        return "left" if rng.random() < 0.5 else "right"
    return "left" if p_left > 0.5 else "right"


def main():
    encoder, desire_head, latent_head = load_observer()
    obs_spec = make_obs_spec()

    table: dict[tuple[str, str], list[int]] = {}
    contrast: dict[tuple[str, str], list[int]] = {}  # (desire, watched_dir) -> [went_left]
    calib: list[tuple[float, int]] = []
    desire_correct = desire_total = 0

    seed = SEED0
    for wc in ["gem_only", "food_only", "both", "none"]:
        for _ in range(N_EPISODES):
            env = make_env(wc, seed=seed, epsilon=EPSILON)
            seed += 1
            frames, disp, marks, door_side, _ = run_watch_phase(env, obs_spec)
            gem_side = env.world.gem_side

            p_left, desires = infer(
                encoder, desire_head, latent_head, frames, disp, marks
            )
            for i, agent in enumerate(env.agents):
                truth = 0 if agent.target_kind == "Gem" else 1
                desire_correct += desires[i] == truth
                desire_total += 1

            for belief, p in [
                ("inferred", p_left),
                ("oracle", 1.0 if gem_side == "left" else 0.0),
                ("uniform", 0.5),
            ]:
                door = choose_door(p)
                table.setdefault((wc, belief), []).append(int(door == gem_side))

            if wc != "none":
                calib.append((p_left, int(gem_side == "left")))

            # Same-direction contrast: single-watched configs only.
            if wc in ("gem_only", "food_only") and door_side[0] is not None:
                d = "gem-lover" if wc == "gem_only" else "food-lover"
                door = choose_door(p_left)
                contrast.setdefault((d, door_side[0]), []).append(
                    int(door == "left")
                )

    print("=" * 76)
    print(f"1. DOOR-CHOICE TABLE ({N_EPISODES} episodes/cell, P(correct door))")
    print("=" * 76)
    print(f"{'watched':<12}{'inferred':>12}{'oracle':>12}{'uniform':>12}")
    for wc in ["gem_only", "food_only", "both", "none"]:
        row = [np.mean(table[(wc, b)]) for b in ["inferred", "oracle", "uniform"]]
        print(f"{wc:<12}{row[0]:>12.3f}{row[1]:>12.3f}{row[2]:>12.3f}")

    print(f"\ndesire accuracy over eval episodes: {desire_correct / max(1, desire_total):.3f}")

    print("\n" + "=" * 76)
    print("2. SAME-DIRECTION CONTRAST — P(observer goes LEFT | watched went dir)")
    print("   (follow the gem-lover; mirror the food-lover — a mimic cannot split these)")
    print("=" * 76)
    for d in ["gem-lover", "food-lover"]:
        for direction in ["left", "right"]:
            vals = contrast.get((d, direction), [])
            if vals:
                print(
                    f"  watched {d:<10} went {direction:<5} (n={len(vals):>3}): "
                    f"P(observer->left) = {np.mean(vals):.3f}"
                )

    print("\n" + "=" * 76)
    print("3. BELIEF CALIBRATION (inferred P(left) vs empirical)")
    print("=" * 76)
    ps = np.array([c[0] for c in calib])
    ys = np.array([c[1] for c in calib])
    for lo in np.arange(0, 1.0, 0.2):
        m = (ps >= lo) & (ps < lo + 0.2)
        if m.sum() > 0:
            print(
                f"  P(left) in [{lo:.1f}, {lo + 0.2:.1f}): n={m.sum():>4}, "
                f"empirical={ys[m].mean():.3f}"
            )


if __name__ == "__main__":
    main()
