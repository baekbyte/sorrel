"""All-pairs and order-swap audit for multi-agent ToM binding.

Every ordered pair of distinct K=3 desires is evaluated with the same
third-person ToM input protocol.  Reversing an order swaps which preference
occupies each otherwise identical initial agent spawn, allowing us to test
whether errors are tied to a preference pair, an agent slot, or a location.

Usage::

    python -m sorrel.examples.treasurehunt.notebooks.norm_pair_sweep 3
"""

import argparse
import csv
from itertools import permutations
from pathlib import Path

import numpy as np

from sorrel.examples.treasurehunt.notebooks.norm_common import make_norm_env
from sorrel.examples.treasurehunt.notebooks.tom3_common import DATA_DIR, make_obs_spec, run_watch_phase
from sorrel.examples.treasurehunt.notebooks.tom3_evaluate import infer, load_observer

EPSILONS = (0.1, 0.3)
SEED0 = 800_000


def non_agent_layout(env) -> tuple[tuple[int, int, str], ...]:
    """Layout fingerprint used to assert true order-swap pairing."""
    return tuple(
        (y, x, entity.kind)
        for y in range(env.world.height)
        for x in range(env.world.width)
        for entity in (env.world.observe((y, x, 1)),)
        if entity.kind != "TreasurehuntAgent"
    )


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Two-sided Wilson interval for a binomial accuracy estimate."""
    if total == 0:
        return 0.0, 0.0
    p = successes / total
    denom = 1.0 + z**2 / total
    center = (p + z**2 / (2 * total)) / denom
    radius = z * np.sqrt((p * (1 - p) + z**2 / (4 * total)) / total) / denom
    return center - radius, center + radius


def run(
    k: int, episodes: int, epsilons: tuple[float, ...], checkpoint: Path | None = None,
) -> list[dict[str, object]]:
    """Run the binding audit for baseline or an explicitly supplied observer."""
    models = load_observer(k, checkpoint)
    encoder, desire_head, gem_head, latent_head = models
    rows: list[dict[str, object]] = []
    # Cache layout fingerprints for each unordered pair/seed/epsilon. Ordered
    # reversals must share the exact world, differing only in desire-to-agent
    # assignment.
    fingerprints: dict[tuple[float, tuple[int, int], int], tuple[tuple[int, int, str], ...]] = {}
    for epsilon_i, epsilon in enumerate(epsilons):
        for desires in permutations(range(k), 2):
            unordered = tuple(sorted(desires))
            for episode in range(episodes):
                # Order is deliberately excluded: a reversal gets the same
                # world and spawn locations for the same epsilon/pair/episode.
                seed = SEED0 + epsilon_i * 100_000 + unordered[0] * 10_000 + unordered[1] * 1_000 + episode
                env = make_norm_env(k, list(desires), seed, epsilon=epsilon)
                fingerprint = non_agent_layout(env)
                fp_key = (epsilon, unordered, episode)
                if fp_key in fingerprints and fingerprints[fp_key] != fingerprint:
                    raise AssertionError(f"layout mismatch for {fp_key}")
                fingerprints[fp_key] = fingerprint
                start_locations = [agent.location for agent in env.agents]
                frames, disp, marks, _, _ = run_watch_phase(env, make_obs_spec())
                _, predicted, desire_probs, _ = infer(
                    k, encoder, desire_head, gem_head, latent_head, frames, disp, marks
                )
                for slot, desire in enumerate(desires):
                    rows.append(
                        {
                            "epsilon": epsilon,
                            "episode": episode,
                            "seed": seed,
                            "ordered_pair": "+".join(str(d) for d in desires),
                            "pair": "+".join(str(d) for d in unordered),
                            "slot": slot,
                            "start_y": start_locations[slot][0],
                            "start_x": start_locations[slot][1],
                            "true_desire": desire,
                            "true_kind": ["Gem", "Food", "Coin", "Berry"][desire],
                            "predicted_desire": predicted[slot],
                            "correct": int(predicted[slot] == desire),
                            "true_probability": float(desire_probs[slot, desire]),
                        }
                    )
    return rows


def summarize(rows: list[dict[str, object]]) -> None:
    print("All-pairs, order-swap desire audit")
    print("\neps  ordered pair  slot  true kind  accuracy [95% CI]    mean P(true)  n")
    keys = sorted({(r["epsilon"], r["ordered_pair"], r["slot"], r["true_kind"]) for r in rows})
    for epsilon, pair, slot, kind in keys:
        subset = [r for r in rows if (r["epsilon"], r["ordered_pair"], r["slot"], r["true_kind"]) == (epsilon, pair, slot, kind)]
        successes = sum(r["correct"] for r in subset)
        accuracy = successes / len(subset)
        lo, hi = wilson_interval(successes, len(subset))
        confidence = np.mean([r["true_probability"] for r in subset])
        print(f"{epsilon:>3.1f}{pair:>14}{slot:>6}{kind:>11}{accuracy:>8.3f} [{lo:.3f}, {hi:.3f}]{confidence:>14.3f}{len(subset):>4}")
    print("\nAggregate by true kind and tracked-agent slot:")
    print("eps  true kind  slot  accuracy [95% CI]    mean P(true)  n")
    keys = sorted({(r["epsilon"], r["true_kind"], r["slot"]) for r in rows})
    for epsilon, kind, slot in keys:
        subset = [r for r in rows if (r["epsilon"], r["true_kind"], r["slot"]) == (epsilon, kind, slot)]
        successes = sum(r["correct"] for r in subset)
        accuracy = successes / len(subset)
        lo, hi = wilson_interval(successes, len(subset))
        print(f"{epsilon:>3.1f}{kind:>11}{slot:>6}{accuracy:>8.3f} [{lo:.3f}, {hi:.3f}]"
              f"{np.mean([r['true_probability'] for r in subset]):>14.3f}{len(subset):>4}")


def write_csv(rows: list[dict[str, object]], k: int, episodes: int, epsilons: tuple[float, ...]) -> None:
    suffix = "-".join(f"e{epsilon:.1f}" for epsilon in epsilons)
    path = DATA_DIR / "reports" / f"norm_pair_sweep_k{k}_{episodes}ep_{suffix}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nCSV: {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("k", nargs="?", type=int, default=3)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--epsilons", nargs="+", type=float, default=EPSILONS)
    parser.add_argument(
        "--checkpoint", type=Path, default=None,
        help="Optional observer checkpoint; defaults to the validated baseline.",
    )
    parser.add_argument("--no-csv", action="store_true")
    args = parser.parse_args()
    if not 2 <= args.k <= 4:
        raise SystemExit("K must be in {2, 3, 4}")
    epsilons = tuple(args.epsilons)
    if any(epsilon < 0.0 or epsilon > 1.0 for epsilon in epsilons):
        raise SystemExit("epsilon values must lie in [0, 1]")
    rows = run(args.k, args.episodes, epsilons, args.checkpoint)
    summarize(rows)
    if not args.no_csv:
        write_csv(rows, args.k, args.episodes, epsilons)


if __name__ == "__main__":
    main()
