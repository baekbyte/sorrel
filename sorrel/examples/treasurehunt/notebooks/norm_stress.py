"""Stress tests for the frozen ToM norm decision rule.

Tests two failure modes that the headline table does not address directly:

* noisy demonstrator behavior, which can degrade desire inference; and
* multiple present partners, which requires binding and aggregating their
  separate inferred preferences rather than protecting one memorized kind.

Usage::

    python -m sorrel.examples.treasurehunt.notebooks.norm_stress 3
    python -m sorrel.examples.treasurehunt.notebooks.norm_stress 3 --episodes 25
"""

import argparse
import csv
from dataclasses import dataclass

import numpy as np

from sorrel.examples.treasurehunt.notebooks.norm_common import (
    ITEM_KINDS,
    forage_step,
    make_norm_env,
    sanction_probabilities,
)
from sorrel.examples.treasurehunt.notebooks.norm_evaluate import N_FORAGE_STEPS, SANCTION
from sorrel.examples.treasurehunt.notebooks.tom3_common import DATA_DIR, make_obs_spec, run_watch_phase
from sorrel.examples.treasurehunt.notebooks.tom3_evaluate import infer, load_observer

EPSILONS = (0.1, 0.2, 0.3)
CONDITIONS = (("single_coin", (2,)), ("gem_and_coin", (0, 2)))
MODES = ("tom", "no_tom", "oracle")
SEED0 = 600_000


@dataclass
class EpisodeResult:
    protected_available: int
    protected_eaten: int
    permitted_available: int
    permitted_eaten: int
    sanctions: int
    reward: float
    desire_correct: int
    desire_total: int


def posterior_for(mode: str, k: int, desires: tuple[int, ...], desire_probs: np.ndarray) -> np.ndarray:
    if mode == "tom":
        probs = desire_probs
    elif mode == "no_tom":
        probs = np.full((len(desires), k), 1.0 / k)
    elif mode == "oracle":
        probs = np.eye(k)[list(desires)]
    else:
        raise ValueError(mode)
    return sanction_probabilities(probs)


def run_episode(k: int, desires: tuple[int, ...], epsilon: float, mode: str, seed: int, models) -> EpisodeResult:
    encoder, desire_head, gem_head, latent_head = models
    env = make_norm_env(k, list(desires), seed, epsilon=epsilon, sanction=SANCTION)
    frames, disp, marks, _, _ = run_watch_phase(env, make_obs_spec())
    _, predicted, desire_probs, _ = infer(
        k, encoder, desire_head, gem_head, latent_head, frames, disp, marks
    )
    posterior = posterior_for(mode, k, desires, desire_probs)
    available = env.count_items()
    for _ in range(N_FORAGE_STEPS):
        forage_step(env, posterior)
    assert env.norm_log is not None
    protected = {ITEM_KINDS[d] for d in desires}
    protected_available = sum(available[kind] for kind in protected)
    protected_eaten = sum(env.norm_log.consumed.get(kind, 0) for kind in protected)
    permitted_available = sum(count for kind, count in available.items() if kind not in protected)
    permitted_eaten = sum(
        env.norm_log.consumed.get(kind, 0) for kind in available if kind not in protected
    )
    return EpisodeResult(
        protected_available,
        protected_eaten,
        permitted_available,
        permitted_eaten,
        env.norm_log.sanctions,
        env.norm_log.total_reward,
        sum(p == d for p, d in zip(predicted, desires)),
        len(desires),
    )


def evaluate(k: int, episodes: int) -> list[dict[str, object]]:
    models = load_observer(k)
    rows: list[dict[str, object]] = []
    for ci, (condition, desires) in enumerate(CONDITIONS):
        if max(desires) >= k:
            continue
        for ei, epsilon in enumerate(EPSILONS):
            for mi, mode in enumerate(MODES):
                results = [
                    run_episode(
                        k, desires, epsilon, mode,
                        SEED0 + 100_000 * ci + 10_000 * ei + 1_000 * mi + ep,
                        models,
                    )
                    for ep in range(episodes)
                ]
                protected_available = sum(r.protected_available for r in results)
                permitted_available = sum(r.permitted_available for r in results)
                rows.append(
                    {
                        "condition": condition,
                        "partners": "+".join(ITEM_KINDS[d] for d in desires),
                        "epsilon": epsilon,
                        "policy": mode,
                        "violation": sum(r.protected_eaten for r in results) / max(1, protected_available),
                        "permitted_eating": sum(r.permitted_eaten for r in results) / max(1, permitted_available),
                        "sanctions_per_episode": sum(r.sanctions for r in results) / episodes,
                        "mean_return": sum(r.reward for r in results) / episodes,
                        "desire_accuracy": sum(r.desire_correct for r in results) / max(1, sum(r.desire_total for r in results)),
                        "episodes": episodes,
                    }
                )
    return rows


def write_csv(rows: list[dict[str, object]], k: int) -> None:
    path = DATA_DIR / "reports" / f"norm_stress_k{k}.csv"
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
    parser.add_argument("--no-csv", action="store_true")
    args = parser.parse_args()
    if args.k < 3:
        raise SystemExit("The configured stress conditions require K >= 3.")
    rows = evaluate(args.k, args.episodes)
    print(f"K={args.k}; {args.episodes} episodes/cell; S={SANCTION:g}; forage={N_FORAGE_STEPS} steps")
    print("\ncondition       eps  policy      violation  permitted eat  sanctions  return  desire acc")
    for row in rows:
        print(
            f"{row['condition']:<16}{row['epsilon']:>4.1f}  {row['policy']:<10}"
            f"{row['violation']:>9.3f}{row['permitted_eating']:>15.3f}"
            f"{row['sanctions_per_episode']:>11.2f}{row['mean_return']:>8.2f}"
            f"{row['desire_accuracy']:>12.3f}"
        )
    if not args.no_csv:
        write_csv(rows, args.k)


if __name__ == "__main__":
    main()
