"""Diagnose multi-partner ToM failures before changing the model or policy.

This evaluates the Gem+Coin condition only.  It separates three questions:

1. Did the desire head identify each partner's preference?
2. Was the posterior confident enough for the sanction decision threshold?
3. Given the aggregate posterior, did the BFS rule consume a protected kind?

The script writes one row per episode and one row per watched partner, making
the multi-agent ToM gap inspectable rather than a single aggregate score.

Usage::

    python -m sorrel.examples.treasurehunt.notebooks.norm_multi_diagnostic 3
"""

import argparse
import csv
from pathlib import Path

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
DESIRES = (0, 2)  # Gem + Coin; the permitted kind at K=3 is Food.
SEED0 = 700_000


def calibration_rows(agent_rows: list[dict[str, object]], k: int) -> list[tuple[float, float, int]]:
    """Return (mean predicted probability, empirical frequency, count) per bin."""
    values: list[tuple[float, int]] = []
    for row in agent_rows:
        true = int(row["true_desire"])
        for kind in range(k):
            values.append((float(row[f"p_{ITEM_KINDS[kind].lower()}"]), int(kind == true)))
    output = []
    for lo in np.arange(0.0, 1.0, 0.2):
        subset = [(p, y) for p, y in values if lo <= p < lo + 0.2]
        if subset:
            output.append((np.mean([p for p, _ in subset]), np.mean([y for _, y in subset]), len(subset)))
    return output


def expected_calibration_error(agent_rows: list[dict[str, object]], k: int) -> float:
    bins = calibration_rows(agent_rows, k)
    total = sum(n for _, _, n in bins)
    return sum(abs(conf - empirical) * n / total for conf, empirical, n in bins)


def run(k: int, episodes: int, checkpoint: Path | None = None) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    models = load_observer(k, checkpoint)
    encoder, desire_head, gem_head, latent_head = models
    episode_rows: list[dict[str, object]] = []
    agent_rows: list[dict[str, object]] = []
    for epsilon_i, epsilon in enumerate(EPSILONS):
        for episode in range(episodes):
            seed = SEED0 + epsilon_i * 10_000 + episode
            env = make_norm_env(k, list(DESIRES), seed, epsilon=epsilon, sanction=SANCTION)
            frames, disp, marks, _, _ = run_watch_phase(env, make_obs_spec())
            _, predicted, desire_probs, _ = infer(
                k, encoder, desire_head, gem_head, latent_head, frames, disp, marks
            )
            aggregate = sanction_probabilities(desire_probs)
            available = env.count_items()
            protected_kinds = {ITEM_KINDS[d] for d in DESIRES}
            protected_available = sum(available[kind] for kind in protected_kinds)
            unprotected_true_kinds = [
                ITEM_KINDS[d] for d in DESIRES if aggregate[d] * SANCTION < 10.0
            ]
            for _ in range(N_FORAGE_STEPS):
                forage_step(env, aggregate)
            assert env.norm_log is not None
            protected_eaten = sum(
                env.norm_log.consumed.get(kind, 0) for kind in protected_kinds
            )
            permitted_kind = next(kind for kind in available if kind not in protected_kinds)
            episode_rows.append(
                {
                    "epsilon": epsilon,
                    "episode": episode,
                    "seed": seed,
                    "protected_available": protected_available,
                    "protected_eaten": protected_eaten,
                    "permitted_available": available[permitted_kind],
                    "permitted_eaten": env.norm_log.consumed.get(permitted_kind, 0),
                    "sanctions": env.norm_log.sanctions,
                    "return": env.norm_log.total_reward,
                    "unprotected_true_kinds": "+".join(unprotected_true_kinds) or "none",
                    **{f"aggregate_{ITEM_KINDS[i].lower()}": aggregate[i] for i in range(k)},
                }
            )
            for partner_i, desire in enumerate(DESIRES):
                probs = desire_probs[partner_i]
                true_prob = float(probs[desire])
                agent_rows.append(
                    {
                        "epsilon": epsilon,
                        "episode": episode,
                        "seed": seed,
                        "partner_index": partner_i,
                        "true_desire": desire,
                        "true_kind": ITEM_KINDS[desire],
                        "predicted_desire": predicted[partner_i],
                        "predicted_kind": ITEM_KINDS[predicted[partner_i]],
                        "correct_argmax": int(predicted[partner_i] == desire),
                        "true_probability": true_prob,
                        "correct_but_under_threshold": int(
                            predicted[partner_i] == desire and true_prob < 0.5
                        ),
                        "aggregate_protects_true_kind": int(aggregate[desire] >= 0.5),
                        **{f"p_{ITEM_KINDS[i].lower()}": probs[i] for i in range(k)},
                    }
                )
    return episode_rows, agent_rows


def print_summary(episode_rows: list[dict[str, object]], agent_rows: list[dict[str, object]], k: int) -> None:
    print("Gem+Coin multi-partner ToM diagnostic")
    print("\neps  desire acc  under-threshold  aggregate misses  violation  permitted eat  ECE")
    for epsilon in EPSILONS:
        eps_agents = [r for r in agent_rows if r["epsilon"] == epsilon]
        eps_episodes = [r for r in episode_rows if r["epsilon"] == epsilon]
        accuracy = np.mean([r["correct_argmax"] for r in eps_agents])
        under_threshold = np.mean([r["correct_but_under_threshold"] for r in eps_agents])
        aggregate_miss = np.mean([r["unprotected_true_kinds"] != "none" for r in eps_episodes])
        violation = sum(r["protected_eaten"] for r in eps_episodes) / max(1, sum(r["protected_available"] for r in eps_episodes))
        permitted = sum(r["permitted_eaten"] for r in eps_episodes) / max(1, sum(r["permitted_available"] for r in eps_episodes))
        ece = expected_calibration_error(eps_agents, k)
        print(f"{epsilon:>3.1f}{accuracy:>12.3f}{under_threshold:>17.3f}{aggregate_miss:>18.3f}{violation:>11.3f}{permitted:>15.3f}{ece:>6.3f}")
    print("\nCalibration (all partner × kind probabilities):")
    for confidence, empirical, n in calibration_rows(agent_rows, k):
        print(f"  mean predicted={confidence:.3f}  empirical={empirical:.3f}  n={n}")


def write_csv(rows: list[dict[str, object]], name: str, k: int) -> None:
    path = DATA_DIR / "reports" / f"norm_multi_{name}_k{k}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV: {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("k", nargs="?", type=int, default=3)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument(
        "--checkpoint", type=Path, default=None,
        help="Optional observer checkpoint; defaults to the validated baseline.",
    )
    parser.add_argument("--no-csv", action="store_true")
    args = parser.parse_args()
    if args.k < 3:
        raise SystemExit("The Gem+Coin diagnostic requires K >= 3.")
    episode_rows, agent_rows = run(args.k, args.episodes, args.checkpoint)
    print_summary(episode_rows, agent_rows, args.k)
    if not args.no_csv:
        write_csv(episode_rows, "episodes", args.k)
        write_csv(agent_rows, "agents", args.k)


if __name__ == "__main__":
    main()
