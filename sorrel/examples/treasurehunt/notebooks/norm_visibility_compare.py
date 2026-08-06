"""Paired baseline-versus-visibility evaluation for the ToM norm policy.

Both observers see exactly the same seeded watch/forage episodes.  The
``hidden_after_8`` condition removes only each tracked partner's marker and
displacement after turn eight, while keeping the shared scene dynamic.  It is
therefore the deployment counterpart of the visibility augmentation used to
train ``tom3_observer_visibility``.

Usage::

    python -m sorrel.examples.treasurehunt.notebooks.norm_visibility_compare 3
    python -m sorrel.examples.treasurehunt.notebooks.norm_visibility_compare 3 --episodes 25
"""

import argparse
import csv

import numpy as np

from sorrel.examples.treasurehunt.notebooks.norm_common import (
    ITEM_KINDS,
    forage_step,
    make_norm_env,
    sanction_probabilities,
)
from sorrel.examples.treasurehunt.notebooks.norm_evaluate import N_FORAGE_STEPS, SANCTION
from sorrel.examples.treasurehunt.notebooks.tom3_common import (
    DATA_DIR,
    DISP_STAY,
    make_obs_spec,
    observer_ckpt,
    run_watch_phase,
    visibility_observer_ckpt,
)
from sorrel.examples.treasurehunt.notebooks.tom3_evaluate import infer, load_observer


SCENARIOS = (
    ("G0 gem", (0,)),
    ("G0 food", (1,)),
    ("G1 held-out coin", (2,)),
    ("multi gem+coin", (0, 2)),
)
EVIDENCE = (("full", 20), ("truncated_8", 8), ("hidden_after_8", 8))
EPSILONS = (0.1, 0.3)
SEED0 = 950_000
N_BOOTSTRAP = 2_000


def corrupt_evidence(frames, disp, marks, evidence: str, watch_steps: int):
    """Apply a defined evidence limitation without modifying the environment."""
    if evidence == "full":
        return frames, disp, marks
    disp = {i: values.copy() for i, values in disp.items()}
    marks = {i: values.copy() for i, values in marks.items()}
    if evidence == "truncated_8":
        frames = frames.copy()
        frames[watch_steps:] = frames[watch_steps - 1]
        for i in disp:
            disp[i][watch_steps:] = DISP_STAY
            marks[i][watch_steps:] = marks[i][watch_steps - 1]
    elif evidence == "hidden_after_8":
        for i in disp:
            disp[i][watch_steps:] = DISP_STAY
            marks[i][watch_steps:] = 0.0
    else:
        raise ValueError(f"Unknown evidence condition: {evidence}")
    return frames, disp, marks


def evaluate_episode(k, desires, epsilon, evidence, watch_steps, seed, models):
    env = make_norm_env(k, list(desires), seed, epsilon=epsilon, sanction=SANCTION)
    frames, disp, marks, _, _ = run_watch_phase(env, make_obs_spec())
    frames, disp, marks = corrupt_evidence(frames, disp, marks, evidence, watch_steps)
    encoder, desire_head, gem_head, latent_head = models
    _, predicted, desire_probs, _ = infer(
        k, encoder, desire_head, gem_head, latent_head, frames, disp, marks
    )
    posterior = sanction_probabilities(desire_probs)
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
    return {
        "protected_available": protected_available,
        "protected_eaten": protected_eaten,
        "permitted_available": permitted_available,
        "permitted_eaten": permitted_eaten,
        "sanctions": env.norm_log.sanctions,
        "return": env.norm_log.total_reward,
        "desire_correct": sum(p == d for p, d in zip(predicted, desires)),
        "desire_total": len(desires),
    }


def summarize(results: list[dict[str, float]]) -> dict[str, float]:
    """Aggregate episode-level outcomes into the reported policy metrics."""
    return {
        "violation": sum(r["protected_eaten"] for r in results) /
        max(1, sum(r["protected_available"] for r in results)),
        "permitted_eating": sum(r["permitted_eaten"] for r in results) /
        max(1, sum(r["permitted_available"] for r in results)),
        "sanctions_per_episode": float(np.mean([r["sanctions"] for r in results])),
        "mean_return": float(np.mean([r["return"] for r in results])),
        "desire_accuracy": sum(r["desire_correct"] for r in results) /
        sum(r["desire_total"] for r in results),
    }


def paired_intervals(
    baseline: list[dict[str, float]], visibility: list[dict[str, float]], seed: int,
) -> dict[str, float]:
    """95% percentile bootstrap intervals for visibility minus baseline.

    Episodes are resampled as matched pairs, preserving the common seeded
    layouts and demonstrator trajectories used by the comparison.
    """
    if len(baseline) != len(visibility):
        raise ValueError("Paired bootstrap requires equally many episodes.")
    rng = np.random.default_rng(seed)
    n = len(baseline)
    index = rng.integers(0, n, size=(N_BOOTSTRAP, n))
    output: dict[str, float] = {}
    specs = {
        "violation": ("protected_eaten", "protected_available"),
        "permitted_eating": ("permitted_eaten", "permitted_available"),
        "sanctions_per_episode": ("sanctions", None),
        "mean_return": ("return", None),
        "desire_accuracy": ("desire_correct", "desire_total"),
    }
    for metric, (numerator, denominator) in specs.items():
        base_num = np.array([r[numerator] for r in baseline], dtype=float)
        vis_num = np.array([r[numerator] for r in visibility], dtype=float)
        if denominator is None:
            samples = vis_num[index].mean(1) - base_num[index].mean(1)
        else:
            base_den = np.array([r[denominator] for r in baseline], dtype=float)
            vis_den = np.array([r[denominator] for r in visibility], dtype=float)
            samples = (
                vis_num[index].sum(1) / np.maximum(1, vis_den[index].sum(1))
                - base_num[index].sum(1) / np.maximum(1, base_den[index].sum(1))
            )
        output[f"delta_{metric}"] = float(samples.mean())
        output[f"delta_{metric}_ci_low"] = float(np.quantile(samples, 0.025))
        output[f"delta_{metric}_ci_high"] = float(np.quantile(samples, 0.975))
    return output


def evaluate(k: int, episodes: int) -> list[dict[str, object]]:
    checkpoints = {
        "baseline": observer_ckpt(k),
        "visibility": visibility_observer_ckpt(k),
    }
    missing = [str(path) for path in checkpoints.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing checkpoint(s): " + ", ".join(missing))
    models = {name: load_observer(k, path) for name, path in checkpoints.items()}
    rows: list[dict[str, object]] = []
    for scenario_i, (scenario, desires) in enumerate(SCENARIOS):
        for epsilon_i, epsilon in enumerate(EPSILONS):
            for evidence_i, (evidence, watch_steps) in enumerate(EVIDENCE):
                totals = {name: [] for name in models}
                for episode in range(episodes):
                    seed = SEED0 + scenario_i * 100_000 + epsilon_i * 10_000 + evidence_i * 1_000 + episode
                    for name, model in models.items():
                        totals[name].append(evaluate_episode(
                            k, desires, epsilon, evidence, watch_steps, seed, model
                        ))
                for name, results in totals.items():
                    row = {
                        "model": name,
                        "scenario": scenario,
                        "partners": "+".join(ITEM_KINDS[d] for d in desires),
                        "epsilon": epsilon,
                        "evidence": evidence,
                        "watch_steps": watch_steps,
                        "episodes": episodes,
                    }
                    row.update(summarize(results))
                    for metric in (
                        "violation", "permitted_eating", "sanctions_per_episode",
                        "mean_return", "desire_accuracy",
                    ):
                        row[f"delta_{metric}"] = ""
                        row[f"delta_{metric}_ci_low"] = ""
                        row[f"delta_{metric}_ci_high"] = ""
                    if name == "visibility":
                        row.update(paired_intervals(
                            totals["baseline"], results,
                            SEED0 + scenario_i * 100_000 + epsilon_i * 10_000 + evidence_i * 1_000,
                        ))
                    rows.append(row)
    return rows


def print_summary(rows: list[dict[str, object]]) -> None:
    print("Paired visibility comparison (visibility - baseline; identical episode seeds)")
    print("\nscenario             eps  evidence          d violation  d permitted  d return  d desire acc")
    index = {(r["scenario"], r["epsilon"], r["evidence"]): r for r in rows if r["model"] == "baseline"}
    for visibility in (r for r in rows if r["model"] == "visibility"):
        baseline = index[(visibility["scenario"], visibility["epsilon"], visibility["evidence"])]
        print(
            f"{visibility['scenario']:<21}{visibility['epsilon']:>4.1f}  {visibility['evidence']:<16}"
            f"{visibility['delta_violation']:>12.3f}"
            f"{visibility['delta_permitted_eating']:>13.3f}"
            f"{visibility['delta_mean_return']:>10.2f}"
            f"{visibility['delta_desire_accuracy']:>14.3f}"
        )
        print(
            f"{'':<21}{'':>4}  {'95% CI, violation':<16}"
            f"[{visibility['delta_violation_ci_low']:.3f}, {visibility['delta_violation_ci_high']:.3f}]"
        )


def write_csv(rows: list[dict[str, object]], k: int) -> None:
    path = DATA_DIR / "reports" / f"norm_visibility_compare_k{k}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
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
        raise SystemExit("The configured scenarios require K >= 3.")
    rows = evaluate(args.k, args.episodes)
    print(f"K={args.k}; {args.episodes} episodes/cell; S={SANCTION:g}; forage={N_FORAGE_STEPS} steps")
    print_summary(rows)
    if not args.no_csv:
        write_csv(rows, args.k)


if __name__ == "__main__":
    main()
