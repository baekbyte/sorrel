"""Test whether multi-agent ToM errors track spatial interference.

The audit is inference-only.  It records the evidence available for each
tracked partner during the 20-step watch phase: FOV exposure, observed motion,
room commitment, and proximity to the other partner when both are visible.
It then compares these measures for correct versus incorrect desire readouts.

Usage::

    python -m sorrel.examples.treasurehunt.notebooks.norm_spatial_audit 3
"""

import argparse
import csv
from itertools import permutations

import numpy as np

from sorrel.examples.treasurehunt.notebooks.norm_common import make_norm_env
from sorrel.examples.treasurehunt.notebooks.tom3_common import (
    DATA_DIR,
    DISP_STAY,
    ITEM_KINDS,
    make_obs_spec,
    run_watch_phase,
)
from sorrel.examples.treasurehunt.notebooks.tom3_evaluate import infer, load_observer

EPSILONS = (0.1, 0.3)
SEED0 = 900_000


def visibility_features(mark: np.ndarray, other_mark: np.ndarray, disp: np.ndarray) -> dict[str, float]:
    """Features available from the strict third-person observation stream."""
    visible = mark.any(axis=(1, 2))
    other_visible = other_mark.any(axis=(1, 2))
    joint = visible & other_visible
    distances = []
    adjacent = 0
    for t in np.flatnonzero(joint):
        y1, x1 = np.argwhere(mark[t])[0]
        y2, x2 = np.argwhere(other_mark[t])[0]
        distance = max(abs(int(y1) - int(y2)), abs(int(x1) - int(x2)))
        distances.append(distance)
        adjacent += distance <= 1
    return {
        "visible_steps": int(visible.sum()),
        "observed_moves": int(((disp != DISP_STAY) & visible).sum()),
        "joint_visible_steps": int(joint.sum()),
        "mean_joint_distance": float(np.mean(distances)) if distances else np.nan,
        "adjacent_joint_fraction": adjacent / len(distances) if distances else np.nan,
    }


def run(k: int, episodes: int, epsilons: tuple[float, ...]) -> list[dict[str, object]]:
    models = load_observer(k)
    encoder, desire_head, gem_head, latent_head = models
    rows: list[dict[str, object]] = []
    for epsilon_i, epsilon in enumerate(epsilons):
        for desires in permutations(range(k), 2):
            unordered = tuple(sorted(desires))
            for episode in range(episodes):
                seed = SEED0 + epsilon_i * 100_000 + unordered[0] * 10_000 + unordered[1] * 1_000 + episode
                env = make_norm_env(k, list(desires), seed, epsilon=epsilon)
                frames, disp, marks, commit_room, commit_step = run_watch_phase(env, make_obs_spec())
                _, predicted, desire_probs, _ = infer(
                    k, encoder, desire_head, gem_head, latent_head, frames, disp, marks
                )
                for slot, desire in enumerate(desires):
                    other_slot = 1 - slot
                    features = visibility_features(marks[slot], marks[other_slot], disp[slot])
                    rows.append(
                        {
                            "epsilon": epsilon,
                            "episode": episode,
                            "seed": seed,
                            "ordered_pair": "+".join(ITEM_KINDS[d] for d in desires),
                            "slot": slot,
                            "true_kind": ITEM_KINDS[desire],
                            "predicted_kind": ITEM_KINDS[predicted[slot]],
                            "correct": int(predicted[slot] == desire),
                            "true_probability": float(desire_probs[slot, desire]),
                            "committed": int(commit_room[slot] is not None),
                            "commit_step": -1 if commit_step[slot] is None else commit_step[slot],
                            **features,
                        }
                    )
    return rows


def mean_or_nan(rows: list[dict[str, object]], key: str) -> float:
    values = np.asarray([float(row[key]) for row in rows], dtype=float)
    return float(np.nanmean(values)) if np.isfinite(values).any() else float("nan")


def summarize(rows: list[dict[str, object]], epsilons: tuple[float, ...]) -> None:
    print("Spatial-interference audit: correct versus incorrect desire inference")
    print("\neps  readout    n  visible  moves  joint vis  joint dist  adjacent  committed")
    for epsilon in epsilons:
        for label, correct in (("correct", 1), ("incorrect", 0)):
            subset = [row for row in rows if row["epsilon"] == epsilon and row["correct"] == correct]
            print(
                f"{epsilon:>3.1f}  {label:<9}{len(subset):>4}"
                f"{mean_or_nan(subset, 'visible_steps'):>9.2f}"
                f"{mean_or_nan(subset, 'observed_moves'):>7.2f}"
                f"{mean_or_nan(subset, 'joint_visible_steps'):>10.2f}"
                f"{mean_or_nan(subset, 'mean_joint_distance'):>12.2f}"
                f"{mean_or_nan(subset, 'adjacent_joint_fraction'):>10.2f}"
                f"{np.mean([row['committed'] for row in subset]):>11.3f}"
            )
    print("\nError rate by close-contact evidence:")
    print("eps  contact class               error rate  n")
    for epsilon in epsilons:
        eps_rows = [row for row in rows if row["epsilon"] == epsilon]
        groups = {
            "never jointly visible": [row for row in eps_rows if row["joint_visible_steps"] == 0],
            "joint, never adjacent": [row for row in eps_rows if row["joint_visible_steps"] > 0 and row["adjacent_joint_fraction"] == 0],
            "adjacent at least once": [row for row in eps_rows if row["joint_visible_steps"] > 0 and row["adjacent_joint_fraction"] > 0],
        }
        for label, subset in groups.items():
            if subset:
                error = 1.0 - np.mean([row["correct"] for row in subset])
                print(f"{epsilon:>3.1f}  {label:<26}{error:>10.3f}{len(subset):>4}")


def write_csv(rows: list[dict[str, object]], k: int, episodes: int, epsilons: tuple[float, ...]) -> None:
    suffix = "-".join(f"e{epsilon:.1f}" for epsilon in epsilons)
    path = DATA_DIR / "reports" / f"norm_spatial_audit_k{k}_{episodes}ep_{suffix}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nCSV: {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("k", nargs="?", type=int, default=3)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--epsilons", nargs="+", type=float, default=EPSILONS)
    parser.add_argument("--no-csv", action="store_true")
    args = parser.parse_args()
    if args.k < 3:
        raise SystemExit("The two-partner spatial audit requires K >= 3.")
    epsilons = tuple(args.epsilons)
    rows = run(args.k, args.episodes, epsilons)
    summarize(rows, epsilons)
    if not args.no_csv:
        write_csv(rows, args.k, args.episodes, epsilons)


if __name__ == "__main__":
    main()
