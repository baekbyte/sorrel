"""ToM rollout experiment: watched-agent config x belief.

The observer's model input is a 15x15 patch; only the inner 9x9 is actually
visible (outer ring is masked, belief fills it via the two-pass action
selection in tom_rollout.py). The manipulation is which watched agent(s) the
observer can see act inside its inner 9x9 FOV:

  WATCHED CONFIG
    gem_only   - gem-lover only (heads west into hidden gem region)
    food_only  - food-lover only (heads east into hidden food region)
    both       - both watched
    none       - observer alone (pure frozen-base baseline)

  BELIEF
    True  - observer uses the FOV-gated first-person belief encoder + two-pass
            (g modulates vision/reconstruction; action comes from the base
            without g, on the belief-completed observation -- the observer's
            OWN preference, not the watched agent's).
    False - observer ignores g and the pass-1 reconstruction; pass-2 still
            runs on the visible (outer-ring-masked) observation alone.

Per cell: N trials with different seeds. Each trial = a 50-turn episode in the
segregated treasurehunt with the observer starting at (10, 10).

Predictions if the spec is now faithfully implemented:
  - 'gem_only, belief on'  -> belief says "gem-lover preference" -> the
                              base's reconstruction tilts toward gems in the
                              outer ring -> the observer's own gem-pref policy
                              heads leftward toward inferred gems.
  - 'food_only, belief on' -> belief says "food-lover preference" -> outer
                              ring filled with food -> the observer's gem-pref
                              policy doesn't want food, so it STILL heads left
                              (the observer's own preference filters which
                              inferred entity it pursues).
  - belief off across configs -> base's intrinsic gem-prior dominates;
                                 observer heads left regardless of who's
                                 visible. (Same direction across all cells.)

"""

from collections import Counter
from pathlib import Path

import numpy as np

from sorrel.examples.treasurehunt.notebooks.tom_rollout import (
    ACTION_NAMES,
    EPISODE_LEN,
    build_env,
)
from sorrel.utils.visualization import ImageRenderer

DATA_DIR = Path(__file__).parent / "../data"
GIF_DIR = DATA_DIR / "gifs"

N_TRIALS = 10
WATCHED_CONFIGS = ["gem_only", "food_only", "both", "none"]
BELIEF_OPTIONS = [True, False]


def run_trial(watched_config: str, belief_on: bool, seed: int, save_gif: bool = False):
    env, observer = build_env(
        watched_config=watched_config, belief_on=belief_on, seed=seed
    )
    start_y, start_x, _ = observer.location
    renderer = (
        ImageRenderer(
            experiment_name=(
                f"TomExpt_{watched_config}_{'belief' if belief_on else 'nobelief'}"
            ),
            record_period=1,
            num_turns=EPISODE_LEN,
        )
        if save_gif
        else None
    )
    actions_taken = []
    for turn in range(EPISODE_LEN):
        env.take_turn()
        actions_taken.append(observer.last_action_name)
        if save_gif:
            renderer.add_image(env.world)
        if env.world.is_done:
            break
    end_y, end_x, _ = observer.location
    if save_gif:
        renderer.save_gif(epoch=seed, folder=GIF_DIR)
    counts = Counter(actions_taken)
    n = max(1, len(actions_taken))
    # Preference-weighted reward (gem-pref observer): gems are worth 10,
    # food only 2, bones -10. Higher = observer ended where it wanted.
    weighted = (
        observer.collected.get("Gem", 0) * 10
        + observer.collected.get("Food", 0) * 2
        + observer.collected.get("Bone", 0) * (-10)
    )
    return {
        "start": (start_y, start_x),
        "end": (end_y, end_x),
        "leftward": start_x - end_x,
        "action_frac": {a: counts.get(a, 0) / n for a in ACTION_NAMES},
        "final_x": end_x,
        "collected": dict(observer.collected),
        "weighted_reward": weighted,
    }


def main():
    print(f"Experiment: {N_TRIALS} trials per cell, episode_len={EPISODE_LEN}")
    print("=" * 90)
    rows = {}
    for wc in WATCHED_CONFIGS:
        for belief in BELIEF_OPTIONS:
            cell = []
            for trial in range(N_TRIALS):
                seed = 100 + trial
                res = run_trial(wc, belief, seed=seed, save_gif=(trial == 0))
                cell.append(res)
                print(
                    f"  [watched={wc:<10s} belief={str(belief):<5s} seed={seed:>3d}] "
                    f"end_x={res['end'][1]:>2d}  lw={res['leftward']:+3d}  "
                    f"gems={res['collected']['Gem']:>2d} food={res['collected']['Food']:>2d} "
                    f"bones={res['collected']['Bone']:>2d} wR={res['weighted_reward']:+4d}"
                )
            rows[(wc, belief)] = cell

    print("\n" + "=" * 100)
    print("CELL SUMMARY (mean over N trials)")
    print("=" * 100)
    print(
        f"{'watched':<12}{'belief':<8}"
        f"{'gems (mean ± std)':>22}"
        f"{'food (mean ± std)':>22}"
        f"{'bones':>12}"
        f"{'gem-pref reward':>20}"
    )
    for wc in WATCHED_CONFIGS:
        for belief in BELIEF_OPTIONS:
            cell = rows[(wc, belief)]
            gems = np.array([r["collected"]["Gem"] for r in cell])
            foods = np.array([r["collected"]["Food"] for r in cell])
            bones = np.array([r["collected"]["Bone"] for r in cell])
            wr = np.array([r["weighted_reward"] for r in cell])
            print(
                f"{wc:<12}{str(belief):<8}"
                f"{gems.mean():>10.2f} ± {gems.std():<6.2f}"
                f"{foods.mean():>10.2f} ± {foods.std():<6.2f}"
                f"{bones.mean():>10.2f}"
                f"{wr.mean():>+13.2f} ± {wr.std():<6.2f}"
            )

    print("\nBelief effect on gem-preferenced reward (positive = belief helps observer find gems):")
    for wc in WATCHED_CONFIGS:
        on = np.mean([r["weighted_reward"] for r in rows[(wc, True)]])
        off = np.mean([r["weighted_reward"] for r in rows[(wc, False)]])
        print(f"  watched={wc:<10s}: belief_on={on:+.2f}  belief_off={off:+.2f}  effect={on - off:+.2f}")

    print("\nBelief effect on gem-vs-food selectivity (gems collected - food collected):")
    for wc in WATCHED_CONFIGS:
        sel_on = np.mean(
            [r["collected"]["Gem"] - r["collected"]["Food"] for r in rows[(wc, True)]]
        )
        sel_off = np.mean(
            [r["collected"]["Gem"] - r["collected"]["Food"] for r in rows[(wc, False)]]
        )
        print(f"  watched={wc:<10s}: belief_on={sel_on:+.2f}  belief_off={sel_off:+.2f}  effect={sel_on - sel_off:+.2f}")

    print("\nWatched-agent contrast under belief_on (gem-pref reward):")
    r_gem = np.mean([r["weighted_reward"] for r in rows[("gem_only", True)]])
    r_food = np.mean([r["weighted_reward"] for r in rows[("food_only", True)]])
    r_none = np.mean([r["weighted_reward"] for r in rows[("none", True)]])
    print(f"  gem_only:  {r_gem:+.2f}")
    print(f"  food_only: {r_food:+.2f}")
    print(f"  none:      {r_none:+.2f}")
    print(f"  gem_only - food_only: {r_gem - r_food:+.2f}  (positive = ToM signal)")


if __name__ == "__main__":
    main()
