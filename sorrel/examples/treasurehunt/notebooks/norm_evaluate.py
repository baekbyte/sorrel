"""Evaluate the decision-rule norm ceiling (G0 and held-out-kind G1).

Run with::

    python -m sorrel.examples.treasurehunt.notebooks.norm_evaluate 3

The policy has no fitted parameters.  ``G1 coin`` is therefore a genuine
zero-shot test: only gem/food partner types are named as the policy's
development conditions, while a coin-loving partner is evaluated with the
same frozen ToM posterior -> expected-sanction rule.
"""

import sys
from dataclasses import dataclass

import numpy as np

from sorrel.examples.treasurehunt.notebooks.norm_common import (
    ITEM_KINDS,
    forage_step,
    make_norm_env,
    sanction_probabilities,
)
from sorrel.examples.treasurehunt.notebooks.tom3_common import make_obs_spec, run_watch_phase
from sorrel.examples.treasurehunt.notebooks.tom3_evaluate import infer, load_observer

N_EPISODES = 100
N_FORAGE_STEPS = 45
SANCTION = 20.0
SEED0 = 300_000


@dataclass
class Scores:
    preferred_available: int = 0
    preferred_eaten: int = 0
    other_available: int = 0
    other_eaten: int = 0
    sanctions: int = 0
    reward: float = 0.0

    def add(self, initial, consumed, preferred: int, log) -> None:
        pref_kind = ITEM_KINDS[preferred]
        self.preferred_available += initial[pref_kind]
        self.preferred_eaten += consumed.get(pref_kind, 0)
        for kind, count in initial.items():
            if kind != pref_kind:
                self.other_available += count
                self.other_eaten += consumed.get(kind, 0)
        self.sanctions += log.sanctions
        self.reward += log.total_reward

    def summary(self) -> tuple[float, float, float, float]:
        violation = self.preferred_eaten / max(1, self.preferred_available)
        nonpreferred = self.other_eaten / max(1, self.other_available)
        return violation, nonpreferred, self.sanctions, self.reward


def evaluate_condition(k: int, desire: int, mode: str, n_episodes: int, seed: int) -> Scores:
    encoder, desire_head, gem_head, latent_head = load_observer(k)
    obs_spec = make_obs_spec()
    scores = Scores()
    for episode in range(n_episodes):
        env = make_norm_env(k, [desire], seed + episode, sanction=SANCTION)
        frames, disp, marks, _, _ = run_watch_phase(env, obs_spec)
        _, _, desire_probs, _ = infer(
            k, encoder, desire_head, gem_head, latent_head, frames, disp, marks
        )
        if mode == "tom":
            posterior = sanction_probabilities(desire_probs)
        elif mode == "oracle":
            posterior = np.eye(k)[desire]
        elif mode == "no_tom":
            posterior = np.full(k, 1.0 / k)
        else:
            raise ValueError(mode)
        assert env.norm_log is not None
        # The partner has legitimately consumed items during the watch phase.
        # Compliance is measured against what was still available when the
        # observer gained control, not against items it could never choose.
        available = env.count_items()
        for _ in range(N_FORAGE_STEPS):
            # Hold demonstrators in place during the choice phase.  Letting a
            # partner continue to vacuum up its own target would erase the
            # very violations we need to measure, making a no-ToM policy look
            # compliant merely because no protected item remains.
            forage_step(env, posterior)
        scores.add(available, env.norm_log.consumed, desire, env.norm_log)
    return scores


def main() -> None:
    k = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    if not 2 <= k <= 4:
        raise SystemExit("K must be in {2, 3, 4}")
    # G0 uses partner types available to the policy-design conditions.  G1
    # holds out Coin, which is present when K >= 3.
    conditions = [("G0 gem", 0), ("G0 food", 1)]
    if k >= 3:
        conditions.append(("G1 held-out coin", 2))
    print(f"K={k}; {N_EPISODES} episodes/row; S={SANCTION:g}; forage={N_FORAGE_STEPS} steps")
    print("\ncondition             policy       violation  non-pref eat  sanctions  mean return")
    for ci, (label, desire) in enumerate(conditions):
        for mi, mode in enumerate(("tom", "no_tom", "oracle")):
            scores = evaluate_condition(k, desire, mode, N_EPISODES, SEED0 + 10_000 * ci + 1_000 * mi)
            violation, other, sanctions, reward = scores.summary()
            print(
                f"{label:<22}{mode:<12}{violation:>9.3f}{other:>14.3f}"
                f"{sanctions / N_EPISODES:>11.2f}{reward / N_EPISODES:>13.2f}"
            )
    print("\nHeadline: norm following is low violation *and* high non-preferred consumption;")
    print("the no-ToM policy uses a uniform desire posterior, while oracle bounds inference cost.")


if __name__ == "__main__":
    main()
