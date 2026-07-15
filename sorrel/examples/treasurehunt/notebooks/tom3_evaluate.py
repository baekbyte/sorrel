"""ToM v3 behavioral evaluation: K-way door choice from inferred belief.

The observer watches for T_WATCH steps, the trained encoder turns each watched
agent's (frames, displacements) into a K-way posterior over "which room holds
the gems" (multi-agent: sum of per-agent log-softmax = uniform-prior naive
Bayes), and a decision rule on the observer's OWN gem preference picks the
door of the argmax room.

Experiments:
  1. Door-choice table: watched in {each single desire, exclusion set
     (all K-1 non-gem-lovers), in-distribution multi-agent set, none} x
     belief {factorized, latent-head, oracle, uniform}. ``factorized`` is the
     standard behavioral readout and uses
     the learned desire prediction plus the visibly committed door and the
     analytic desire x door rule.  It separates perception of desire from
     the harder end-to-end latent head.  Single non-gem desires have an
     analytic ceiling of 1/(K-1) (they only EXCLUDE their own room).
  2. Exclusion headline (K >= 3): watching only non-gem-lovers, the observer
     should pick the door NOBODY entered — impossible for a mimic.
  3. Same-direction contrast, K-way: watched agent committed to door X ->
     P(observer picks X) high iff it is a gem-lover; otherwise the observer
     avoids X and spreads over the rest.
  4. Belief calibration (multiclass) + desire accuracy.
  5. epsilon-robustness: eval at demonstrator noise 0.2/0.3 with the
     0.1-trained observer.

Usage:
    python -m sorrel.examples.treasurehunt.notebooks.tom3_evaluate [K ...]
    (default: 2 3 4)
"""

import sys

import numpy as np
import torch

from sorrel.examples.treasurehunt.notebooks.tom3_common import (
    ITEM_KINDS,
    agent_window,
    make_env,
    make_obs_spec,
    observer_ckpt,
    run_watch_phase,
)
from sorrel.models.pytorch.transformer import BeliefEncoder

N_EPISODES = 300
N_EPISODES_EPS = 150
EPSILON = 0.1
SEED0 = 100_000  # disjoint from training seeds

rng = np.random.default_rng(0)


def load_observer(k: int):
    ckpt = torch.load(observer_ckpt(k), map_location="cpu")
    encoder = BeliefEncoder(**ckpt["arch"], device="cpu")
    encoder.load_state_dict(ckpt["encoder"])
    encoder.eval()
    desire_head = torch.nn.Linear(ckpt["arch"]["layer_size"], k)
    desire_head.load_state_dict(ckpt["desire_head"])
    gem_head = None
    # Checkpoints before the binary gem auxiliary head remain evaluable; their
    # factorized readout falls back to the K-way desire head's gem probability.
    if "gem_head" in ckpt:
        gem_head = torch.nn.Linear(ckpt["arch"]["layer_size"], 2)
        gem_head.load_state_dict(ckpt["gem_head"])
    latent_head = torch.nn.Linear(ckpt["arch"]["layer_size"], k)
    latent_head.load_state_dict(ckpt["latent_head"])
    return encoder, desire_head, gem_head, latent_head


def infer(k, encoder, desire_head, gem_head, latent_head, frames, disp_by_agent, marks):
    """Per-agent desire predictions/probabilities + legacy latent posterior.

    The returned latent posterior is retained as a diagnostic.  Behavioral
    inference should use :func:`factorized_posterior`, which combines the
    desire probabilities with the observed door commits without assuming that
    the overlapping per-agent windows are conditionally independent.
    """
    log_post = np.zeros(k)
    desires = []
    desire_probs = []
    gem_probs = []
    with torch.no_grad():
        for i, disp in disp_by_agent.items():
            window = agent_window(frames, marks[i])
            f = torch.tensor(window, dtype=torch.float32).unsqueeze(0)
            a = torch.tensor(disp, dtype=torch.long).view(1, -1, 1)
            g = encoder(f, a)
            d_prob = torch.softmax(desire_head(g)[0], dim=0).numpy()
            desire_probs.append(d_prob)
            desires.append(int(d_prob.argmax()))
            if gem_head is None:
                gem_probs.append(float(d_prob[0]))
            else:
                gem_probs.append(float(torch.softmax(gem_head(g)[0], dim=0)[1]))
            log_post += torch.log_softmax(latent_head(g)[0], dim=0).numpy()
    post = np.exp(log_post - log_post.max())
    post /= post.sum()
    return (
        post,
        desires,
        np.stack(desire_probs) if desire_probs else np.empty((0, k)),
        np.array(gem_probs),
    )


def choose_door(post: np.ndarray) -> int:
    """Argmax with random tie-breaking (uniform posterior -> random door)."""
    best = np.flatnonzero(post >= post.max() - 1e-12)
    return int(best[rng.integers(len(best))])


def factorized_posterior(
    k: int, desire_probs: np.ndarray | list[int], commit_room: dict[int, int | None]
) -> np.ndarray:
    """Combine visible door commits using the task's analytic model.

    A gem-lover identifies its committed room; a non-gem-lover rules its
    committed room out.  The likelihood marginalizes the *soft* learned
    desire posterior: ``P(gem room = door) = P(gem-lover)`` and every other
    room receives ``P(non-gem-lover)/(K-1)``.  This is essential for multiple
    agents: using hard argmax desires lets one uncertain gem-lover become a
    false exclusion that eliminates the true room.

    Integer desire labels are accepted for oracle/unit-test use and converted
    to one-hot posteriors.
    """
    log_post = np.zeros(k)
    eps = 1e-6
    desire_probs = np.asarray(desire_probs)
    if desire_probs.ndim == 1:
        # Integer vectors are oracle desire labels; floating vectors are the
        # dedicated binary gem probabilities from the auxiliary head.
        if np.issubdtype(desire_probs.dtype, np.integer):
            gem_probs = np.eye(k)[desire_probs.astype(np.int64), 0]
        else:
            gem_probs = desire_probs
    else:
        assert desire_probs.shape[1] == k
        gem_probs = desire_probs[:, 0]
    for i, p_gem in enumerate(gem_probs):
        door = commit_room[i]
        if door is None:
            continue
        p_gem = float(p_gem)
        likelihood = np.full(k, max((1.0 - p_gem) / (k - 1), eps))
        likelihood[door] = max(p_gem, eps)
        log_post += np.log(likelihood)
    post = np.exp(log_post - log_post.max())
    return post / post.sum()


def eval_configs(k: int) -> dict[str, list[int]]:
    configs = {f"single_{ITEM_KINDS[d].lower()}": [d] for d in range(k)}
    if k >= 3:
        configs["exclusion"] = list(range(1, k))
    # Generation deliberately caps simultaneous demonstrators at three.  Do
    # not present K=4's four-agent result as a normal generalization result.
    # This representative 3-agent mixture is sampled during K=4 training.
    if k <= 3:
        configs["all"] = list(range(k))
    else:
        configs["three_including_gem"] = [0, 1, 2]
    configs["none"] = []
    return configs


def evaluate_k(k: int) -> None:
    print("=" * 76)
    print(f"K = {k}")
    print("=" * 76)
    encoder, desire_head, gem_head, latent_head = load_observer(k)
    obs_spec = make_obs_spec()
    configs = eval_configs(k)

    table: dict[tuple[str, str], list[int]] = {}
    contrast: dict[int, list[int]] = {}  # desire -> [observer picked watched door]
    calib: list[tuple[float, int]] = []  # (posterior_i, room i is gem room)
    excl_post: list[tuple[float, float]] = (
        []
    )  # single non-gem: (p on own room, p max rest)
    desire_correct = desire_total = 0

    seed = SEED0
    for name, desires_cfg in configs.items():
        for _ in range(N_EPISODES):
            env = make_env(k, desires_cfg, seed=seed, epsilon=EPSILON)
            seed += 1
            frames, disp, marks, commit_room, _ = run_watch_phase(env, obs_spec)
            gem_room = env.world.gem_room
            assert gem_room is not None

            latent_post, desires_hat, desire_probs, gem_probs = infer(
                k, encoder, desire_head, gem_head, latent_head, frames, disp, marks
            )
            if not desires_cfg:
                factor_post = np.full(k, 1.0 / k)
            else:
                factor_post = factorized_posterior(k, gem_probs, commit_room)
            # The desired behavioral policy is explicitly compositional:
            # perceived desire + observed committed door -> gem-room belief.
            # Keep the end-to-end latent head as a diagnostic, rather than
            # multiplying its correlated per-agent posteriors as the policy.
            post = factor_post
            for i, d in enumerate(desires_cfg):
                desire_correct += desires_hat[i] == d
                desire_total += 1

            oracle = np.eye(k)[gem_room]
            uniform = np.full(k, 1.0 / k)
            for belief, p in [
                ("factorized", post),
                ("latent_head", latent_post),
                ("oracle", oracle),
                ("uniform", uniform),
            ]:
                door = choose_door(p)
                table.setdefault((name, belief), []).append(int(door == gem_room))

            if desires_cfg:
                for i in range(k):
                    calib.append((float(post[i]), int(i == gem_room)))

            if len(desires_cfg) == 1 and commit_room[0] is not None:
                d = desires_cfg[0]
                door = choose_door(post)
                contrast.setdefault(d, []).append(int(door == commit_room[0]))
                if d != 0:
                    rest = np.delete(post, commit_room[0])
                    excl_post.append((float(post[commit_room[0]]), float(rest.max())))

    print(
        f"\n1. DOOR-CHOICE TABLE ({N_EPISODES} episodes/cell, P(correct door); "
        f"chance={1 / k:.3f})"
    )
    print(
        f"{'watched':<22}{'factorized':>12}{'latent-head':>12}{'oracle':>10}{'uniform':>10}{'ceiling':>10}"
    )
    for name, desires_cfg in configs.items():
        row = [
            np.mean(table[(name, b)])
            for b in ["factorized", "latent_head", "oracle", "uniform"]
        ]
        if not desires_cfg:
            ceiling = 1.0 / k
        elif 0 in desires_cfg or len(desires_cfg) >= k - 1:
            ceiling = 1.0
        else:
            ceiling = 1.0 / (k - len(desires_cfg))
        print(
            f"{name:<22}{row[0]:>12.3f}{row[1]:>12.3f}{row[2]:>10.3f}{row[3]:>10.3f}{ceiling:>10.2f}"
        )

    print(
        f"\ndesire accuracy over eval episodes: "
        f"{desire_correct / max(1, desire_total):.3f} (chance={1 / k:.3f})"
    )

    print("\n2. SAME-DIRECTION CONTRAST — P(observer picks the watched agent's door)")
    print("   (follow the gem-lover; avoid everyone else — a mimic cannot split these)")
    for d, vals in sorted(contrast.items()):
        print(
            f"  watched {ITEM_KINDS[d]:<6}-lover (n={len(vals):>3}): "
            f"P(follow) = {np.mean(vals):.3f}"
        )
    if excl_post:
        own = np.mean([p for p, _ in excl_post])
        rest = np.mean([q for _, q in excl_post])
        print(
            f"  single non-gem posterior: mean P(gems in agent's room) = {own:.3f} "
            f"(should be ~0), mean max over rest = {rest:.3f} "
            f"(~{1 / (k - 1):.2f} = graded exclusion)"
        )

    print("\n3. BELIEF CALIBRATION (posterior on a room vs empirical P(gems there))")
    ps = np.array([c[0] for c in calib])
    ys = np.array([c[1] for c in calib])
    for lo in np.arange(0, 1.0, 0.2):
        m = (ps >= lo) & (ps < lo + 0.2)
        if m.sum() > 0:
            print(
                f"  P in [{lo:.1f}, {lo + 0.2:.1f}): n={m.sum():>5}, "
                f"empirical={ys[m].mean():.3f}"
            )

    print("\n4. EPSILON ROBUSTNESS (factorized P(correct door), observer trained at 0.1)")
    for eps in (0.2, 0.3):
        accs = []
        for name, desires_cfg in configs.items():
            if not desires_cfg:
                continue
            correct = []
            for _ in range(N_EPISODES_EPS):
                env = make_env(k, desires_cfg, seed=seed, epsilon=eps)
                seed += 1
                frames, disp, marks, commit_room, _ = run_watch_phase(env, obs_spec)
                _, desires_hat, _, gem_probs = infer(
                    k, encoder, desire_head, gem_head, latent_head, frames, disp, marks
                )
                post = factorized_posterior(k, gem_probs, commit_room)
                correct.append(int(choose_door(post) == env.world.gem_room))
            accs.append(f"{name}={np.mean(correct):.3f}")
        print(f"  eps={eps}: " + "  ".join(accs))
    print()


def main():
    ks = [int(a) for a in sys.argv[1:]] or [2, 3, 4]
    for k in ks:
        evaluate_k(k)


if __name__ == "__main__":
    main()
