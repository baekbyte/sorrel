"""Generate the ToM v3 observer datasets (K rooms / K preferences / N agents).

Per episode: frozen observer at the chamber center records its own 9x9 POV
for T_WATCH steps while 1-3 scripted watched agents (distinct desires,
sampled per episode) pick through the chamber preview items (desire signal)
and head through a door (latent signal). One training example per watched
agent per episode:

    frames (T, 9, 9, 9) uint8   observer POV (8 entity channels) +
                                tracked-agent marker channel
    disp   (T,) int64           watched agent's OBSERVED displacement tokens
    desire in 0..K-1            index into ITEM_KINDS
    latent in 0..K-1            which room holds the gems (used_rooms index)

Also runs the dataset-floor checks: a majority-vote probe on the watched
agent's committed door must be near-chance (1/K) on the LATENT when desires
are mixed, and near the 1/(K-1) exclusion ceiling given a non-gem desire —
the task requires the desire x door interaction, not a direction shortcut.

Usage:
    python -m sorrel.examples.treasurehunt.notebooks.tom3_generate_data [K ...]
    (default: 2 3 4)
"""

import sys

import numpy as np

from sorrel.examples.treasurehunt.notebooks.tom3_common import (
    ITEM_KINDS,
    T_WATCH,
    agent_window,
    dataset_path,
    make_env,
    make_obs_spec,
    run_watch_phase,
)

# Per-K episode counts. K >= 3 needs more data: clean single-agent windows
# per desire class fall as 1/K while the discrimination gets K-way harder
# (6k episodes left the K=3/4 observers data-starved: desire 77%/58%).
N_EPISODES = {2: 6000, 3: 15000, 4: 15000}
EPSILON = 0.1
SEED0 = 1000


def majority_acc(feat: np.ndarray, y: np.ndarray) -> float:
    """Accuracy of the best per-feature-value majority classifier."""
    correct = 0
    for v in np.unique(feat):
        m = feat == v
        correct += np.bincount(y[m]).max()
    return correct / len(y)


def generate(k: int) -> None:
    obs_spec = make_obs_spec()
    meta_rng = np.random.default_rng(k)  # samples each episode's watched set
    max_agents = min(3, k)

    frames_all, disp_all = [], []
    desire_all, latent_all, door_all, episode_all = [], [], [], []

    entered = n_windows = 0
    n_episodes = N_EPISODES[k]
    for ep in range(n_episodes):
        seed = SEED0 + ep
        n_agents = int(meta_rng.integers(1, max_agents + 1))
        desires = sorted(meta_rng.choice(k, size=n_agents, replace=False).tolist())
        env = make_env(k, desires, seed=seed, epsilon=EPSILON)
        frames, disp, marks, commit_room, _ = run_watch_phase(env, obs_spec)
        latent = env.world.gem_room
        for i, d in enumerate(desires):
            frames_all.append(agent_window(frames, marks[i]).astype(np.uint8))
            disp_all.append(disp[i])
            desire_all.append(d)
            latent_all.append(latent)
            door_all.append(-1 if commit_room[i] is None else commit_room[i])
            episode_all.append(ep)
            entered += commit_room[i] is not None
            n_windows += 1

    frames_arr = np.stack(frames_all)
    disp_arr = np.stack(disp_all)
    desire_arr = np.array(desire_all, dtype=np.int64)
    latent_arr = np.array(latent_all, dtype=np.int64)
    door_arr = np.array(door_all, dtype=np.int64)
    episode_arr = np.array(episode_all, dtype=np.int64)

    path = dataset_path(k)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        frames=frames_arr,
        disp=disp_arr,
        desire=desire_arr,
        latent=latent_arr,
        door=door_arr,
        episode=episode_arr,
    )
    print(
        f"K={k}: {n_episodes} episodes, {n_windows} windows, "
        f"door-commit rate {entered / n_windows:.2f}"
    )
    print(f"  dataset: {frames_arr.shape} -> {path}")
    print(
        "  label balance: desire "
        + np.array2string(np.bincount(desire_arr, minlength=k) / n_windows, precision=2)
        + ", latent "
        + np.array2string(np.bincount(latent_arr, minlength=k) / n_windows, precision=2)
    )

    # ---- dataset-floor checks (majority-vote probes on the committed door) ----
    print(f"  Floors (door-feature majority probes; chance = {1 / k:.3f}):")
    print(
        f"    latent from door alone (mixed desires): "
        f"{majority_acc(door_arr, latent_arr):.3f}  (must be ~{1 / k:.2f})"
    )
    for d in range(k):
        m = desire_arr == d
        ceiling = 1.0 if d == 0 else 1.0 / (k - 1)
        print(
            f"    latent from door | desire={ITEM_KINDS[d]:<6}: "
            f"{majority_acc(door_arr[m], latent_arr[m]):.3f}  "
            f"(analytic ceiling ~{ceiling:.2f})"
        )
    print(
        f"    desire from door alone: "
        f"{majority_acc(door_arr, desire_arr):.3f}  (must be ~{1 / k:.2f})"
    )


def main():
    ks = [int(a) for a in sys.argv[1:]] or [2, 3, 4]
    for k in ks:
        generate(k)
        print()


if __name__ == "__main__":
    main()
