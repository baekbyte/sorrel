"""Generate the ToM v2 observer dataset (third-person, scripted demonstrators).

Per episode: frozen observer at corridor center records its own 9x9 POV for
T_WATCH steps while scripted watched agents pick through the corridor preview
items (desire signal) and head through a door (latent signal). One training
example per watched agent per episode:

    frames (T, 7, 9, 9) uint8   observer POV + tracked-agent marker channel
                                (third person; never the watched agent's own
                                view)
    disp   (T,) int64           watched agent's OBSERVED displacement tokens
                                (up/down/left/right/stay), derived from its
                                position changes inside the observer's view
    desire {0=gem-lover, 1=food-lover}
    latent {0=gems left, 1=gems right}

Also runs the dataset-floor check: a logistic probe on displacement summary
features must be near-chance on the LATENT when desires are mixed (the task
requires the desire x direction interaction, not a direction shortcut).

Usage:
    python -m sorrel.examples.treasurehunt.notebooks.tom2_generate_data
"""

import numpy as np

from sorrel.examples.treasurehunt.notebooks.tom2_common import (
    DATASET_PATH,
    T_WATCH,
    WATCHED_CONFIGS,
    agent_window,
    make_env,
    make_obs_spec,
    run_watch_phase,
)

N_EPISODES_PER_CONFIG = 2000
EPSILON = 0.1
SEED0 = 1000


def main():
    obs_spec = make_obs_spec()
    frames_all, disp_all, desire_all, latent_all = [], [], [], []
    config_all, episode_all, door_all = [], [], []

    seed = SEED0
    for wc in WATCHED_CONFIGS:
        entered = 0
        n_windows = 0
        for _ in range(N_EPISODES_PER_CONFIG):
            env = make_env(wc, seed=seed, epsilon=EPSILON)
            frames, disp, marks, door_side, _ = run_watch_phase(env, obs_spec)
            latent = 0 if env.world.gem_side == "left" else 1
            for i, agent in enumerate(env.agents):
                desire = 0 if agent.target_kind == "Gem" else 1
                frames_all.append(agent_window(frames, marks[i]).astype(np.uint8))
                disp_all.append(disp[i])
                desire_all.append(desire)
                latent_all.append(latent)
                config_all.append(wc)
                episode_all.append(seed)
                door_all.append(
                    {"left": 0, "right": 1, None: -1}[door_side[i]]
                )
                entered += door_side[i] is not None
                n_windows += 1
            seed += 1
        print(
            f"{wc}: {N_EPISODES_PER_CONFIG} episodes, {n_windows} windows, "
            f"door-commit rate {entered / max(1, n_windows):.2f}"
        )

    frames_arr = np.stack(frames_all)          # (N, T, 6, 9, 9) uint8
    disp_arr = np.stack(disp_all)              # (N, T)
    desire_arr = np.array(desire_all, dtype=np.int64)
    latent_arr = np.array(latent_all, dtype=np.int64)
    door_arr = np.array(door_all, dtype=np.int64)
    episode_arr = np.array(episode_all, dtype=np.int64)
    config_arr = np.array(config_all)

    DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        DATASET_PATH,
        frames=frames_arr,
        disp=disp_arr,
        desire=desire_arr,
        latent=latent_arr,
        door=door_arr,
        episode=episode_arr,
        config=config_arr,
    )
    print(f"\nDataset: {frames_arr.shape} frames -> {DATASET_PATH}")
    print(f"label balance: desire mean={desire_arr.mean():.2f}, latent mean={latent_arr.mean():.2f}")

    # ---- dataset-floor checks ------------------------------------------------
    # Summary feature: net horizontal displacement of the watched agent.
    net_dx = (disp_arr == 3).sum(1) - (disp_arr == 2).sum(1)  # right - left
    went_right = (net_dx > 0).astype(int)

    def acc(pred, y):
        return max((pred == y).mean(), (1 - pred == y).mean())

    print("\nFloors (logistic-style single-feature probes):")
    print(f"  latent from direction alone (mixed desires): {acc(went_right, latent_arr):.3f}  (must be ~0.5)")
    for d, name in [(0, "gem-lover"), (1, "food-lover")]:
        m = desire_arr == d
        print(f"  latent from direction | desire={name}: {acc(went_right[m], latent_arr[m]):.3f}  (should be high)")
    print(f"  desire from direction alone: {acc(went_right, desire_arr):.3f}  (must be ~0.5)")


if __name__ == "__main__":
    main()
