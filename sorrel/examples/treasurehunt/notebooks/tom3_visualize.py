"""Render ToM v3 episodes as GIFs: watch phase + belief-driven door choice.

For each eval config, runs one episode: the observer sits at the chamber
center while the scripted demonstrator(s) reveal desire (preview pickups) and
latent (door commit); then the trained observer picks a door from its
inferred K-way posterior and walks there, continuing to the nearest gem once
inside. The `exclusion` config (K >= 3) is the money shot: the observer walks
through the one door nobody used.

Output: data/gifs/Tom3Demo_k<K>_<config>_epoch<seed>.gif

Usage:
    python -m sorrel.examples.treasurehunt.notebooks.tom3_visualize [K]
    (default: 4)
"""

import sys
from collections import deque

import numpy as np

from sorrel.examples.treasurehunt.notebooks.tom3_common import (
    DATA_DIR,
    ITEM_KINDS,
    T_WATCH,
    make_env,
    make_obs_spec,
    run_watch_phase,
)
from sorrel.examples.treasurehunt.notebooks.tom3_evaluate import (
    choose_door,
    eval_configs,
    factorized_posterior,
    infer,
    load_observer,
)
from sorrel.utils.visualization import ImageRenderer

GIF_DIR = DATA_DIR / "gifs"
SEED = 4242
N_ACT_STEPS = 20


def bfs_path(world, start, goal_test, traversable):
    """Shortest path from start to the nearest cell satisfying goal_test."""
    frontier = deque([start])
    came_from = {start: None}
    goal = None
    while frontier:
        node = frontier.popleft()
        if node != start and goal_test(node):
            goal = node
            break
        y, x = node
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nxt = (y + dy, x + dx)
            if nxt in came_from:
                continue
            if not (0 <= nxt[0] < world.height and 0 <= nxt[1] < world.width):
                continue
            if not traversable(nxt):
                continue
            came_from[nxt] = node
            frontier.append(nxt)
    if goal is None:
        return []
    path = [goal]
    while came_from[path[-1]] is not None:
        path.append(came_from[path[-1]])
    return list(reversed(path))[1:]  # drop start


def observer_route(world, start, room: str):
    """Route: through the chosen room's door, then to the nearest gem."""

    def walkable(n):
        kind = world.observe((n[0], n[1], 1)).kind
        return kind in ("EmptyEntity", *ITEM_KINDS)

    in_room = lambda n: world.room_of((n[0], n[1], 1)) == room
    path = bfs_path(world, start, in_room, walkable)
    if path:
        tail = bfs_path(
            world,
            path[-1],
            lambda n: world.observe((n[0], n[1], 1)).kind == "Gem",
            lambda n: walkable(n) and in_room(n),
        )
        path += tail
    return path


def main():
    k = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    encoder, desire_head, gem_head, latent_head = load_observer(k)
    obs_spec = make_obs_spec()

    for name, desires_cfg in eval_configs(k).items():
        env = make_env(k, desires_cfg, seed=SEED, epsilon=0.05)
        world = env.world
        rooms = world.used_rooms()
        renderer = ImageRenderer(
            experiment_name=f"Tom3Demo_k{k}_{name}",
            record_period=1,
            num_turns=T_WATCH + N_ACT_STEPS,
        )
        renderer.add_image(world)

        # --- watch phase (renders each turn) --------------------------------
        frames, disp, marks, commit_room, _ = run_watch_phase(
            env, obs_spec, on_turn=renderer.add_image
        )

        # --- inference + belief-driven walk ---------------------------------
        _, desires_hat, _, gem_probs = infer(
            k, encoder, desire_head, gem_head, latent_head, frames, disp, marks
        )
        post = (
            factorized_posterior(k, gem_probs, commit_room)
            if desires_cfg
            else np.full(k, 1.0 / k)
        )
        door = choose_door(post)
        correct = door == world.gem_room
        post_str = " ".join(f"{rooms[i]}={post[i]:.2f}" for i in range(k))
        print(
            f"{name:<14}: gem_room={rooms[world.gem_room]:<6} P=[{post_str}] "
            f"-> door={rooms[door]:<6} ({'CORRECT' if correct else 'WRONG'}) "
            f"desires={[ITEM_KINDS[d] for d in desires_hat]}"
        )
        start = (env.observer.location[0], env.observer.location[1])
        route = observer_route(world, start, rooms[door])
        for step_loc in route[:N_ACT_STEPS]:
            env.take_turn()  # watched agents keep foraging
            world.move(env.observer, (step_loc[0], step_loc[1], 1))
            renderer.add_image(world)

        renderer.save_gif(epoch=SEED, folder=GIF_DIR)
        print(f"  -> {GIF_DIR}/Tom3Demo_k{k}_{name}_epoch{SEED}.gif")


if __name__ == "__main__":
    main()
