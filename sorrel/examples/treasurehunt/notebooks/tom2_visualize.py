"""Render ToM v2 episodes as GIFs: watch phase + belief-driven door choice.

For each watched config, runs one episode: the observer sits at corridor
center while the scripted demonstrator(s) reveal desire (preview pickups) and
latent (door choice); then the trained observer picks a door from its
inferred posterior and walks there, continuing to the nearest gem once inside.

Output: data/gifs/Tom2Demo_<config>_epoch<seed>.gif

Usage:
    python -m sorrel.examples.treasurehunt.notebooks.tom2_visualize
"""

from collections import deque

from sorrel.examples.treasurehunt.notebooks.tom2_common import (
    DATA_DIR,
    T_WATCH,
    make_env,
    make_obs_spec,
    run_watch_phase,
)
from sorrel.examples.treasurehunt.notebooks.tom2_evaluate import (
    choose_door,
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


def observer_route(world, start, door: str):
    """Route: through the chosen door, then to the nearest gem (if any)."""

    def walkable(n):
        kind = world.observe((n[0], n[1], 1)).kind
        return kind in ("EmptyEntity", "Gem", "Food")

    if door == "left":
        in_room = lambda n: n[1] < world.left_wall_x
    else:
        in_room = lambda n: n[1] > world.right_wall_x
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
    encoder, desire_head, latent_head = load_observer()
    obs_spec = make_obs_spec()

    for wc in ["gem_only", "food_only", "both", "none"]:
        env = make_env(wc, seed=SEED, epsilon=0.05)
        world = env.world
        renderer = ImageRenderer(
            experiment_name=f"Tom2Demo_{wc}",
            record_period=1,
            num_turns=T_WATCH + N_ACT_STEPS,
        )
        renderer.add_image(world)

        # --- watch phase (renders each turn) --------------------------------
        frames, disp, marks, _, _ = run_watch_phase(
            env, obs_spec, on_turn=renderer.add_image
        )

        # --- inference + belief-driven walk ---------------------------------
        p_left, desires = infer(encoder, desire_head, latent_head, frames, disp, marks)
        door = choose_door(p_left)
        correct = door == world.gem_side
        print(
            f"{wc:<10}: gem_side={world.gem_side:<5} P(left)={p_left:.3f} "
            f"-> door={door:<5} ({'CORRECT' if correct else 'WRONG'}) "
            f"desires={desires}"
        )
        start = (env.observer.location[0], env.observer.location[1])
        route = observer_route(world, start, door)
        for step_loc in route[:N_ACT_STEPS]:
            env.take_turn()  # watched agents keep foraging
            world.move(env.observer, (step_loc[0], step_loc[1], 1))
            renderer.add_image(world)

        renderer.save_gif(epoch=SEED, folder=GIF_DIR)
        print(f"  -> {GIF_DIR}/Tom2Demo_{wc}_epoch{SEED}.gif")


if __name__ == "__main__":
    main()
