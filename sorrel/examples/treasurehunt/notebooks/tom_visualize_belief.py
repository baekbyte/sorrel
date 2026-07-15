"""Visualize what the belief module fills the observer's outer ring with.

For each watched-agent condition (gem_only, food_only, both, none) we run a
short rollout with belief=on and render a three-panel GIF per condition:

    [ world ]   [ what observer SEES ]   [ what observer THINKS ]

  world                  : 20x20 grid with all agents marked. The observer's
                           15x15 input window is outlined; its inner 9x9 actual
                           FOV is outlined in green.
  what observer SEES     : the observer's masked 15x15 POV (raw model input).
                           Outer ring is black (cells the observer cannot see).
  what observer THINKS   : the belief-completed POV that pass 2 acts on.
                           Outer-ring cells are filled by the belief
                           reconstruction; their background is tinted so it
                           is obvious they came from belief, not direct sight.

This is the key diagnostic for whether `g` is preference-SELECTIVE: if the
outer ring fills the same way under gem_only and food_only, the architecture
is not differentiating watched-agent identity at the spatial level (which
matches the non-segregated experiment's null on `gem_only - food_only`).
If the outer ring fills with gem-channel under gem_only and food-channel
under food_only, the encoder IS preference-selective at the spatial level.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from sorrel.examples.treasurehunt.notebooks.tom_rollout import (
    ACTION_NAMES,
    ACTUAL_FOV_RADIUS,
    ARCH,
    EPISODE_LEN,
    build_env,
)

DATA_DIR = Path(__file__).parent / "../data"
GIF_DIR = DATA_DIR / "gifs"
EPISODE_LEN_VIZ = 60  # long enough for window to fill even when watched agents wander
SEED = 100
WATCHED_CONFIGS = ["gem_only", "food_only", "both", "none"]

# Channel order matches ENTITY_LIST in tom_rollout. Channel 6 (mask) is the
# observer's "unknown" indicator and is rendered specially (not via this list).
ENTITY_NAMES = ["EmptyEntity", "Wall", "Gem", "Bone", "Food", "Agent", "Mask"]
COLORS = [
    (0.94, 0.92, 0.84),  # 0 EmptyEntity - sand
    (0.30, 0.30, 0.30),  # 1 Wall        - dark grey
    (0.20, 0.55, 0.95),  # 2 Gem         - blue
    (0.85, 0.30, 0.30),  # 3 Bone        - red
    (0.30, 0.75, 0.40),  # 4 Food        - green
    (0.95, 0.85, 0.30),  # 5 Agent       - yellow
    (0.08, 0.08, 0.10),  # 6 Mask        - near-black (drawn explicitly below)
]
NUM_ENTITY_CHANNELS = 6  # channels 0..5 are entities; channel 6 is the mask
MASKED_COLOR = (0.08, 0.08, 0.10)  # near-black for "observer cannot see this cell"

# Subtle red tint on outer-ring cells in the belief-completed panel so it's
# obvious which cells came from belief rather than direct sight.
RING_TINT_STRENGTH = 0.28


def pov_to_rgb(pov: np.ndarray, mark_outer_ring_as_belief: bool = False) -> np.ndarray:
    """Render a (C, H, W) POV to an RGB image (H, W, 3).

    Channels 0..5 are entity channels (one-hot); channel 6 is the mask channel
    (1 where the observer cannot see the cell). Rendering rules:
      - mask channel = 1: render as near-black (truly hidden).
      - entity channels all 0 AND mask channel = 0: EmptyEntity (sand).
      - otherwise: argmax over entity channels gives the entity color.
    If mark_outer_ring_as_belief is True, outer-ring cells that DO have an
    entity (i.e. reconstructed content) are tinted red so they're visually
    distinct from directly observed cells.
    """
    C, H, W = pov.shape
    cy, cx = H // 2, W // 2
    rgb = np.full((H, W, 3), MASKED_COLOR, dtype=np.float32)
    has_mask_channel = C > NUM_ENTITY_CHANNELS
    for y in range(H):
        for x in range(W):
            # If the dedicated mask channel says this cell is hidden, draw black.
            if has_mask_channel and pov[NUM_ENTITY_CHANNELS, y, x] > 0.5:
                rgb[y, x] = MASKED_COLOR
                continue
            entity_sum = pov[:NUM_ENTITY_CHANNELS, y, x].sum()
            if entity_sum < 0.5:
                # No entity, not masked -> EmptyEntity (sand)
                rgb[y, x] = np.array(COLORS[0], dtype=np.float32)
                continue
            ch = int(pov[:NUM_ENTITY_CHANNELS, y, x].argmax())
            base_color = np.array(COLORS[ch], dtype=np.float32)
            in_outer_ring = max(abs(y - cy), abs(x - cx)) > ACTUAL_FOV_RADIUS
            if mark_outer_ring_as_belief and in_outer_ring:
                tint = np.array([0.95, 0.40, 0.40], dtype=np.float32)
                rgb[y, x] = (1 - RING_TINT_STRENGTH) * base_color + RING_TINT_STRENGTH * tint
            else:
                rgb[y, x] = base_color
    return rgb


def world_to_rgb(world, observer, watched_agents) -> np.ndarray:
    """Render the 20x20 world's top layer as an RGB image. Each agent is
    drawn as a small yellow / cyan / magenta dot on top of the cell color."""
    H, W = world.height, world.width
    rgb = np.full((H, W, 3), COLORS[0], dtype=np.float32)
    for y in range(H):
        for x in range(W):
            entity = world.observe((y, x, 1))
            kind = entity.kind
            if kind == "Wall":
                rgb[y, x] = COLORS[1]
            elif kind == "Gem":
                rgb[y, x] = COLORS[2]
            elif kind == "Bone":
                rgb[y, x] = COLORS[3]
            elif kind == "Food":
                rgb[y, x] = COLORS[4]
    return rgb


def render_frame(
    world,
    observer,
    watched,
    turn: int,
    watched_label: str,
    save_to: Path | None = None,
) -> np.ndarray:
    """Compose the three-panel frame as an RGB ndarray."""
    fig = plt.figure(figsize=(14, 5.3), dpi=110)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.4, 1.0, 1.0])
    ax_world = fig.add_subplot(gs[0, 0])
    ax_seen = fig.add_subplot(gs[0, 1])
    ax_thinks = fig.add_subplot(gs[0, 2])

    # ---------- Panel 1: world ----------
    ax_world.imshow(world_to_rgb(world, observer, watched), interpolation="nearest")
    # Mark agents.
    for ag, label, color in (
        [(observer, "OBS", "yellow")]
        + [
            (
                a,
                "gem" if a.preferences.get("Gem", 0) > 0.5 else "food",
                "cyan" if a.preferences.get("Gem", 0) > 0.5 else "magenta",
            )
            for a in watched
        ]
    ):
        y, x, _ = ag.location
        ax_world.scatter([x], [y], s=120, c=color, edgecolors="black", linewidths=0.8)
        ax_world.text(x, y, label, fontsize=6, ha="center", va="center", color="black")

    # Outline the observer's 15x15 model input and its inner 9x9 FOV.
    oy, ox, _ = observer.location
    full_r = ARCH["state_size"][1] // 2          # 7 for a 15x15 patch
    fov_r = ACTUAL_FOV_RADIUS                    # 4
    ax_world.add_patch(
        plt.Rectangle(
            (ox - full_r - 0.5, oy - full_r - 0.5),
            2 * full_r + 1,
            2 * full_r + 1,
            fill=False,
            edgecolor="orange",
            lw=1.4,
            label="15x15 input",
        )
    )
    ax_world.add_patch(
        plt.Rectangle(
            (ox - fov_r - 0.5, oy - fov_r - 0.5),
            2 * fov_r + 1,
            2 * fov_r + 1,
            fill=False,
            edgecolor="lime",
            lw=1.4,
            label="9x9 actual FOV",
        )
    )
    ax_world.set_xticks([])
    ax_world.set_yticks([])
    ax_world.set_title(f"World — turn {turn}", fontsize=11)

    # ---------- Panel 2: what observer SEES (masked POV) ----------
    if observer.last_masked_pov is not None:
        rgb_seen = pov_to_rgb(observer.last_masked_pov, mark_outer_ring_as_belief=False)
    else:
        rgb_seen = np.full((*ARCH["state_size"][1:], 3), MASKED_COLOR)
    ax_seen.imshow(rgb_seen, interpolation="nearest")
    H = ARCH["state_size"][1]
    cy = H // 2
    ax_seen.add_patch(
        plt.Rectangle(
            (cy - fov_r - 0.5, cy - fov_r - 0.5),
            2 * fov_r + 1,
            2 * fov_r + 1,
            fill=False,
            edgecolor="lime",
            lw=1.4,
        )
    )
    ax_seen.set_xticks([])
    ax_seen.set_yticks([])
    ax_seen.set_title("What observer SEES\n(masked 15x15)", fontsize=10)

    # ---------- Panel 3: what observer THINKS (belief-completed) ----------
    if observer.last_completed_pov is not None:
        rgb_thinks = pov_to_rgb(
            observer.last_completed_pov, mark_outer_ring_as_belief=True
        )
    else:
        rgb_thinks = np.full((*ARCH["state_size"][1:], 3), MASKED_COLOR)
    ax_thinks.imshow(rgb_thinks, interpolation="nearest")
    ax_thinks.add_patch(
        plt.Rectangle(
            (cy - fov_r - 0.5, cy - fov_r - 0.5),
            2 * fov_r + 1,
            2 * fov_r + 1,
            fill=False,
            edgecolor="lime",
            lw=1.4,
        )
    )
    ax_thinks.set_xticks([])
    ax_thinks.set_yticks([])
    title_thinks = "What observer THINKS\n(belief-completed; ring tinted red)"
    if not observer.last_belief_was_used:
        title_thinks += "  [belief OFF]"
    ax_thinks.set_title(title_thinks, fontsize=10)

    # Top-level annotation: watched config + per-agent inferred g norms + action.
    parts = [f"Watched: {watched_label}"]
    if observer.inferences:
        parts.extend(
            f"||{k.replace('_norm', '')}||={v:.1f}" for k, v in observer.inferences.items()
        )
    if observer.last_belief_recon is not None:
        recon = observer.last_belief_recon
        parts.append(
            f"ring: gem={recon.get('gem_in_ring', 0):.2f}  food={recon.get('food_in_ring', 0):.2f}"
        )
    parts.append(f"action: {observer.last_action_name.upper()}")
    fig.suptitle("    |    ".join(parts), fontsize=10, y=0.99)

    # Legend (tiny, bottom-right).
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, fc=COLORS[i]) for i in (2, 3, 4, 5, 0, 1)
    ]
    legend_labels = ["Gem", "Bone", "Food", "Agent", "Empty/sand", "Wall"]
    fig.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        ncol=6,
        fontsize=7,
        frameon=False,
        bbox_to_anchor=(0.5, -0.01),
    )

    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    if save_to is not None:
        Image.fromarray(img).save(save_to)
    return img


def run_one(watched_config: str) -> dict:
    env, observer = build_env(
        watched_config=watched_config, belief_on=True, seed=SEED
    )
    frames = []
    # Aggregate per-step outer-ring channel composition (the diagnostic).
    # ring_argmax_counts[c] = how many cells got argmax channel c summed over
    # all frames in which a belief was computed.
    ring_argmax_counts = np.zeros(NUM_ENTITY_CHANNELS, dtype=np.int64)
    n_belief_frames = 0
    H = ARCH["state_size"][1]
    cy = H // 2
    yy, xx = np.meshgrid(np.arange(H), np.arange(H), indexing="ij")
    outer_ring_mask = np.maximum(np.abs(yy - cy), np.abs(xx - cy)) > ACTUAL_FOV_RADIUS

    print(
        f"[{watched_config}] observer at {observer.location}, "
        f"watched at {[a.location for a in observer.watched_agents]}"
    )
    for turn in range(EPISODE_LEN_VIZ):
        env.take_turn()
        frames.append(
            render_frame(env.world, observer, observer.watched_agents, turn, watched_config)
        )
        if observer.last_belief_was_used and observer.last_completed_pov is not None:
            # argmax over ENTITY channels (0..5) only; the mask channel is a
            # separate book-keeping channel and shouldn't enter the histogram.
            entity_only = observer.last_completed_pov[:NUM_ENTITY_CHANNELS]
            ring_argmax_counts += np.bincount(
                entity_only.argmax(0)[outer_ring_mask],
                minlength=NUM_ENTITY_CHANNELS,
            )
            n_belief_frames += 1
        if env.world.is_done:
            break

    GIF_DIR.mkdir(parents=True, exist_ok=True)
    out = GIF_DIR / f"BeliefViz_{watched_config}.gif"
    pil_frames = [Image.fromarray(f) for f in frames]
    pil_frames[0].save(
        out, save_all=True, append_images=pil_frames[1:], duration=400, loop=0
    )
    print(f"  saved: {out}")
    print(
        f"  observer collected: gems={observer.collected['Gem']} "
        f"food={observer.collected['Food']} bones={observer.collected['Bone']}"
    )
    return {
        "watched": watched_config,
        "ring_counts": ring_argmax_counts,
        "n_belief_frames": n_belief_frames,
    }


def main() -> None:
    summary = [run_one(wc) for wc in WATCHED_CONFIGS]
    # Aggregate report: what does the belief module fill the outer ring with?
    print()
    print("=" * 78)
    print("OUTER-RING ARGMAX COMPOSITION (over the rollout, % of outer-ring cells)")
    print("=" * 78)
    # Only entity-channel columns are reported in the diagnostic; the mask
    # channel is bookkeeping, not a real entity.
    header = f"{'watched':<12s}" + "".join(
        f"{name:>10s}" for name in ENTITY_NAMES[:NUM_ENTITY_CHANNELS]
    )
    print(header)
    print("-" * len(header))
    for row in summary:
        wc = row["watched"]
        counts = row["ring_counts"]
        total = counts.sum()
        if total == 0:
            print(f"{wc:<12s}" + "  (no belief frames in this rollout)")
            continue
        pcts = counts / total * 100
        print(
            f"{wc:<12s}"
            + "".join(f"{p:>9.1f}%" for p in pcts)
            + f"   (n={row['n_belief_frames']} belief frames)"
        )
    print()
    print(
        "Read this table to answer: does the outer ring fill the SAME way "
        "regardless of who's watched (architecture not selective), or does "
        "watching the gem-lover fill it with more gems and watching the "
        "food-lover with more food (architecture IS selective)?"
    )


if __name__ == "__main__":
    main()
