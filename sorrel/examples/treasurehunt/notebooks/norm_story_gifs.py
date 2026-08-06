"""Create annotated GIFs that communicate the three Phase 1 norm results.

The generated animations are intended for a report or presentation rather
than raw debugging. They preserve the actual environment rollouts, then add
only explanatory overlays: watch/forage phase, inferred protected kinds, and
the final consumption outcome.

Outputs (under ``data/gifs``):

* ``NormStory_k3_matched_contrast.gif`` — matched Gem versus Coin partner;
* ``NormStory_k3_visibility_recovery.gif`` — baseline versus visibility model
  when a tracked partner's evidence disappears after step eight; and
* ``NormStory_k3_multi_partner.gif`` — Gem+Coin partners, with Food as the
  selectively permitted kind.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from sorrel.examples.treasurehunt.notebooks.norm_common import (
    ITEM_KINDS,
    allowed_kinds,
    forage_step,
    make_norm_env,
    sanction_probabilities,
)
from sorrel.examples.treasurehunt.notebooks.norm_evaluate import N_FORAGE_STEPS, SANCTION
from sorrel.examples.treasurehunt.notebooks.norm_visibility_compare import corrupt_evidence
from sorrel.examples.treasurehunt.notebooks.tom3_common import (
    DATA_DIR,
    make_obs_spec,
    observer_ckpt,
    run_watch_phase,
    visibility_observer_ckpt,
)
from sorrel.examples.treasurehunt.notebooks.tom3_evaluate import infer, load_observer
from sorrel.utils.visualization import image_from_array, render_sprite

K = 3
WATCH_STEPS = 20
GIF_SEED = 500_042
VISIBILITY_SEED0 = 1_252_000  # Gem+Coin, epsilon=.1, hidden-after-8 comparison cell.
MULTI_SEED0 = 700_000


@dataclass
class Rollout:
    frames: list[Image.Image]
    posterior: np.ndarray
    allowed: set[str]
    consumed: dict[str, int]
    sanctions: int
    desires: tuple[int, ...]
    evidence: str


def font(size: int) -> ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/Library/Fonts/Arial.ttf",
    ):
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


FONT_TITLE = font(17)
FONT_TEXT = font(13)


def world_image(world) -> Image.Image:
    return image_from_array(render_sprite(world)).convert("RGB")


def short_counts(consumed: dict[str, int]) -> str:
    return ", ".join(f"{kind}:{consumed.get(kind, 0)}" for kind in ITEM_KINDS[:K])


def posterior_text(posterior: np.ndarray) -> str:
    return "  ".join(f"{kind}={prob:.2f}" for kind, prob in zip(ITEM_KINDS[:K], posterior))


def record_rollout(
    desires: tuple[int, ...], seed: int, models, *, epsilon: float = 0.1,
    evidence: str = "full", watch_steps: int = WATCH_STEPS, capture: bool = True,
) -> Rollout:
    """Run one exact norm rollout and retain every world frame."""
    env = make_norm_env(K, list(desires), seed, epsilon=epsilon, sanction=SANCTION)
    recorded = [world_image(env.world)] if capture else []
    frames, disp, marks, _, _ = run_watch_phase(
        env, make_obs_spec(),
        on_turn=(lambda world: recorded.append(world_image(world))) if capture else None,
    )
    frames, disp, marks = corrupt_evidence(frames, disp, marks, evidence, watch_steps)
    encoder, desire_head, gem_head, latent_head = models
    _, _, desire_probs, _ = infer(
        K, encoder, desire_head, gem_head, latent_head, frames, disp, marks
    )
    posterior = sanction_probabilities(desire_probs)
    allowed = allowed_kinds(posterior, SANCTION)
    for _ in range(N_FORAGE_STEPS):
        forage_step(env, posterior)
        if capture:
            recorded.append(world_image(env.world))
    assert env.norm_log is not None
    return Rollout(
        frames=recorded,
        posterior=posterior,
        allowed=allowed,
        consumed=dict(env.norm_log.consumed),
        sanctions=env.norm_log.sanctions,
        desires=desires,
        evidence=evidence,
    )


def protected_eaten(rollout: Rollout) -> int:
    return sum(rollout.consumed.get(ITEM_KINDS[d], 0) for d in rollout.desires)


def find_visibility_example(baseline_models, visibility_models) -> int:
    """Find a deterministic Gem+Coin episode where visibility avoids an error."""
    for offset in range(250):
        seed = VISIBILITY_SEED0 + offset
        baseline = record_rollout(
            (0, 2), seed, baseline_models, evidence="hidden_after_8", watch_steps=8, capture=False
        )
        visibility = record_rollout(
            (0, 2), seed, visibility_models, evidence="hidden_after_8", watch_steps=8, capture=False
        )
        if protected_eaten(baseline) > protected_eaten(visibility) and protected_eaten(visibility) == 0:
            return seed
    raise RuntimeError("No visibility-recovery episode found in the configured seed range.")


def find_multi_example(models) -> int:
    """Find a clear selective-compliance Gem+Coin rollout with Food consumption."""
    for offset in range(250):
        seed = MULTI_SEED0 + offset
        rollout = record_rollout((0, 2), seed, models, capture=False)
        if protected_eaten(rollout) == 0 and rollout.consumed.get("Food", 0) > 0:
            return seed
    raise RuntimeError("No selective multi-partner episode found in the configured seed range.")


def phase(frame_index: int) -> str:
    if frame_index <= WATCH_STEPS:
        return f"WATCH  {frame_index:02d}/{WATCH_STEPS}"
    return f"FORAGE  {frame_index - WATCH_STEPS:02d}/{N_FORAGE_STEPS}"


def draw_panel(canvas: Image.Image, rollout: Rollout, frame_index: int, x: int, label: str) -> None:
    image = rollout.frames[min(frame_index, len(rollout.frames) - 1)]
    canvas.paste(image, (x, 54))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((x, 35, x + image.width, 54), fill=(31, 41, 55))
    draw.text((x + 6, 37), label, fill="white", font=FONT_TEXT)
    if rollout.evidence == "hidden_after_8" and frame_index > 8:
        draw.text((x + image.width - 144, 37), "partner evidence hidden", fill=(255, 205, 90), font=FONT_TEXT)


def compose_side_by_side(title: str, left: Rollout, right: Rollout, left_label: str, right_label: str, output: Path) -> None:
    count = max(len(left.frames), len(right.frames))
    image = left.frames[0]
    width, height = image.width * 2, image.height + 100
    composed: list[Image.Image] = []
    for i in range(count):
        canvas = Image.new("RGB", (width, height), (14, 23, 36))
        draw = ImageDraw.Draw(canvas)
        draw.text((8, 7), title, fill="white", font=FONT_TITLE)
        draw.text((8, 25), phase(i), fill=(178, 208, 245), font=FONT_TEXT)
        draw_panel(canvas, left, i, 0, left_label)
        draw_panel(canvas, right, i, image.width, right_label)
        composed.append(canvas)
    draw = ImageDraw.Draw(composed[-1])
    draw.rectangle((0, height - 46, width, height), fill=(19, 51, 48))
    left_summary = f"LEFT  inferred: {posterior_text(left.posterior)} | protect: {', '.join(ITEM_KINDS[d] for d in left.desires)} | ate {short_counts(left.consumed)} | sanctions {left.sanctions}"
    right_summary = f"RIGHT  inferred: {posterior_text(right.posterior)} | protect: {', '.join(ITEM_KINDS[d] for d in right.desires)} | ate {short_counts(right.consumed)} | sanctions {right.sanctions}"
    draw.text((8, height - 43), left_summary, fill="white", font=FONT_TEXT)
    draw.text((8, height - 23), right_summary, fill="white", font=FONT_TEXT)
    composed.extend([composed[-1]] * 8)
    output.parent.mkdir(parents=True, exist_ok=True)
    composed[0].save(output, save_all=True, append_images=composed[1:], duration=100, loop=0)


def compose_single(title: str, rollout: Rollout, output: Path) -> None:
    image = rollout.frames[0]
    width, height = image.width, image.height + 120
    composed: list[Image.Image] = []
    for i, raw in enumerate(rollout.frames):
        canvas = Image.new("RGB", (width, height), (14, 23, 36))
        draw = ImageDraw.Draw(canvas)
        draw.text((8, 7), title, fill="white", font=FONT_TITLE)
        draw.text((8, 25), phase(i), fill=(178, 208, 245), font=FONT_TEXT)
        canvas.paste(raw, (0, 54))
        composed.append(canvas)
    draw = ImageDraw.Draw(composed[-1])
    protected = ", ".join(ITEM_KINDS[d] for d in rollout.desires)
    draw.rectangle((0, height - 66, width, height), fill=(19, 51, 48))
    draw.text((8, height - 61), f"inferred: {posterior_text(rollout.posterior)}", fill="white", font=FONT_TEXT)
    draw.text((8, height - 41), f"protect: {protected} | allowed: {', '.join(sorted(rollout.allowed))}", fill="white", font=FONT_TEXT)
    draw.text((8, height - 21), f"ate {short_counts(rollout.consumed)} | sanctions {rollout.sanctions}", fill="white", font=FONT_TEXT)
    composed.extend([composed[-1]] * 8)
    output.parent.mkdir(parents=True, exist_ok=True,)
    composed[0].save(output, save_all=True, append_images=composed[1:], duration=100, loop=0)


def main() -> None:
    folder = DATA_DIR / "gifs"
    baseline = load_observer(K, observer_ckpt(K))
    visibility = load_observer(K, visibility_observer_ckpt(K))

    gem = record_rollout((0,), GIF_SEED, baseline)
    coin = record_rollout((2,), GIF_SEED, baseline)
    compose_side_by_side(
        "Matched layout: partner preference changes the protected kind",
        gem, coin,
        "Gem partner",
        "Coin partner",
        folder / "NormStory_k3_matched_contrast.gif",
    )

    seed = find_visibility_example(baseline, visibility)
    baseline_rollout = record_rollout(
        (0, 2), seed, baseline, evidence="hidden_after_8", watch_steps=8
    )
    visibility_rollout = record_rollout(
        (0, 2), seed, visibility, evidence="hidden_after_8", watch_steps=8
    )
    compose_side_by_side(
        f"Partial evidence recovery (Gem+Coin, seed {seed})",
        baseline_rollout, visibility_rollout,
        "Baseline observer",
        "Visibility-trained observer",
        folder / "NormStory_k3_visibility_recovery.gif",
    )

    seed = find_multi_example(baseline)
    multi = record_rollout((0, 2), seed, baseline)
    compose_single(
        f"Two partners: selectively defer to Gem + Coin (seed {seed})",
        multi,
        folder / "NormStory_k3_multi_partner.gif",
    )
    print(f"saved GIFs -> {folder}")


if __name__ == "__main__":
    main()
