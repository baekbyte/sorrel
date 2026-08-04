"""Paired same-layout contrast for preference-conditioned norm following.

For every seed this script creates two episodes with byte-identical initial
layouts.  Only the watched partner's desire changes (Gem versus Coin).  The
observer must therefore reverse which of those two kinds it leaves behind.

Usage::

    python -m sorrel.examples.treasurehunt.notebooks.norm_contrast 3
    python -m sorrel.examples.treasurehunt.notebooks.norm_contrast 3 --gif
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
from sorrel.examples.treasurehunt.notebooks.norm_evaluate import (
    N_FORAGE_STEPS,
    SANCTION,
)
from sorrel.examples.treasurehunt.notebooks.tom3_common import (
    DATA_DIR,
    make_obs_spec,
    run_watch_phase,
)
from sorrel.examples.treasurehunt.notebooks.tom3_evaluate import infer, load_observer
from sorrel.utils.visualization import ImageRenderer

N_PAIRS = 100
SEED0 = 500_000
GIF_SEED = 500_042


@dataclass
class Episode:
    available: dict[str, int]
    consumed: dict[str, int]
    posterior: np.ndarray
    fingerprint: tuple[tuple[int, int, str], ...]


def layout_fingerprint(env) -> tuple[tuple[int, int, str], ...]:
    """Initial non-agent layout, used to guarantee a real paired contrast."""
    return tuple(
        (y, x, entity.kind)
        for y in range(env.world.height)
        for x in range(env.world.width)
        for entity in (env.world.observe((y, x, 1)),)
        if entity.kind != "TreasurehuntAgent"
    )


def run_episode(k: int, desire: int, seed: int, *, models, renderer=None) -> Episode:
    encoder, desire_head, gem_head, latent_head = models
    env = make_norm_env(k, [desire], seed, sanction=SANCTION)
    fingerprint = layout_fingerprint(env)
    if renderer is not None:
        renderer.add_image(env.world)
    frames, disp, marks, _, _ = run_watch_phase(
        env, make_obs_spec(), on_turn=(renderer.add_image if renderer else None)
    )
    _, _, desire_probs, _ = infer(
        k, encoder, desire_head, gem_head, latent_head, frames, disp, marks
    )
    posterior = sanction_probabilities(desire_probs)
    available = env.count_items()
    for _ in range(N_FORAGE_STEPS):
        forage_step(env, posterior)
        if renderer is not None:
            renderer.add_image(env.world)
    assert env.norm_log is not None
    return Episode(available, dict(env.norm_log.consumed), posterior, fingerprint)


def rate(episode: Episode, kind: str) -> float:
    return episode.consumed.get(kind, 0) / max(1, episode.available[kind])


def evaluate(k: int) -> None:
    if k < 3:
        raise SystemExit("The Gem-versus-Coin contrast requires K >= 3.")
    models = load_observer(k)
    gem_rates: list[float] = []
    coin_rates: list[float] = []
    for offset in range(N_PAIRS):
        seed = SEED0 + offset
        gem_partner = run_episode(k, 0, seed, models=models)
        coin_partner = run_episode(k, 2, seed, models=models)
        if gem_partner.fingerprint != coin_partner.fingerprint:
            raise AssertionError(f"initial layouts differ for seed {seed}")
        gem_rates.append(rate(coin_partner, "Gem") - rate(gem_partner, "Gem"))
        coin_rates.append(rate(gem_partner, "Coin") - rate(coin_partner, "Coin"))
    print(f"K={k}; {N_PAIRS} paired initial layouts; S={SANCTION:g}")
    print("\nOnly partner desire differs between each pair.")
    print("consumption-rate change when partner changes Gem -> Coin")
    print(f"  Gem  (should rise): {np.mean(gem_rates):+.3f}")
    print(f"  Coin (should fall): {-np.mean(coin_rates):+.3f}")
    print(
        "\nPositive Gem and negative Coin changes are the same-layout preference "
        "contrast: the observed world cannot explain the policy flip."
    )


def make_gifs(k: int) -> None:
    folder = DATA_DIR / "gifs"
    models = load_observer(k)
    for label, desire in (("gem_partner", 0), ("coin_partner", 2)):
        renderer = ImageRenderer(
            experiment_name=f"NormContrast_k{k}_{label}",
            record_period=1,
            num_turns=20 + N_FORAGE_STEPS + 1,
        )
        episode = run_episode(k, desire, GIF_SEED, models=models, renderer=renderer)
        renderer.save_gif(GIF_SEED, folder)
        probs = " ".join(
            f"{kind}={episode.posterior[i]:.2f}"
            for i, kind in enumerate(ITEM_KINDS[:k])
        )
        print(
            f"{label}: posterior [{probs}], consumed={episode.consumed}\n"
            f"  -> {folder}/NormContrast_k{k}_{label}_epoch{GIF_SEED}.gif"
        )


def main() -> None:
    args = [arg for arg in sys.argv[1:] if arg != "--gif"]
    k = int(args[0]) if args else 3
    evaluate(k)
    if "--gif" in sys.argv[1:]:
        make_gifs(k)


if __name__ == "__main__":
    main()
