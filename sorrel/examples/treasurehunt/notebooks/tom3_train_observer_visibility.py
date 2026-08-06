"""Train the visibility-robust ToM v3 desire-inference variant.

The baseline encoder is trained with whole-window truncation. This variant
adds tracked-partner suffix masking while retaining the evolving shared scene,
matching the partial FOV-gated evidence associated with multi-agent norm
errors. It writes a separate checkpoint and never overwrites the baseline.

Usage:
    python -m sorrel.examples.treasurehunt.notebooks.tom3_train_observer_visibility 3
"""

import argparse

from sorrel.examples.treasurehunt.notebooks.tom3_train_observer import train


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("ks", nargs="*", type=int, default=[3])
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Override the normal per-K training budget (for smoke tests only).",
    )
    args = parser.parse_args()
    for k in args.ks:
        train(k, variant="visibility", steps_override=args.steps)


if __name__ == "__main__":
    main()
