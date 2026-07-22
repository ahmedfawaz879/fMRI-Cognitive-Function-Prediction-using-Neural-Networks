"""Command-line entrypoint for the fMRI tract-based cognition pipeline.

Original script behavior (predict_cognition_nn.py, pre-refactor), for
comparison:

* Hardcoded ``MANIFEST_CSV = "tract_manifest.csv"``
* Hardcoded ``OUTPUT_DIR = "results_nn"``
* No CLI arguments at all -- always invoked as ``python predict_cognition_nn.py``
* Unconditionally called ``plt.show()`` at the end, which blocks when run
  headless/automated (e.g. in CI or a test suite)

This CLI keeps the same default manifest/output-dir *values* so running
it with no arguments reproduces the original script's paths, while
adding the ability to override them and making interactive plot display
opt-in (``--show-plot``) instead of automatic, so the pipeline doesn't
block when run non-interactively.
"""
from __future__ import annotations

import argparse
import logging

import yaml

from .data import load_manifest
from .train import run_cross_validation

DEFAULT_MANIFEST = "tract_manifest.csv"
DEFAULT_OUTPUT_DIR = "results_nn"
DEFAULT_CONFIG = "configs/default.yaml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fmri-cognition",
        description=(
            "Predict cognitive function from tract-level NIfTI features "
            "using an MLP trained with 5-fold cross-validation."
        ),
    )
    parser.add_argument(
        "--manifest",
        default=DEFAULT_MANIFEST,
        help=f"Path to the tract manifest CSV (default: {DEFAULT_MANIFEST!r}, "
        "same default the original script hardcoded).",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to write predictions/model/scaler/plot to "
        f"(default: {DEFAULT_OUTPUT_DIR!r}, same default the original script hardcoded).",
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help=f"Path to the hyperparameter config YAML (default: {DEFAULT_CONFIG!r}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override the seed defined in the config file.",
    )
    parser.add_argument(
        "--show-plot",
        action="store_true",
        help="Display the prediction scatter plot interactively. The original "
        "script always did this (plt.show()); it is now opt-in so automated "
        "runs don't block waiting on a GUI window.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level: DEBUG, INFO, WARNING, ERROR (default: INFO).",
    )
    return parser


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logger = logging.getLogger(__name__)

    config = load_config(args.config)
    if args.seed is not None:
        config["seed"] = args.seed

    logger.info("Loading manifest from %s", args.manifest)
    df = load_manifest(args.manifest)

    metrics = run_cross_validation(df, config, args.output_dir, show_plot=args.show_plot)
    logger.info("Final metrics: R2=%.3f, MAE=%.3f", metrics["r2"], metrics["mae"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
