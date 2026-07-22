#!/usr/bin/env python
"""Repo-root CLI entrypoint.

Thin wrapper around ``fmri_cognition.cli.main`` -- see
``src/fmri_cognition/cli.py`` for argument parsing and orchestration.
Requires the package to be installed (``pip install -e .``) so that
``fmri_cognition`` is importable.

Usage:
    python main.py --help
    python main.py --manifest tract_manifest.csv --output-dir results_nn
"""
from fmri_cognition.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
