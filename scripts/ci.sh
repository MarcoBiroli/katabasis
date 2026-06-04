#!/usr/bin/env bash
# Local CI: the same checks the GitHub Actions workflow runs (lint, format,
# the Phase-0 equivariance canary, and the test suite). Run from the repo root.
#
# A ready-to-use GitHub Actions workflow is provided at
# `.github/workflows-ci.yml.template`; copy it to `.github/workflows/ci.yml`
# to enable CI on push/PR. (It ships as a template because the automation token
# used to create this repo lacked the `workflow` OAuth scope.)
set -euo pipefail

echo "==> ruff"
ruff check src tests scripts

echo "==> black --check"
black --check src tests scripts

echo "==> equivariance smoke (Phase-0 canary)"
python scripts/smoke_equivariance.py --group SE3

echo "==> pytest"
pytest -q

echo "All CI checks passed."
