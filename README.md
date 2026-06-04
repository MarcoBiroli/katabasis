# Katabasis

Physics-informed equivariant prediction of transition-state (TS) geometries and
reaction pathways from reactant (R) and product (P) structures.

The central inductive bias: **contractive descent flows from a saddle into a
basin are easier to learn than expansive ascent flows out of one**, and the
saddle is a privileged anchor that factorizes a reaction path into two such
descents. Two networks:

- **Network A** maps the R↔P-symmetric midpoint `M = (R + P) / 2` to the saddle.
  It is architecturally symmetric under the R↔P swap (shared encoder + symmetric
  combine) and predicts a *displacement* from `M` (residual output).
- **Network B** is a target-conditioned, contractive descent flow `v(x, target, t)`.
  At inference it is run twice — from the predicted saddle toward R and toward P —
  giving the full pathway and a cycle-consistency confidence score.

At inference only R and P are observed; the saddle is a latent. **The headline
TS-RMSD comes from Network A alone** — B contributes to it only as an auxiliary
training signal. Whether that signal beats a direct `M → saddle` baseline is the
load-bearing experiment, front-loaded into Phases 2 and 4 (see `PLAN.md`).

See `CLAUDE.md` for the scientific premise and the non-negotiable correctness
requirements, and `PLAN.md` for the phased plan.

## Install

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu   # or a CUDA wheel
pip install "omegaconf>=2.4.0.dev3"                                  # see note below
pip install -e ".[dev]"
```

> **Config note.** Configuration is built on **OmegaConf** directly (composable
> YAML + `key=value` / `group=option` CLI overrides) rather than full
> `hydra-core`, whose pinned `antlr4-python3-runtime` has no prebuilt wheel on
> some platforms. `omegaconf>=2.4` works with the modern antlr wheel.

## Quickstart (no dataset required)

Everything runs on CPU against a built-in **synthetic** reaction generator, so
the full pipeline is exercisable without downloading the 20M-structure dataset:

```bash
# Phase-0 equivariance canary
python scripts/smoke_equivariance.py --group SE3

# Tiny end-to-end runs on synthetic fixtures (seconds)
python scripts/run.py data=synthetic model=small train=smoke train.phase=3 run_dir=runs/b
python scripts/run.py data=synthetic model=small train=smoke train.phase=2 run_dir=runs/a_direct
python scripts/run.py data=synthetic model=small train=smoke train.phase=4 \
    train.aux_enabled=true ckpt_b=runs/b/net_b.pt run_dir=runs/a_aux
python scripts/evaluate.py data=synthetic model=small train=smoke \
    ckpt_a=runs/a_aux/net_a.pt ckpt_b=runs/b/net_b.pt
```

## Real data (Halo8)

```bash
python scripts/download_halo8.py --dry-run          # inspect Zenodo files
python scripts/download_halo8.py                     # download (~large)
python scripts/prepare_data.py data=t1x_core         # cache + filter + split
python scripts/run.py data=t1x_core model=default train=phase3 run_dir=runs/phase3
python scripts/run.py data=t1x_core train=phase2 run_dir=runs/phase2          # baseline
python scripts/run.py data=t1x_core train=phase4 ckpt_b=runs/phase3/net_b.pt run_dir=runs/phase4
python scripts/run.py data=t1x_core train=phase5 \
    ckpt_a=runs/phase4/net_a.pt ckpt_b=runs/phase3/net_b.pt run_dir=runs/phase5
```

Halo8 is an ASE database. The reader (`katabasis.data.halo8`) is **defensive
about the key-value schema** (configurable key names + fallbacks) because the
exact layout can only be confirmed against the real record — inspect a sample of
reactions before trusting the full set (see `PLAN.md` risk register).

## Repository layout

```
src/katabasis/
  data/        alignment (Kabsch + connectivity-constrained permutation),
               connectivity grouping, midpoint, filters, by-reaction splits,
               arc-length descent bridges, Halo8 reader, synthetic fixtures,
               batching, torch datasets, pipeline orchestration
  models/      irreps/group config, equivariant message-passing backbone,
               Network A (saddle), Network B (descent flow), atom embedding
  losses/      aligned RMSD (differentiable), flow matching (velocity-scale
               normalized), contractivity (symmetrized numerical abscissa)
  eval/        RK4 flow integration, metrics (saddle/path/cycle), FD Hessian
  train/       phase loops (2/3/4/5), shared steps, run logging (wandb optional)
configs/       hydra-style composable YAML (data/ model/ train/)
scripts/       smoke_equivariance, download_halo8, prepare_data, run, evaluate
tests/         CI-enforced correctness tests (see below)
```

## Equivariance scope: SE(3) vs O(3)

Pinned in `configs/config.yaml` (`group: SE3` by default). A point cloud carrying
only positions yields, through equivariant tensor products, natural-parity
features, so the e3nn backbone is O(3)-equivariant (hence also SE(3)). **The
group choice controls what the equivariance test asserts**: `O3` includes
reflections; `SE3` (chirality-preserving) deliberately excludes them, since
forcing reflection-equivariance would map a molecule to the TS of its mirror
image. Genuinely *discriminating* enantiomers further requires injecting a
pseudoscalar feature — a documented v1 limitation (`models/irreps.py`).

## Correctness guarantees (CI-enforced)

Run all checks locally with `bash scripts/ci.sh` (ruff + black + the Phase-0
equivariance canary + pytest). A GitHub Actions workflow is provided at
`.github/workflows-ci.yml.template`; copy it to `.github/workflows/ci.yml` to
enable CI (it ships as a template because the token used to create the repo
lacked the `workflow` OAuth scope).

The mandatory tests from `CLAUDE.md`'s "Critical correctness requirements":

| Requirement | Test |
|---|---|
| R/P alignment, never raw RMSD | `test_alignment.py` |
| Connectivity-constrained permutation (no cross-heavy-atom H swaps) | `test_connectivity.py` |
| End-to-end SE(3) equivariance (midpoint + both networks) | `test_midpoint_equivariance.py`, `test_network_equivariance.py` |
| R↔P symmetry of Network A | `test_network_equivariance.py` |
| No saddle leakage at inference | `test_inference_no_saddle_leakage.py` |
| Splits by reaction (no leakage) | `test_splits.py` |
| Contractivity = numerical abscissa, **not** spectral radius/abscissa | `test_contractivity.py` |
| Velocity-scale normalization balances short/long legs | `test_flow_matching.py` |
| Single-barrier (single-step) filtering | `test_filters.py` |

The contractivity test is worth highlighting: it includes a non-normal matrix
whose eigenvalues are all negative (spectral abscissa < 0) but whose symmetric
part has a positive eigenvalue — the surrogate correctly flags it as expansive,
where penalizing the spectral radius or abscissa would silently pass it.

## Implementation status

This repository is the **complete, tested software** for the method: data
pipeline, both equivariant networks, all three loss terms, the four phase
training loops, evaluation/ablation tooling, configs, and CI.

What is **validated here** (CPU, no GPU/data needed): exact equivariance and
R↔P symmetry (to float64 precision), the connectivity-constrained alignment, the
numerical-abscissa surrogate against analytic matrices, the arc-length bridge,
split/filtering logic, and that all four phases train end-to-end on synthetic
fixtures and produce finite metrics + a cycle-consistency correlation.

What **requires GPU + the real Halo8 dataset** (and is therefore not a number in
this repo): the actual saddle-RMSD results, the Phase-2-vs-Phase-4 comparison,
the halogen-transfer experiment, and the React-OT / OA-ReactDiff benchmarks.
The plan's acceptance gates are written to be honest either way — if the descent
machinery does not beat the direct baseline on TS geometry, its value is pathway
generation and confidence scoring, and that must be reported as such.

## License

MIT.
