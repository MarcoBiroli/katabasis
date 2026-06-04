# Katabasis

## Project overview

Katabasis is a physics-informed equivariant model for predicting transition state (TS) geometries and reaction pathways from reactant (R) and product (P) structures. The name comes from the Greek for "descent": the architecture is built on the inductive bias that *contractive* descent flows from a saddle into a basin are easier to learn than expansive ascent flows from a basin out to a saddle, and that the saddle is a privileged anchor that factorizes the transition path into two such descents.

The model is a two-network pipeline:

- **Network A** maps the symmetric midpoint `M = (R + P) / 2` to a predicted saddle structure. Trained against the ground-truth TS in the dataset. **A must be architecturally symmetric under the R↔P swap** (see "R↔P symmetry of Network A" below) — feeding it a symmetric `M` is not sufficient.
- **Network B** is target-conditioned: it takes `(current_state, target_geometry)` and produces an equivariant velocity / displacement field, supervised to flow contractively into the target's basin. At inference it is run twice — from the predicted saddle toward R and toward P. R↔P symmetry is enforced *by construction* via the conditioning mechanism (a single network handles both descents).

At inference time only R and P are observed. The saddle is a **latent** in the model. The descent flows serve three roles: (a) auxiliary supervision for A during training, (b) full-pathway generation, and (c) a cycle-consistency self-check (descent endpoints should land on R and P; the discrepancy is a usable confidence score).

**What determines the headline metric.** At inference the predicted TS geometry comes out of Network A alone; B does not refine the saddle at test time. So B's contribution to the benchmark TS-RMSD is *only* the auxiliary training signal (and joint fine-tuning). Whether that signal actually improves the saddle is the central hypothesis of the project, and it is tested as early as possible — see PLAN.md Phases 2 and 4. Treat that comparison as the load-bearing experiment, not a Phase-5 afterthought.

## Scientific premise — and what it isn't

The motivating intuition is **not** the loose claim that "downhill is easier than uphill" thermodynamically. The defensible claim is geometric: descent flows from a saddle into a basin are **contractive** — nearby trajectories converge onto the minimum, and the linearized dynamics has negative-real-part eigenvalues. Ascent flows from a basin are expansive and have to make a measure-zero choice about which saddle to exit through. Network B should be regularized to enforce contractivity explicitly.

Note that contractivity is something we *impose* on B, not something B discovers: B is target-conditioned, so it transports to whichever target it is given regardless of which side of the true separatrix it starts on. The contractivity regularizer is therefore a prior whose value is itself an experimental question — the "remove contractivity regularization" ablation tests it directly.

The midpoint `M = (R + P) / 2` is **not** assumed to be the saddle, nor close to it for reactions with curved reaction coordinates (SN2 with backside attack, eliminations, halogen abstractions). M is treated as a deterministic, R↔P-symmetric initialization that A corrects toward the true saddle; for these curved cases M can be a chemically unphysical structure, so A must lean on R and P (which it also receives), not just on M. This subtlety is load-bearing — diagnostics that conflate M with the saddle are wrong.

## Tech stack

- Python ≥ 3.11
- PyTorch ≥ 2.4
- `e3nn` for SE(3)-equivariant primitives. Alternatives (NequIP, MACE) are reasonable swaps; default to `e3nn` unless there's a strong reason.
- ASE for molecular structure handling and Halo8 access (Halo8 ships as an ASE database)
- `hydra-core` for experiment configs
- `wandb` for experiment tracking
- `pytest` for tests, `ruff` for linting, `black` for formatting

## Data

**Halo8** (Zenodo record 16737590; Nature Sci. Data 2025, doi 10.1038/s41597-025-05944-3): ASE database of ~20M structures from ~19,000 reaction pathways at ωB97X-3c. Combines recalculated Transition1x reactions (C/N/O, ≤7 heavy atoms) with halogen-substituted reactions from GDB-13 (≤8 heavy atoms; F/Cl/Br substitutions). Each reaction provides:

- Reactant geometry (R)
- Product geometry (P)
- Transition state geometry (saddle)
- NEB images along the minimum energy pathway
- Energies, forces, dipole moments, partial charges

Halo8 should ship with R, P, TS, and NEB images sharing a coordinate frame within each reaction (it comes from a single NEB calculation), but **verify, don't assume**.

**Single-step reactions only.** The whole `M → single saddle → two descents` premise assumes one first-order saddle between R and P. NEB occasionally returns paths with a hidden intermediate minimum (two barriers) or a kinked band; these are multi-step reactions masquerading as single-step and they break the factorization. Verify each reaction has a single-barrier energy profile (monotone up to the saddle, monotone down to the endpoint) and exclude the rest in v1. This is a data filter, specified in PLAN.md Phase 1.

Three subsets to define:
- `t1x_core`: recalculated Transition1x reactions only (no halogens). Phase 1 sanity checks and ablations.
- `halogen_only`: halogen-substituted reactions only. Transfer experiments. Note the transfer test probes *geometric* transfer; the electronic structure of heavy halogens (polarizability, relativistic effects, bonding) differs from C/N/O, so do not expect the energetics — and thus the PES shape near the saddle — to transfer as cleanly as the geometry.
- `full`: complete Halo8.

Splits are **by reaction**, not by image — every NEB image from a held-out reaction goes to the same split, otherwise validation leaks. Enforce this in the loader and add a test.

## Code conventions

- Type hints everywhere; `from __future__ import annotations` at module top.
- Tensor shape annotations as comments: `# (B, N, 3)`.
- Configs in `configs/` (Hydra), code in `src/katabasis/`, scripts in `scripts/`, runs logged to `runs/`.
- No magic numbers in code — everything from config.
- Tests next to the module they test as `test_*.py`.
- Use Kabsch + permutation-aware alignment from `src/katabasis/data/alignment.py` everywhere two structures are compared. **Never raw Cartesian RMSD.** The permutation matching must be **connectivity-constrained**: only atoms that are genuinely interchangeable may be permuted (the three H on a single methyl, the three F on a CF₃), and the assignment must respect element *and* bonded environment — a global Hungarian match over all atoms of one element will happily swap an H on carbon A with an H on carbon B and silently corrupt every RMSD in the project.

## Equivariance scope: rotations, translations, parity

The pipeline must be SE(3)-equivariant end to end (see Critical correctness requirement 2). One subtlety to decide *explicitly* and document: SE(3) (rotations + translations) preserves chirality, whereas O(3) (adding parity/reflections) does not. `e3nn` models frequently include odd-parity irreps by default, which makes them reflection-equivariant — they would map a molecule to the TS of its mirror image. For reactions where stereochemistry matters this is wrong, and you want to *preserve* chirality (no improper operations). Pin the intended group in config, choose irrep parities accordingly, and have the equivariance test cover (or deliberately exclude) reflections to match that choice.

## Conditioning mechanism for Network B

Target conditioning is via the **target-displacement vector field**: at each atom, `(target_position - current_position)` is fed in as an equivariant l=1 feature alongside standard atomic features. This is automatically SE(3)-equivariant and encodes "how far am I from the target" at every node. Optionally augment with FiLM conditioning on a scalar invariant (e.g., per-atom RMSD-to-target) at each layer for stronger global awareness.

**Do not** simply concatenate target Cartesian coordinates as additional channels — that breaks equivariance unless very carefully handled.

## R↔P symmetry of Network A

The transition state is the same structure regardless of which endpoint we label "reactant," so A's prediction must be invariant under swapping R and P. The midpoint `M = (R + P) / 2` is symmetric, but A also consumes R and P directly (Phase 4 architecture: `(R, P, M) → displacement from M`), and an ordered `(R, P)` input lets the network treat the two channels differently and break the symmetry. Enforce symmetry **architecturally**: encode R and P with a *shared* encoder and combine them with a symmetric operation (sum/mean of the two encodings, or symmetric features such as functions of `|R − P|`), so swapping R and P leaves the predicted saddle unchanged by construction. Add a unit test: predict from `(R, P)` and from `(P, R)`; the aligned saddle predictions must match within float tolerance.

## Training objective for Network B

Flow matching along an arc-length parameterized bridge through the NEB descent images. The supervision is sitting in the data — the NEB images sample the descent path, and flow matching regresses the network's velocity prediction against the path tangent at sampled times.

- **Time parameterization: arc length, not NEB-image index.** Compute cumulative arc length per descent leg once at preprocessing time and persist. NEB-index parameterization is a uniform subdivision of *something*, but that something depends on optimizer details and is noisy across reactions.
- **Bridge construction.** For a descent leg (saddle → endpoint), let `s ∈ [0, L]` be arc length and `t = s / L ∈ [0, 1]`. At training, sample `t`, interpolate `x_t` between adjacent NEB images linearly in arc length (or cubic spline if smoothness matters — verify on actual images first; cubic can overshoot near high curvature), and compute the path tangent `dx/dt` by finite differences on the same images. Regress `v_θ(x_t, endpoint, t)` against `dx/dt`.
- **Velocity-scale normalization.** Since `dx/dt = L · (unit tangent)`, the magnitude of the regression target scales with the leg's total arc length `L`, which varies by orders of magnitude across the dataset; left unaddressed, the loss is dominated by long-path reactions. Either predict the unit tangent (direction) and a scalar speed separately, or normalize positions by a per-reaction characteristic scale and report RMSD after un-normalizing. Decide and fix this before training B.
- **Time sampling: bias toward small `t`.** Most of the interesting dynamics is near the saddle, where the trajectory commits to a basin. Default: `t ∼ Beta(0.5, 1.5)`. Uniform sampling is wrong — it overweights the harmonic-relaxation tail near the endpoint. (Note the `0.5` shape puts a singularity at `t = 0`; if saddle-proximal targets prove ill-conditioned, soften toward both shape parameters > 1 or use a mixture.)
- **Equivariance check.** Linear interpolation between NEB images preserves equivariance only if the images share a frame within the reaction. Halo8 satisfies this; verify with a unit test.

Loss is three terms:

```
L_B =        E_{reaction, t ~ p(t), endpoint ∈ {R, P}}  || v_θ(x_t, endpoint, t) − dx/dt ||²
    + λ_c · E[ ReLU( λ_max( sym(J_v) evaluated at target ) ) ]      # sym(M) = (M + Mᵀ)/2
    + λ_e · (late-training only) endpoint RMSD after integrating v_θ from x_0 to t = 1
```

with `p(t)` biased toward `t = 0`, `λ_c ∈ [0.01, 0.1]` tuned so the regularizer settles to ~10% of the velocity loss at convergence, and `λ_e = 0` for the first ~80% of training, `~0.1` thereafter. The contractivity regularizer is evaluated only at `t` values away from the saddle (e.g., `t > 0.2`) — at the saddle, the relevant mode is expansive by construction.

**Contractivity surrogate — penalize the symmetric part, not the spectral radius.** Contractivity here means *monotone* convergence of nearby trajectories, which is governed by the numerical abscissa `μ(J_v) = λ_max(sym(J_v))`, not by the eigenvalues of `J_v` itself. The relation `‖exp(J_v · τ)‖ ≤ exp(μ(J_v) · τ)` makes `μ < 0` the correct contraction condition. Two traps to avoid: (1) plain power iteration on `J_v` returns the largest-*magnitude* eigenvalue (spectral radius), which controls nothing relevant; (2) even the largest-*real-part* eigenvalue (spectral abscissa) being negative does **not** imply contraction — equivariant Jacobians are typically non-normal, and a non-normal `J_v` with all eigenvalues in the left half-plane can still grow transiently. So estimate `μ` by power iteration on the **symmetrized** Jacobian-vector product `u ↦ ½(J_v u + J_vᵀ u)` (both directions are available via autograd), and optionally use Hutchinson's estimator on `tr(J_v)` as a cheap averaged-contractivity signal. The JVP-based estimate adds extra backward passes per step — see PLAN.md for applying it on a batch subsample / every k steps to bound cost.

**Evaluation integrates the flow, and scores the path, not just the endpoint.** Pointwise velocity error is necessary but not sufficient. At evaluation, integrate `v_θ` with a fixed-step solver (RK4 with ~32 steps is the default) from `x_0` to `t = 1` and report endpoint RMSD against ground truth. But endpoint RMSD alone is nearly tautological for a *target-conditioned* flow — B is told where to go, so a network that teleports to the target along a non-physical path can still score well, which would quietly invalidate pathway generation (role b). So also report **intermediate path RMSD**: at several `t ∈ (0, 1)`, compare the integrated `x_t` against the arc-length-matched NEB image. A good descent matches the path, not only its endpoint. If the gap between training velocity error and integrated endpoint error grows during training, turn on the late-training endpoint correction earlier than the default schedule.

## Critical correctness requirements

These are non-negotiable; bugs here invalidate results silently:

1. **R/P alignment before midpoint construction.** `M = (R + P)/2` is only meaningful if R and P share a frame. Verify on the dataset; if any reactions are misaligned, Kabsch-align P to R as preprocessing. Atom permutation must be consistent across (R, P, TS, NEB) within a reaction; spot-check on symmetric scaffolds (CF₃, CCl₃, t-Bu).
2. **Equivariance end-to-end.** The full pipeline including midpoint construction and Network B's conditioning must be SE(3)-equivariant. CI-enforced unit test: rotate (R, P) jointly by a random rotation; all downstream predictions must rotate identically. This test is mandatory. Decide explicitly whether the intended group is SE(3) (chirality-preserving) or O(3) (parity-aware) per "Equivariance scope" above, and make the test cover reflections accordingly.
3. **Atom permutation invariance** for indistinguishable atoms. Use permutation-aware matching in any RMSD computation, and constrain it by connectivity and element so only genuinely interchangeable atoms are permuted (no swapping H across different heavy atoms).
4. **No saddle leakage at inference.** Network A must not see ground-truth saddles at test time. Have a single inference pathway taking only (R, P) and assert no saddle is in scope.
5. **Splits by reaction.** Add a test asserting no reaction ID appears in two splits.
6. **R↔P symmetry of Network A.** A's saddle prediction must be invariant to swapping R and P, enforced architecturally (shared encoder + symmetric combine), with a unit test comparing `(R, P)` and `(P, R)` predictions. See "R↔P symmetry of Network A."

## Things to avoid

- Training the full (auxiliary-supervised) Network A before Network B (the pipeline schedule is B-first; see PLAN.md). The early *direct* `M → saddle` baseline in Phase 2 is a separate control and is meant to run before B.
- Treating `(R+P)/2` as a saddle approximation. It is a starting point for A's correction.
- Computing RMSD without alignment, or with an unconstrained (connectivity-blind) permutation match.
- Breaking equivariance for marginal performance gains — it's a load-bearing inductive bias here.
- Letting the auxiliary descent-consistency loss in Network A training dominate saddle RMSD; saddle weight stays ≥10× larger.
- Using the spectral radius (or even the spectral abscissa) of `J_v` as the contractivity penalty — use the numerical abscissa `λ_max(sym(J_v))`.
- Comparing to React-OT or OA-ReactDiff on different splits than they used originally.

## Key references

- Halo8: https://www.nature.com/articles/s41597-025-05944-3, https://zenodo.org/records/16737590
- Transition1x: predecessor dataset (ωB97X DFT, NEB pathways, C/N/O chemistry).
- React-OT, OA-ReactDiff (Duan et al.): equivariant TS prediction baselines.
- TSDiff (Kim et al.): diffusion-based TS prediction.
- `e3nn` library (Geiger et al.): SE(3)-equivariant primitives.
- NequIP, MACE: equivariant message-passing baselines for chemical PESs.

## Glossary

- **MEP**: minimum energy path — steepest-descent connection between R, saddle, and P.
- **Saddle / TS**: stationary point on the PES with one negative Hessian eigenvalue.
- **NEB**: nudged elastic band — relaxes a chain of images to recover the MEP.
- **Contractive flow**: a vector field whose trajectories converge — formally, one whose Jacobian has negative numerical abscissa `μ(J) = λ_max((J + Jᵀ)/2) < 0` at the relevant points (monotone contraction). Negative-real-part eigenvalues alone give only asymptotic stability, which for non-normal `J` permits transient expansion.
- **Numerical abscissa**: `μ(J) = λ_max((J + Jᵀ)/2)`; bounds the growth rate of `‖exp(Jτ)‖`. The correct contractivity target here.
- **Equivariance**: f(g·x) = g·f(x) for all g in a symmetry group (here SE(3) — or O(3) if parity is included — and atom permutations).
- **Kabsch alignment**: optimal rigid-body alignment minimizing RMSD between two point sets with known correspondences.
