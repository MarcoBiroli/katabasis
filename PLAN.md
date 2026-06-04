# Katabasis Implementation Plan

This plan is sequenced. Each phase has a clear acceptance criterion; do not advance to the next phase until the criterion is met. Open design decisions are flagged at the end of each phase — surface them before deciding.

**A note on sequencing.** The project's central claim is that decomposing TS prediction through the saddle into two contractive descents beats predicting the saddle directly. At inference the TS geometry comes from Network A alone — the descent flows contribute to the headline metric only as an auxiliary training signal. So the cheapest, most important experiment is the *direct* `M → saddle` baseline with no Network B at all, and it is run early (Phase 2) to set the bar the full method must clear. The "does the descent machinery actually help the saddle" question is answered as soon as Network A trains (Phase 4), not deferred to the Phase 6 ablation table.

---

## Phase 0 — Project setup

**Goal.** Repo skeleton, dependencies, smoke tests for everything downstream.

**Tasks.**
- Initialize repo with `pyproject.toml`, `src/katabasis/` layout, `tests/`, `configs/`, `scripts/`.
- Pin PyTorch 2.4+, e3nn, ASE, hydra-core, wandb, pytest, ruff, black.
- Set up CI (GitHub Actions or equivalent) to run tests + lint on every push.
- Implement the **equivariance smoke test**: load a single Halo8 reaction, compute `M = (R + P) / 2`, apply a random SO(3) rotation jointly to R and P, verify the rotated midpoint equals the rotation of the unrotated midpoint within float tolerance. This is the canary for everything that follows.
- Decide and pin the symmetry group in config: SE(3) (chirality-preserving) vs O(3) (parity-aware). See CLAUDE.md "Equivariance scope." This choice drives irrep parities and whether the equivariance tests include reflections.

**Acceptance.** CI green; equivariance smoke test passes.

---

## Phase 1 — Data pipeline

**Goal.** Clean, equivariance-preserving data loaders for the three subsets.

**Tasks.**
- Halo8 ASE-database reader. Cache parsed reactions to disk (one file per reaction, indexed).
- Verify R/P alignment within each reaction; flag and Kabsch-fix any misalignments.
- Implement Kabsch alignment + **connectivity-constrained** permutation-aware matching for indistinguishable atoms: only genuinely interchangeable atoms may be permuted (the H on a single methyl, the F on a CF₃), and the assignment must respect element *and* bonded environment. A global Hungarian match over all atoms of one element will swap H across different heavy atoms and silently corrupt every RMSD — guard against this and unit-test it on symmetric scaffolds.
- Implement equivariant `M = (R + P) / 2` midpoint construction. Unit test: rotation equivariance.
- Define `t1x_core`, `halogen_only`, `full` subsets. Splits **by reaction**, 80/10/10. Persist split assignments as JSON for reproducibility.
- Loader returns `(R, P, saddle, neb_images, energies, forces)` per reaction with consistent atom indexing.
- Filter pathological reactions:
  - barriers below a low threshold (likely conformer-only "reactions");
  - reactions where R-to-P RMSD is below a low threshold (no real chemistry happening);
  - reactions where the listed TS does not have one dominant negative Hessian mode along a bond-making/breaking coordinate (use forces or computed Hessians; if Hessians aren't stored, compute small-displacement finite-difference Hessians on a sample to validate);
  - **multi-step reactions**: NEB paths with a hidden intermediate minimum (two barriers) or a kinked band. The single-saddle factorization assumes one first-order saddle, so require a single-barrier energy profile (monotone up to the saddle along the band, monotone down to each endpoint) and exclude the rest in v1. Detect from the per-image NEB energies.
- Document the filter thresholds and counts.

**Acceptance.**
- Loader returns clean batches at >1000 reactions/sec.
- Equivariance unit test passes for the midpoint and for a representative network forward pass.
- Filtering removes <30% of reactions; remaining set is documented and counted.
- Split-leakage test passes (no reaction ID in two splits).

**Open decisions.**
1. Filter thresholds (barrier height floor, R↔P RMSD floor). Default starting points: barrier ≥ 5 kcal/mol, R↔P RMSD ≥ 0.5 Å. Tune by inspection of removed reactions.
2. Whether to compute Hessians on the fly for filtering or trust dataset metadata.
3. Single-barrier detection tolerance (how much non-monotonicity in the NEB energy profile is noise vs. a real intermediate). Inspect borderline cases.

---

## Phase 2 — Direct saddle baseline (thesis control)

**Goal.** Establish, with the cheapest possible experiments, the saddle-RMSD bar that the full descent-decomposition method must beat — *before* investing in Network B. This phase needs nothing from Phase 3+.

**Tasks.**
- **Trivial control:** predict `M` itself (zero displacement) and report aligned saddle RMSD. Calibrates how far the midpoint is from the TS and how much work A actually has to do.
- **Direct `M → saddle` network:** train a bare equivariant network — the same architecture intended for Network A in Phase 4 — with the primary saddle-RMSD loss only, **no Network B, no auxiliary loss**. This *is* Network A v0 and it *is* the "remove Network B entirely" ablation, run first as the control rather than last as an afterthought. It must be R↔P symmetric by construction (see Phase 4 / CLAUDE.md).
- Train and evaluate on `t1x_core` to keep it cheap.
- (Reference only, not an inference-valid baseline: the energy-max NEB image is a path-derived estimate of the saddle, but it requires the NEB band, which is unavailable at inference. Use it only to sanity-check the data, not as a method baseline.)

**Acceptance.**
- Direct `M → saddle` beats the zero-displacement trivial baseline by a clear margin (otherwise A is not learning useful structure and something is wrong before B is even involved).
- Record the direct-baseline saddle RMSD. **This is the number the full method (Phase 4) must beat.** If it does not, the descent machinery contributes pathway generation and confidence scoring only — not better TS geometries — and that must be reported honestly.

**Open decisions.**
1. Network A architecture details (depth, l_max, multiplicities) — these are shared with Phase 4; fix them here so Phase 4 measures the auxiliary-loss effect cleanly, not an architecture change.

---

## Phase 3 — Network B (descent flow), in isolation

**Goal.** A target-conditioned, contractive descent flow trained on ground-truth saddles, validated first on `t1x_core`.

**Tasks.**
- SE(3)-equivariant message-passing backbone with `e3nn`. Start with 3–4 layers, l_max=2, hidden multiplicities ~32. Treat as hyperparameters.
- Target conditioning via per-atom displacement vector features `(target − current)` as l=1 inputs. Add FiLM-on-RMSD-to-target as a complementary scalar signal at each layer.
- **Contractivity regularization — penalize the numerical abscissa, not the spectral radius.** Penalize positive values of `λ_max(sym(J_v))` where `sym(M) = (M + Mᵀ)/2` and `J_v = ∂v/∂x` is evaluated at the target. Estimate it by power iteration on the **symmetrized** Jacobian-vector product `u ↦ ½(J_v u + J_vᵀ u)` (both directions via autograd). Do **not** run plain power iteration on `J_v` — that returns the largest-magnitude eigenvalue (spectral radius), which does not control contraction; and even the largest-real-part eigenvalue is insufficient because equivariant Jacobians are non-normal and can grow transiently. Optionally add Hutchinson's estimator on `tr(J_v)` as a cheap averaged-contractivity signal. Apply only away from the saddle (`t`-gated); near the saddle the unstable mode is by construction expansive. **Cost control:** the JVP estimate adds extra backward passes per step — apply it on a batch subsample and/or every k steps, not every example every step.
- **Loss function: flow matching with arc-length bridge.** Full spec in CLAUDE.md ("Training objective for Network B"). Implementation tasks: precompute cumulative arc length per descent leg at preprocessing; piecewise-linear interpolation between NEB images for `x_t`; finite-difference path tangent `dx/dt`; **velocity-scale normalization** (predict direction + scalar speed, or normalize positions by a per-reaction scale — `dx/dt = L·tangent` spans orders of magnitude across reactions otherwise); `t`-biased sampling (default `Beta(0.5, 1.5)`); contractivity regularizer with `t`-gating (`t > 0.2`) using the symmetrized estimator above; endpoint-RMSD term gated by epoch threshold (off until ~80% through training). RK4-32 integrator for evaluation.
- Training data construction: for each reaction, every `(NEB_image_i, endpoint)` pair plus `(saddle, R)` and `(saddle, P)`. Endpoint = R if image is on the R-side of the saddle (lower index), P otherwise.
- **Non-learned physics baseline (free, from the data):** Halo8 stores forces, so integrate steepest descent on the actual ωB97X-3c forces from the saddle toward each basin and compare the resulting path to the NEB descent. This grounds the "descent flow" premise against real physics and gives B a path-fidelity baseline to beat.
- Robustifying B to imperfect saddles from A: noise injection on the starting state during training (Gaussian, σ ≈ typical NEB image spacing — measure from the dataset). **Caveat:** isotropic Gaussian noise is a weak proxy for A's actual errors, which are structured (correlated across atoms, larger along the reaction coordinate). Plan to also expose B to A's *real* predicted saddles — at minimum via the Phase 5 joint fine-tuning, and consider a dedicated "B on A's outputs" pass once a Phase 4 A exists.

**Sanity-check experiment.** Train B on `t1x_core` only. Evaluate on held-out reactions: starting from the ground-truth saddle and the held-out target, integrate B's flow and check endpoint distance to ground truth — *and* path fidelity (see acceptance).

**Acceptance.**
- Endpoint RMSD < 0.1 Å on `t1x_core` held-out reactions.
- **Path fidelity:** at several `t ∈ (0, 1)`, the integrated `x_t` matches the arc-length-matched NEB image. Endpoint RMSD alone is nearly tautological for a target-conditioned flow (B is told where to go); a flow that teleports to the target along a non-physical path would invalidate pathway generation, so this metric is required, not optional. Target: intermediate-path RMSD comparable to (or better than) the steepest-descent baseline above.
- Contractivity verified empirically: linearization at the target has negative `λ_max(sym(J_v))` for >95% of test reactions.
- Equivariance unit test passes on the trained model.

Then scale to `full` Halo8 and re-validate. Quantify the `t1x_core` vs. `halogen_only` performance gap. **If `halogen_only` RMSD is more than 3× worse, do not advance to Phase 4 yet** — diagnose first (likely culprits: backside-attack geometries, heavy-halogen relativistic effects, or insufficient model capacity).

**Open decisions.**
1. **Backbone:** plain e3nn message-passing vs. NequIP-style vs. MACE-style. Start plain; revisit only if accuracy is insufficient at the Phase 3 acceptance gate.
2. **Velocity-scale normalization scheme:** direction+speed split vs. per-reaction coordinate scaling. Decide by which keeps the velocity-matching loss balanced across short and long paths.
3. **Hessian regularization at the saddle:** if Halo8 stores TS Hessians, use them to enforce that B's behavior near the saddle has one expansive direction (the reaction coordinate) and contractive orthogonal modes. If not, skip in v1.
4. **Time-sampling distribution.** `Beta(0.5, 1.5)` is a reasonable default; tune by checking that saddle-proximal accuracy improves vs. uniform `t`. Watch the `t = 0` singularity if saddle-proximal targets are ill-conditioned. The right answer is dataset-dependent.
5. **Spline degree for `x_t` interpolation.** Linear is the default; cubic is more accurate but can overshoot near sharp curvature. Verify smoothness on the actual NEB images before deciding. Note descent legs may be sparse (4–6 images), which biases the finite-difference tangent at the saddle end — the region you care most about.
6. **Contractivity `t`-gating threshold.** Default `t > 0.2`; the right value depends on how rapidly the linearization stabilizes away from the saddle. Diagnose by computing the `sym(J_v)` eigenvalue spectrum along training paths.

---

## Phase 4 — Network A (saddle predictor) with frozen B

**Goal.** A model predicting the saddle from `M = (R + P) / 2` with auxiliary supervision from the frozen Network B — and a direct measurement of whether that auxiliary signal beats the Phase 2 baseline.

**Tasks.**
- Architecture: equivariant network mapping `(R, P, M)` → **displacement from M**. Predicted saddle = M + displacement. Residual-style output is much easier to learn than absolute coordinates. **R↔P symmetric by construction:** encode R and P with a *shared* encoder and combine symmetrically (sum/mean, or features of `|R − P|`) so swapping R and P leaves the prediction unchanged; unit-test `(R, P)` vs `(P, R)`. Use the same architecture as the Phase 2 direct baseline so the only change measured here is the auxiliary loss.
- Primary loss: Kabsch- and permutation-aligned saddle RMSD against ground truth.
- Auxiliary loss: with B frozen, run two descents from the predicted saddle (toward R and toward P); penalize endpoint mismatch against ground-truth R, P.
- Loss weighting: primary:auxiliary = 10:1. Search in [5:1, 20:1] but never let auxiliary dominate.
- Optional: Hessian-eigenvalue loss — predicted saddle should have one negative eigenvalue along the bond-making/breaking mode. Skip in v1 unless cheap to compute.

**Acceptance.**
- Saddle RMSD competitive with React-OT and OA-ReactDiff on the Transition1x core split (target: within 20% of best baseline; ideally beat them).
- **Beats the Phase 2 direct `M → saddle` baseline.** This delta is the core "does the descent decomposition help the saddle" measurement — surfaced here, not deferred to Phase 6. If A-with-auxiliary-B does *not* beat the direct baseline, stop and report it honestly: the descent machinery then earns its place via pathway generation and confidence scoring, not TS accuracy, and the paper's framing must reflect that.
- Saddle predictions have correct topology: one dominant negative Hessian mode on >90% of test cases (numerical Hessian on the predicted geometry).

**Open decisions.**
1. Whether to include energy supervision if Halo8 barrier heights are reliable enough to use as a weak supervisory signal for A.
2. Whether to share any backbone weights between A and B. **Default: no.** Different tasks, different inductive biases.

---

## Phase 5 — Joint fine-tuning

**Goal.** Joint optimization to fix systematic A/B mismatches without compromising A's saddle prediction. This is also where B finally sees A's *real* error distribution rather than synthetic noise.

**Tasks.**
- Unfreeze both networks; LR ÷ 10.
- Same loss as Phase 4.
- Run for ~10% of Phase 4 epoch count.
- Monitor: A's saddle RMSD must not degrade by more than 10% relative to the Phase 4 endpoint. If it does, the auxiliary weight is too high — back off.

**Acceptance.**
- Saddle RMSD ≤ Phase 4 result.
- Cycle-consistency error (descent endpoints vs. ground-truth R/P) reduced relative to Phase 4.

---

## Phase 6 — Evaluation and ablations

**Goal.** Benchmark vs. baselines and ablate the architecture for the paper.

**Tasks.**
- Evaluate against React-OT and OA-ReactDiff on Transition1x core. **Match their evaluation protocol exactly** — same split, same metrics, same alignment procedure. Otherwise reviewers will (justifiably) reject the comparison. Prefer published numbers on identical splits; scope carefully whether *retraining* the baselines (a large lift, especially for the halogen protocol) is actually necessary.
- Halogen transfer: train on `t1x_core`, test on `halogen_only`. Compare to baselines on the same protocol. This is the headline experiment if the physical inductive bias is real — but note it tests *geometric* transfer; halogen energetics differ, so frame the claim accordingly.
- Ablations:
  - **Remove Network B entirely** (direct `M → saddle`). This is the Phase 2 baseline, already established — fold its number into the table. Tests whether the descent decomposition helps the saddle.
  - **Remove auxiliary descent loss** (B trained, but not used in A's training). The headline comparison was already made at the Phase 4 gate; here, formalize it across the full splits. Tests whether B helps A *during training*.
  - **Replace `M` with random midpoint** (uniform in a box around M). Tests whether the symmetric initialization matters.
  - **Remove contractivity regularization.** Tests whether the physical prior matters.
- **Cycle consistency as a confidence score.** Measure correlation between (descent endpoint error from predicted saddle) and (saddle prediction error). Positive correlation → usable confidence score and a clean paper headline.

**Acceptance.** Publication-grade table with all baselines, ablations, halogen transfer, and the cycle-consistency confidence analysis.

---

## Cross-cutting concerns

- **Reproducibility.** All experiments seeded; configs versioned; data splits persisted to disk with hash.
- **Compute budget.** Estimate before each phase. Phase 3 (full Halo8) and Phase 6 (baselines) are expensive — budget GPU-days, not GPU-hours. The contractivity regularizer multiplies per-step cost (extra backward passes for the symmetrized power iteration); subsample it and measure its overhead before committing to full-dataset runs.
- **Logging.** Every run logs to wandb with full config, validation metrics every epoch, gradient norms (sparingly).

## Risk register

- **Halo8 quality issues.** If the dataset has systematic problems (mislabeled saddles, broken NEB images, inconsistent atom indexing, hidden multi-step reactions), Phase 1 filtering needs to be aggressive. Allocate time for manual inspection of a sample of ~50 reactions across the three subsets before trusting the loader.
- **The descent machinery may not improve the saddle.** Since the inference-time TS comes from A alone, it is entirely possible the auxiliary-B signal does not beat the Phase 2 direct baseline. This is why the comparison is front-loaded (Phases 2 and 4). If it holds, B's value is pathway generation + confidence; plan the paper to be honest either way.
- **Equivariance bugs.** Silent but catastrophic. CI-enforced unit tests are mandatory and should be added the moment a new module touches geometric inputs. Include the R↔P symmetry test for A and (per the pinned group) reflection behavior.
- **Wrong contractivity surrogate.** Penalizing the spectral radius or spectral abscissa instead of the numerical abscissa `λ_max(sym(J_v))` would let non-normal flows pass the regularizer while still expanding transiently. Use the symmetrized estimator and verify the empirical contractivity acceptance gate in Phase 3.
- **Contractivity regularization fights the data near the saddle.** NEB images near the saddle are not on a contractive flow because the saddle has an unstable direction. The regularization must be applied only away from the saddle; verify in a Phase 3 diagnostic.
- **Train-test distribution shift on B's saddle inputs.** Isotropic noise injection is a weak proxy for A's structured errors. Mitigated by exposing B to A's real predictions in Phase 5 joint fine-tuning; watch for it in the Phase 5 monitoring.
- **Velocity-scale imbalance.** Without per-reaction normalization, `dx/dt = L·tangent` makes long-path reactions dominate B's loss. Addressed in Phase 3; confirm the loss is balanced across path lengths.
- **(R+P)/2 lands in pathological regions for halogen reactions.** Expected. The whole point of A is to correct this, and A also sees R and P directly. If A fails, consider an equivariant pre-relaxation step (a few iterations of a learned "midpoint sanitizer") before feeding to A.
