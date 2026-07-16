# Particle MCMC for alan — implementation checklist

## Context

alan's K-dim tensor-contraction machinery already computes `P_MP(z)`, an *unbiased* estimator
of the true marginal likelihood (proved in Heap et al. 2023, Appendix 3.1.3, and again in
Bowyer/Heap/Aitchison 2024, Appendix A.2). Unbiasedness is the only property Particle MCMC
(Andrieu, Doucet & Holenstein 2010 — cited as related work in both papers) needs to guarantee
an MCMC chain built around that estimator has the *exact* true posterior as its stationary
distribution, even though each step uses a noisy estimate.

This means alan already has the hard part of two well-established exact-inference schemes
half-built:

- **Particle Marginal Metropolis-Hastings (PMMH)** — an outer Metropolis-Hastings chain over
  some "global" variables θ, where the acceptance ratio uses `P_MP(x|θ)` (nuisance latents
  marginalized out via the existing forward pass) in place of the intractable true marginal
  likelihood.
- **Particle Gibbs** — reuses alan's existing posterior-index sampling
  (`Sample.importance_sample`/`_importance_sample_idxs`/`sample_logpq.py`, which already
  implements "Algorithm 3: joint distributions using the source term trick" from the autodiff
  paper) as the resampling step in a conditional-SMC loop.

Both would give **exact** posteriors (unlike VI/RWS/QEM, which only ever approximate), while
still using the K-dim machinery — rather than plain HMC, which would sidestep it entirely.
PMMH in particular is also a more principled alternative to the "demote a shared hyperprior to
a point-estimated `Param`" workaround discussed for the non-nested-priors problem, since PMMH
gives that hyperprior a proper posterior instead of a point estimate.

Not started — this branch just parks the plan.

## Empirical findings since this doc was written, and what they change

A separate empirical pass (`explore/mp-is-bias-empirical`, toy hierarchical Gaussian model,
real alan `Problem`/`Sample` objects throughout) measured two things directly relevant to the
Particle Gibbs design above: per-variable marginal ESS via `Marginals.ess()`, and a jackknife
bias correction on the plain MP-IS estimator. Neither changes the PMMH plan; both bear on
Particle Gibbs.

- **ESS is a mixing-rate warning, not just a bias number.** Real `Marginals.ess()` values across
  K=10..3000: `ESS(mu)` ≈ 0.60×K, stable across the whole range. `ESS(theta)` ≈ 0.48×K on
  average, but with real ~4x spread *across the 6 plate replicates* (K=3000: min 503.85, mean
  1442.33, max 1947.80 — i.e. worst-replicate ESS/K≈0.17 vs best≈0.65). Kish's ESS≤K is a hard
  bound; the finding is that the efficiency ratio doesn't improve with K, and it's markedly
  uneven across plate elements. In a conditional-SMC / Particle Gibbs loop this uneven weight
  degeneracy is exactly the mechanism behind "reference trajectory stickiness" (the known
  slow-mixing failure mode of plain Particle Gibbs): whichever plate replicate has the worst
  ESS/K on a given step will tend to dominate how often the reference trajectory's ancestry gets
  displaced, and the whole chain's mixing rate is bottlenecked by that worst replicate, not the
  average one. This is a concrete, now-quantified reason to treat the "(Optional, better
  mixing) Ancestor sampling (PGAS)" item below as a near-term priority rather than a nice-to-have
  — plain Particle Gibbs on a model with this much cross-replicate ESS spread should be expected
  to mix slowly.
- **Order B (see `experiments/mp_is_bias/README.md` / the plate-reduction artifact) doesn't
  require redesigning the conditional-sampling plan.** The independent per-replicate
  marginalization that Order B established is a *description of how the existing forward pass
  already reduces*, not a new constraint — the "pin a reference trajectory at a fixed particle
  index (e.g. k=0) per latent" approach in the Particle-Gibbs checklist below is unaffected by
  it either way.
- **Jackknife is a narrow, cheap patch — not a substitute for Particle Gibbs.** Redone under the
  corrected Order-B reduction (`jackknife_orderB.py`), the result is more nuanced than the
  original (pre-fix) "none of it helps" finding: jackknifing `mu` (the flat, linearly-reduced
  ratio) gives real bias reduction at small K (>10x at K=3), fading to a no-op by K≈30, because
  it matches the classical delete-one SNIS-jackknife theory exactly. Jackknifing `theta` (reduced
  via a per-plate-element `logsumexp`, not a flat ratio) never helps at any K tested — that
  reduction shape falls outside what delete-one jackknife theory covers. Practically: jackknife
  needs no new Markov-chain, burn-in, or conditional-resampling machinery, so it's worth keeping
  as a cheap correction for outer/point-estimated quantities (the same structural category as a
  PMMH-style outer θ). But it only ever patches a point estimate's bias for flat-ratio-reduced
  variables — no exactness, no calibrated posterior uncertainty, and no help for plated/nested
  latents. Particle Gibbs remains the tool for the actual problem (exact posteriors over nested,
  plated latents); jackknife and Particle Gibbs are complementary, not competing, and jackknife
  is not a reason to deprioritize this plan.
- **BR-SNIS likely doesn't have the theta-vs-mu asymmetry above.** Its chain-recycling mechanism
  (mixing-time-based i-SIR candidate pool recycling, not delete-one-and-extrapolate) doesn't need
  to know which reduction shape/axis it's correcting, so it should apply uniformly across both
  flat-ratio and nested-logsumexp reductions where jackknife only covers the former. This
  reinforces treating BR-SNIS + Particle Gibbs (rather than jackknife + Particle Gibbs) as the
  more general long-run combination, consistent with the roadmap artifact's existing "practical
  read."

## Shared prerequisites (needed by both)

- [ ] Expose `P_MP(z)`/its log as a standalone callable outside `_elbo`'s log-sum path, in a
      form suitable for pseudo-marginal ratios — a version of the `reduce_Ks.py`/`logpq.py`
      forward pass that returns the estimator directly rather than only as an ELBO term.
- [ ] Decide the θ (outer, MCMC-controlled) vs. nuisance-latent (K-dim-marginalized) split
      mechanism. Nothing today marks a `Plate` variable as "outer" — everything sampled by Q
      either goes through the K-dim machinery or is a plain `Param`. Mirrors Geffner & Domke's
      "local IWAE" single-level decomposition (cited in both papers): one outer `z0`, the rest
      integrated out.
- [ ] Represent θ as a single point in latent space per chain (not a torchdim/K-dim tensor),
      with unconstrained-space transforms for proposals — reuse `dist.dist.support`, already
      inspected by `checking.py`'s `check_support`.
- [ ] Per-chain RNG control. Sampling currently goes through global torch RNG state via
      `TorchDimDist`; need seedable, independent generators to run multiple chains for R-hat.
- [ ] MCMC diagnostics: acceptance rate, effective sample size on the *chain* (distinct from
      `Marginals.ess()`, which measures K-particle weight degeneracy, not chain autocorrelation),
      R-hat across chains. None of this exists yet.

## PMMH-specific

- [ ] Proposal `q(θ'|θ)` — random-walk Gaussian in unconstrained space is the standard start —
      plus step-size adaptation (dual-averaging or simple acceptance-rate targeting).
- [ ] Wire a proposed θ' into the model as a substituted value for one forward pass, rather than
      something sampled — probably means treating θ like a `BoundPlate` `input` for that call,
      not a `Dist`.
- [ ] Accept/reject loop using the log-likelihood-estimate difference
      (`log P̂(x|θ') - log P̂(x|θ) + log π(θ') - log π(θ) + log q(θ|θ') - log q(θ'|θ)`),
      compared against `log(uniform)` — this is the standard PMMH acceptance step; the real
      subtlety is just that the nuisance particles must be freshly redrawn for every proposed
      θ' (satisfied automatically by re-running the forward pass), not reused from the previous
      step.
- [ ] Cache the accepted `log P̂(x|θ)` between steps (standard PMMH: only re-estimate the
      likelihood for the *proposal*, keep the current chain state's estimate fixed until the
      next accepted move) — needs a small chain-state object (θ, cached log-likelihood).
- [ ] Chain driver, e.g. `run_pmmh(problem, n_steps, proposal, ...)` returning θ samples +
      diagnostics.
- [ ] Validation: a small model with a known analytic posterior over θ (e.g. conjugate
      Normal-Normal hyperprior) to check against.

## Particle Gibbs-specific

- [ ] Conditional sampling: extend `Plate.sample`/`sample_gdt`/`Sampler.resample_scope` so that,
      given a reference trajectory (one full latent-space sample, pinned at a fixed particle
      index, e.g. k=0), the other K−1 particles are drawn as usual but the reference survives
      every resampling step unchanged. This is the standard "conditional SMC" construction —
      touches `Plate.py`'s recursive sampling and `Sampler.py` directly, more invasive than
      PMMH.
- [ ] (Optional, better mixing) Ancestor sampling (PGAS, Lindsten et al.) — instead of always
      keeping the reference's original lineage, resample which ancestor the reference is
      attached to at each plate-tree level, reusing the existing backward/marginal machinery
      (`logPQ_sample`, Algorithm 3) to get those ancestor weights. More work, avoids the slow
      mixing plain particle Gibbs is known for on state-space/timeseries models.
- [ ] After conditional resampling, draw the new reference trajectory by reusing
      `Sample.importance_sample()`/`_importance_sample_idxs()` (already samples from `P_z(k)`
      via the source-term trick) to pick a new k, then `index_into_sample(...)` to extract z^k.
- [ ] Chain driver returning a sequence of full latent-space samples + diagnostics.
- [ ] Validation: compare Particle Gibbs posterior samples against alan's own
      `importance_sample`/`marginals` output on a model small enough to also check against
      brute-force enumeration.

## Both — validation & write-up

- [ ] Sanity-check both against a known-correct sampler on a toy model (e.g. NumPyro NUTS or a
      hand-rolled Gibbs sampler on a conjugate model).
- [ ] Benchmark against the existing VI/RWS/QEM paths and the hand-rewritten blackjax NUTS
      baselines in `examples/models/blackjax/` — this would be the first MCMC method that runs
      directly against an alan-native model instead of a hand-duplicated JAX model.
- [ ] Decide where this lives — likely a new `src/alan/PMCMC.py` alongside `Sample.py`/
      `Sampler.py`, exposed via `Problem`.
