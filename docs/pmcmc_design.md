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
