# MP-IS vs. Global-IS vs. HMC: an empirical bias check

Prompted by a colleague's comment: *"probably not going to beat HMC per se, what with
the posterior estimates being biased by self-normalization, but mostly seems to do a
reasonable job of trying to catch up to it eventually."* This runs that check directly,
plus three attempts at a jackknife bias correction.

## Setup

A small hierarchical Gaussian model with a closed-form posterior (so "ground truth" isn't
itself an approximation):

```
mu ~ N(0, 1)
theta_i | mu ~ N(mu, 1),  i = 1..6
x_i | theta_i ~ N(theta_i, 0.7)     (observed, fixed synthetic data)
```

- `compare_model.py` — the model, fixed data, and an exact posterior via the Hessian of
  the (exactly quadratic) negative log joint.
- `hmc.py` — a minimal hand-rolled leapfrog HMC sampler (no blackjax/numpyro dependency).
  Validated against the analytic posterior: max abs mean error ~0.013 over 20k samples.
- `alan_compare.py` — builds the same model as an alan `Problem` (P = the generative
  model above; Q = a deliberately simple/mismatched factorised proposal, independent
  `N(0,1)` per latent, matching the "one-shot IS" methodology in both alan papers'
  experiments), and extracts posterior moments via `sample.moments()`.
- `sweep.py` — runs `Problem.sample` (MP-IS) and `Problem.sample_nonmp` (Global IS) across
  `K in [3, 10, 30, 100, 300, 1000, 3000]`, 40 repeats each, and an HMC run, all against the
  same analytic ground truth. Results in `results.json`.
- `jackknife.py`, `jackknife2.py`, `jackknife3.py` — a hand-rolled (white-box) reimplementation
  of MP-IS for this model (so intermediate sums are directly accessible), with three
  variants of a delete-one jackknife bias correction: over mu's K-dim, over theta's K-dim,
  and the mathematically-correct pairwise delete-one-of-K&sup2; version.

## Results

**The core claim holds.** MP-IS bias on `mu` shrinks from +0.089 (K=3) to +0.002 (K=3000),
non-monotonically at small K (a sign flip between K=10 and K=30) — the classic signature of
finite-sample self-normalization bias, not just plain MC noise. By K=3000 it's essentially
matching HMC's own bias (+0.0018 over 20k samples). Global IS, at matched K, is consistently
~5-10x worse on both `mu` and the `theta` group across the whole sweep — matching both
papers' headline result.

**None of the three jackknife variants helped.** All three left bias roughly unchanged
and slightly *increased* RMSE, most visibly at small K. Diagnosis: the K&sup2; combinatorial
grid isn't K&sup2; exchangeable i.i.d. terms — it's an outer-product-like construction from
only 2K underlying draws (K mu-particles &times; K theta-particles), and its effective
sample size scales like O(K), not O(K&sup2;) (checked directly: ESS/K &asymp; 0.9-3.5 across
K=30..1000, consistent with the "ESS is capped at K, not K^n" finding elsewhere in this
project). The classical delete-one jackknife's bias-cancellation algebra assumes flat,
roughly-exchangeable terms; none of the three "which axis to delete along" choices tried
here (mu-row, theta-row, single-pair) match that assumption for this rank-structured
construction, regardless of which nominal `n` is plugged into the correction formula.

**Practical read:** don't invest in a naive jackknife adaptation of MP-IS without first
deriving a version of the theory for this specific outer-product/tensor-contracted
structure — it isn't a "just try it" fix, per what's tested here. BR-SNIS-style debiasing
(mixing-time-based, not delete-one-and-extrapolate-based) remains the more promising
direction, and is already scoped on `explore/particle-mcmc` (`docs/pmcmc_design.md`) since
it needs the same Particle-Gibbs infrastructure.

## Reproducing

```
cd experiments/mp_is_bias
python3 sweep.py        # MP-IS / Global-IS / HMC vs analytic truth
python3 jackknife3.py   # the (best-attempted) pairwise jackknife check
```

Needs `torch` (2.1-2.4ish — this repo needs `functorch.dim` *and* named-tensor APIs,
both of which are deprecated/removed in the newest PyTorch releases; 2.3.1 is confirmed
working) and `opt_einsum`.
