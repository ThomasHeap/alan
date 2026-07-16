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
- `diagnose_handroll_mismatch.py` — isolates the discrepancy described in "Correction" below.
- `marginal_ess.py` — real per-variable marginal ESS from alan's own `Marginals.ess()`
  (not the hand-roll), replacing the ad hoc joint-grid ESS check the first pass used.

## Resolved: the hand-roll implemented the wrong reduction order

**The `sweep.py` headline numbers (MP-IS/Global-IS/HMC bias vs. K) were never affected by
this** — they call `sample.moments()` on real alan `Sample` objects throughout, never the
hand-roll. But the jackknife experiments and the original ESS check were built entirely on
the hand-rolled reimplementation in `jackknife.py`/`jackknife2.py`/`jackknife3.py`, which
turned out not to reproduce alan's own `Sample.moments()` output — confirmed to be a real
discrepancy, localized precisely to the aggregation step, not any individual log p / log q
term (each of those matched up to the expected dropped normalizing constant).

**Root cause (see `diagnose_handroll_mismatch.py`): a wrong mental model of what "one
K-dim shared across a plate" means, not a bug in alan.** I had assumed a plate-shared
K-dim means every replicate uses the *same* particle index in lockstep — sum the six
plate elements' log-densities first, *then* `logsumexp` once over the shared K-dim
("Order A", what the hand-roll implemented). alan actually does something more powerful
("Order B"): each plate element independently picks whichever of the K particles best
explains *its own* observation, via a `logsumexp` **per plate element**, and only sums
those already-reduced per-element values across the plate afterward. Concretely, in
`reduce_Ks(lps, all_Ks)` (`src/alan/reduce_Ks.py`) the plate dimension is still a live,
broadcast axis while the K-dim gets reduced — the actual plate sum
(`lp.sum(new_platedim)` in `_logPQ_plate`, `src/alan/logpq.py`) happens *afterward*, on the
already-`logsumexp`'d values. `logsumexp` does not distribute over an inner sum, so Order A
and Order B are genuinely different computations, not two ways of writing the same one.
This is a real extra bit of Rao-Blackwellization the plate structure buys you — worth
noting elsewhere in this project's write-ups on how plates work.

Once corrected to Order B, both the ELBO (`log P_MP(z)`, matching `elbo_nograd()`) and the
posterior mean of `mu` (via the identical source-term-trick autodiff alan uses internally)
match alan **exactly**, to float32 precision (`-11.456429` / `-11.456429`, `0.437142` /
`0.437142`).

**Practical consequence:** the original "none of the three jackknife variants helped"
finding was run against the Order-A (wrong) reduction, so it isn't validated for alan's
actual computation — it would need to be redone against the corrected Order-B formula in
`diagnose_handroll_mismatch.py` to know whether it holds. Not yet redone. The ESS numbers
were pulled directly from alan's own `Marginals.ess()` throughout (never the hand-roll) and
are unaffected.

## Results

**The core claim holds.** MP-IS bias on `mu` shrinks from +0.089 (K=3) to +0.002 (K=3000),
non-monotonically at small K (a sign flip between K=10 and K=30) — the classic signature of
finite-sample self-normalization bias, not just plain MC noise. By K=3000 it's essentially
matching HMC's own bias (+0.0018 over 20k samples). Global IS, at matched K, is consistently
~5-10x worse on both `mu` and the `theta` group across the whole sweep — matching both
papers' headline result.

**None of the three jackknife variants helped — but run against the wrong (Order-A)
reduction (see "Resolved" above), so this needs to be redone against the corrected Order-B
formula before it's a trustworthy statement about alan itself.** As originally run, all
three left bias roughly unchanged and slightly *increased* RMSE, most visibly at small K.
Working hypothesis, unchanged in spirit but not re-verified post-fix: the K&sup2;
combinatorial grid isn't K&sup2; exchangeable i.i.d. terms — it's an outer-product-like
construction from only 2K underlying draws (K mu-particles &times; K theta-particles), and
the classical delete-one jackknife's bias-cancellation algebra assumes flat,
roughly-exchangeable terms, which a rank-structured construction doesn't provide regardless
of which axis gets deleted or which nominal `n` goes into the correction formula. Whether
that story survives Order B (which itself adds another layer of per-plate-element
structure) is now the open question.

This is consistent with alan's *real*, verified marginal ESS (`marginal_ess.py`, straight
from `Marginals.ess()`, not the hand-roll): `ESS(mu)` sits at a strikingly stable
**&asymp;0.60&times;K** across the entire K=10..3000 range tested, and `ESS(theta)`
averages **&asymp;0.48&times;K&mdash;but with real spread across the 6 plate replicates**
(e.g. at K=3000: min 504, mean 1442, max 1948 &mdash; roughly a 4&times; gap between the
best- and worst-represented group). Kish's ESS for an n-length normalized weight vector is
always &le; n, so `ESS(mu) &le; K` is a hard bound, not an empirical observation &mdash; what's
empirical is the *efficiency ratio* (&asymp;0.5-0.6 here) staying roughly constant as K
grows, meaning the proposal/target mismatch this ratio reflects isn't getting relatively
worse or better with more particles, just more finely resolved. See the artifact's ESS
section for the fuller derivation.

**Practical read:** don't invest in a naive jackknife adaptation of MP-IS without first
deriving a version of the theory for this specific outer-product/tensor-contracted
structure — it isn't a "just try it" fix, per what's tested here. BR-SNIS-style debiasing
(mixing-time-based, not delete-one-and-extrapolate-based) remains the more promising
direction, and is already scoped on `explore/particle-mcmc` (`docs/pmcmc_design.md`) since
it needs the same Particle-Gibbs infrastructure.

## Reproducing

```
cd experiments/mp_is_bias
python3 sweep.py                        # MP-IS / Global-IS / HMC vs analytic truth
python3 jackknife3.py                   # the (best-attempted) pairwise jackknife check
python3 marginal_ess.py                 # real per-variable ESS from alan's Marginals.ess()
python3 diagnose_handroll_mismatch.py   # reproduces the unresolved hand-roll discrepancy
```

Needs `torch` (2.1-2.4ish — this repo needs `functorch.dim` *and* named-tensor APIs,
both of which are deprecated/removed in the newest PyTorch releases; 2.3.1 is confirmed
working) and `opt_einsum`.
