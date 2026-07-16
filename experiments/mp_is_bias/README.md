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

**Redone against the corrected Order-B reduction (`jackknife_orderB.py`), and the story is
more interesting than "jackknife doesn't work here."** The two K-dims now play structurally
different roles in the computation (see `alan-plate-reduction.html`, the companion
explainer, for the full derivation): `mu` only ever enters a flat, linear ratio-of-sums at
the root — exactly the classical self-normalized-IS shape the jackknife's bias-cancellation
theory was derived for. `theta` enters through a `logsumexp` applied independently *per
plate element*, a structurally different reduction. Rerunning the same delete-one jackknife
against each:

| K | plain bias | jack(mu) bias | jack(theta) bias |
|---|---|---|---|
| 3 | &minus;0.0468 | **&minus;0.0036** | &minus;0.0566 |
| 10 | +0.0148 | **+0.0131** | +0.0211 |
| 30 | +0.0071 | +0.0069 | +0.0142 |
| 100 | +0.0239 | +0.0238 | +0.0246 |
| 1000 | +0.0030 | +0.0030 | +0.0030 |

Jackknifing `mu` shows a real effect exactly where SNIS bias theory predicts it should be
largest: at K=3, bias drops more than 10&times; (&minus;0.047 &rarr; &minus;0.004), fading
to a no-op by K&asymp;30 as the underlying bias becomes small relative to Monte Carlo noise
(RMSE gets worse at K=3 specifically, the known variance-for-bias tradeoff of jackknife at
very small sample counts). Jackknifing `theta` never helps, at any K — consistent with the
classical jackknife's bias-cancellation algebra being derived for flat self-normalized
ratios, which the per-element `logsumexp` reduction structurally isn't.

Cross-validated two ways against real alan (`cross_check_orderB.py`): matching means and
standard deviations across 300 independent draws at K=50 and K=200 (e.g. K=200: std 0.0446
real alan vs. 0.0410 here — compare to the *unfixed* version's 4&times; mismatch), on top of
the exact match already established in `diagnose_handroll_mismatch.py`.

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

**Practical read:** a jackknife correction is worth applying specifically to whichever part
of a moment estimate reduces via a flat, linear self-normalized ratio (like `mu`'s root-level
reduction here) — and specifically *not* worth applying to parts that reduce via a nested
per-element `logsumexp` (like `theta`'s plate-level reduction), where it doesn't help and
can mildly hurt. That's a more actionable, structural distinction than "does jackknife work
on MP-IS," and it only fell out once the Order-A/Order-B bug was fixed. BR-SNIS-style
debiasing (mixing-time-based, not delete-one-and-extrapolate-based) remains the more
promising direction for the reduction that doesn't jackknife well, and is already scoped on
`explore/particle-mcmc` (`docs/pmcmc_design.md`) since it needs the same Particle-Gibbs
infrastructure.

For the full derivation of *why* Order B is the exact answer (not just the one that happens
to match), with a from-scratch, hand-checkable K=2/2-plate-element worked example, see the
companion piece: `alan-plate-reduction.html`.

## Reproducing

```
cd experiments/mp_is_bias
python3 sweep.py                        # MP-IS / Global-IS / HMC vs analytic truth
python3 jackknife_orderB.py             # the corrected (Order-B) jackknife check
python3 cross_check_orderB.py           # statistical validation of jackknife_orderB.py vs real alan
python3 jackknife3.py                   # superseded: the original (Order-A, wrong) pairwise jackknife check
python3 marginal_ess.py                 # real per-variable ESS from alan's Marginals.ess()
python3 diagnose_handroll_mismatch.py   # reproduces the unresolved hand-roll discrepancy
```

Needs `torch` (2.1-2.4ish — this repo needs `functorch.dim` *and* named-tensor APIs,
both of which are deprecated/removed in the newest PyTorch releases; 2.3.1 is confirmed
working) and `opt_einsum`.
