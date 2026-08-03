# Particle Gibbs / PGAS prototype

A working, validated implementation of the Particle-Gibbs checklist items in
`docs/pmcmc_design.md`, built on a toy linear-Gaussian state-space model rather
than alan's own plate machinery (see that doc's "Empirical findings" section
for why: Particle Gibbs is fundamentally about resampling *across a sequence*
with a pinned reference trajectory, and alan's own toy model in
`experiments/mp_is_bias/` is a flat 2-level hierarchy, not a sequence — it has
no genuine "time" axis for CSMC to resample across). This is deliberately
scoped as a prototype, per the design doc's own validation checklist item
("Sanity-check ... on a toy model ... small enough to also check against
brute-force enumeration" / closed-form ground truth) — it does not touch
alan's core `Plate.py`/`Sampler.py`.

## Setup

```
x_0 ~ N(0, 1)
x_t = 0.9 x_{t-1} + N(0, 0.3),  t = 1..20
y_t = x_t + N(0, 0.5)            (observed)
```

Linear-Gaussian, so the exact posterior is available in closed form via the
Kalman filter (forward) + RTS smoother (backward) — no approximate ground
truth needed anywhere in this experiment.

- `ssm_model.py` — the model, data simulation, Kalman filter, RTS smoother
  (exact `E[x_t | y_1:T]` for every `t`), exact `log p(y_1:T)`. Cross-checked
  against a brute-force joint-Gaussian `log_prob` on a small `T` — matches to
  6 decimal places.
- `bootstrap_pf.py` — a standard bootstrap particle filter (systematic
  resampling every step), used two ways: (a) its log-likelihood estimate is
  checked for the expected unbiasedness-in-the-limit / shrinking-bias-with-K
  behavior against the exact Kalman value; (b) it demonstrates the classic
  **path degeneracy** problem — tracing genealogies back from the final time
  step collapses onto a small fraction of surviving ancestors at `t=0` (e.g.
  K=10: only 1/10 unique ancestors survive one sweep; K=30: only 3/30). This
  is exactly the failure mode Particle Gibbs exists to fix.
- `particle_gibbs.py` — conditional SMC (`csmc_sweep`) with particle slot 0
  permanently reserved for a pinned reference trajectory — guaranteed to
  survive every resampling step by construction, unlike the plain bootstrap
  filter above. Implements both plain CSMC (the reference always keeps its
  own historical lineage) and PGAS (Lindsten, Jordan & Schön 2014 — the
  reference's ancestor is instead resampled each step, weighted by how well
  each candidate ancestor's transition would have predicted the pinned
  value). `run_particle_gibbs` is the chain driver.
- `pg_validate.py` — validates posterior-mean accuracy against the RTS
  smoother, and directly measures mixing speed (chain effective sample size
  of `x_0` via Geyer's initial-positive-sequence estimator) for plain PG vs.
  PGAS.

## Results

**Posterior-mean accuracy** (RMSE of the chain's mean trajectory vs. the exact
RTS-smoothed mean, 2000 iterations, 200 burn-in):

| K | plain PG | PGAS | plain bootstrap PF (fresh run, no chain) |
|---|---|---|---|
| 5 | 0.1843 | **0.0164** | 0.4338 |
| 10 | 0.0467 | **0.0141** | 0.3435 |
| 30 | 0.0304 | **0.0114** | 0.2278 |

Both Particle Gibbs variants beat a single fresh bootstrap-PF genealogy
estimate by a wide margin at every K (as expected — the whole point of
conditioning on a surviving reference is to stop throwing away the
population's history), and PGAS is consistently more accurate than plain PG
at matched K.

**Mixing speed** — chain ESS of `x_0` (the point furthest from any
observation, so the most exposed to reference-trajectory stickiness):

| K | plain PG chain ESS | PGAS chain ESS | speedup |
|---|---|---|---|
| 5 | **1.0** / 1800 (0.1%) | 772.1 / 1800 (42.9%) | 772x |
| 10 | 15.9 / 1800 (0.9%) | 1174.9 / 1800 (65.3%) | 74x |
| 30 | 422.2 / 1800 (23.5%) | 1616.1 / 1800 (89.8%) | 3.8x |

At K=5, plain Particle Gibbs' reference trajectory **never moves at all** —
`x_0` is bit-for-bit identical across all 1800 post-burn-in iterations (chain
ESS=1 by definition, not a numerical artifact — confirmed by inspecting the
raw samples). This is the "reference stickiness" pathology in its most acute
form: at small K, the reference's own historical lineage from `t=0` almost
always survives every resampling step (there just aren't enough competing
particles to displace it), so the chain effectively gets stuck re-sampling
the *same trajectory* forever. PGAS fixes this specifically because it lets
the reference's future be re-attached to a *different* candidate past at
every sweep, decoupling "this trajectory survives" from "this trajectory's
early history never changes." The gap narrows as K grows (more particles
means plain PG's own stickiness eases too) but stays substantial even at
K=30.

This is a direct, concrete confirmation of the prediction in
`docs/pmcmc_design.md`'s "Empirical findings" section — that the uneven
per-replicate ESS spread found in alan's own MP-IS (`experiments/mp_is_bias/`)
was a real warning sign for plain Particle Gibbs mixing, and that PGAS should
be treated as a near-term priority rather than a nice-to-have. It's a
different toy model (a state-space model here vs. the flat hierarchical model
there), but it's the standard setting in the literature where this exact
failure mode is documented, and the numbers bear it out unambiguously.

**Path degeneracy**, for context on *why* CSMC's reference-pinning guarantee
matters at all: a single ordinary bootstrap-PF sweep (`bootstrap_pf.py`)
already loses almost all ancestor diversity by `t=0` (K=10: 1/10 unique
ancestors survive; K=30: 3/30). CSMC's reference slot is what stops this from
mattering for the *pinned* trajectory specifically — but as the mixing
results above show, that guarantee alone (without ancestor sampling) doesn't
prevent the reference itself from becoming the sole survivor of every sweep
in turn, which is a different, chain-level version of the same underlying
degeneracy phenomenon.

## Reproducing

```
cd experiments/particle_gibbs
python3 ssm_model.py        # model + Kalman filter + RTS smoother sanity check
python3 bootstrap_pf.py     # log-lik bias vs K, path degeneracy demo
python3 particle_gibbs.py   # PG/PGAS smoke test
python3 pg_validate.py      # full accuracy + mixing-speed comparison (~a few minutes)
```

Needs only `torch` (no alan dependency — this experiment is fully standalone).

## Harder model: nonlinear/non-Gaussian (Gordon, Salmond & Smith 1993)

The linear-Gaussian model above is unimodal and has a closed-form posterior —
too easy to be a stress test on its own. `nonlinear_*.py` repeats the same
comparison on the standard nonlinear/non-Gaussian benchmark from the
particle-filtering literature:

```
x_0 ~ N(0, 5)
x_t = 0.5 x_{t-1} + 25 x_{t-1}/(1+x_{t-1}^2) + 8 cos(1.2 t) + N(0, 10),  t = 1..20
y_t = x_t^2 / 20 + N(0, 1)                                               (observed)
```

No closed-form posterior (Kalman/RTS don't apply), so ground truth instead
comes from a grid-based/point-mass HMM forward-backward filter (Kitagawa
1987) in `nonlinear_ssm_model.py` — discretize the 1D state onto a fine grid
(800 points over [-40, 40]) and run exact forward-backward on the resulting
HMM. And because `y_t` is quadratic in `x_t`, the filtering/smoothing
marginals are genuinely **bimodal** at several timesteps (e.g. `x_0`, `x_1`:
~88%/12% split) — a much sterner test of whether a single reference
trajectory can represent the posterior at all, on top of the usual mixing
question.

`nonlinear_bootstrap_pf.py`, `nonlinear_particle_gibbs.py`, and
`nonlinear_pg_validate.py` mirror the linear-model files' structure exactly,
with the transition/observation densities swapped in — CSMC/PGAS themselves
are model-agnostic.

### Results (2000 iterations, 200 burn-in, unless noted)

**Posterior-mean accuracy** (RMSE vs. the grid-smoother mean):

| K | plain PG | PGAS | plain bootstrap PF (fresh run, no chain) |
|---|---|---|---|
| 50 | 0.2742 | **0.0254** | 2.9919 |
| 200 | 0.0934 | **0.0249** | 0.7296 |
| 500 | 0.0523 | **0.0266** | 0.8181 |

Same qualitative story as the linear case, but the gaps are far larger — a
fresh bootstrap-PF genealogy is off by whole units even at K=500, since path
degeneracy here is severe enough (see below) that it usually locks onto
*one* mode's history and never recovers the other.

**Bimodality check** — `P(x_t > 0)` at `t=0,1`, grid ground truth vs. chain, K=200:

| | t | grid | plain PG | PGAS |
|---|---|---|---|---|
| | 0 | 0.885 | 0.885 (diff 0.001) | 0.880 (diff 0.006) |
| | 1 | 0.888 | 0.885 (diff 0.003) | 0.883 (diff 0.005) |

Both variants recover the true mode weights closely at this K — the
reference trajectory does successfully hop between modes over the course of
the chain, for both plain PG and PGAS. The real difference between them
shows up in mixing speed, not asymptotic accuracy.

**Mixing speed** — chain ESS of `x_0` (bimodal *and* furthest from the data):

| K | plain PG chain ESS | PGAS chain ESS | speedup |
|---|---|---|---|
| 50 | 29.0 / 1800 (1.6%) | **1421.7** / 1800 (79.0%) | 49.1x |
| 200 | 607.2 / 1800 (33.7%) | **1709.8** / 1800 (95.0%) | 2.8x |
| 500 | 1090.2 / 1800 (60.6%) | **1801.0** / 1800 (100.1%) | 1.7x |

**Path degeneracy** is much worse than the linear case at matched K (K=50:
1/50 unique ancestors survive to `t=0` in one bootstrap-PF sweep; K=200:
3/200) — consistent with this being a notoriously hard benchmark precisely
*because* of its nonlinearity and near-uninformative observations.

Bottom line: PGAS's advantage over plain PG, already clear on the easy linear
model, gets substantially larger on a harder, genuinely multimodal target —
and, encouragingly, neither variant silently collapses onto a single mode at
the K values tested here.

### Reproducing

```
cd experiments/particle_gibbs
python3 nonlinear_ssm_model.py        # model + grid filter/smoother sanity check
python3 nonlinear_bootstrap_pf.py     # log-lik bias vs K, path degeneracy demo
python3 nonlinear_particle_gibbs.py   # PG/PGAS smoke test
python3 nonlinear_pg_validate.py      # full accuracy + bimodality + mixing-speed comparison (~5 minutes)
```

## What this doesn't cover

This validates the *algorithm*, not an alan integration — `csmc_sweep` is
hand-written against the toy SSM's specific transition/observation densities,
not wired into `Plate.py`/`Sampler.py`'s general recursive sampling. Turning
this into something usable on arbitrary alan `Problem`s is still the scope of
the (unstarted) "Particle Gibbs-specific" checklist in
`docs/pmcmc_design.md` — this prototype exists to de-risk that plan by
confirming the PGAS-over-plain-PG mixing advantage is real and large before
investing in the more invasive `Plate.py`/`Sampler.py` integration work.
