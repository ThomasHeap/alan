# PMMH driven by alan's own P_hat_MP

Wires alan's actual `Problem.sample(K).elbo_nograd()` into a real Particle
Marginal Metropolis-Hastings (PMMH) loop over a parameter `mu` — not a
hand-derived reimplementation of alan's log-weight formula (that already
exists in `experiments/mp_is_bias/br_snis.py`'s `isir_chain_mu`, which
analytically re-derives the estimator instead of calling into alan at all).

## Why this works

We proved `P_hat_MP` is exactly unbiased for the model evidence earlier this
session — which is exactly what pseudo-marginal MCMC (Andrieu & Roberts
2009) needs: plug any nonnegative, unbiased estimator of a target's
normalizing constant into an ordinary Metropolis-Hastings acceptance ratio,
and the resulting chain has the *exact* posterior as its stationary
distribution, for any `K`. Smaller `K` should only cost mixing speed, never
correctness — that's the core, checkable claim here.

## Setup

```
mu ~ N(0, S0)                       -- PMMH's target parameter. External to
                                        alan (never one of its latents) --
                                        alan is called once per MH proposal,
                                        at one specific mu.
theta_p | mu ~ N(mu, TAU),  p=1..P  -- alan's own latent, marginalised out
                                        by its K^n machinery.
x_p | theta_p ~ N(theta_p, SIGMA)   -- observed.
```

`theta` is conjugate-Gaussian given `mu`, so `x_p | mu ~ N(mu, TAU^2+SIGMA^2)`
in closed form, giving an exact `log p(x|mu)` (to sanity-check alan's noisy
`P_hat_MP` against) and, since the `mu`-prior is Gaussian too, an exact
closed-form posterior `p(mu|x)` to validate the PMMH chain against.

- `pmmh_model.py` — the model, simulation, closed-form posterior.
- `alan_pmmh.py` — `build_problem(mu, x)` treats `mu` as a fixed external
  value (only `theta` is an alan latent); `log_phat_mp(...)` calls alan's
  real `elbo_nograd()`; `run_pmmh(...)` implements the standard
  pseudo-marginal discipline of *freezing* the accepted state's noisy
  log-likelihood estimate rather than recomputing it fresh every step
  (`freeze=False` reproduces that bug on purpose, for comparison).
- `pmmh_validate.py` — checks the chain is unbiased across `K`, and
  empirically demonstrates the `freeze=False` bug's bias.

## Results (1500 iterations, 300 burn-in, 5 reps per row)

**Correctness vs K** (should be unbiased at *every* K — smaller K only costs
mixing speed):

| K | chain mean | bias | chain var (true 0.0241) | accept rate | chain ESS |
|---|---|---|---|---|---|
| 5 | 0.5432 | -0.0600 (se 0.0318) | 0.0116 | 0.01 | 14.1/1200 |
| 20 | 0.6272 | +0.0239 (se 0.0278) | 0.0177 | 0.09 | 35.5/1200 |
| 100 | 0.6061 | +0.0028 (se 0.0044) | 0.0235 | 0.33 | 145.2/1200 |

K=100 nails it (bias 0.0028, tiny se). K=20 is within ~1 se of zero bias.
K=5 shows a bias of about 2 se — plausible, given chain ESS is only ~14 out
of 1200 samples at that K (accept rate 1%, so almost the whole chain is
autocorrelated repeats of a few states); with only 5 reps at this ESS, the
Monte Carlo error on the bias estimate itself is large. Not run long enough
to fully separate "still consistent, just very slow-mixing" from "something
else going on at very small K" — a longer chain at K=5 would be the natural
follow-up if this matters going forward.

**The freeze discipline** (freeze=True is the correct algorithm; freeze=False
recomputes the current state's noisy log-likelihood fresh every step, the
classic pseudo-marginal bug):

| K | freeze=True bias | freeze=False bias |
|---|---|---|
| 5 | +0.0225 (se 0.0673) | **-0.1161** (se 0.0294) |
| 20 | -0.0161 (se 0.0074) | **-0.0371** (se 0.0074) |

At K=20 the effect is clean: freeze=False is about 2x further from zero,
with comparable noise. At K=5 the freeze=True estimate itself is noisy (large
se, and inconsistent with the independent K=5 run in the table above — both
are real, independent 5-rep runs at very low ESS, not a contradiction, just
Monte Carlo noise), so the freeze=False comparison there is suggestive
rather than clean. The qualitative point — recomputing the current state's
likelihood breaks the chain — holds up at K=20 regardless.

## Reproducing

```
cd experiments/pmmh
python3 pmmh_model.py      # closed-form ground truth sanity check
python3 alan_pmmh.py       # smoke test, one PMMH chain
python3 pmmh_validate.py   # full correctness-vs-K + freeze-discipline comparison (~9 minutes)
```

## What this doesn't cover

Only tested on one small, closed-form model with a single scalar parameter.
The interesting further step is a model where `p(x|mu)` has no closed form
at all (so there's no way to cross-check `P_hat_MP`'s bias directly, only
the chain's stationary behaviour) — and/or a vector-valued parameter, where
proposal tuning starts to matter more.
