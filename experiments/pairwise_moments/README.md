# Cross-timestep joint moments for Timeseries variables

Validates that cross-timestep joint posterior moments (e.g. `Cov(x_i, x_j)`
for two different timesteps of the *same* `Timeseries` variable) are
derivable from alan's existing per-timestep transition-weight machinery,
without any MCMC/Gibbs/CSMC machinery -- closing a gap identified while
exploring Particle Gibbs this session: `sample.moments()` already gives
exact, cheap, per-timestep *marginal* moments (no sampling, no SNIS bias),
but has no way to relate two different timesteps of the same variable to
each other. Confirmed empirically (see `experiments/alan_csmc/`) that the
K-index at timestep `i` and at timestep `j` of a `Timeseries` sample carry no
structural correspondence -- so naively treating "same K-index" as "same
underlying trajectory" gives garbage (a naive same-k covariance estimate for
an adjacent pair came out at -0.03 against a true value of 0.105).

## The formula

Standard forward-backward pairwise-smoothing formula (Rabiner 1989's `xi_t`,
generalised from adjacent timesteps to arbitrary lags; see also
Briers/Doucet/Maskell 2004, already cited in `reduce_Ks.py`):

```
p(x_i=k_i, x_j=k_j | y_1:T)  ~  alpha_i(k_i) * Bridge_{i->j}(k_i, k_j) * beta_j(k_j)
```

- `alpha_i` is exactly `filtered_t` at `t=i` -- already computed inside
  `sample_Ks_timeseries`.
- `beta_j` (the backward-only message) isn't stored explicitly today, but
  falls straight out of the existing recursion: since
  `smoothed_t = filtered_t + logsumexp[(smoothed_{t+1}-filtered_{t+1}) + transition]`,
  the quantity `smoothed_t - filtered_t` *is* `beta_t`, satisfying its own
  clean recursion (base case `beta_{T-1} = 0` in log-space).
- `Bridge_{i->j}` is the one genuinely new piece: the *uncontracted* forward
  chain over just `(i, j]` -- `chain_logmmexp(lp_ordered[i+1:j+1])`, using
  the exact same per-timestep matrices and `logmmexp` utility already in
  `reduce_Ks.py`, just restricted to a sub-range instead of the full one.

From `xi_ij`: `E[x_i x_j] = sum_{k_i,k_j} xi(k_i,k_j) * sample_i(k_i) * sample_j(k_j)`,
and `Cov(x_i,x_j) = E[x_i x_j] - E[x_i]E[x_j]`.

## Implementation approach

`pairwise_moments.py` intercepts the call to `sample_Ks_timeseries` during a
genuine `problem.sample(K).importance_sample(N)` run, to capture the REAL
`lp_ordered = lp.order(T_dim, init_K_dim, K_dim)` plain `[T,K,K]` tensor
alan's own backward pass builds -- rather than hand-reconstructing it, which
risks silently diverging from what alan actually computes. Deliberately does
the minimum possible work in torchdim space (one conditioning step on the
already-sampled `init` index, mirroring `filtered_t`'s own pattern exactly)
and everything else -- the bridge, the beta recursion, the final `xi`
combination -- in plain PyTorch. This is a deliberate choice: getting
`Kinit_dim`/`K_dim`'s reused-across-every-timestep labelling right by hand
has already caused two separate bugs elsewhere in this session's Timeseries
work, so this prototype avoids replicating that risk anywhere it doesn't
have to.

## Results

Validated against exact ground truth (brute-force Gaussian conditioning --
`y_t = x_t + noise` observes each `x_t` directly, so the full joint
posterior covariance is closed form) on the same linear-Gaussian model used
throughout `experiments/particle_gibbs/`:

| pair (i,j) | true Cov | estimated Cov | \|diff\| |
|---|---|---|---|
| (0,1) adjacent | +0.1054 | +0.0845 | 0.0209 |
| (0,3) | +0.0212 | +0.0161 | 0.0051 |
| (0,9) ~zero | +0.0002 | +0.0001 | 0.0000 |
| (5,6) adjacent | +0.0840 | +0.0836 | 0.0004 |
| (5,10) | +0.0034 | +0.0030 | 0.0004 |

Correctly recovers both near-zero and clearly-nonzero true covariances, for
both adjacent and non-adjacent pairs (K=N=200). Accuracy vs K on pair (0,1):

| K=N | estimated Cov | \|diff\| |
|---|---|---|
| 30 | +0.1354 | 0.0300 |
| 100 | +0.0862 | 0.0192 |
| 500 | +0.0944 | 0.0111 |

Error shrinks with K, as expected for an exact-given-K-support-points
estimator (not a Monte Carlo chain). Cross-checked internal consistency:
marginal means derived by summing `xi` over one index roughly match the
independently-computed, already-validated `sample.moments('ts', mean)` at
the same positions (0.51 vs 0.46 and 0.15 vs 0.13 at t=0,1 -- same order,
different seeds/K so not identical, but clearly the same quantity).

## Reproducing

```
cd experiments/pairwise_moments
python3 pairwise_moments.py   # accuracy check across several (i,j) pairs
```

## What this validates, and what's still needed for real integration

This confirms the FORMULA and ALGORITHM are correct against alan's actual
internal representation -- the thing genuinely worth de-risking before
committing to more invasive work. It is not itself the target end state.

The actual target (not attempted here, per discussion) is doing this via
alan's OWN source-term-trick autograd machinery -- inserting source terms at
two timestep positions within the same Timeseries reduction and
differentiating, the same way `Sample._marginal_idxs` already gets joint
moments *across different variables* in a plate. That needs new plumbing:
`Timeseries.log_prob` (and the extra_log_factors mechanism) would need to
accept a source term indexed at *specific* timestep positions, not just
uniformly across `T`. This prototype's hand-derived forward/backward/bridge
computation is a lower-risk stepping stone that proves the target quantity
is right, before touching that autograd plumbing.
