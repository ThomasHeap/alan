"""
BR-SNIS (Cardoso et al. 2022) style debiasing, prototyped on the toy hierarchical
model in compare_model.py. BR-SNIS replaces a fixed-K self-normalized importance
sample with an i-SIR (iterated sampling-importance-resampling) Markov chain: at
each step, the current state joins a fresh pool of K-1 proposal draws, and a new
state is drawn from the pool proportional to importance weight. This i-SIR kernel
is a standard exact MCMC kernel targeting the true posterior (Andrieu-Doucet-
Holenstein / independent-proposal MH-with-pool) for any K >= 2, so the *ergodic
average* of the chain converges to the true posterior expectation as the chain
length T grows -- regardless of pool size K. That's the source of the bias
reduction: plain SNIS has O(1/K) bias for any fixed K; the i-SIR chain average's
bias instead decays with T (chain mixing), and can be driven arbitrarily low
without needing K -> infinity.

Two variants, matching the two structurally different reductions identified in
jackknife_orderB.py:

  - isir_chain_full: exact i-SIR over the FULL joint (mu, theta) -- the target
    density here is closed-form (no inner Monte Carlo noise), since it's just
    the toy model's unnormalized posterior. This is the natural, well-posed
    home for BR-SNIS: it's a flat, single-stage SNIS problem, exactly what
    Global-IS (alan_compare.global_is_moments) already is. Unlike jackknife,
    which could only debias mu (the flat-ratio reduction) and not theta (the
    nested logsumexp reduction), this version debiases BOTH mu and theta,
    because at the full-joint level neither variable is reduced through a
    nested logsumexp -- that nesting is an artifact of alan's *plate-wise*
    marginalization (Order B), which this flat i-SIR chain sidesteps entirely
    by not doing any plate-level marginalization at all.

  - isir_chain_mu: a pseudo-marginal i-SIR chain over mu ALONE, whose per-
    candidate "likelihood" is alan's actual noisy K_theta-particle logsumexp
    estimate of p(x|mu) (mirrors compute_log_r_mu in jackknife_orderB.py, not
    the closed-form marginal). This is the honest test of "can BR-SNIS-style
    chain recycling debias the specific quantity jackknife_mu was already
    debiasing, but via chain mixing instead of delete-one extrapolation."
    Correctness requires standard pseudo-marginal MCMC discipline: once a
    candidate is accepted as the new reference, its (noisy) log-weight must be
    FROZEN and reused on subsequent steps, not recomputed -- refreshing it
    would break the chain's stationary distribution.

No attempt is made to apply i-SIR directly to theta's own K_theta particles in
isolation (matching mu's treatment) -- theta's posterior marginal is only
well-defined jointly with mu (it enters the model as theta_i | mu), so there's
no flat, self-contained SNIS problem for theta alone the way there is for mu.
The full-joint chain above is the correct way to reach theta.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import math
import torch as t
from compare_model import P_SIZE, S0, TAU, SIGMA, x_data, analytic_posterior

mean, cov = analytic_posterior()
true_mu = mean[0].item()
true_theta = mean[1:]


# ---------------------------------------------------------------------------
# Full-joint i-SIR (exact target, no inner MC noise)
# ---------------------------------------------------------------------------

def log_target_minus_proposal(mu, theta):
    """log[p(mu) p(theta|mu) p(x|theta)] - log q(mu,theta), q = N(0,1) per latent
    (same Q as alan_compare.build_problem). mu: [N], theta: [N, P_SIZE]."""
    logp_mu = -0.5 * (mu / S0) ** 2
    logp_theta_given_mu = -0.5 * (((theta - mu.unsqueeze(-1)) / TAU) ** 2).sum(-1)
    logp_x_given_theta = -0.5 * (((x_data - theta) / SIGMA) ** 2).sum(-1)
    logq = -0.5 * mu ** 2 - 0.5 * (theta ** 2).sum(-1)
    return logp_mu + logp_theta_given_mu + logp_x_given_theta - logq


def isir_chain_full(K, T, burn_in, seed):
    """Exact i-SIR MCMC chain over the full joint (mu, theta_1..P). Returns the
    post-burn-in chain as (mu_chain [T-burn_in], theta_chain [T-burn_in, P_SIZE])."""
    g = t.Generator().manual_seed(seed)
    mu_ref = t.randn(1, generator=g)
    theta_ref = t.randn(1, P_SIZE, generator=g)

    mu_chain, theta_chain = [], []
    for step in range(T):
        mu_fresh = t.randn(K - 1, generator=g)
        theta_fresh = t.randn(K - 1, P_SIZE, generator=g)
        mu_pool = t.cat([mu_ref, mu_fresh])
        theta_pool = t.cat([theta_ref, theta_fresh])

        logw = log_target_minus_proposal(mu_pool, theta_pool)
        logw = logw - logw.max()
        w = logw.exp()
        idx = t.multinomial(w, 1, generator=g).item()

        mu_ref = mu_pool[idx:idx + 1]
        theta_ref = theta_pool[idx:idx + 1]
        if step >= burn_in:
            mu_chain.append(mu_ref.item())
            theta_chain.append(theta_ref.squeeze(0).clone())
    return t.tensor(mu_chain), t.stack(theta_chain)


def br_snis_full_estimate(K, T, burn_in, seed):
    mu_chain, theta_chain = isir_chain_full(K, T, burn_in, seed)
    return mu_chain.mean().item(), theta_chain.mean(0)


# ---------------------------------------------------------------------------
# Pseudo-marginal i-SIR over mu only (noisy K_theta-particle likelihood, as
# alan's own MP-IS actually computes it -- see jackknife_orderB.compute_log_r_mu)
# ---------------------------------------------------------------------------

def mu_candidate_logweight(mu_val, K_theta, g):
    """Order-B pseudo-marginal log-weight for one mu candidate: logA(mu) plus a
    K_theta-particle logsumexp estimate of log p(x|mu), summed over the plate.
    This is a NOISY estimator (fresh randomness each call) -- callers must
    freeze the returned value once a candidate is accepted, per standard
    pseudo-marginal MCMC discipline."""
    theta_p = t.randn(K_theta, P_SIZE, generator=g)                        # [K_theta, P]
    logp_theta_given_mu = -0.5 * ((theta_p - mu_val) / TAU) ** 2           # [K_theta, P]
    logq_theta = -0.5 * theta_p ** 2                                       # [K_theta, P]
    x_term = -0.5 * ((x_data[None, :] - theta_p) / SIGMA) ** 2             # [K_theta, P]
    combined = logp_theta_given_mu - logq_theta + x_term                   # [K_theta, P]
    per_elem = t.logsumexp(combined, dim=0) - math.log(K_theta)            # [P]
    logA = -0.5 * (mu_val / S0) ** 2 - (-0.5 * mu_val ** 2)
    return (logA + per_elem.sum()).item()


def isir_chain_mu(K, K_theta, T, burn_in, seed):
    """Pseudo-marginal i-SIR chain over mu alone. Returns post-burn-in mu chain."""
    g = t.Generator().manual_seed(seed)
    mu_ref = t.randn(1, generator=g).item()
    logw_ref = mu_candidate_logweight(mu_ref, K_theta, g)

    chain = []
    for step in range(T):
        mus = [mu_ref] + [t.randn(1, generator=g).item() for _ in range(K - 1)]
        logws = [logw_ref] + [mu_candidate_logweight(m, K_theta, g) for m in mus[1:]]
        logws_t = t.tensor(logws)
        logws_t = logws_t - logws_t.max()
        w = logws_t.exp()
        idx = t.multinomial(w, 1, generator=g).item()

        mu_ref = mus[idx]
        logw_ref = logws[idx]   # freeze -- do NOT recompute on future steps
        if step >= burn_in:
            chain.append(mu_ref)
    return t.tensor(chain)


def br_snis_mu_estimate(K, K_theta, T, burn_in, seed):
    return isir_chain_mu(K, K_theta, T, burn_in, seed).mean().item()


if __name__ == "__main__":
    print(f"true_mu={true_mu:.4f}  true_theta={true_theta}")

    print("\n--- full-joint i-SIR smoke test, K=5, T=500, burn_in=100 ---")
    mu_hat, theta_hat = br_snis_full_estimate(K=5, T=500, burn_in=100, seed=0)
    print(f"mu_hat={mu_hat:.4f}  theta_hat={theta_hat}")

    print("\n--- pseudo-marginal mu-only i-SIR smoke test, K=5, K_theta=30, T=500, burn_in=100 ---")
    mu_hat2 = br_snis_mu_estimate(K=5, K_theta=30, T=500, burn_in=100, seed=0)
    print(f"mu_hat={mu_hat2:.4f}")
