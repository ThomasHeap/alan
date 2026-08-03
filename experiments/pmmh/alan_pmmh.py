"""
Particle Marginal Metropolis-Hastings (PMMH) over mu, using alan's own
Problem/Sample machinery to compute an unbiased estimate of log p(x|mu) at
each proposed mu -- literally `problem.sample(K).elbo_nograd()`, not a hand-
derived reimplementation (contrast experiments/mp_is_bias/br_snis.py's
isir_chain_mu, which analytically re-derives alan's log-weight formula by
hand instead of calling into alan at all).

Standard pseudo-marginal MCMC discipline (Andrieu & Roberts 2009): once a mu
is accepted as the current state, its (noisy) log-likelihood estimate must
be FROZEN and reused on subsequent steps -- recomputing a fresh estimate for
the current state at every step biases the stationary distribution, because
the acceptance ratio would then compare a fresh noisy estimate against a
fresh noisy estimate rather than a fixed one against a fresh one, breaking
detailed balance. `run_pmmh(..., freeze=False)` reproduces this bug on
purpose, to demonstrate the failure mode empirically rather than just
asserting it (see pmmh_validate.py).
"""
import math
import torch as t
from alan import Normal, Plate, BoundPlate, Problem, Data, PermutationSampler
from pmmh_model import P_SIZE, TAU, SIGMA, log_prior


def build_problem(mu, x):
    P = Plate(p1=Plate(
        theta=Normal(mu, TAU),
        x=Normal('theta', SIGMA),
    ))
    P = BoundPlate(P, {'p1': P_SIZE})

    Q = Plate(p1=Plate(
        theta=Normal(0., 1.),
        x=Data(),
    ))
    Q = BoundPlate(Q, {'p1': P_SIZE})

    return Problem(P, Q, {'x': x.rename('p1')})


def log_phat_mp(mu, x, K, seed):
    """log P_hat_MP(x | mu), alan's own unbiased evidence estimator, computed
    by actually constructing and sampling an alan Problem at this specific,
    externally-fixed mu (mu is a plain float baked into P, not one of alan's
    latents -- only theta is)."""
    t.manual_seed(seed)
    problem = build_problem(mu, x)
    sample = problem.sample(K, reparam=False, sampler=PermutationSampler)
    return float(sample.elbo_nograd())


def run_pmmh(x, n_iters, K, proposal_scale, burn_in, seed, freeze=True):
    """PMMH chain over mu. freeze=True is the correct algorithm; freeze=False
    recomputes the current state's log-likelihood fresh every step (the
    classic pseudo-marginal bug) for comparison. Returns (chain, accept_rate)."""
    g = t.Generator().manual_seed(seed)
    seed_counter = [0]

    def fresh_seed():
        seed_counter[0] += 1
        return seed * 7919 + seed_counter[0]

    mu_ref = 0.0
    logpost_ref = log_phat_mp(mu_ref, x, K, fresh_seed()) + log_prior(mu_ref)

    chain, n_accept = [], 0
    for it in range(n_iters):
        if not freeze:
            logpost_ref = log_phat_mp(mu_ref, x, K, fresh_seed()) + log_prior(mu_ref)

        mu_prop = mu_ref + proposal_scale * t.randn(1, generator=g).item()
        logpost_prop = log_phat_mp(mu_prop, x, K, fresh_seed()) + log_prior(mu_prop)

        log_alpha = logpost_prop - logpost_ref
        if math.log(t.rand(1, generator=g).item()) < log_alpha:
            mu_ref, logpost_ref = mu_prop, logpost_prop  # freeze the accepted estimate
            n_accept += 1

        if it >= burn_in:
            chain.append(mu_ref)

    return t.tensor(chain), n_accept / n_iters


if __name__ == "__main__":
    from pmmh_model import simulate, exact_posterior

    x = simulate()
    post_mean, post_var = exact_posterior(x)
    print(f"exact posterior: mu | x ~ N({post_mean:.4f}, {post_var:.4f})\n")

    print("--- PMMH smoke test, K=50, 1000 iterations, 200 burn-in ---")
    chain, acc = run_pmmh(x, n_iters=1000, K=50, proposal_scale=0.3, burn_in=200, seed=0)
    print(f"chain mean={chain.mean():.4f}  chain var={chain.var():.4f}  accept_rate={acc:.2f}")
