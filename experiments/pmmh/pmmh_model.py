"""
A linear-Gaussian hierarchical model with a closed-form target posterior,
used to validate Particle Marginal Metropolis-Hastings (PMMH) driven by
alan's OWN evidence estimator -- not a hand-derived reimplementation of it
(that already exists in experiments/mp_is_bias/br_snis.py's isir_chain_mu,
which analytically re-derives alan's log-weight formula by hand). The point
here is to plug alan's actual `Problem.sample(K).elbo_nograd()` into a real
Metropolis-Hastings loop and check it recovers the exact posterior.

    mu ~ N(0, S0)                       -- PMMH's target parameter. Held
                                            FIXED/external to alan at sample
                                            time (never one of alan's own
                                            latents) -- alan is called once
                                            per MH proposal, at a specific mu.
    theta_p | mu ~ N(mu, TAU),  p=1..P  -- alan's own latent, marginalised
                                            out by its K^n machinery.
    x_p | theta_p ~ N(theta_p, SIGMA)   -- observed.

theta is conjugate-Gaussian given mu, so x_p | mu ~ N(mu, TAU^2 + SIGMA^2)
in closed form -- giving both an exact log p(x|mu) (to sanity-check alan's
noisy P_hat_MP against) and, since the mu-prior is also Gaussian, an exact
closed-form posterior p(mu | x) to validate the PMMH chain against.
"""
import math
import torch as t

P_SIZE = 30
S0, TAU, SIGMA = 1.0, 0.7, 0.5


def simulate(seed=0, true_mu=0.6):
    g = t.Generator().manual_seed(seed)
    theta = true_mu + TAU * t.randn(P_SIZE, generator=g)
    x = theta + SIGMA * t.randn(P_SIZE, generator=g)
    return x


def exact_posterior(x):
    """Closed-form p(mu | x): x_p | mu ~ N(mu, TAU^2+SIGMA^2) iid, mu ~ N(0, S0^2)."""
    obs_var = TAU ** 2 + SIGMA ** 2
    prior_prec = 1.0 / S0 ** 2
    lik_prec = x.numel() / obs_var
    post_prec = prior_prec + lik_prec
    post_var = 1.0 / post_prec
    post_mean = post_var * (x.sum().item() / obs_var)
    return post_mean, post_var


def log_marginal_likelihood(mu, x):
    """Exact log p(x | mu), closed form -- for cross-checking alan's noisy
    P_hat_MP(x|mu) is centered correctly, independent of the PMMH chain."""
    obs_var = TAU ** 2 + SIGMA ** 2
    return (-0.5 * (x - mu) ** 2 / obs_var - 0.5 * math.log(2 * math.pi * obs_var)).sum().item()


def log_prior(mu):
    return -0.5 * (mu / S0) ** 2 - 0.5 * math.log(2 * math.pi * S0 ** 2)


if __name__ == "__main__":
    x = simulate()
    post_mean, post_var = exact_posterior(x)
    print(f"data mean={x.mean():.4f}")
    print(f"exact posterior: mu | x ~ N({post_mean:.4f}, {post_var:.4f})  (sd={post_var ** 0.5:.4f})")
