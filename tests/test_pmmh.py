import math

import torch as t

from alan import Normal, Plate, BoundPlate, Problem, Data, pmmh

P_SIZE = 30
S0, TAU, SIGMA = 1.0, 0.7, 0.5


def _simulate(seed=0, true_mu=0.6):
    """
    A linear-Gaussian hierarchical model with a closed-form target posterior:

        mu ~ N(0, S0)                       -- PMMH's target parameter, external
                                                to alan (never one of its latents).
        theta_p | mu ~ N(mu, TAU),  p=1..P  -- alan's own latent, marginalised out
                                                by its own K^n machinery.
        x_p | theta_p ~ N(theta_p, SIGMA)   -- observed.

    theta is conjugate-Gaussian given mu, so p(mu | x) is also closed-form
    Gaussian -- exact ground truth to validate the PMMH chain against.
    """
    g = t.Generator().manual_seed(seed)
    theta = true_mu + TAU * t.randn(P_SIZE, generator=g)
    x = theta + SIGMA * t.randn(P_SIZE, generator=g)
    return x


def _exact_posterior(x):
    obs_var = TAU ** 2 + SIGMA ** 2
    prior_prec = 1.0 / S0 ** 2
    lik_prec = x.numel() / obs_var
    post_prec = prior_prec + lik_prec
    post_var = 1.0 / post_prec
    post_mean = post_var * (x.sum().item() / obs_var)
    return post_mean, post_var


def _log_prior(mu):
    return -0.5 * (mu / S0) ** 2 - 0.5 * math.log(2 * math.pi * S0 ** 2)


def _build_problem(mu, x):
    P = Plate(p1=Plate(theta=Normal(mu, TAU), x=Normal('theta', SIGMA)))
    P = BoundPlate(P, {'p1': P_SIZE})
    Q = Plate(p1=Plate(theta=Normal(0., 1.), x=Data()))
    Q = BoundPlate(Q, {'p1': P_SIZE})
    return Problem(P, Q, {'x': x.rename('p1')})


def _propose(scale):
    def propose(mu, g):
        return mu + scale * t.randn(1, generator=g).item()
    return propose


def test_pmmh_matches_closed_form_posterior():
    x = _simulate()
    post_mean, post_var = _exact_posterior(x)

    t.manual_seed(0)
    result = pmmh(
        build_problem=lambda mu: _build_problem(mu, x),
        log_prior=_log_prior,
        propose=_propose(0.3),
        theta_init=0.0,
        n_iters=1500,
        K=100,
        burn_in=300,
        seed=0,
    )
    chain = t.tensor(result.chain)

    assert abs(chain.mean().item() - post_mean) < 0.05
    assert abs(chain.var().item() - post_var) < 0.03
    assert 0.0 < result.accept_rate < 1.0


def test_pmmh_unbiased_at_low_K():
    """The core pseudo-marginal claim: even a small, noisy K should give an
    unbiased chain, just a slower-mixing one -- not a biased one."""
    x = _simulate()
    post_mean, _ = _exact_posterior(x)

    t.manual_seed(1)
    result = pmmh(
        build_problem=lambda mu: _build_problem(mu, x),
        log_prior=_log_prior,
        propose=_propose(0.3),
        theta_init=0.0,
        n_iters=1500,
        K=10,
        burn_in=300,
        seed=1,
    )
    chain = t.tensor(result.chain)

    # Loose tolerance: small K means high variance, but should still be centered
    # in roughly the right place, not systematically biased to one side.
    assert abs(chain.mean().item() - post_mean) < 0.3
