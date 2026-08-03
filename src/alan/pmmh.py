"""
Particle Marginal Metropolis-Hastings (PMMH), driven by alan's own unbiased
evidence estimator.

Andrieu & Roberts (2009)'s pseudo-marginal MCMC result: plug any
nonnegative, unbiased estimator of a target's normalising constant into an
ordinary Metropolis-Hastings acceptance ratio, and the resulting chain has
the *exact* posterior as its stationary distribution, for any number of
importance samples K. Smaller K only costs mixing speed, never correctness.

alan's own `Problem.sample(K).elbo_nograd()` (i.e. log P_hat_MP) is exactly
such an estimator -- it's exactly unbiased for the model evidence. This
module wires it directly into a Metropolis-Hastings loop over a parameter
`theta`, rather than requiring a hand-derived reimplementation of alan's own
log-weight computation.
"""
import math
from collections import namedtuple

import torch as t

from .Sampler import PermutationSampler
from .Split import checkpoint


PMMHResult = namedtuple('PMMHResult', ['chain', 'accept_rate'])


def _log_phat_mp(build_problem, theta, K, sampler, computation_strategy):
    problem = build_problem(theta)
    sample = problem.sample(K, reparam=False, sampler=sampler)
    return float(sample.elbo_nograd(computation_strategy=computation_strategy))


def pmmh(
        build_problem,
        log_prior,
        propose,
        theta_init,
        n_iters:int,
        K:int,
        burn_in:int=0,
        seed:int=None,
        sampler=PermutationSampler,
        computation_strategy=checkpoint):
    """
    pmmh(build_problem, log_prior, propose, theta_init, n_iters, K, burn_in=0, seed=None, sampler=PermutationSampler, computation_strategy=checkpoint)

    A Metropolis-Hastings chain over a parameter `theta`, using alan's own
    unbiased evidence estimator as the marginal-likelihood plug-in at every
    proposed theta.

    Correctly implements the standard pseudo-marginal discipline of
    *freezing* the current state's (noisy) log-likelihood estimate once
    accepted, rather than recomputing it fresh at every step -- recomputing
    it breaks the chain's stationary distribution (this is a real, easy
    mistake to make, not a theoretical nicety).

    Arguments:
        build_problem (callable):
            theta -> alan.Problem. theta must be baked into the returned
            Problem as a plain, fixed value (NOT one of the Problem's own
            latents) -- alan's own latents are marginalised out by
            ``Problem.sample(K)``; only theta is externally controlled here.
        log_prior (callable):
            theta -> float, the log prior density on theta.
        propose (callable):
            (theta, generator) -> theta, a SYMMETRIC proposal (e.g. a random
            walk). This implementation assumes a symmetric proposal density,
            so it doesn't include a Hastings correction term -- an
            asymmetric proposal would need one.
        theta_init:
            initial value of theta.
        n_iters (int):
            number of Metropolis-Hastings iterations.
        K (int):
            number of importance samples used for each evidence estimate.
            Smaller K only costs mixing speed, never correctness -- see
            the module docstring.
        burn_in (int):
            number of initial iterations to discard from the returned chain.
        seed (int, optional):
            seeds a fresh ``torch.Generator`` for reproducibility. If not
            given, uses the default (global) generator.
        sampler:
            alan ``Sampler`` class used inside ``Problem.sample`` (see
            :mod:`alan.Sampler`).
        computation_strategy:
            passed to ``Sample.elbo_nograd`` (see :ref:`Computation Strategy`).

    Returns:
        PMMHResult(chain, accept_rate), where ``chain`` has length
        ``n_iters - burn_in``.
    """
    g = t.Generator().manual_seed(seed) if seed is not None else t.Generator()

    theta = theta_init
    logpost = _log_phat_mp(build_problem, theta, K, sampler, computation_strategy) + log_prior(theta)

    chain = []
    n_accept = 0
    for it in range(n_iters):
        theta_prop = propose(theta, g)
        logpost_prop = _log_phat_mp(build_problem, theta_prop, K, sampler, computation_strategy) + log_prior(theta_prop)

        log_alpha = logpost_prop - logpost
        if math.log(t.rand(1, generator=g).item()) < log_alpha:
            theta, logpost = theta_prop, logpost_prop  # freeze the accepted estimate
            n_accept += 1

        if it >= burn_in:
            chain.append(theta)

    return PMMHResult(chain=chain, accept_rate=n_accept / n_iters)
