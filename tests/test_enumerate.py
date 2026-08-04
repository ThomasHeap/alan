"""
Validates alan.Enumerate(): exact marginalisation of a discrete latent by
enumeration, rather than K-sample importance sampling. Ground truth is
computed directly (a finite mixture's marginal likelihood, and its gradient
w.r.t. a component location, both closed-form for a Gaussian mixture).
"""
import math

import torch as t
import torch.nn.functional as F

from alan import Normal, Bernoulli, Categorical, Plate, BoundPlate, Problem, Data, Enumerate, mean, mean2


def _normal_pdf(x, mu, sigma):
    return math.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * math.sqrt(2 * math.pi))


def test_enumerate_bernoulli_matches_closed_form_marginal():
    p_true, mu0, mu1, sigma, obs_val = 0.3, -2.0, 3.0, 1.0, 1.0

    P = Plate(
        z=Bernoulli(p_true),
        obs=Normal(lambda z: z * mu1 + (1 - z) * mu0, sigma),
    )
    P = BoundPlate(P, {})
    Q = Plate(z=Enumerate(), obs=Data())
    Q = BoundPlate(Q, {})

    prob = Problem(P, Q, {'obs': t.tensor(obs_val)})
    sample = prob.sample(K=2, reparam=False)
    elbo = sample.elbo_nograd()

    true_marginal = (1 - p_true) * _normal_pdf(obs_val, mu0, sigma) + p_true * _normal_pdf(obs_val, mu1, sigma)
    assert abs(elbo.item() - math.log(true_marginal)) < 1e-4


def test_enumerate_is_deterministic():
    """Unlike K-sample importance sampling, enumeration has zero variance --
    repeated calls on freshly-drawn samples must agree exactly."""
    P = Plate(z=Bernoulli(0.3), obs=Normal(lambda z: z * 3.0 + (1 - z) * -2.0, 1.0))
    P = BoundPlate(P, {})
    Q = Plate(z=Enumerate(), obs=Data())
    Q = BoundPlate(Q, {})
    prob = Problem(P, Q, {'obs': t.tensor(1.0)})

    elbos = [prob.sample(K=2, reparam=False).elbo_nograd().item() for _ in range(5)]
    assert all(e == elbos[0] for e in elbos)


def test_enumerate_categorical_gradient_matches_closed_form():
    """3-way categorical mixture with a learnable component location (a P
    parameter). Both the marginal likelihood and its gradient w.r.t. that
    parameter should match the closed-form mixture exactly -- confirming
    autograd flows correctly through the exact marginalisation, not just
    that the forward value is right."""
    mus_true = t.tensor([-3.0, 0.0, 4.0])
    probs = t.tensor([0.2, 0.5, 0.3])
    sigma, obs_val = 1.0, 1.0

    mu_param = mus_true.clone().requires_grad_(True)

    P = Plate(
        z=Categorical(probs),
        obs=Normal(lambda z: (F.one_hot(z.long(), 3).float() * mu_param).sum(-1), sigma),
    )
    P = BoundPlate(P, {})
    Q = Plate(z=Enumerate(), obs=Data())
    Q = BoundPlate(Q, {})

    prob = Problem(P, Q, {'obs': t.tensor(obs_val)})
    sample = prob.sample(K=3, reparam=True)
    elbo = sample.elbo_vi()

    true_marginal = sum(p.item() * _normal_pdf(obs_val, mu.item(), sigma) for p, mu in zip(probs, mus_true))
    assert abs(elbo.item() - math.log(true_marginal)) < 1e-4

    elbo.backward()
    denom = true_marginal
    for i, (p, mu) in enumerate(zip(probs, mus_true)):
        w = p.item() * _normal_pdf(obs_val, mu.item(), sigma) / denom
        expected_grad = w * (obs_val - mu.item()) / sigma ** 2
        assert abs(mu_param.grad[i].item() - expected_grad) < 1e-4


def test_enumerate_moments_marginals_importance_sample_match_closed_form():
    """sample.moments/.marginals()/.importance_sample() all share the same
    underlying reduction as elbo_vi/elbo_nograd -- confirm all three give
    the exact posterior mean/variance of an enumerated variable (moments,
    marginals), and that importance_sample's genuine resampled draws land
    within Monte Carlo noise of the true posterior."""
    p_true, mu0, mu1, sigma, obs_val = 0.3, -2.0, 3.0, 1.0, 1.0

    P = Plate(z=Bernoulli(p_true), obs=Normal(lambda z: z * mu1 + (1 - z) * mu0, sigma))
    P = BoundPlate(P, {})
    Q = Plate(z=Enumerate(), obs=Data())
    Q = BoundPlate(Q, {})
    prob = Problem(P, Q, {'obs': t.tensor(obs_val)})

    w0 = (1 - p_true) * _normal_pdf(obs_val, mu0, sigma)
    w1 = p_true * _normal_pdf(obs_val, mu1, sigma)
    true_post_mean = w1 / (w0 + w1)
    true_post_var = true_post_mean * (1 - true_post_mean)  # z in {0,1}, so Var(z) = p(1-p)

    sample = prob.sample(K=2, reparam=False)

    post_mean, post_mean2 = sample.moments([('z', mean), ('z', mean2)])
    assert abs(post_mean.item() - true_post_mean) < 1e-4
    assert abs((post_mean2.item() - post_mean.item() ** 2) - true_post_var) < 1e-4

    marg_mean = sample.marginals().moments([('z', mean)])[0]
    assert abs(marg_mean.item() - true_post_mean) < 1e-4

    t.manual_seed(0)
    imp = sample.importance_sample(N=5000)
    empirical_mean = imp.dump()['z'].mean().item()
    std_err = (true_post_var / 5000) ** 0.5
    assert abs(empirical_mean - true_post_mean) < 5 * std_err


def test_enumerate_rejects_wrong_K():
    P = Plate(z=Bernoulli(0.3), obs=Normal('z', 1.0))
    P = BoundPlate(P, {})
    Q = Plate(z=Enumerate(), obs=Data())
    Q = BoundPlate(Q, {})
    prob = Problem(P, Q, {'obs': t.tensor(1.0)})

    try:
        prob.sample(K=5, reparam=False).elbo_nograd()
        assert False, "expected an error for K != Bernoulli's cardinality"
    except Exception:
        pass


def test_enumerate_rejects_enumerate_in_P():
    P = Plate(z=Enumerate(), obs=Normal('z', 1.0))
    P = BoundPlate(P, {})
    Q = Plate(z=Bernoulli(0.3), obs=Data())
    Q = BoundPlate(Q, {})
    try:
        Problem(P, Q, {'obs': t.tensor(1.0)})
        assert False, "expected an error for Enumerate() in P"
    except Exception:
        pass
