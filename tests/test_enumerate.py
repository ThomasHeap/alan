"""
Validates alan.Enumerate(): exact marginalisation of a discrete latent by
enumeration, rather than K-sample importance sampling. Ground truth is
computed directly (a finite mixture's marginal likelihood, and its gradient
w.r.t. a component location, both closed-form for a Gaussian mixture).
"""
import math

import torch as t
import torch.nn.functional as F

from alan import Normal, Bernoulli, Categorical, Plate, BoundPlate, Problem, Data, Enumerate


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
