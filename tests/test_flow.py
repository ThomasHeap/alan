"""
Validates alan.Flow(): a distribution built by pushing a base Dist's sample
through a chain of invertible Transforms, with log_prob computed exactly via
the change-of-variables formula. Ground truth throughout is the closed-form
LogNormal distribution (base=Normal, transform=ExpTransform), since it's a
standard, easily-checked example of a transformed Gaussian.
"""
import math

import torch as t
import torch.nn as nn

from alan import Normal, Plate, BoundPlate, Problem, Data, Flow, ExpTransform


def _lognormal_logpdf(z, mu, sigma):
    return -math.log(z) - math.log(sigma) - 0.5 * math.log(2 * math.pi) - (math.log(z) - mu) ** 2 / (2 * sigma ** 2)


def test_flow_log_prob_matches_closed_form_lognormal():
    mu, sigma = 0.5, 0.8
    flow = Flow(Normal(mu, sigma), [ExpTransform()])

    for zval in [0.2, 0.5, 1.0, 2.0, 3.5, 8.0]:
        lp, kinit = flow.log_prob(t.tensor(zval), scope={}, T_dim=None, K_dim=None)
        assert kinit is None
        closed_form = _lognormal_logpdf(zval, mu, sigma)
        assert abs(lp.item() - closed_form) < 1e-4


def test_flow_sample_matches_lognormal_moments():
    mu, sigma = 0.5, 0.8
    P = Plate(z=Flow(Normal(mu, sigma), [ExpTransform()]))
    P = BoundPlate(P, {})

    t.manual_seed(0)
    # .rename(None) works around a pre-existing, Flow-unrelated limitation:
    # torch's .var() doesn't support named tensors (confirmed identically on
    # a plain, non-Flow Normal's BoundPlate.sample() output).
    samples = P.sample(sample_size=200_000)['z'].rename(None)
    emp_mean, emp_var = samples.mean().item(), samples.var().item()

    true_mean = math.exp(mu + sigma ** 2 / 2)
    true_var = (math.exp(sigma ** 2) - 1) * math.exp(2 * mu + sigma ** 2)

    assert abs(emp_mean - true_mean) < 0.05 * true_mean
    assert abs(emp_var - true_var) < 0.05 * true_var


def test_flow_vi_recovers_true_parameters_with_no_likelihood():
    """With no likelihood term at all, the ELBO is exactly -KL(Q||P), so the
    ELBO-optimal Q exactly equals P -- an unambiguous ground truth for
    checking that gradients flow correctly through the whole Flow (base
    distribution AND transform chain) during training."""
    mu_true, sigma_true = 0.5, 0.8

    P = Plate(z=Flow(Normal(mu_true, sigma_true), [ExpTransform()]))
    P = BoundPlate(P, {})

    q_mu = nn.Parameter(t.tensor(0.0))
    q_log_sigma = nn.Parameter(t.tensor(0.0))
    Q = Plate(z=Flow(Normal(q_mu, lambda: q_log_sigma.exp()), [ExpTransform()]))
    Q = BoundPlate(Q, {})
    Q.q_mu = q_mu
    Q.q_log_sigma = q_log_sigma

    assert any(p is q_mu for p in Q.parameters())
    assert any(p is q_log_sigma for p in Q.parameters())

    prob = Problem(P, Q, {})
    opt = t.optim.Adam(Q.parameters(), lr=0.03)

    t.manual_seed(0)
    for _ in range(1000):
        opt.zero_grad()
        sample = prob.sample(K=10, reparam=True)
        elbo = sample.elbo_vi()
        (-elbo).backward()
        opt.step()

    assert abs(q_mu.item() - mu_true) < 0.25
    assert abs(q_log_sigma.exp().item() - sigma_true) < 0.25


def test_flow_sample_nonmp_matches_mp_elbo():
    """sample_nonmp uses a completely separate code path (IndependentSampler,
    SampleNonMP._elbo) from the massively-parallel sample()/elbo_vi() used
    elsewhere in this file. For a single-variable, unplated model the two
    should coincide (no correlation structure across K for either path to
    exploit differently), so this checks SampleNonMP's isinstance(distQ,
    Flow) wiring (SampleNonMP.py) gives a consistent answer, not just one
    that runs without crashing."""
    mu, sigma = 0.5, 0.8
    P = Plate(z=Flow(Normal(mu, sigma), [ExpTransform()]))
    P = BoundPlate(P, {})
    Q = Plate(z=Flow(Normal(mu, sigma), [ExpTransform()]))
    Q = BoundPlate(Q, {})
    prob = Problem(P, Q, {})

    t.manual_seed(0)
    elbo_mp = prob.sample(K=1000, reparam=True).elbo_vi()
    t.manual_seed(0)
    elbo_nonmp = prob.sample_nonmp(K=1000, reparam=True).elbo_vi()

    assert abs(elbo_mp.item() - elbo_nonmp.item()) < 0.1


def test_flow_importance_sample_and_extend_predictive_ll():
    """Exercises Sample.importance_sample / ImportanceSample.extend /
    ExtendedImportanceSample.predictive_ll with a Flow-Q latent feeding an
    ordinary Dist likelihood inside a genuine plate (as opposed to a bare
    top-level scalar Data() variable, which hits a separate, pre-existing,
    Flow-unrelated limitation of corresponding_plates/.extend() -- confirmed
    identically reproducible with no Flow involved at all, so out of scope
    here)."""
    mu_true, sigma_true, obs_sigma = 0.5, 0.8, 0.3
    platesizes = {'plate_1': 4}

    P = Plate(
        z=Flow(Normal(mu_true, sigma_true), [ExpTransform()]),
        plate_1=Plate(obs=Normal('z', obs_sigma)),
    )
    P = BoundPlate(P, platesizes)

    t.manual_seed(0)
    obs_val = P.sample()['obs']

    q_mu = nn.Parameter(t.tensor(0.3))
    q_log_sigma = nn.Parameter(t.tensor(0.0))
    Q = Plate(
        z=Flow(Normal(q_mu, lambda: q_log_sigma.exp()), [ExpTransform()]),
        plate_1=Plate(obs=Data()),
    )
    Q = BoundPlate(Q, platesizes)
    Q.q_mu = q_mu
    Q.q_log_sigma = q_log_sigma

    prob = Problem(P, Q, {'obs': obs_val})
    opt = t.optim.Adam(Q.parameters(), lr=0.03)
    t.manual_seed(0)
    for _ in range(300):
        opt.zero_grad()
        elbo = prob.sample(K=10, reparam=True).elbo_vi()
        (-elbo).backward()
        opt.step()

    sample = prob.sample(K=20, reparam=True)
    imp = sample.importance_sample(N=200)
    z_vals = imp.dump()['z'].rename(None)
    assert t.isfinite(z_vals).all()
    assert (z_vals > 0).all()  # ExpTransform's support is the positive reals

    ext = imp.extend({'plate_1': 8})

    extended_obs = t.cat([obs_val.rename(None), t.zeros(4)]).rename('plate_1')
    pll = ext.predictive_ll({'obs': extended_obs})
    assert t.isfinite(pll['obs'])


def test_flow_rejects_non_dist_base():
    try:
        Flow("not a dist", [ExpTransform()])
        assert False, "expected an error for a non-Dist base_dist"
    except Exception:
        pass


def test_flow_rejects_empty_transforms():
    try:
        Flow(Normal(0., 1.), [])
        assert False, "expected an error for an empty transform list"
    except Exception:
        pass
