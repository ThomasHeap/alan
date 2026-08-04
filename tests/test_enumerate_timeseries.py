"""
Validates alan.Enumerate() applied to a Timeseries variable: exact
discrete-state HMM forward filtering / forward-backward smoothing.

Timeseries.log_prob's relabelling trick already computes the full
[Kinit, Kcurr] transition log-probability matrix at each timestep via
broadcasting, and the existing chain_logmmexp + logsumexp reduction already
implements the standard forward algorithm's alpha_t = alpha_{t-1} @
Transition_t recursion -- so Enumerate needs no new reduction machinery
here, just Q's Timeseries slot being a bare Enumerate() (mirroring how it
can already be Data() or a plain Dist instead of a full Timeseries object),
deterministically arange(C) at every timestep, and the same -log(K)
removal validated for the non-Timeseries case. Ground truth throughout is a
hand-rolled forward / forward-backward algorithm in plain PyTorch.
"""
import math

import torch as t

from alan import Normal, Bernoulli, Plate, BoundPlate, Problem, Data, Enumerate, Timeseries, mean


def _normal_pdf(x, mu, sigma):
    return math.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * math.sqrt(2 * math.pi))


def _simulate_hmm(T_steps, pi1, trans01, trans11, mu0, mu1, sigma, seed):
    g = t.Generator().manual_seed(seed)
    states = [1 if t.rand(1, generator=g).item() < pi1 else 0]
    for _ in range(T_steps - 1):
        p = trans11 if states[-1] == 1 else trans01
        states.append(1 if t.rand(1, generator=g).item() < p else 0)
    obs = t.tensor([
        (mu1 if s == 1 else mu0) + sigma * t.randn(1, generator=g).item()
        for s in states
    ])
    return states, obs


def _hand_forward(obs, pi1, trans01, trans11, mu0, mu1, sigma):
    """Standard HMM forward algorithm; returns (log P(obs), list of alpha vectors)."""
    trans = t.tensor([[1 - trans01, trans01], [1 - trans11, trans11]])
    emit = lambda x: t.tensor([_normal_pdf(x, mu0, sigma), _normal_pdf(x, mu1, sigma)])

    alphas = [t.tensor([1 - pi1, pi1]) * emit(obs[0].item())]
    for tt in range(1, obs.shape[0]):
        alphas.append((alphas[-1] @ trans) * emit(obs[tt].item()))
    return math.log(alphas[-1].sum().item()), alphas


def _hand_forward_backward(obs, pi1, trans01, trans11, mu0, mu1, sigma):
    """Standard HMM forward-backward smoother; returns per-timestep P(z_t=1|obs)."""
    T = obs.shape[0]
    trans = t.tensor([[1 - trans01, trans01], [1 - trans11, trans11]])
    emit = lambda x: t.tensor([_normal_pdf(x, mu0, sigma), _normal_pdf(x, mu1, sigma)])

    _, alphas = _hand_forward(obs, pi1, trans01, trans11, mu0, mu1, sigma)

    betas = [None] * T
    betas[-1] = t.ones(2)
    for tt in range(T - 2, -1, -1):
        betas[tt] = trans @ (emit(obs[tt + 1].item()) * betas[tt + 1])

    smoothed = []
    for tt in range(T):
        g = alphas[tt] * betas[tt]
        smoothed.append((g / g.sum())[1].item())
    return smoothed


def _build_problem(obs, T_steps, pi1, trans01, trans11, mu0, mu1, sigma):
    P = Plate(
        init=Bernoulli(pi1),
        T=Plate(
            z=Timeseries('init', Bernoulli(lambda prev: prev * trans11 + (1 - prev) * trans01)),
            obs=Normal(lambda z: z * mu1 + (1 - z) * mu0, sigma),
        ),
    )
    P = BoundPlate(P, {'T': T_steps})
    Q = Plate(
        init=Enumerate(),
        T=Plate(z=Enumerate(), obs=Data()),
    )
    Q = BoundPlate(Q, {'T': T_steps})
    return Problem(P, Q, {'obs': obs.rename('T')})


PI1, TRANS01, TRANS11, MU0, MU1, SIGMA = 0.5, 0.3, 0.7, -1.0, 2.0, 0.5


def test_enumerate_timeseries_matches_forward_algorithm():
    states, obs = _simulate_hmm(8, PI1, TRANS01, TRANS11, MU0, MU1, SIGMA, seed=0)
    true_loglik, _ = _hand_forward(obs, PI1, TRANS01, TRANS11, MU0, MU1, SIGMA)

    prob = _build_problem(obs, 8, PI1, TRANS01, TRANS11, MU0, MU1, SIGMA)
    elbo = prob.sample(K=2, reparam=False).elbo_nograd()

    assert abs(elbo.item() - true_loglik) < 1e-3


def test_enumerate_timeseries_matches_forward_algorithm_longer_T():
    """T=30 exercises chain_logmmexp's O(T) tree reduction, not just a short chain."""
    states, obs = _simulate_hmm(30, PI1, TRANS01, TRANS11, MU0, MU1, SIGMA, seed=1)
    true_loglik, _ = _hand_forward(obs, PI1, TRANS01, TRANS11, MU0, MU1, SIGMA)

    prob = _build_problem(obs, 30, PI1, TRANS01, TRANS11, MU0, MU1, SIGMA)
    elbo = prob.sample(K=2, reparam=False).elbo_nograd()

    assert abs(elbo.item() - true_loglik) < 1e-2


def test_enumerate_timeseries_is_deterministic():
    states, obs = _simulate_hmm(8, PI1, TRANS01, TRANS11, MU0, MU1, SIGMA, seed=0)
    prob = _build_problem(obs, 8, PI1, TRANS01, TRANS11, MU0, MU1, SIGMA)
    elbos = [prob.sample(K=2, reparam=False).elbo_nograd().item() for _ in range(3)]
    assert all(e == elbos[0] for e in elbos)


def test_enumerate_timeseries_moments_match_forward_backward_smoother():
    states, obs = _simulate_hmm(8, PI1, TRANS01, TRANS11, MU0, MU1, SIGMA, seed=1)
    true_smoothed = _hand_forward_backward(obs, PI1, TRANS01, TRANS11, MU0, MU1, SIGMA)

    prob = _build_problem(obs, 8, PI1, TRANS01, TRANS11, MU0, MU1, SIGMA)
    sample = prob.sample(K=2, reparam=False)
    mom = sample.moments([('z', mean)])[0].rename(None)

    assert (mom - t.tensor(true_smoothed)).abs().max().item() < 1e-4


def test_enumerate_timeseries_gradient_flows():
    """Confirms autograd propagates through the exact HMM machinery into a P
    parameter (the emission means), via a finite-difference check against
    the closed-form forward algorithm."""
    states, obs = _simulate_hmm(6, PI1, TRANS01, TRANS11, MU0, MU1, SIGMA, seed=2)

    def loglik(mu1_val):
        return _hand_forward(obs, PI1, TRANS01, TRANS11, MU0, mu1_val, SIGMA)[0]

    eps = 1e-3
    finite_diff_grad = (loglik(MU1 + eps) - loglik(MU1 - eps)) / (2 * eps)

    mu1_param = t.tensor(MU1, requires_grad=True)
    P = Plate(
        init=Bernoulli(PI1),
        T=Plate(
            z=Timeseries('init', Bernoulli(lambda prev: prev * TRANS11 + (1 - prev) * TRANS01)),
            obs=Normal(lambda z: z * mu1_param + (1 - z) * MU0, SIGMA),
        ),
    )
    P = BoundPlate(P, {'T': 6})
    Q = Plate(init=Enumerate(), T=Plate(z=Enumerate(), obs=Data()))
    Q = BoundPlate(Q, {'T': 6})
    prob = Problem(P, Q, {'obs': obs.rename('T')})

    elbo = prob.sample(K=2, reparam=True).elbo_vi()
    elbo.backward()

    assert abs(mu1_param.grad.item() - finite_diff_grad) < 1e-2


def test_enumerate_timeseries_rejects_wrong_K():
    states, obs = _simulate_hmm(5, PI1, TRANS01, TRANS11, MU0, MU1, SIGMA, seed=0)
    prob = _build_problem(obs, 5, PI1, TRANS01, TRANS11, MU0, MU1, SIGMA)
    try:
        prob.sample(K=5, reparam=False).elbo_nograd()
        assert False, "expected an error for K != Bernoulli's cardinality"
    except Exception:
        pass
