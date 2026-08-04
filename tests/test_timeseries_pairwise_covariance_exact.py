import torch as t

from alan import Normal, Timeseries, Plate, BoundPlate, Problem, Data, PermutationSampler

A, Q_VAR, R_VAR, P0, T_STEPS = 0.9, 0.3, 0.5, 1.0, 20


def _simulate(seed=0):
    g = t.Generator().manual_seed(seed)
    x = t.zeros(T_STEPS + 1)
    x[0] = P0 ** 0.5 * t.randn(1, generator=g)
    for tt in range(1, T_STEPS + 1):
        x[tt] = A * x[tt - 1] + Q_VAR ** 0.5 * t.randn(1, generator=g)
    y = x[1:] + R_VAR ** 0.5 * t.randn(T_STEPS, generator=g)
    return x, y


def _exact_posterior_covariance():
    """Brute-force Gaussian conditioning: y_t = x_t + noise observes each x_t
    directly, so the full joint posterior covariance is closed form."""
    prior_cov = t.zeros(T_STEPS, T_STEPS)
    diag_var = P0
    for k in range(T_STEPS):
        diag_var = diag_var * A ** 2 + Q_VAR
        prior_cov[k, k] = diag_var
        fut = diag_var * A ** t.arange(T_STEPS - k)
        prior_cov[k, k:] = fut
        prior_cov[k:, k] = fut
    return prior_cov - prior_cov @ t.linalg.inv(prior_cov + R_VAR * t.eye(T_STEPS)) @ prior_cov


def _build_problem(y):
    P = Plate(
        init=Normal(0., P0 ** 0.5),
        T=Plate(ts=Timeseries('init', Normal(lambda prev: A * prev, Q_VAR ** 0.5)), obs=Normal('ts', R_VAR ** 0.5)),
    )
    P = BoundPlate(P, {'T': T_STEPS})
    Q = Plate(
        init=Normal(0., 1.),
        T=Plate(ts=Timeseries('init', Normal(lambda prev: 0. * prev, 1.0)), obs=Data()),
    )
    Q = BoundPlate(Q, {'T': T_STEPS})
    return Problem(P, Q, {'obs': y.rename('T')})


def test_timeseries_pairwise_covariance_exact_matches_closed_form():
    """
    Cov(x_i, x_j) between two different timesteps of the same Timeseries
    variable, computed via the autodiff source-term trick
    (logpq._timeseries_pairwise_source_term), should match the exact
    closed-form posterior covariance on this linear-Gaussian model -- for
    both adjacent and distant pairs, and both clearly-nonzero and near-zero
    true covariances. Error here is purely from K being finite (a discrete
    approximation to the posterior), with no additional N-sampling noise.
    """
    x_true, y = _simulate()
    post_cov = _exact_posterior_covariance()
    problem = _build_problem(y)

    t.manual_seed(0)
    sample = problem.sample(K=200, reparam=False, sampler=PermutationSampler)

    for (i, j) in [(0, 1), (0, 3), (0, 9), (5, 6), (5, 10)]:
        cov_est, _, _ = sample.timeseries_pairwise_covariance_exact('ts', i, j)
        cov_true = post_cov[i, j].item()
        assert abs(cov_est - cov_true) < 0.01, (
            f"pair ({i},{j}): estimated {cov_est:.4f} vs true {cov_true:.4f}"
        )


def test_timeseries_pairwise_covariance_exact_accuracy_improves_with_K():
    """Exact-given-K-support-points estimator: error should shrink monotonically
    as K grows, since (unlike the N-sampling based version) there's no extra
    Monte Carlo noise at all -- only K-truncation error."""
    x_true, y = _simulate()
    post_cov = _exact_posterior_covariance()
    true_cov_01 = post_cov[0, 1].item()

    errors = []
    for K in [30, 100, 500]:
        problem = _build_problem(y)
        t.manual_seed(0)
        sample = problem.sample(K=K, reparam=False, sampler=PermutationSampler)
        cov_est, _, _ = sample.timeseries_pairwise_covariance_exact('ts', 0, 1)
        errors.append(abs(cov_est - true_cov_01))

    assert errors[0] > errors[1] > errors[2]


def test_timeseries_pairwise_covariance_exact_is_deterministic():
    """Unlike timeseries_pairwise_covariance (which averages N sampled
    resolutions of upstream uncertainty), the autodiff version involves no
    sampling at all, so repeated calls on the same Sample must agree exactly."""
    x_true, y = _simulate()
    problem = _build_problem(y)
    t.manual_seed(0)
    sample = problem.sample(K=30, reparam=False, sampler=PermutationSampler)

    results = [sample.timeseries_pairwise_covariance_exact('ts', 0, 1) for _ in range(3)]
    assert results[0] == results[1] == results[2]


def test_timeseries_pairwise_covariance_exact_more_accurate_than_sampled_at_large_K():
    """At large, matched K/N, the exact autodiff version should be at least as
    accurate as the N-conditioned hand-derived version, since it has no
    N-sampling noise on top of the K-truncation error they both share."""
    x_true, y = _simulate()
    post_cov = _exact_posterior_covariance()
    true_cov_01 = post_cov[0, 1].item()

    problem = _build_problem(y)
    t.manual_seed(0)
    sample = problem.sample(K=500, reparam=False, sampler=PermutationSampler)

    exact_cov, _, _ = sample.timeseries_pairwise_covariance_exact('ts', 0, 1)
    sampled_cov, _, _ = sample.timeseries_pairwise_covariance('ts', 0, 1, N=500)

    exact_err = abs(exact_cov - true_cov_01)
    sampled_err = abs(sampled_cov - true_cov_01)
    assert exact_err <= sampled_err + 1e-6


def test_timeseries_pairwise_covariance_exact_rejects_bad_indices():
    x_true, y = _simulate()
    problem = _build_problem(y)
    t.manual_seed(0)
    sample = problem.sample(K=20, reparam=False, sampler=PermutationSampler)

    try:
        sample.timeseries_pairwise_covariance_exact('ts', 5, 5)
        assert False, "expected an exception for i == j"
    except Exception as e:
        assert "i < j" in str(e)
