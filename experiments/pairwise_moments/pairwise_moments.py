"""
Prototype: cross-timestep joint posterior moments (e.g. Cov(x_i, x_j) for a
Timeseries variable) using the SAME building blocks alan's own
sample_Ks_timeseries (reduce_Ks.py) already computes -- a forward-filtered
alpha_i, a backward message beta_j, and one genuinely new piece, an
un-contracted "bridge" matrix product over the (i, j) sub-range:

    p(x_i=k_i, x_j=k_j | y_1:T)  ~  alpha_i(k_i) * Bridge_{i->j}(k_i, k_j) * beta_j(k_j)

This is the standard forward-backward pairwise-smoothing formula (Rabiner
1989's xi_t, generalised from adjacent pairs to arbitrary lags; see also
Briers/Doucet/Maskell 2004, already cited in reduce_Ks.py). alpha_i is
exactly filtered_t at t=i. beta_j is the backward-only message, extractable
from the existing smoothed_t recursion since smoothed_t - filtered_t
satisfies its own clean recursion. Bridge_{i->j} = chain_logmmexp of the
per-timestep transition matrices over just (i, j], not summed down to a
scalar the way the existing all-the-way-through cumulative product is.

Captures the REAL lp_ordered tensor alan's own importance-sampling machinery
builds (via sample_Ks_timeseries's exact `.order(T_dim, init_K_dim, K_dim)`
call, intercepted from a genuine problem.sample(K).importance_sample(N) run)
rather than hand-reconstructing it -- so any discrepancy against ground
truth reflects the pairwise-marginal FORMULA, not a mismatch against alan's
actual internal representation. Deliberately does the minimum possible work
in torchdim space (one conditioning step, mirroring the existing code
exactly) and everything else -- the bridge, beta recursion, and final
combination -- in plain PyTorch, to avoid the dim-labelling subtleties that
have proven error-prone elsewhere in this Timeseries code this session.

Eventual target (not attempted here): do this via alan's OWN source-term-
trick autograd machinery (insert source terms at two timestep positions,
differentiate) instead of this hand-derived computation -- this prototype
exists to validate the formula/algorithm first, cheaply, before that.
"""
import sys
import warnings
warnings.filterwarnings("ignore")

import torch as t
import alan  # noqa: F401 -- ensures the package (and its submodules) are initialised
from alan import Normal, Timeseries, Plate, BoundPlate, Problem, Data, PermutationSampler
from alan.utils import generic_dims, generic_order
from alan.reduce_Ks import logmmexp, chain_logmmexp

from ssm_model import A, Q_VAR, R_VAR, P0, T_STEPS, simulate


def build_problem(y, T=T_STEPS):
    P = Plate(init=Normal(0., P0 ** 0.5), T=Plate(
        ts=Timeseries('init', Normal(lambda prev: A * prev, Q_VAR ** 0.5)),
        obs=Normal('ts', R_VAR ** 0.5)))
    P = BoundPlate(P, {'T': T})
    Q = Plate(init=Normal(0., 1.), T=Plate(
        ts=Timeseries('init', Normal(lambda prev: 0. * prev, 1.0)),
        obs=Data()))
    Q = BoundPlate(Q, {'T': T})
    return Problem(P, Q, {'obs': y.rename('T')})


def capture_lp_ordered(problem, K, N, seed):
    """Runs a real problem.sample(K).importance_sample(N) call, intercepting
    sample_Ks_timeseries's own arguments so we can rebuild the EXACT same
    lp_ordered = lp.order(T_dim, init_K_dim, K_dim) plain [T,K,K] tensor
    alan's real backward pass uses -- instead of reconstructing it by hand."""
    SLQ = sys.modules['alan.sample_logpq']
    orig_fn = SLQ.sample_Ks_timeseries
    captured = {}

    def wrapper(lps, Ks_to_sum, ts_init_Ks, N_dim, num_samples, T_dim, indices):
        K_dim = Ks_to_sum[0]
        init_K_dim = ts_init_Ks[0]
        lp = sum(lps)
        for dim in list(set(generic_dims(lp)).intersection(indices.keys()).difference({init_K_dim})):
            lp = lp.order(dim)[indices[dim]]
        lp_ordered = lp.order(T_dim, init_K_dim, K_dim)  # plain [T, K, K]

        # condition on (and average over N_dim, mirroring filtered_t/smoothed_t)
        # the already-sampled init index, to get a plain [K] alpha_0 prior-ish start
        captured['lp_ordered'] = lp_ordered
        captured['init_K_dim'] = init_K_dim
        captured['K_dim'] = K_dim
        captured['N_dim'] = N_dim
        captured['indices_init'] = indices[init_K_dim]

        return orig_fn(lps, Ks_to_sum, ts_init_Ks, N_dim, num_samples, T_dim, indices)

    SLQ.sample_Ks_timeseries = wrapper
    try:
        t.manual_seed(seed)
        sample = problem.sample(K, reparam=False, sampler=PermutationSampler)
        imp = sample.importance_sample(N)
    finally:
        SLQ.sample_Ks_timeseries = orig_fn

    return sample, imp, captured


def alpha_beta_bridge_xi(captured, i, j):
    """All plain-PyTorch from here on. lp_ordered[t] is [K,K]: row = index at
    t-1 (or, for t=0, init's own index), col = index at t -- this convention
    is what makes cumulative[t]=logmmexp(cumulative[t-1], lp_ordered[t])
    chain correctly as ordinary matrix multiplication (already validated
    elsewhere this session), and is all this function relies on."""
    lp_ordered = captured['lp_ordered']
    init_K_dim, K_dim, N_dim = captured['init_K_dim'], captured['K_dim'], captured['N_dim']
    T = lp_ordered.shape[0]
    assert 0 <= i < j <= T - 1

    # alpha_i: forward filtered log-mass at i, conditioned on the
    # already-sampled init index (the one torchdim step in this function).
    cumulative_i = lp_ordered[0]
    for t_idx in range(1, i + 1):
        cumulative_i = logmmexp(cumulative_i, lp_ordered[t_idx])
    alpha_i_torchdim = cumulative_i[init_K_dim, K_dim]
    alpha_i = alpha_i_torchdim.order(init_K_dim)[captured['indices_init']]
    alpha_i = alpha_i.order(N_dim)
    alpha_i = t.logsumexp(alpha_i, 0).order(K_dim)  # plain [K], index at i
    alpha_i = alpha_i - t.logsumexp(alpha_i, 0)      # normalise

    # beta_j: backward message, beta_T-1 = 0 (log-space), plain [K] throughout.
    K = alpha_i.shape[0]
    beta = t.zeros(K)
    for t_idx in range(T - 1, j, -1):
        # beta at t_idx-1: sum over "index at t_idx" (columns of lp_ordered[t_idx])
        beta = t.logsumexp(lp_ordered[t_idx] + beta[None, :], dim=1)
    beta_j = beta  # plain [K], index at j

    # Bridge_{i->j}: un-contracted chain over (i, j], plain [K,K],
    # row = index at i, col = index at j.
    bridge = lp_ordered[i + 1]
    for t_idx in range(i + 2, j + 1):
        bridge = logmmexp(bridge, lp_ordered[t_idx])

    log_xi = alpha_i[:, None] + bridge + beta_j[None, :]
    log_xi = log_xi - t.logsumexp(log_xi.reshape(-1), 0)
    return log_xi.exp()  # [K, K], sums to 1, xi[k_i, k_j]


def joint_moment(sample, captured, i, j, T_dim):
    """E[ts_i ts_j] and Cov(ts_i, ts_j) from xi_ij and the actual sampled values."""
    xi = alpha_beta_bridge_xi(captured, i, j)
    ts = sample.detached_sample['T']['ts']
    K_dim = captured['K_dim']
    ts_ordered = generic_order(ts, [T_dim, K_dim])  # [T, K]
    s_i, s_j = ts_ordered[i], ts_ordered[j]

    mean_i = (xi.sum(1) * s_i).sum()
    mean_j = (xi.sum(0) * s_j).sum()
    e_ij = (xi * s_i[:, None] * s_j[None, :]).sum()
    return (e_ij - mean_i * mean_j).item(), mean_i.item(), mean_j.item()


if __name__ == "__main__":
    x_true, y = simulate()

    # Exact ground truth via brute-force Gaussian conditioning (y_t = x_t + noise
    # observes each x_t directly, so this is closed form).
    prior_cov = t.zeros(T_STEPS, T_STEPS)
    diag_var = P0
    for k in range(T_STEPS):
        diag_var = diag_var * A ** 2 + Q_VAR
        prior_cov[k, k] = diag_var
        fut = diag_var * A ** t.arange(T_STEPS - k)
        prior_cov[k, k:] = fut
        prior_cov[k:, k] = fut
    post_cov = prior_cov - prior_cov @ t.linalg.inv(prior_cov + R_VAR * t.eye(T_STEPS)) @ prior_cov

    problem = build_problem(y)
    sample, imp, captured = capture_lp_ordered(problem, K=200, N=200, seed=0)
    T_dim = problem.all_platedims['T']

    print("pair (i,j)   true Cov      estimated Cov   |diff|")
    for (i, j) in [(0, 1), (0, 3), (0, 9), (5, 6), (5, 10)]:
        cov_est, _, _ = joint_moment(sample, captured, i, j, T_dim)
        cov_true = post_cov[i, j].item()
        print(f"({i:2d},{j:2d})      {cov_true:+.4f}       {cov_est:+.4f}         {abs(cov_true - cov_est):.4f}")
