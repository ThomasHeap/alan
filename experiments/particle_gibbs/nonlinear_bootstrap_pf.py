"""
Bootstrap particle filter on the nonlinear/non-Gaussian growth model in
nonlinear_ssm_model.py, mirroring bootstrap_pf.py's structure but against a
model with no closed-form ground truth. Two things this establishes, both
needed to motivate and validate Particle Gibbs on this harder model:

  1. An unbiased-in-expectation estimator of log p(y_1:T), checked against
     the exact grid-based filter's log-likelihood.
  2. Path degeneracy, same failure mode as in the linear-Gaussian case -- but
     here it interacts with the genuine bimodality of the filtering
     distribution (see nonlinear_ssm_model.py's docstring): a single
     bootstrap-PF sweep tends to collapse onto whichever mode happened to
     get lucky early weights, not just onto few ancestors.
"""
import torch as t
from nonlinear_ssm_model import (
    Q_VAR, T_STEPS, f, obs_logpdf, simulate, grid_filter_smoother, marginal_mean_var,
)


def systematic_resample(weights, g):
    K = weights.shape[0]
    u0 = t.rand(1, generator=g).item()
    positions = (u0 + t.arange(K, dtype=t.float64)) / K
    cumsum = t.cumsum(weights.double(), dim=0)
    cumsum[-1] = 1.0
    idx = t.searchsorted(cumsum, positions)
    return idx.clamp(max=K - 1)


def bootstrap_pf(y, K, T=T_STEPS, seed=0):
    """Returns (particle_history, ancestry, log_lik), same conventions as
    bootstrap_pf.py's version."""
    g = t.Generator().manual_seed(seed)
    x = 5.0 ** 0.5 * t.randn(K, generator=g)  # X0_VAR = 5.0
    particle_history = [x.clone()]
    ancestry = []

    log_lik = 0.0
    for tt in range(1, T + 1):
        x_pred = f(x, tt) + Q_VAR ** 0.5 * t.randn(K, generator=g)
        logw = obs_logpdf(y[tt - 1], x_pred)
        w_max = logw.max()
        w = (logw - w_max).exp()
        log_lik += (w_max + t.log(w.mean())).item()

        w_norm = w / w.sum()
        idx = systematic_resample(w_norm, g)
        ancestry.append(idx.clone())
        x = x_pred[idx]
        particle_history.append(x.clone())

    return particle_history, ancestry, log_lik


def reconstruct_trajectories(particle_history, ancestry, T=T_STEPS):
    K = particle_history[0].shape[0]
    full = t.zeros(K, T + 1)
    idx = t.arange(K)
    full[:, T] = particle_history[T][idx]
    for tt in range(T, 0, -1):
        idx = ancestry[tt - 1][idx]
        full[:, tt - 1] = particle_history[tt - 1][idx]
    return full


def unique_ancestors_at_t0(particle_history, ancestry, T=T_STEPS):
    K = particle_history[0].shape[0]
    idx = t.arange(K)
    for tt in range(T, 0, -1):
        idx = ancestry[tt - 1][idx]
    return len(idx.unique())


if __name__ == "__main__":
    x_true, y = simulate()
    grid, alpha, gamma, log_lik_exact = grid_filter_smoother(y)
    mean_s, _ = marginal_mean_var(grid, gamma)

    print(f"exact log p(y_1:T) (grid) = {log_lik_exact:.4f}\n")
    for K in [20, 100, 500, 2000]:
        ests = []
        n_unique = []
        for rep in range(30):
            ph, anc, ll = bootstrap_pf(y, K, seed=100 * K + rep)
            ests.append(ll)
            n_unique.append(unique_ancestors_at_t0(ph, anc))
        ests = t.tensor(ests)
        print(f"K={K:4d}  log_lik_hat mean={ests.mean():.4f} (bias {ests.mean() - log_lik_exact:+.4f}), "
              f"std={ests.std():.4f}   |  unique ancestors surviving to t=0: "
              f"mean={sum(n_unique) / len(n_unique):.1f} / K={K} (min {min(n_unique)}, max {max(n_unique)})")

    print("\n--- genealogy-based smoothing mean vs grid ground truth, K=500 ---")
    ph, anc, _ = bootstrap_pf(y, 500, seed=0)
    full = reconstruct_trajectories(ph, anc)
    pf_mean = full.mean(0)
    print("t     grid smoother    PF genealogy    |diff|")
    for tt in [0, 1, 2, 5, 10, 15, 20]:
        print(f"{tt:2d}    {mean_s[tt]:+.4f}         {pf_mean[tt]:+.4f}        {abs(mean_s[tt] - pf_mean[tt]):.4f}")
