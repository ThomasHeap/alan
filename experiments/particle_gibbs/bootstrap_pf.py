"""
Standard bootstrap particle filter (SIS + systematic resampling every step) on
the toy SSM in ssm_model.py. Two things this establishes, both needed to
motivate and validate Particle Gibbs:

  1. An unbiased estimator of log p(y_1:T) (standard PF theory) -- checked
     against the exact Kalman filter log-likelihood.
  2. The classic PATH DEGENERACY problem: because every resampling step prunes
     the particle population down to whichever ancestors happened to get high
     weight, tracing genealogies back from the final time step collapses onto
     very few (often exactly 1) surviving ancestors at early t. This is
     PRECISELY the failure mode Particle Gibbs / PGAS exist to fix -- CSMC's
     pinned reference trajectory guarantees at least ONE full trajectory
     survives every resampling step, by construction, no matter how much the
     rest of the population degenerates.
"""
import torch as t
from ssm_model import A, Q_VAR, R_VAR, P0, T_STEPS, simulate, exact_posterior


def systematic_resample(weights, g):
    """weights: normalized [K]. Returns ancestor indices [K]."""
    K = weights.shape[0]
    u0 = t.rand(1, generator=g).item()
    positions = (u0 + t.arange(K, dtype=t.float64)) / K
    cumsum = t.cumsum(weights.double(), dim=0)
    cumsum[-1] = 1.0
    idx = t.searchsorted(cumsum, positions)
    return idx.clamp(max=K - 1)


def bootstrap_pf(y, K, T=T_STEPS, seed=0):
    """Returns (particle_history, ancestry, log_lik).
    particle_history[t]: [K] particle VALUES at time t, post-resampling (t=0..T).
    ancestry[t-1]: [K] indices into particle_history[t-1], for t=1..T.
    log_lik: unbiased estimate of log p(y_1:T)."""
    g = t.Generator().manual_seed(seed)
    x = P0 ** 0.5 * t.randn(K, generator=g)
    particle_history = [x.clone()]
    ancestry = []

    log_lik = 0.0
    for tt in range(1, T + 1):
        x_pred = A * x + Q_VAR ** 0.5 * t.randn(K, generator=g)
        logw = -0.5 * (y[tt - 1] - x_pred) ** 2 / R_VAR   # constants cancel in the mean-of-w step below
        w_max = logw.max()
        w = (logw - w_max).exp()
        log_lik += (w_max + t.log(w.mean()) - 0.5 * t.log(2 * t.pi * t.tensor(R_VAR))).item()

        w_norm = w / w.sum()
        idx = systematic_resample(w_norm, g)
        ancestry.append(idx.clone())
        x = x_pred[idx]
        particle_history.append(x.clone())

    return particle_history, ancestry, log_lik


def reconstruct_trajectories(particle_history, ancestry, T=T_STEPS):
    """Trace full [K, T+1] trajectories back through the ancestry lineage from
    the (equally-weighted, post-resampling) final particle set."""
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
    m_smooth, P_smooth, log_lik_exact = exact_posterior(y)

    print(f"exact log p(y_1:T) = {log_lik_exact:.4f}\n")
    for K in [20, 100, 500]:
        ests = []
        n_unique = []
        for rep in range(30):
            ph, anc, ll = bootstrap_pf(y, K, seed=100 * K + rep)
            ests.append(ll)
            n_unique.append(unique_ancestors_at_t0(ph, anc))
        ests = t.tensor(ests)
        print(f"K={K:4d}  log_lik_hat mean={ests.mean():.4f} (bias {ests.mean()-log_lik_exact:+.4f}), "
              f"std={ests.std():.4f}   |  unique ancestors surviving to t=0: "
              f"mean={sum(n_unique)/len(n_unique):.1f} / K={K} (min {min(n_unique)}, max {max(n_unique)})")

    print("\n--- genealogy-based smoothing mean vs RTS smoother ground truth, K=200 ---")
    ph, anc, _ = bootstrap_pf(y, 200, seed=0)
    full = reconstruct_trajectories(ph, anc)
    pf_mean = full.mean(0)
    print("t     RTS smoother    PF genealogy    |diff|")
    for tt in [0, 1, 2, 5, 10, 15, 20]:
        print(f"{tt:2d}    {m_smooth[tt]:+.4f}         {pf_mean[tt]:+.4f}        {abs(m_smooth[tt]-pf_mean[tt]):.4f}")
