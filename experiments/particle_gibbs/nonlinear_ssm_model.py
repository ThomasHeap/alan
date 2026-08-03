"""
Gordon, Salmond & Smith (1993) nonlinear/non-Gaussian growth-model benchmark
-- the standard stress test in the particle-filtering literature, chosen here
specifically because it breaks the two things the linear-Gaussian toy model in
ssm_model.py can't exercise:

  1. No closed-form posterior (the transition is nonlinear in x, so
     Kalman/RTS don't apply). Ground truth instead comes from a grid-based /
     point-mass HMM forward-backward filter (Kitagawa 1987) -- discretize the
     1D state onto a fine grid and run exact forward-backward on the
     resulting (large but finite) HMM. This converges to the true continuous
     posterior as the grid is refined and needs no closed-form assumption
     about the transition or observation densities.
  2. The observation y_t = x_t^2/20 + noise is symmetric in x_t, so the
     filtering/smoothing marginals are genuinely (at least locally) BIMODAL
     -- a much harder target for a single reference trajectory to represent
     than the unimodal linear-Gaussian case in ssm_model.py.

    x_0 ~ N(0, X0_VAR)
    x_t = 0.5 x_{t-1} + 25 x_{t-1}/(1+x_{t-1}^2) + 8 cos(1.2 t) + w_t,  w_t ~ N(0, Q_VAR)
    y_t = x_t^2 / 20 + v_t,                                            v_t ~ N(0, R_VAR)
"""
import math
import torch as t

Q_VAR, R_VAR, X0_VAR = 10.0, 1.0, 5.0
T_STEPS = 20

# Grid for the exact point-mass filter/smoother (ground truth). Wide and fine
# enough that the simulated trajectory and the bulk of the posterior mass sit
# comfortably inside it -- see __main__ below for the check.
GRID_LIM, N_GRID = 40.0, 800


def f(x, tt):
    """Transition mean. tt is the integer time index (1..T), matching the
    model's explicit t-dependence via the cos term."""
    return 0.5 * x + 25 * x / (1 + x ** 2) + 8 * math.cos(1.2 * tt)


def simulate(T=T_STEPS, seed=0):
    g = t.Generator().manual_seed(seed)
    x = t.zeros(T + 1)
    x[0] = X0_VAR ** 0.5 * t.randn(1, generator=g)
    for tt in range(1, T + 1):
        x[tt] = f(x[tt - 1], tt) + Q_VAR ** 0.5 * t.randn(1, generator=g)
    y = x[1:] ** 2 / 20 + R_VAR ** 0.5 * t.randn(T, generator=g)
    return x, y


def _normal_pdf(x, mean, var):
    return t.exp(-0.5 * (x - mean) ** 2 / var) / math.sqrt(2 * math.pi * var)


def transition_logpdf(x_new, x_prev, tt):
    return -0.5 * (x_new - f(x_prev, tt)) ** 2 / Q_VAR - 0.5 * math.log(2 * math.pi * Q_VAR)


def obs_logpdf(y_t, x_t):
    return -0.5 * (y_t - x_t ** 2 / 20) ** 2 / R_VAR - 0.5 * math.log(2 * math.pi * R_VAR)


def _grid():
    grid = t.linspace(-GRID_LIM, GRID_LIM, N_GRID, dtype=t.float64)
    dx = (grid[-1] - grid[0]) / (N_GRID - 1)
    return grid, dx


def _transition_matrix(grid, dx, tt):
    """M[i, j] ~= p(x_tt = grid[j] | x_{tt-1} = grid[i]) * dx, a nearly-row-
    stochastic [N_GRID, N_GRID] matrix (rows sum to ~1, exactly 1 as
    GRID_LIM -> inf and N_GRID -> inf)."""
    means = f(grid, tt)  # [N_GRID], broadcast over the whole grid as x_{tt-1}
    return _normal_pdf(grid[None, :], means[:, None], Q_VAR) * dx


def grid_filter_smoother(y, T=T_STEPS):
    """Exact (up to grid discretization) forward-filtering + backward-
    smoothing posterior marginals, plus log p(y_1:T), via a point-mass HMM
    (Kitagawa 1987). Returns (grid, alpha [T+1, N_GRID] filtered marginals,
    gamma [T+1, N_GRID] smoothed marginals, log_lik). Both alpha and gamma
    rows are probability masses over the grid, summing to 1."""
    grid, dx = _grid()
    y = y.double()

    alpha = t.zeros(T + 1, N_GRID, dtype=t.float64)
    alpha[0] = _normal_pdf(grid, 0.0, X0_VAR)
    alpha[0] = alpha[0] / alpha[0].sum()

    log_lik = 0.0
    for tt in range(1, T + 1):
        M = _transition_matrix(grid, dx, tt)
        pred = alpha[tt - 1] @ M
        lik = _normal_pdf(y[tt - 1], grid ** 2 / 20, R_VAR)
        unnorm = pred * lik
        total = unnorm.sum()
        log_lik += total.log().item()
        alpha[tt] = unnorm / total

    beta = t.zeros(T + 1, N_GRID, dtype=t.float64)
    beta[T] = 1.0
    for tt in range(T, 0, -1):
        M = _transition_matrix(grid, dx, tt)
        lik = _normal_pdf(y[tt - 1], grid ** 2 / 20, R_VAR)
        unnorm = M @ (lik * beta[tt])
        beta[tt - 1] = unnorm / unnorm.sum()  # rescale to avoid underflow; gamma is a ratio, so this is exact

    gamma = alpha * beta
    gamma = gamma / gamma.sum(dim=1, keepdim=True)

    return grid, alpha, gamma, log_lik


def marginal_mean_var(grid, dist):
    """dist: [..., N_GRID] grid probability masses (rows sum to 1)."""
    mean = (dist * grid).sum(-1)
    var = (dist * (grid - mean.unsqueeze(-1)) ** 2).sum(-1)
    return mean, var


if __name__ == "__main__":
    x_true, y = simulate()
    print(f"simulated x range: [{x_true.min():.2f}, {x_true.max():.2f}]  "
          f"(grid covers [{-GRID_LIM:.0f}, {GRID_LIM:.0f}])")

    grid, alpha, gamma, log_lik = grid_filter_smoother(y)
    print(f"filtered-mass rows sum to 1: {t.allclose(alpha.sum(-1), t.ones(T_STEPS + 1, dtype=t.float64))}")
    print(f"smoothed-mass rows sum to 1: {t.allclose(gamma.sum(-1), t.ones(T_STEPS + 1, dtype=t.float64))}")

    mean_f, var_f = marginal_mean_var(grid, alpha)
    mean_s, var_s = marginal_mean_var(grid, gamma)
    print(f"\nlog p(y_1:T) (grid) = {log_lik:.4f}")
    print("t   true_x    filt_mean   filt_sd   smooth_mean   smooth_sd")
    for tt in [0, 1, 2, 5, 10, 15, 20]:
        print(f"{tt:2d}  {x_true[tt]:+.3f}   {mean_f[tt]:+.3f}     {var_f[tt] ** 0.5:.3f}    "
              f"{mean_s[tt]:+.3f}        {var_s[tt] ** 0.5:.3f}")
