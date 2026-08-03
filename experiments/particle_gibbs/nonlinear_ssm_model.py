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
from grid_hmm import grid_filter_smoother as _grid_filter_smoother, marginal_mean_var

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


def x0_logpdf(x):
    return -0.5 * x ** 2 / X0_VAR - 0.5 * math.log(2 * math.pi * X0_VAR)


def transition_logpdf(x_new, x_prev, tt):
    return -0.5 * (x_new - f(x_prev, tt)) ** 2 / Q_VAR - 0.5 * math.log(2 * math.pi * Q_VAR)


def obs_logpdf(y_t, x_t):
    return -0.5 * (y_t - x_t ** 2 / 20) ** 2 / R_VAR - 0.5 * math.log(2 * math.pi * R_VAR)


def grid_filter_smoother(y, T=T_STEPS):
    """Thin wrapper around grid_hmm's model-agnostic filter/smoother, fixing
    in this model's own densities and grid settings. See grid_hmm.py for the
    general recipe (Kitagawa 1987 point-mass HMM forward-backward)."""
    return _grid_filter_smoother(y, T, GRID_LIM, N_GRID, x0_logpdf, transition_logpdf, obs_logpdf)


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
