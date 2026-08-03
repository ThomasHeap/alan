"""
Stochastic volatility (SV) model -- the headline benchmark in Andrieu, Doucet
& Holenstein's (2010) original PMCMC paper (already cited in
docs/pmcmc_design.md), chosen as a third, differently-hard test after the
linear-Gaussian model (ssm_model.py, unimodal + closed form) and the
Gordon/Salmond/Smith growth model (nonlinear_ssm_model.py, bimodal). SV's
latent transition is linear-Gaussian and *not* bimodal, but the standard
parameterisation is strongly persistent (phi close to 1), which is the
classic source of slow PMCMC mixing in the literature -- a different axis of
difficulty than nonlinearity/multimodality, and closer to how these models
get used in practice (financial return series).

    x_0 ~ N(mu, sigma_eta^2 / (1 - phi^2))                   (stationary dist.)
    x_t = mu + phi (x_{t-1} - mu) + sigma_eta * eta_t,        eta_t ~ N(0, 1)
    y_t = exp(x_t / 2) * eps_t,                               eps_t ~ N(0, 1)   (observed)

y_t | x_t is Gaussian with variance exp(x_t) -- linear-Gaussian latent, but a
nonlinear (multiplicative-in-x) observation variance, so there's still no
closed-form posterior and Kalman/RTS don't apply; ground truth again comes
from the point-mass grid HMM in grid_hmm.py.

Parameters (mu=0, phi=0.98, sigma_eta=0.15) match the standard
Kim/Shephard/Chib (1998) benchmark used throughout the SV/PMCMC literature.
"""
import math
import torch as t
from grid_hmm import grid_filter_smoother as _grid_filter_smoother, marginal_mean_var

MU, PHI, SIGMA_ETA = 0.0, 0.98, 0.15
T_STEPS = 100

X0_VAR = SIGMA_ETA ** 2 / (1 - PHI ** 2)  # stationary variance of the AR(1) latent

GRID_LIM, N_GRID = 6.0, 600  # log-volatility stays in a much tighter range than the growth model's state


def f(x, tt):
    """Transition mean. tt is unused (the SV latent is time-homogeneous) but
    kept for interface parity with nonlinear_ssm_model.f."""
    return MU + PHI * (x - MU)


def simulate(T=T_STEPS, seed=0):
    g = t.Generator().manual_seed(seed)
    x = t.zeros(T + 1)
    x[0] = MU + X0_VAR ** 0.5 * t.randn(1, generator=g)
    for tt in range(1, T + 1):
        x[tt] = f(x[tt - 1], tt) + SIGMA_ETA * t.randn(1, generator=g)
    y = t.exp(x[1:] / 2) * t.randn(T, generator=g)
    return x, y


def x0_logpdf(x):
    return -0.5 * (x - MU) ** 2 / X0_VAR - 0.5 * math.log(2 * math.pi * X0_VAR)


def transition_logpdf(x_new, x_prev, tt):
    return -0.5 * (x_new - f(x_prev, tt)) ** 2 / SIGMA_ETA ** 2 - 0.5 * math.log(2 * math.pi * SIGMA_ETA ** 2)


def obs_logpdf(y_t, x_t):
    var = t.exp(x_t)
    return -0.5 * y_t ** 2 / var - 0.5 * (math.log(2 * math.pi) + x_t)


def grid_filter_smoother(y, T=T_STEPS):
    """Thin wrapper around grid_hmm's model-agnostic filter/smoother, fixing
    in this model's own densities and grid settings."""
    return _grid_filter_smoother(y, T, GRID_LIM, N_GRID, x0_logpdf, transition_logpdf, obs_logpdf)


if __name__ == "__main__":
    x_true, y = simulate()
    print(f"simulated log-vol range: [{x_true.min():.2f}, {x_true.max():.2f}]  "
          f"(grid covers [{-GRID_LIM:.0f}, {GRID_LIM:.0f}])")

    grid, alpha, gamma, log_lik = grid_filter_smoother(y)
    print(f"filtered-mass rows sum to 1: {t.allclose(alpha.sum(-1), t.ones(T_STEPS + 1, dtype=t.float64))}")
    print(f"smoothed-mass rows sum to 1: {t.allclose(gamma.sum(-1), t.ones(T_STEPS + 1, dtype=t.float64))}")

    mean_f, var_f = marginal_mean_var(grid, alpha)
    mean_s, var_s = marginal_mean_var(grid, gamma)
    print(f"\nlog p(y_1:T) (grid) = {log_lik:.4f}")
    print("t     true_x    filt_mean   filt_sd   smooth_mean   smooth_sd")
    for tt in [0, 1, 2, 10, 25, 50, 75, 100]:
        print(f"{tt:3d}  {x_true[tt]:+.3f}    {mean_f[tt]:+.3f}     {var_f[tt] ** 0.5:.3f}    "
              f"{mean_s[tt]:+.3f}        {var_s[tt] ** 0.5:.3f}")
