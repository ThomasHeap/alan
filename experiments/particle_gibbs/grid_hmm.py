"""
Generic grid-based / point-mass HMM forward-backward filter+smoother
(Kitagawa 1987), factored out of nonlinear_ssm_model.py because sv_model.py
needs the identical numerical recipe against a different transition/
observation density. Model-agnostic: takes the prior/transition/observation
log-densities as callables, discretizes the (assumed scalar) state onto a
fine grid, and runs exact forward-backward on the resulting HMM -- this
converges to the true continuous posterior as the grid is refined, and needs
no closed-form assumption about the densities involved.
"""
import torch as t


def grid_filter_smoother(y, T, grid_lim, n_grid, x0_logpdf, transition_logpdf, obs_logpdf):
    """
    x0_logpdf(x) -> log p(x_0 = x)
    transition_logpdf(x_new, x_prev, tt) -> log p(x_tt = x_new | x_{tt-1} = x_prev), tt in 1..T
    obs_logpdf(y_t, x) -> log p(y_t | x_t = x)
    All three must broadcast elementwise over a grid-shaped `x`/`x_new`/`x_prev`.

    Returns (grid, alpha [T+1, n_grid], gamma [T+1, n_grid], log_lik). alpha
    (filtered) and gamma (smoothed) rows are probability masses over the
    grid, each summing to 1.
    """
    grid = t.linspace(-grid_lim, grid_lim, n_grid, dtype=t.float64)
    dx = (grid[-1] - grid[0]) / (n_grid - 1)
    y = y.double()

    def transition_matrix(tt):
        # M[i, j] ~= p(x_tt=grid[j] | x_{tt-1}=grid[i]) * dx, nearly row-stochastic
        return transition_logpdf(grid[None, :], grid[:, None], tt).exp() * dx

    alpha = t.zeros(T + 1, n_grid, dtype=t.float64)
    alpha[0] = x0_logpdf(grid).exp()
    alpha[0] = alpha[0] / alpha[0].sum()

    log_lik = 0.0
    Ms = []
    for tt in range(1, T + 1):
        M = transition_matrix(tt)
        Ms.append(M)
        pred = alpha[tt - 1] @ M
        lik = obs_logpdf(y[tt - 1], grid).exp()
        unnorm = pred * lik
        total = unnorm.sum()
        log_lik += total.log().item()
        alpha[tt] = unnorm / total

    beta = t.zeros(T + 1, n_grid, dtype=t.float64)
    beta[T] = 1.0
    for tt in range(T, 0, -1):
        lik = obs_logpdf(y[tt - 1], grid).exp()
        unnorm = Ms[tt - 1] @ (lik * beta[tt])
        beta[tt - 1] = unnorm / unnorm.sum()  # rescale to avoid underflow; gamma is a ratio, so this is exact

    gamma = alpha * beta
    gamma = gamma / gamma.sum(dim=1, keepdim=True)

    return grid, alpha, gamma, log_lik


def marginal_mean_var(grid, dist):
    """dist: [..., n_grid] grid probability masses (rows sum to 1)."""
    mean = (dist * grid).sum(-1)
    var = (dist * (grid - mean.unsqueeze(-1)) ** 2).sum(-1)
    return mean, var
