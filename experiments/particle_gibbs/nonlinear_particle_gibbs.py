"""
Particle Gibbs (conditional SMC) + PGAS on the nonlinear/non-Gaussian growth
model, mirroring particle_gibbs.py's structure and the same CSMC/PGAS
construction (Andrieu, Doucet & Holenstein 2010; Lindsten, Jordan & Schoen
2014) -- only the transition/observation densities change, both algorithms
are otherwise model-agnostic. The point of running this on the harder model
is that the reference trajectory now has to represent a genuinely BIMODAL
target (see nonlinear_ssm_model.py's docstring), which is a much sterner test
of whether a single pinned trajectory can mix between modes at all.
"""
import torch as t
from nonlinear_ssm_model import Q_VAR, T_STEPS, f, transition_logpdf, obs_logpdf

X0_VAR = 5.0


def csmc_sweep(y, K, ref_traj, ancestor_sampling, g, T=T_STEPS):
    x = t.empty(K)
    x[0] = ref_traj[0]
    x[1:] = X0_VAR ** 0.5 * t.randn(K - 1, generator=g)
    particle_history = [x.clone()]
    w = t.full((K,), 1.0 / K)
    weight_history = [w.clone()]
    ancestry = []

    for tt in range(1, T + 1):
        anc = t.zeros(K, dtype=t.long)
        anc[1:] = t.multinomial(w, K - 1, replacement=True, generator=g)

        x_new = t.zeros(K)
        x_new[1:] = f(x[anc[1:]], tt) + Q_VAR ** 0.5 * t.randn(K - 1, generator=g)
        x_new[0] = ref_traj[tt]

        if ancestor_sampling:
            logw_as = t.log(w + 1e-300) + transition_logpdf(ref_traj[tt], x, tt)
            logw_as = logw_as - logw_as.max()
            w_as = logw_as.exp()
            w_as = w_as / w_as.sum()
            anc[0] = t.multinomial(w_as, 1, generator=g).item()
        else:
            anc[0] = 0
        ancestry.append(anc.clone())

        logw = obs_logpdf(y[tt - 1], x_new)
        logw = logw - logw.max()
        w_new = logw.exp()
        w_new = w_new / w_new.sum()

        particle_history.append(x_new.clone())
        weight_history.append(w_new.clone())
        x, w = x_new, w_new

    return particle_history, weight_history, ancestry


def sample_new_reference(particle_history, weight_history, ancestry, g, T=T_STEPS):
    j = t.multinomial(weight_history[T], 1, generator=g).item()
    new_traj = t.zeros(T + 1)
    idx = j
    new_traj[T] = particle_history[T][idx]
    for tt in range(T, 0, -1):
        idx = ancestry[tt - 1][idx].item()
        new_traj[tt - 1] = particle_history[tt - 1][idx]
    return new_traj


def run_particle_gibbs(y, n_iters, K, ancestor_sampling=False, T=T_STEPS, seed=0, init_traj=None):
    g = t.Generator().manual_seed(seed)
    if init_traj is None:
        ref_traj = t.zeros(T + 1)
        ref_traj[0] = X0_VAR ** 0.5 * t.randn(1, generator=g).item()
        for tt in range(1, T + 1):
            ref_traj[tt] = f(ref_traj[tt - 1].item(), tt) + Q_VAR ** 0.5 * t.randn(1, generator=g).item()
    else:
        ref_traj = init_traj.clone()

    chain = [ref_traj.clone()]
    for _ in range(n_iters):
        ph, wh, anc = csmc_sweep(y, K, ref_traj, ancestor_sampling, g, T)
        ref_traj = sample_new_reference(ph, wh, anc, g, T)
        chain.append(ref_traj.clone())
    return t.stack(chain)


if __name__ == "__main__":
    from nonlinear_ssm_model import simulate, grid_filter_smoother, marginal_mean_var

    x_true, y = simulate()
    grid, _, gamma, _ = grid_filter_smoother(y)
    mean_s, _ = marginal_mean_var(grid, gamma)

    print("--- Particle Gibbs smoke test, K=200, 400 iterations, 100 burn-in ---")
    for label, ancestor_sampling in [("plain PG", False), ("PGAS", True)]:
        chain = run_particle_gibbs(y, n_iters=400, K=200, ancestor_sampling=ancestor_sampling, seed=0)
        post_burn = chain[100:]
        pg_mean = post_burn.mean(0)
        rmse = (pg_mean - mean_s).pow(2).mean().sqrt().item()
        print(f"{label:10s} RMSE vs grid-smoother mean: {rmse:.4f}")
