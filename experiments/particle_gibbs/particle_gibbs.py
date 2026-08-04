"""
Particle Gibbs (conditional SMC) on the toy SSM, matching the design-doc
checklist item: "extend sampling so that, given a reference trajectory, the
other K-1 particles are drawn as usual but the reference survives every
resampling step unchanged." Standard construction (Andrieu, Doucet &
Holenstein 2010, Algorithm 2), plus the optional ancestor-sampling variant
(PGAS, Lindsten, Jordan & Schön 2014) flagged in the design doc as the
higher-priority item once the mp_is_bias ESS findings showed real cross-plate
weight-degeneracy spread.

Particle slot 0 is always reserved for the reference trajectory at every time
step -- it is never resampled away, guaranteeing (by the standard CSMC
argument) that the chain of reference trajectories produced by repeatedly
running csmc_sweep + sample_new_reference has the true smoothing posterior
p(x_0:T | y_1:T) as its stationary distribution, exactly, regardless of K.

Plain CSMC always re-attaches the reference's future to its OWN historical
lineage (ancestor index 0 at every step) -- this is what causes the classic
slow-mixing "reference stickiness" failure mode. PGAS instead resamples which
t-1 particle the reference's future gets attached to, weighted by how well
that particle's transition would have predicted the pinned x*_t value, letting
the reference borrow a different, better-fitting past each sweep.
"""
import torch as t
from ssm_model import A, Q_VAR, R_VAR, P0, T_STEPS


def csmc_sweep(y, K, ref_traj, ancestor_sampling, g, T=T_STEPS):
    """One conditional-SMC sweep given a pinned reference trajectory. Returns
    (particle_history, weight_history, ancestry) covering t=0..T."""
    x = t.empty(K)
    x[0] = ref_traj[0]
    x[1:] = P0 ** 0.5 * t.randn(K - 1, generator=g)
    particle_history = [x.clone()]
    w = t.full((K,), 1.0 / K)
    weight_history = [w.clone()]
    ancestry = []

    for tt in range(1, T + 1):
        anc = t.zeros(K, dtype=t.long)
        anc[1:] = t.multinomial(w, K - 1, replacement=True, generator=g)

        x_new = t.zeros(K)
        x_new[1:] = A * x[anc[1:]] + Q_VAR ** 0.5 * t.randn(K - 1, generator=g)
        x_new[0] = ref_traj[tt]

        if ancestor_sampling:
            logw_as = t.log(w + 1e-300) - 0.5 * (ref_traj[tt] - A * x) ** 2 / Q_VAR
            logw_as = logw_as - logw_as.max()
            w_as = logw_as.exp()
            w_as = w_as / w_as.sum()
            anc[0] = t.multinomial(w_as, 1, generator=g).item()
        else:
            anc[0] = 0
        ancestry.append(anc.clone())

        logw = -0.5 * (y[tt - 1] - x_new) ** 2 / R_VAR
        logw = logw - logw.max()
        w_new = logw.exp()
        w_new = w_new / w_new.sum()

        particle_history.append(x_new.clone())
        weight_history.append(w_new.clone())
        x, w = x_new, w_new

    return particle_history, weight_history, ancestry


def sample_new_reference(particle_history, weight_history, ancestry, g, T=T_STEPS):
    """Draw a new full reference trajectory: pick a final-time index ~ its
    weight, then trace its ancestry back through the sweep."""
    j = t.multinomial(weight_history[T], 1, generator=g).item()
    new_traj = t.zeros(T + 1)
    idx = j
    new_traj[T] = particle_history[T][idx]
    for tt in range(T, 0, -1):
        idx = ancestry[tt - 1][idx].item()
        new_traj[tt - 1] = particle_history[tt - 1][idx]
    return new_traj


def run_particle_gibbs(y, n_iters, K, ancestor_sampling=False, T=T_STEPS, seed=0, init_traj=None):
    """Chain driver. Returns [n_iters+1, T+1] tensor of reference trajectories
    (row 0 is the initial trajectory)."""
    g = t.Generator().manual_seed(seed)
    if init_traj is None:
        ref_traj = t.zeros(T + 1)
        ref_traj[0] = P0 ** 0.5 * t.randn(1, generator=g).item()
        for tt in range(1, T + 1):
            ref_traj[tt] = A * ref_traj[tt - 1].item() + Q_VAR ** 0.5 * t.randn(1, generator=g).item()
    else:
        ref_traj = init_traj.clone()

    chain = [ref_traj.clone()]
    for _ in range(n_iters):
        ph, wh, anc = csmc_sweep(y, K, ref_traj, ancestor_sampling, g, T)
        ref_traj = sample_new_reference(ph, wh, anc, g, T)
        chain.append(ref_traj.clone())
    return t.stack(chain)


if __name__ == "__main__":
    from ssm_model import simulate, exact_posterior

    x_true, y = simulate()
    m_smooth, P_smooth, _ = exact_posterior(y)

    print("--- Particle Gibbs smoke test, K=10, 200 iterations, 50 burn-in ---")
    for label, ancestor_sampling in [("plain PG", False), ("PGAS", True)]:
        chain = run_particle_gibbs(y, n_iters=200, K=10, ancestor_sampling=ancestor_sampling, seed=0)
        post_burn = chain[50:]
        pg_mean = post_burn.mean(0)
        rmse = (pg_mean - m_smooth).pow(2).mean().sqrt().item()
        print(f"{label:10s} RMSE vs RTS smoother mean: {rmse:.4f}")
