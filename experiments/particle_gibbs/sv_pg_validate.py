"""
Validate Particle Gibbs / PGAS against the grid-based ground truth on the
stochastic volatility model, and compare mixing speed between plain PG and
PGAS -- same comparisons as nonlinear_pg_validate.py, but here the stress
test is PHI=0.98 persistence over a long horizon (T=100) rather than
multimodality, which is the classic PMCMC-mixing failure mode this model is
famous for in the literature.

Run: python sv_pg_validate.py
"""
import torch as t
from sv_model import T_STEPS, simulate, grid_filter_smoother, marginal_mean_var
from sv_particle_gibbs import run_particle_gibbs
from sv_bootstrap_pf import bootstrap_pf, unique_ancestors_at_t0, reconstruct_trajectories


def autocorr(x, max_lag):
    x = x - x.mean()
    var = (x ** 2).mean().clamp_min(1e-12)
    n = len(x)
    return t.tensor([(x[:n - lag] * x[lag:]).mean() / var for lag in range(max_lag + 1)])


def chain_ess(x, max_lag=200):
    if x.var().item() < 1e-10:
        return 1.0
    max_lag = min(max_lag, len(x) // 2 - 2)
    acf = autocorr(x, max_lag)
    s, lag = 1.0, 1
    while lag + 1 <= max_lag:
        pair = (acf[lag] + acf[lag + 1]).item()
        if pair < 0:
            break
        s += 2 * pair
        lag += 2
    return len(x) / max(s, 1e-6)


if __name__ == "__main__":
    x_true, y = simulate()
    grid, _, gamma, _ = grid_filter_smoother(y)
    mean_s, _ = marginal_mean_var(grid, gamma)

    N_ITERS = 1200
    BURN_IN = 200
    Ks = [50, 200, 400]
    n = N_ITERS - BURN_IN

    # Run each (K, label) chain once and reuse it for both the accuracy and
    # mixing-speed comparisons below (they're different statistics of the
    # same chain, not different experiments -- no need to pay for two runs).
    chains = {}
    for K in Ks:
        for label, ancestor_sampling in [("PG", False), ("PGAS", True)]:
            chains[K, label] = run_particle_gibbs(y, n_iters=N_ITERS, K=K, ancestor_sampling=ancestor_sampling, seed=42)

    print("=== Posterior-mean accuracy: plain PG vs PGAS vs a matched-K bootstrap PF, across K ===")
    for K in Ks:
        row = f"K={K:3d}  "
        for label in ["PG", "PGAS"]:
            post = chains[K, label][BURN_IN:]
            rmse = (post.mean(0) - mean_s).pow(2).mean().sqrt().item()
            row += f"{label} RMSE={rmse:.4f}   "

        pf_rmses = []
        for rep in range(10):
            ph, anc, _ = bootstrap_pf(y, K, seed=900 * K + rep)
            full = reconstruct_trajectories(ph, anc)
            pf_rmses.append((full.mean(0) - mean_s).pow(2).mean().sqrt().item())
        row += f"plain-PF RMSE={sum(pf_rmses) / len(pf_rmses):.4f} (avg of 10 fresh runs, no chain)"
        print(row)

    print("\n=== Mixing speed: chain ESS of x_0 (the point furthest from any observation,")
    print("    so most exposed to persistence-driven reference-trajectory stickiness) ===")
    for K in Ks:
        pg_ess = chain_ess(chains[K, "PG"][BURN_IN:, 0])
        pgas_ess = chain_ess(chains[K, "PGAS"][BURN_IN:, 0])
        print(f"K={K:3d}   plain PG: chain ESS(x_0) = {pg_ess:6.1f} / {n} ({100 * pg_ess / n:.1f}%)"
              f"   |   PGAS: chain ESS(x_0) = {pgas_ess:6.1f} / {n} ({100 * pgas_ess / n:.1f}%)"
              f"   ({pgas_ess / max(pg_ess, 1e-6):.1f}x)")

    print("\n=== Path degeneracy in a single bootstrap-PF sweep vs CSMC's guarantee ===")
    for K in [50, 200]:
        ph, anc, _ = bootstrap_pf(y, K, seed=0)
        n_unique = unique_ancestors_at_t0(ph, anc)
        print(f"K={K:3d}  plain bootstrap PF: {n_unique}/{K} unique ancestors survive to t=0 in one sweep"
              f"   |   CSMC: exactly 1 trajectory (the reference) is GUARANTEED to survive, by construction,"
              f" every sweep")
