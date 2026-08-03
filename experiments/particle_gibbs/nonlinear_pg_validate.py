"""
Validate Particle Gibbs / PGAS against the grid-based ground truth on the
nonlinear/non-Gaussian growth model, and compare mixing speed between plain
PG and PGAS -- same comparisons as pg_validate.py, plus a bimodality check
that the linear-Gaussian model can't exercise: the grid ground truth shows
x_0 and x_1's marginals are genuinely bimodal (~88% mass on one side, ~12%
on the other -- see nonlinear_ssm_model.py's docstring), so a chain that
gets stuck on one mode's reference trajectory will visibly under/over-shoot
P(x_t > 0) even if its posterior *mean* happens to look reasonable.

Run: python nonlinear_pg_validate.py
"""
import torch as t
from nonlinear_ssm_model import T_STEPS, simulate, grid_filter_smoother, marginal_mean_var
from nonlinear_particle_gibbs import run_particle_gibbs
from nonlinear_bootstrap_pf import bootstrap_pf, unique_ancestors_at_t0, reconstruct_trajectories

BIMODAL_TS = [0, 1]  # picked by inspection of the grid marginals: P(x_t>0) ~ 0.88 at both


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
    p_pos_grid = {tt: gamma[tt][grid > 0].sum().item() for tt in BIMODAL_TS}

    N_ITERS = 2000
    BURN_IN = 200
    Ks = [50, 200, 500]

    print("=== Posterior-mean accuracy: plain PG vs PGAS vs a matched-K bootstrap PF, across K ===")
    for K in Ks:
        row = f"K={K:3d}  "
        chains = {}
        for label, ancestor_sampling in [("PG", False), ("PGAS", True)]:
            chain = run_particle_gibbs(y, n_iters=N_ITERS, K=K, ancestor_sampling=ancestor_sampling, seed=42)
            chains[label] = chain
            post = chain[BURN_IN:]
            rmse = (post.mean(0) - mean_s).pow(2).mean().sqrt().item()
            row += f"{label} RMSE={rmse:.4f}   "

        pf_rmses = []
        for rep in range(10):
            ph, anc, _ = bootstrap_pf(y, K, seed=900 * K + rep)
            full = reconstruct_trajectories(ph, anc)
            pf_rmses.append((full.mean(0) - mean_s).pow(2).mean().sqrt().item())
        row += f"plain-PF RMSE={sum(pf_rmses) / len(pf_rmses):.4f} (avg of 10 fresh runs, no chain)"
        print(row)

    print(f"\n=== Bimodality check: P(x_t > 0) at t in {BIMODAL_TS}, grid ground truth vs PG/PGAS chains, K=200 ===")
    K = 200
    for label, ancestor_sampling in [("plain PG", False), ("PGAS", True)]:
        chain = run_particle_gibbs(y, n_iters=N_ITERS, K=K, ancestor_sampling=ancestor_sampling, seed=7)
        post = chain[BURN_IN:]
        for tt in BIMODAL_TS:
            p_chain = (post[:, tt] > 0).float().mean().item()
            print(f"{label:10s} t={tt}: grid P(x_t>0)={p_pos_grid[tt]:.3f}   "
                  f"chain P(x_t>0)={p_chain:.3f}   |diff|={abs(p_pos_grid[tt] - p_chain):.3f}")

    print("\n=== Mixing speed: chain ESS of x_0 (bimodal AND furthest from the observations,")
    print("    so most exposed to both reference-trajectory stickiness and mode-switching failure) ===")
    for K in Ks:
        pg_chain = run_particle_gibbs(y, n_iters=N_ITERS, K=K, ancestor_sampling=False, seed=1)
        pgas_chain = run_particle_gibbs(y, n_iters=N_ITERS, K=K, ancestor_sampling=True, seed=1)
        pg_ess = chain_ess(pg_chain[BURN_IN:, 0])
        pgas_ess = chain_ess(pgas_chain[BURN_IN:, 0])
        n = N_ITERS - BURN_IN
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
