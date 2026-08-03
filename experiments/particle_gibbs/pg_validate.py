"""
Validate Particle Gibbs / PGAS against the RTS smoother ground truth, and
compare mixing speed between plain PG and PGAS -- the comparison the
mp_is_bias ESS findings predicted would favor PGAS (see docs/pmcmc_design.md's
"Empirical findings" section: uneven per-replicate weight degeneracy in MP-IS
was flagged as a concrete reference-stickiness risk for plain Particle Gibbs).

Run: python pg_validate.py
"""
import torch as t
from ssm_model import T_STEPS, simulate, exact_posterior
from particle_gibbs import run_particle_gibbs
from bootstrap_pf import bootstrap_pf, unique_ancestors_at_t0


def autocorr(x, max_lag):
    x = x - x.mean()
    var = (x ** 2).mean().clamp_min(1e-12)   # near-frozen chain -> var~0; floor avoids nan, ESS still ~0
    n = len(x)
    return t.tensor([(x[:n - lag] * x[lag:]).mean() / var for lag in range(max_lag + 1)])


def chain_ess(x, max_lag=200):
    """Geyer's initial positive sequence estimator: n / (1 + 2*sum of paired ACF).
    A chain that never moves at all (var=0, e.g. plain PG stuck on one reference
    trajectory for the whole run) has ESS=1 by definition -- report that
    directly rather than letting the ACF's variance floor produce a spurious
    "well-mixed" estimate from float-rounding noise."""
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
    m_smooth, P_smooth, _ = exact_posterior(y)

    N_ITERS = 2000
    BURN_IN = 200

    print("=== Posterior-mean accuracy: plain PG vs PGAS vs a matched-K bootstrap PF, across K ===")
    for K in [5, 10, 30]:
        row = f"K={K:3d}  "
        for label, ancestor_sampling in [("PG", False), ("PGAS", True)]:
            chain = run_particle_gibbs(y, n_iters=N_ITERS, K=K, ancestor_sampling=ancestor_sampling, seed=42)
            post = chain[BURN_IN:]
            rmse = (post.mean(0) - m_smooth).pow(2).mean().sqrt().item()
            row += f"{label} RMSE={rmse:.4f}   "

        pf_rmses = []
        for rep in range(20):
            ph, anc, _ = bootstrap_pf(y, K, seed=900 * K + rep)
            from bootstrap_pf import reconstruct_trajectories
            full = reconstruct_trajectories(ph, anc)
            pf_rmses.append((full.mean(0) - m_smooth).pow(2).mean().sqrt().item())
        row += f"plain-PF RMSE={sum(pf_rmses)/len(pf_rmses):.4f} (avg of 20 fresh runs, no chain)"
        print(row)

    print("\n=== Mixing speed: chain ESS of x_0 (the point furthest from the observations,")
    print("    hence most exposed to reference-trajectory stickiness), plain PG vs PGAS ===")
    for K in [5, 10, 30]:
        pg_chain = run_particle_gibbs(y, n_iters=N_ITERS, K=K, ancestor_sampling=False, seed=1)
        pgas_chain = run_particle_gibbs(y, n_iters=N_ITERS, K=K, ancestor_sampling=True, seed=1)
        pg_ess = chain_ess(pg_chain[BURN_IN:, 0])
        pgas_ess = chain_ess(pgas_chain[BURN_IN:, 0])
        n = N_ITERS - BURN_IN
        print(f"K={K:3d}   plain PG: chain ESS(x_0) = {pg_ess:6.1f} / {n} ({100*pg_ess/n:.1f}%)"
              f"   |   PGAS: chain ESS(x_0) = {pgas_ess:6.1f} / {n} ({100*pgas_ess/n:.1f}%)"
              f"   ({pgas_ess/pg_ess:.1f}x)")

    print("\n=== Path degeneracy in a single bootstrap-PF sweep (motivating CSMC) vs CSMC's guarantee ===")
    for K in [10, 30]:
        ph, anc, _ = bootstrap_pf(y, K, seed=0)
        n_unique = unique_ancestors_at_t0(ph, anc)
        print(f"K={K:3d}  plain bootstrap PF: {n_unique}/{K} unique ancestors survive to t=0 in one sweep"
              f"   |   CSMC: exactly 1 trajectory (the reference) is GUARANTEED to survive, by construction,"
              f" every sweep")
