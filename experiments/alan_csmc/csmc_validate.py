"""
Validate alan-native CSMC (alan_csmc.py) against the RTS-smoother ground
truth, directly comparable to experiments/particle_gibbs/pg_validate.py's
already-documented hand-rolled plain-PG and PGAS numbers on the SAME model
and parameters -- the point is to test the hypothesis that alan's dense (no
intermediate resampling) Timeseries machinery doesn't need PGAS's
ancestor-sampling fix to mix well, since it never prunes anything mid-
sequence in the first place.

Run: python csmc_validate.py
"""
import warnings
warnings.filterwarnings("ignore")

import torch as t
from csmc_model import T_STEPS, simulate, exact_posterior
from alan_csmc import run_alan_csmc


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
    m_smooth, _ = exact_posterior(y)

    N_ITERS, BURN_IN = 2000, 200
    n = N_ITERS - BURN_IN

    print("=== alan-native CSMC: posterior-mean accuracy + mixing speed vs K ===")
    print(f"(compare directly to experiments/particle_gibbs/README.md's plain-PG/PGAS tables")
    print(f" on the same model: K=5 PG=0.1843/PGAS=0.0164, K=10 PG=0.0467/PGAS=0.0141, K=30 PG=0.0304/PGAS=0.0114)\n")

    for K in [5, 10, 30]:
        chain = run_alan_csmc(y, n_iters=N_ITERS, K=K, burn_in=BURN_IN, seed=42)
        rmse = (chain.mean(0) - m_smooth[1:]).pow(2).mean().sqrt().item()
        ess_x1 = chain_ess(chain[:, 0])  # ts's first timestep -- furthest from the end of the chain's reasoning
        print(f"K={K:3d}   RMSE={rmse:.4f}   chain ESS(ts[t=1])={ess_x1:6.1f}/{n} ({100 * ess_x1 / n:.1f}%)")
