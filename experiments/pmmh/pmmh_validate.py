"""
Validate PMMH (over mu, driven by alan's own P_hat_MP) against the closed-
form posterior, across K -- the core claim of pseudo-marginal MCMC is that
the chain is asymptotically EXACT for *any* K >= 1 (unlike plug-in SNIS,
which has O(1/K) bias for finite K): smaller K should only cost mixing
speed, never correctness. Also demonstrates the "must freeze the accepted
log-likelihood" discipline empirically, by comparing against the same chain
with that discipline turned off (freeze=False).

Run: python pmmh_validate.py
"""
import warnings
warnings.filterwarnings("ignore")

import torch as t
from pmmh_model import simulate, exact_posterior
from alan_pmmh import run_pmmh


def autocorr(x, max_lag):
    x = x - x.mean()
    var = (x ** 2).mean().clamp_min(1e-12)
    n = len(x)
    return t.tensor([(x[:n - lag] * x[lag:]).mean() / var for lag in range(max_lag + 1)])


def chain_ess(x, max_lag=100):
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
    x = simulate()
    post_mean, post_var = exact_posterior(x)
    print(f"exact posterior: mu | x ~ N({post_mean:.4f}, {post_var:.4f})  (sd={post_var ** 0.5:.4f})\n")

    N_ITERS, BURN_IN, N_REPS = 1500, 300, 5
    n = N_ITERS - BURN_IN

    print("=== Correctness vs K (freeze=True, the correct algorithm): should be unbiased at EVERY K ===")
    for K in [5, 20, 100]:
        means, variances, accs, esss = [], [], [], []
        for rep in range(N_REPS):
            chain, acc = run_pmmh(x, n_iters=N_ITERS, K=K, proposal_scale=0.3, burn_in=BURN_IN, seed=100 * K + rep)
            means.append(chain.mean().item())
            variances.append(chain.var().item())
            accs.append(acc)
            esss.append(chain_ess(chain))
        means = t.tensor(means)
        print(f"K={K:4d}  chain mean={means.mean():.4f} (bias {means.mean() - post_mean:+.4f}, "
              f"se {means.std() / N_REPS ** 0.5:.4f})   chain var={sum(variances) / len(variances):.4f} "
              f"(true {post_var:.4f})   accept_rate={sum(accs) / len(accs):.2f}   "
              f"chain ESS={sum(esss) / len(esss):.1f}/{n}")

    print("\n=== The freeze discipline: freeze=True (correct) vs freeze=False (classic pseudo-marginal bug) ===")
    for K in [5, 20]:
        for freeze in [True, False]:
            means = []
            for rep in range(N_REPS):
                chain, _ = run_pmmh(x, n_iters=N_ITERS, K=K, proposal_scale=0.3, burn_in=BURN_IN,
                                     seed=900 * K + rep, freeze=freeze)
                means.append(chain.mean().item())
            means = t.tensor(means)
            label = "freeze=True (correct)" if freeze else "freeze=False (BUG)"
            print(f"K={K:3d}  {label:22s}  chain mean={means.mean():.4f} "
                  f"(bias {means.mean() - post_mean:+.4f}, se {means.std() / N_REPS ** 0.5:.4f})")
