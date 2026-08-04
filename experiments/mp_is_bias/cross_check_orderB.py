"""
Statistical (unmatched-seed, many-repeat) validation of jackknife_orderB.py's
plain estimator against real alan, complementing the exact-match validation
in diagnose_handroll_mismatch.py (which used alan's own drawn particles).
Compares distributions, not individual draws -- seed-matching across the two
codebases was never meaningful (see diagnose_handroll_mismatch.py).
"""
import sys
sys.path.insert(0, "/home/user/alan/src")
import warnings
warnings.filterwarnings("ignore")

import torch as t
from alan_compare import build_problem, mp_is_moments
from jackknife_orderB import compute_log_r_mu, mu_hat_from_log_r
from compare_model import analytic_posterior

if __name__ == "__main__":
    true_mu = analytic_posterior()[0][0].item()
    prob = build_problem()

    for K in [50, 200]:
        N = 300
        alan_vals, mine_vals = [], []
        for rep in range(N):
            mu_a, _ = mp_is_moments(prob, K, seed=20000 + rep)
            log_r_mu, mu_p, _ = compute_log_r_mu(K, seed=70000 + rep)
            mine_vals.append(mu_hat_from_log_r(log_r_mu, mu_p).item())
            alan_vals.append(mu_a)
        alan_vals, mine_vals = t.tensor(alan_vals), t.tensor(mine_vals)
        print(f"K={K}")
        print(f"  alan (real):    mean={alan_vals.mean():.4f} (se {alan_vals.std()/N**0.5:.4f})  std={alan_vals.std():.4f}")
        print(f"  mine (order B): mean={mine_vals.mean():.4f} (se {mine_vals.std()/N**0.5:.4f})  std={mine_vals.std():.4f}")
        print(f"  true_mu={true_mu:.4f}")
