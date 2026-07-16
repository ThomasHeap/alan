"""
Hand-rolled (white-box) reimplementation of MP-IS for the toy hierarchical model,
matching alan's tensor-contraction structure exactly (K_mu x K_theta matrix,
theta's K-dim shared across the plate, contracted inner-plate-first), so we have
full access to intermediate sums for a delete-one jackknife bias correction on
the mu K-dim.

r(k_mu, k_theta) = A[k_mu] * B[k_theta] * C[k_theta, k_mu]
  A = p(mu)/q(mu),  B = p(x|theta)/q(theta),  C = p(theta|mu)
"""
import torch as t
from compare_model import P_SIZE, S0, TAU, SIGMA, x_data, analytic_posterior

mean, cov = analytic_posterior()
true_mu = mean[0].item()


def mp_is_mu_with_jackknife(K, seed):
    t.manual_seed(seed)
    mu_p = t.randn(K)                       # Q(mu) = N(0,1)
    theta_p = t.randn(K, P_SIZE)             # Q(theta_i) = N(0,1), shared K-dim across plate

    logA = (-0.5 * (mu_p / S0) ** 2) - (-0.5 * mu_p ** 2)          # log p(mu) - log q(mu)
    logB = (-0.5 * ((x_data[None, :] - theta_p) / SIGMA) ** 2).sum(-1) \
           - (-0.5 * (theta_p ** 2).sum(-1))                       # log p(x|theta) - log q(theta)
    # log p(theta_j | mu_k), summed over the plate -> [K_theta, K_mu]
    logC = (-0.5 * ((theta_p[:, None, :] - mu_p[None, :, None]) / TAU) ** 2).sum(-1)

    log_r = logA[None, :] + logB[:, None] + logC        # [K_theta, K_mu]
    log_r = log_r - log_r.max()
    r = log_r.exp()                                      # [K_theta, K_mu]

    col_sum = r.sum(0)                                   # sum over theta -> [K_mu], matches inner-plate-first fold
    num_total = (mu_p * col_sum).sum()
    den_total = col_sum.sum()
    mu_hat = (num_total / den_total).item()

    # Leave-one-mu-particle-out, vectorized: total minus each term
    num_loo = num_total - mu_p * col_sum                 # [K_mu]
    den_loo = den_total - col_sum                         # [K_mu]
    mu_hat_loo = num_loo / den_loo                        # [K_mu], K-1 particle estimates

    mu_hat_jack = K * mu_hat - (K - 1) * mu_hat_loo.mean().item()
    return mu_hat, mu_hat_jack


if __name__ == "__main__":
    # cross-check against alan's own MP-IS output at a fixed seed, same model/data
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
    import warnings
    warnings.filterwarnings("ignore")
    from alan_compare import build_problem, mp_is_moments

    prob = build_problem()
    for K in [5, 50]:
        mu_alan, _ = mp_is_moments(prob, K, seed=42)
        mu_mine, _ = mp_is_mu_with_jackknife(K, seed=42)
        print(f"K={K}: alan MP-IS mu_hat={mu_alan:.6f}  hand-rolled mu_hat={mu_mine:.6f}  (should match closely)")

    print()
    Ks = [3, 10, 30, 100, 300, 1000, 3000]
    N_REPEATS = 60
    for K in Ks:
        plain_errs, jack_errs = [], []
        for rep in range(N_REPEATS):
            mu_hat, mu_jack = mp_is_mu_with_jackknife(K, seed=2000 * K + rep)
            plain_errs.append(mu_hat - true_mu)
            jack_errs.append(mu_jack - true_mu)
        plain_errs = t.tensor(plain_errs)
        jack_errs = t.tensor(jack_errs)
        print(f"K={K:5d}  plain bias={plain_errs.mean().item():+.4f} (se {plain_errs.std().item()/N_REPEATS**0.5:.4f})"
              f"   jackknife bias={jack_errs.mean().item():+.4f} (se {jack_errs.std().item()/N_REPEATS**0.5:.4f})"
              f"   |plain RMSE|={plain_errs.pow(2).mean().sqrt().item():.4f}"
              f"   |jack RMSE|={jack_errs.pow(2).mean().sqrt().item():.4f}")
