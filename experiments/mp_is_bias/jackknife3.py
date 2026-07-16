"""The mathematically correct delete-one jackknife: leave out ONE (k_mu, k_theta)
PAIR at a time (K^2 leave-one-out terms, matching the classical delete-1 theory,
which is defined over the flat sum of M=K^2 combinatorial terms) rather than a
whole row/column (which deletes K terms at once -- a "delete-d" jackknife that
needs a different formula, not the standard one)."""
import torch as t
from compare_model import P_SIZE, S0, TAU, SIGMA, x_data, analytic_posterior

mean, cov = analytic_posterior()
true_mu = mean[0].item()


def mp_is_mu_pairwise_jackknife(K, seed):
    t.manual_seed(seed)
    mu_p = t.randn(K)
    theta_p = t.randn(K, P_SIZE)

    logA = (-0.5 * (mu_p / S0) ** 2) - (-0.5 * mu_p ** 2)
    logB = (-0.5 * ((x_data[None, :] - theta_p) / SIGMA) ** 2).sum(-1) - (-0.5 * (theta_p ** 2).sum(-1))
    logC = (-0.5 * ((theta_p[:, None, :] - mu_p[None, :, None]) / TAU) ** 2).sum(-1)

    log_r = logA[None, :] + logB[:, None] + logC
    log_r = log_r - log_r.max()
    r = log_r.exp()                       # [K_theta, K_mu]

    mu_grid = mu_p[None, :].expand_as(r)  # [K_theta, K_mu]
    num_total = (mu_grid * r).sum()
    den_total = r.sum()
    mu_hat = (num_total / den_total).item()

    M = K * K
    num_loo = num_total - mu_grid * r     # [K_theta, K_mu], one term removed each
    den_loo = den_total - r
    mu_hat_loo = num_loo / den_loo

    jack = M * mu_hat - (M - 1) * mu_hat_loo.mean().item()
    return mu_hat, jack


if __name__ == "__main__":
    Ks = [3, 10, 30, 100, 300, 1000]
    N_REPEATS = 60
    for K in Ks:
        plain, jack = [], []
        for rep in range(N_REPEATS):
            p, j = mp_is_mu_pairwise_jackknife(K, seed=7000 * K + rep)
            plain.append(p - true_mu); jack.append(j - true_mu)
        plain, jack = t.tensor(plain), t.tensor(jack)
        print(f"K={K:5d}  plain bias={plain.mean():+.4f} RMSE={plain.pow(2).mean().sqrt():.4f}"
              f"   |  pairwise-jack bias={jack.mean():+.4f} RMSE={jack.pow(2).mean().sqrt():.4f}")
