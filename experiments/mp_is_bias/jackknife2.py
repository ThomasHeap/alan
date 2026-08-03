"""Try jackknifing over theta's K-dim instead of mu's, since theta is 6-dimensional
collectively (via the plate) and likely dominates the self-normalizing denominator's
variance/bias far more than the scalar mu does."""
import torch as t
from compare_model import P_SIZE, S0, TAU, SIGMA, x_data, analytic_posterior

mean, cov = analytic_posterior()
true_mu = mean[0].item()


def mp_is_mu_both_jackknifes(K, seed):
    t.manual_seed(seed)
    mu_p = t.randn(K)
    theta_p = t.randn(K, P_SIZE)

    logA = (-0.5 * (mu_p / S0) ** 2) - (-0.5 * mu_p ** 2)
    logB = (-0.5 * ((x_data[None, :] - theta_p) / SIGMA) ** 2).sum(-1) - (-0.5 * (theta_p ** 2).sum(-1))
    logC = (-0.5 * ((theta_p[:, None, :] - mu_p[None, :, None]) / TAU) ** 2).sum(-1)

    log_r = logA[None, :] + logB[:, None] + logC
    log_r = log_r - log_r.max()
    r = log_r.exp()                      # [K_theta, K_mu]

    col_sum = r.sum(0)                   # [K_mu]  (sum over theta)
    row_sum = r.sum(1)                   # [K_theta] (sum over mu)
    rowmu_sum = (mu_p[None, :] * r).sum(1)   # [K_theta]: sum_kmu mu_p[kmu]*r[ktheta,kmu]

    num_total = (mu_p * col_sum).sum()
    den_total = col_sum.sum()
    mu_hat = (num_total / den_total).item()

    # jackknife over mu's K-dim
    num_loo_mu = num_total - mu_p * col_sum
    den_loo_mu = den_total - col_sum
    jack_mu = K * mu_hat - (K - 1) * (num_loo_mu / den_loo_mu).mean().item()

    # jackknife over theta's K-dim
    num_loo_th = num_total - rowmu_sum
    den_loo_th = den_total - row_sum
    jack_theta = K * mu_hat - (K - 1) * (num_loo_th / den_loo_th).mean().item()

    return mu_hat, jack_mu, jack_theta


if __name__ == "__main__":
    Ks = [3, 10, 30, 100, 300, 1000, 3000]
    N_REPEATS = 60
    for K in Ks:
        plain, jm, jt = [], [], []
        for rep in range(N_REPEATS):
            p, m, th = mp_is_mu_both_jackknifes(K, seed=5000 * K + rep)
            plain.append(p - true_mu); jm.append(m - true_mu); jt.append(th - true_mu)
        plain, jm, jt = t.tensor(plain), t.tensor(jm), t.tensor(jt)
        print(f"K={K:5d}  plain bias={plain.mean():+.4f} RMSE={plain.pow(2).mean().sqrt():.4f}"
              f"   |  jack(mu) bias={jm.mean():+.4f} RMSE={jm.pow(2).mean().sqrt():.4f}"
              f"   |  jack(theta) bias={jt.mean():+.4f} RMSE={jt.pow(2).mean().sqrt():.4f}")
