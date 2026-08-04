"""
Jackknife bias correction, redone against the CORRECT (Order B) reduction:
each plate element independently logsumexp's over its own K_theta particles,
and those per-element results get summed across the plate afterward -- not
the other way round. See diagnose_handroll_mismatch.py on
explore/mp-is-bias-empirical for how this was found.

Two separate jackknifes, since K_mu and K_theta now enter the computation
structurally differently:

  - jack(mu): K_mu is only ever combined via a LINEAR sum (the final
    ratio-of-sums for E[mu]), so leave-one-out is the simple "total minus one
    term" trick, same as before.

  - jack(theta): K_theta is reduced via a LOGSUMEXP, independently per plate
    element, before anything is summed. Leave-one-out here means removing one
    term from *each* of the 6 per-element logsumexps and re-propagating --
    done via the exact identity logsumexp_{-j}(v) = log(exp(LSE(v)) - exp(v_j)),
    vectorized over all K choices of j at once.
"""
import sys
sys.path.insert(0, "/home/user/alan/src")
import warnings
warnings.filterwarnings("ignore")

import math
import torch as t
from compare_model import P_SIZE, S0, TAU, SIGMA, x_data, analytic_posterior

mean, cov = analytic_posterior()
true_mu = mean[0].item()


def compute_log_r_mu(K, seed):
    """Order-B log_r_mu[kmu] = logA[kmu] + sum_i logsumexp_ktheta(theta_term[i,ktheta,kmu] + x_term[i,ktheta])"""
    t.manual_seed(seed)
    mu_p = t.randn(K)
    theta_p = t.randn(K, P_SIZE)   # [K_theta, P_SIZE]

    logA = (-0.5 * (mu_p / S0) ** 2) - (-0.5 * mu_p ** 2) - math.log(K)   # [K_mu]

    # theta_term[i, ktheta, kmu] = log p(theta_i^ktheta | mu^kmu) - log q(theta_i^ktheta) - log(K)
    logp_theta_given_mu = -0.5 * ((theta_p[None, :, :] - mu_p[:, None, None]) / TAU) ** 2  # [K_mu, K_theta, P]
    logp_theta_given_mu = logp_theta_given_mu.permute(2, 1, 0)                              # [P, K_theta, K_mu]
    logq_theta = (-0.5 * theta_p ** 2).T[:, :, None]   # [P, K_theta, 1]
    theta_term = logp_theta_given_mu - logq_theta - math.log(K)   # [P, K_theta, K_mu]

    # x_term[i, ktheta] = log p(x_i | theta_i^ktheta)
    x_term = (-0.5 * ((x_data[:, None] - theta_p.T) / SIGMA) ** 2)   # [P, K_theta]

    combined = theta_term + x_term[:, :, None]     # [P, K_theta, K_mu]
    per_elem_reduced = t.logsumexp(combined, dim=1)  # reduce K_theta per plate element -> [P, K_mu]
    p1_summed = per_elem_reduced.sum(0)              # sum across the plate -> [K_mu]

    log_r_mu = logA + p1_summed
    return log_r_mu, mu_p, combined  # combined kept for the theta-jackknife


def mu_hat_from_log_r(log_r_mu, mu_p):
    m = log_r_mu.max()
    w = (log_r_mu - m).exp()
    return (mu_p * w).sum() / w.sum()


def jackknife_mu(log_r_mu, mu_p, K):
    """Leave-one-mu-particle-out: simple linear 'total minus one term' trick."""
    m = log_r_mu.max()
    w = (log_r_mu - m).exp()
    num_total, den_total = (mu_p * w).sum(), w.sum()
    num_loo = num_total - mu_p * w
    den_loo = den_total - w
    loo_estimates = num_loo / den_loo
    plain = (num_total / den_total).item()
    jack = K * plain - (K - 1) * loo_estimates.mean().item()
    return plain, jack


def jackknife_theta(combined, logA, mu_p, K):
    """Leave-one-theta-particle-out, exact via the LSE leave-one-out identity,
    vectorized over all K choices of the excluded particle at once."""
    per_elem_reduced = t.logsumexp(combined, dim=1)               # [P, K_mu]
    exp_full = per_elem_reduced.exp()                              # [P, K_mu]
    exp_term = combined.exp()                                      # [P, K_theta, K_mu]
    # leave out ktheta=j (the K_theta axis IS the "j" index here)
    exp_loo = exp_full[:, None, :] - exp_term                      # [P, K_theta(=j), K_mu]
    exp_loo = exp_loo.clamp_min(1e-300)                            # numerical floor
    lse_loo = exp_loo.log()                                        # [P, j, K_mu]
    p1_summed_loo = lse_loo.sum(0)                                 # sum over plate -> [j, K_mu]

    log_r_mu_loo = logA[None, :] + p1_summed_loo                   # [j, K_mu]
    m = log_r_mu_loo.max(dim=1, keepdim=True).values
    w_loo = (log_r_mu_loo - m).exp()                                # [j, K_mu]
    loo_estimates = (mu_p[None, :] * w_loo).sum(1) / w_loo.sum(1)  # [j]

    plain = mu_hat_from_log_r(logA + per_elem_reduced.sum(0), mu_p).item()
    jack = K * plain - (K - 1) * loo_estimates.mean().item()
    return plain, jack


if __name__ == "__main__":
    # Note: don't expect this script's output to numerically match alan's under a *shared*
    # seed -- it draws mu_p/theta_p directly via t.randn(), not through alan's Q._sample()
    # pipeline, which consumes RNG state differently (see diagnose_handroll_mismatch.py).
    # The formula itself was validated there, exactly, against alan's real per-component
    # log-probs; see cross_check_orderB.py for a statistical (unmatched-seed, many-repeat)
    # validation of *this* script's distribution against real alan.

    Ks = [3, 10, 30, 100, 300, 1000]
    N_REPEATS = 60
    for K in Ks:
        plain_e, jm_e, jt_e = [], [], []
        for rep in range(N_REPEATS):
            seed = 8000 * K + rep
            log_r_mu, mu_p, combined = compute_log_r_mu(K, seed)
            logA = (-0.5 * (mu_p / S0) ** 2) - (-0.5 * mu_p ** 2) - math.log(K)

            plain, jm = jackknife_mu(log_r_mu, mu_p, K)
            _, jt = jackknife_theta(combined, logA, mu_p, K)

            plain_e.append(plain - true_mu)
            jm_e.append(jm - true_mu)
            jt_e.append(jt - true_mu)

        plain_e, jm_e, jt_e = t.tensor(plain_e), t.tensor(jm_e), t.tensor(jt_e)
        print(f"K={K:5d}  plain bias={plain_e.mean():+.4f} RMSE={plain_e.pow(2).mean().sqrt():.4f}"
              f"   |  jack(mu) bias={jm_e.mean():+.4f} RMSE={jm_e.pow(2).mean().sqrt():.4f}"
              f"   |  jack(theta) bias={jt_e.mean():+.4f} RMSE={jt_e.pow(2).mean().sqrt():.4f}")
