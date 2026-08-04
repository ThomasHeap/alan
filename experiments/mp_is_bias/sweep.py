import sys, time, json
sys.path.insert(0, "/home/user/alan/src")
import warnings
warnings.filterwarnings("ignore")

import torch as t
from alan_compare import build_problem, mp_is_moments, global_is_moments
from compare_model import analytic_posterior
from hmc import hmc

t.set_grad_enabled(True)

mean, cov = analytic_posterior()
true_mu = mean[0].item()
true_theta = mean[1:]

Ks = [3, 10, 30, 100, 300, 1000, 3000]
N_REPEATS = 40

prob = build_problem()

results = {"true_mu": true_mu, "true_theta": true_theta.tolist(), "Ks": Ks, "mp": {}, "global": {}}

t0 = time.time()
for K in Ks:
    mp_mu_errs, mp_theta_errs = [], []
    gl_mu_errs, gl_theta_errs = [], []
    for rep in range(N_REPEATS):
        seed = 1000 * K + rep
        mu_hat, theta_hat = mp_is_moments(prob, K, seed=seed)
        mp_mu_errs.append(mu_hat - true_mu)
        mp_theta_errs.append((theta_hat - true_theta).abs().mean().item())

        mu_hat_g, theta_hat_g = global_is_moments(prob, K, seed=seed)
        gl_mu_errs.append(mu_hat_g - true_mu)
        gl_theta_errs.append((theta_hat_g - true_theta).abs().mean().item())

    mp_mu_errs = t.tensor(mp_mu_errs)
    gl_mu_errs = t.tensor(gl_mu_errs)
    results["mp"][K] = {
        "mean_bias_mu": mp_mu_errs.mean().item(),
        "se_mu": (mp_mu_errs.std() / (N_REPEATS ** 0.5)).item(),
        "mean_abs_theta_err": sum(mp_theta_errs) / len(mp_theta_errs),
    }
    results["global"][K] = {
        "mean_bias_mu": gl_mu_errs.mean().item(),
        "se_mu": (gl_mu_errs.std() / (N_REPEATS ** 0.5)).item(),
        "mean_abs_theta_err": sum(gl_theta_errs) / len(gl_theta_errs),
    }
    print(f"K={K:5d}  MP-IS bias(mu)={results['mp'][K]['mean_bias_mu']:+.4f} (se {results['mp'][K]['se_mu']:.4f})"
          f"  theta|err|={results['mp'][K]['mean_abs_theta_err']:.4f}"
          f"   |  Global bias(mu)={results['global'][K]['mean_bias_mu']:+.4f} (se {results['global'][K]['se_mu']:.4f})"
          f"  theta|err|={results['global'][K]['mean_abs_theta_err']:.4f}"
          f"   [{time.time()-t0:.1f}s]")

# HMC cross-check (independent ground truth already validated against analytic solution)
hmc_samples, acc = hmc(n_samples=20000, burn_in=2000)
hmc_mu = hmc_samples[:, 0].mean().item()
hmc_theta_err = (hmc_samples[:, 1:].mean(0) - true_theta).abs().mean().item()
results["hmc"] = {"mu_bias": hmc_mu - true_mu, "theta_abs_err": hmc_theta_err, "accept_rate": acc}
print(f"\nHMC (20000 samples): bias(mu)={hmc_mu - true_mu:+.4f}  theta|err|={hmc_theta_err:.4f}  accept={acc:.2f}")

with open("results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved results.json")
