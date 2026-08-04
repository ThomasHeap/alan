"""
Validation sweep for br_snis.py: bias/RMSE of both i-SIR chains vs Global-IS,
MP-IS, and jackknife_mu, at roughly matched cost (K_pool * T_steps model
evaluations for the i-SIR chains vs K draws for Global-IS/MP-IS).

Run: python br_snis_validate.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import warnings
warnings.filterwarnings("ignore")

import torch as t
from compare_model import analytic_posterior
from alan_compare import build_problem, mp_is_moments, global_is_moments
from jackknife_orderB import compute_log_r_mu, jackknife_mu
from br_snis import br_snis_full_estimate, br_snis_mu_estimate

mean, cov = analytic_posterior()
true_mu = mean[0].item()
true_theta = mean[1:]


def summarize(name, errs):
    errs = t.tensor(errs)
    print(f"  {name:28s} bias={errs.mean():+.4f} (se {errs.std()/len(errs)**0.5:.4f})"
          f"   RMSE={errs.pow(2).mean().sqrt():.4f}")


if __name__ == "__main__":
    prob = build_problem()
    N_REPEATS = 40

    print("=== mu: full-joint i-SIR vs Global-IS vs MP-IS vs jackknife(mu), matched budget ===")
    for K_pool, T in [(3, 100), (5, 200), (10, 300)]:
        budget = K_pool * T
        print(f"\n-- i-SIR K_pool={K_pool}, T={T} (budget~{budget})  vs Global-IS/MP-IS at K={budget} --")

        isir_mu_errs, isir_theta_errs = [], []
        global_errs, mp_errs, jack_errs = [], [], []
        for rep in range(N_REPEATS):
            mu_hat, theta_hat = br_snis_full_estimate(K_pool, T, burn_in=T // 5, seed=1000 * K_pool + rep)
            isir_mu_errs.append(mu_hat - true_mu)
            isir_theta_errs.append((theta_hat - true_theta).pow(2).mean().sqrt().item())

            mu_g, _ = global_is_moments(prob, budget, seed=2000 * K_pool + rep)
            global_errs.append(mu_g - true_mu)

            mu_m, _ = mp_is_moments(prob, budget, seed=3000 * K_pool + rep)
            mp_errs.append(mu_m - true_mu)

            log_r_mu, mu_p, _ = compute_log_r_mu(budget, seed=4000 * K_pool + rep)
            _, mu_jack = jackknife_mu(log_r_mu, mu_p, budget)
            jack_errs.append(mu_jack - true_mu)

        summarize("i-SIR (full-joint, chain avg)", isir_mu_errs)
        summarize("Global-IS", global_errs)
        summarize("MP-IS (plain)", mp_errs)
        summarize("MP-IS + jackknife(mu)", jack_errs)
        print(f"  {'i-SIR theta RMSE (per-elem, avg)':28s} = {t.tensor(isir_theta_errs).mean():.4f}")

    print("\n=== mu: pseudo-marginal i-SIR (alan's noisy K_theta likelihood) vs jackknife(mu) ===")
    print("(jackknife shown two ways: at the SAME tiny K as the i-SIR outer pool -- matching how")
    print(" small alan's own K_mu would be -- and at a K matched to the i-SIR's TOTAL model-eval")
    print(" cost K_pool*T*K_theta, for a cost-fair comparison.)")
    for K_pool, K_theta, T in [(3, 30, 100), (5, 30, 200)]:
        print(f"\n-- i-SIR K_pool={K_pool}, K_theta={K_theta}, T={T} --")
        pm_errs = []
        for rep in range(N_REPEATS):
            mu_hat = br_snis_mu_estimate(K_pool, K_theta, T, burn_in=T // 5, seed=5000 * K_pool + rep)
            pm_errs.append(mu_hat - true_mu)
        summarize("i-SIR (pseudo-marginal, chain avg)", pm_errs)

        jack_errs_small = []
        for rep in range(N_REPEATS):
            log_r_mu, mu_p, _ = compute_log_r_mu(K_pool, seed=7000 * K_pool + rep)
            _, mu_jack = jackknife_mu(log_r_mu, mu_p, K_pool)
            jack_errs_small.append(mu_jack - true_mu)
        summarize(f"MP-IS + jackknife(mu), K={K_pool}", jack_errs_small)

        K_matched = K_pool * K_theta
        jack_errs_matched = []
        for rep in range(N_REPEATS):
            log_r_mu, mu_p, _ = compute_log_r_mu(K_matched, seed=7500 * K_pool + rep)
            _, mu_jack = jackknife_mu(log_r_mu, mu_p, K_matched)
            jack_errs_matched.append(mu_jack - true_mu)
        summarize(f"MP-IS + jackknife(mu), K={K_matched} (cost-matched)", jack_errs_matched)

    print("\n=== theta: full-joint i-SIR vs plain MP-IS (jackknife(theta) never helped -- see jackknife_orderB.py) ===")
    for K_pool, T in [(5, 200), (10, 300)]:
        budget = K_pool * T
        isir_theta_rmse = []
        mp_theta_rmse = []
        for rep in range(N_REPEATS):
            _, theta_hat = br_snis_full_estimate(K_pool, T, burn_in=T // 5, seed=8000 * K_pool + rep)
            isir_theta_rmse.append((theta_hat - true_theta).pow(2).mean().sqrt().item())

            _, theta_m = mp_is_moments(prob, budget, seed=9000 * K_pool + rep)
            mp_theta_rmse.append((theta_m.rename(None) - true_theta).pow(2).mean().sqrt().item())
        print(f"\n-- K_pool={K_pool}, T={T} (budget~{budget}) --")
        print(f"  i-SIR (full-joint) theta RMSE: {t.tensor(isir_theta_rmse).mean():.4f}")
        print(f"  MP-IS (plain)      theta RMSE: {t.tensor(mp_theta_rmse).mean():.4f}")
