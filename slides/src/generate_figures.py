"""
Generates the figures used in the presentation, from real experiment output
(experiments/mp_is_bias/results.json + fresh reruns of marginal_ess.py /
jackknife_orderB.py on explore/mp-is-bias-empirical, and the numbers reported
in experiments/particle_gibbs/README.md on explore/particle-mcmc). No
fabricated numbers -- every value here was produced by an actual script run.
"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.size": 13,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#444444",
    "axes.labelcolor": "#222222",
    "text.color": "#222222",
    "xtick.color": "#444444",
    "ytick.color": "#444444",
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})

TEAL = "#1f6f78"
CORAL = "#c05746"
GOLD = "#d1a344"
GREY = "#8a8a8a"

# ---------------------------------------------------------------------------
# 1. bias vs K: MP-IS vs Global-IS vs HMC
# ---------------------------------------------------------------------------
with open("/home/user/alan/slides/src/results.json") as f:
    results = json.load(f)

Ks = results["Ks"]
mp_bias = [results["mp"][str(k)]["mean_bias_mu"] for k in Ks]
mp_se = [results["mp"][str(k)]["se_mu"] for k in Ks]
global_bias = [results["global"][str(k)]["mean_bias_mu"] for k in Ks]
global_se = [results["global"][str(k)]["se_mu"] for k in Ks]
hmc_bias = results["hmc"]["mu_bias"]

fig, ax = plt.subplots(figsize=(9, 5))
ax.errorbar(Ks, mp_bias, yerr=mp_se, marker="o", color=TEAL, label="MP-IS", capsize=3, linewidth=2)
ax.errorbar(Ks, global_bias, yerr=global_se, marker="s", color=CORAL, label="Global-IS", capsize=3, linewidth=2)
ax.axhline(hmc_bias, color=GOLD, linestyle="--", linewidth=2, label="HMC (20k samples)")
ax.axhline(0, color=GREY, linestyle=":", linewidth=1)
ax.set_xscale("log")
ax.set_xlabel("K (particles)")
ax.set_ylabel(r"bias of $\hat{E}[\mu]$")
ax.set_title("Self-normalization bias shrinks with K")
ax.legend(frameon=False, loc="upper right")
fig.tight_layout()
fig.savefig("/home/user/alan/slides/figures/bias_vs_K.pdf")
plt.close(fig)

# ---------------------------------------------------------------------------
# 2. ESS efficiency vs K, mu and theta (mean + min/max band)
# ---------------------------------------------------------------------------
ess_Ks = [10, 30, 100, 300, 1000, 3000]
ess_mu_ratio = [0.5617, 0.6061, 0.5905, 0.6175, 0.6128, 0.5965]
ess_theta_mean_ratio = [0.5858, 0.5132, 0.4702, 0.5015, 0.4823, 0.4808]
ess_theta_min = [3.40, 5.77, 19.12, 48.93, 165.15, 503.85]
ess_theta_max = [7.28, 21.18, 65.37, 205.50, 658.35, 1947.80]
ess_theta_min_ratio = [m / k for m, k in zip(ess_theta_min, ess_Ks)]
ess_theta_max_ratio = [m / k for m, k in zip(ess_theta_max, ess_Ks)]

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(ess_Ks, ess_mu_ratio, marker="o", color=TEAL, linewidth=2, label=r"ESS($\mu$) / K")
ax.plot(ess_Ks, ess_theta_mean_ratio, marker="s", color=CORAL, linewidth=2, label=r"ESS($\theta$) / K, mean over 6 plate elements")
ax.fill_between(ess_Ks, ess_theta_min_ratio, ess_theta_max_ratio, color=CORAL, alpha=0.18,
                 label=r"ESS($\theta$) / K, range across plate elements")
ax.set_xscale("log")
ax.set_ylim(0, 1.0)
ax.set_xlabel("K (particles)")
ax.set_ylabel("efficiency ratio (ESS / K)")
ax.set_title("Marginal ESS efficiency stays flat -- but uneven across the plate")
ax.legend(frameon=False, loc="lower left", fontsize=11)
fig.tight_layout()
fig.savefig("/home/user/alan/slides/figures/ess_vs_K.pdf")
plt.close(fig)

# ---------------------------------------------------------------------------
# 3. jackknife: |bias| plain vs jack(mu) vs jack(theta), grouped by K
# ---------------------------------------------------------------------------
jk_Ks = [3, 10, 30, 100, 300, 1000]
plain = [0.0468, 0.0148, 0.0071, 0.0239, 0.0064, 0.0030]
jack_mu = [0.0036, 0.0131, 0.0069, 0.0238, 0.0064, 0.0030]
jack_theta = [0.0566, 0.0211, 0.0142, 0.0246, 0.0066, 0.0030]

x = np.arange(len(jk_Ks))
width = 0.26
fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(x - width, plain, width, label="plain MP-IS", color=GREY)
ax.bar(x, jack_mu, width, label=r"+ jackknife($\mu$)", color=TEAL)
ax.bar(x + width, jack_theta, width, label=r"+ jackknife($\theta$)", color=CORAL)
ax.set_xticks(x)
ax.set_xticklabels([str(k) for k in jk_Ks])
ax.set_xlabel("K (particles)")
ax.set_ylabel(r"|bias| of $\hat{E}[\mu]$")
ax.set_title("Jackknife helps mu, not theta")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig("/home/user/alan/slides/figures/jackknife_bias.pdf")
plt.close(fig)

# ---------------------------------------------------------------------------
# 4. Particle Gibbs vs PGAS: chain ESS %, grouped by K
# ---------------------------------------------------------------------------
pg_Ks = [5, 10, 30]
pg_ess_pct = [0.1, 0.9, 23.5]
pgas_ess_pct = [42.9, 65.3, 89.8]

x = np.arange(len(pg_Ks))
width = 0.32
fig, ax = plt.subplots(figsize=(8, 5))
bars1 = ax.bar(x - width / 2, pg_ess_pct, width, label="plain Particle Gibbs", color=CORAL)
bars2 = ax.bar(x + width / 2, pgas_ess_pct, width, label="PGAS (ancestor sampling)", color=TEAL)
ax.set_xticks(x)
ax.set_xticklabels([f"K={k}" for k in pg_Ks])
ax.set_ylabel("chain ESS of $x_0$ (% of chain length)")
ax.set_title("PGAS fixes reference-trajectory stickiness")
ax.legend(frameon=False, loc="upper left")
ax.text(0 - width / 2, pg_ess_pct[0] + 1.5, "frozen\n(0 moves)", ha="center", fontsize=10, color=CORAL)
fig.tight_layout()
fig.savefig("/home/user/alan/slides/figures/pg_mixing.pdf")
plt.close(fig)

print("Figures written to slides/figures/")
