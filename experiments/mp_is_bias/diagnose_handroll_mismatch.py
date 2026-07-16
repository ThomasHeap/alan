"""
Diagnostic: the hand-rolled MP-IS reimplementation in jackknife.py does NOT
reproduce alan's own Sample.moments() output, even when fed alan's *exact*
drawn particle values and alan's *exact* per-component log-probs. This
isolates the discrepancy precisely: it is not a random-seed-alignment
artifact (matching seeds across two independent codebases was never going to
align anyway, since alan's sample_gdt draws an extra, provably-unused
permutation via Sampler.perm() for every group regardless of whether it has
a parent -- see dist.py's Dist.sample(), which ignores the timeseries_perm
argument entirely for a plain Dist). It's a real, unresolved discrepancy in
aggregation, localized here to NOT be in any individual log p / log q
component. Root cause not yet found -- this script exists to reproduce the
isolation, not to fix it. See README's "Correction" section.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import warnings
warnings.filterwarnings("ignore")

import torch as t
from alan import PermutationSampler, mean as MEAN
from alan.Sample import Sample
from alan_compare import build_problem

prob = build_problem()
K = 8
t.manual_seed(3)
sample, groupvarname2Kdim = prob.Q._sample(K, False, PermutationSampler, prob.all_platedims)
K_mu, K_theta = groupvarname2Kdim["mu"], groupvarname2Kdim["theta"]
p1dim = prob.all_platedims["p1"]

mu_dist_P = prob.P.plate.flat_prog["mu"]
mu_dist_Q = prob.Q.plate.flat_prog["mu"]
theta_dist_P = prob.P.plate.flat_prog["p1"].flat_prog["theta"]
theta_dist_Q = prob.Q.plate.flat_prog["p1"].flat_prog["theta"]
x_dist_P = prob.P.plate.flat_prog["p1"].flat_prog["x"]

mu_t, theta_t = sample["mu"], sample["p1"]["theta"]
x_named = prob.data["p1"]["x"]

# Pull every log-prob component straight from alan's own Dist objects (ground truth).
lp_mu, _ = mu_dist_P.log_prob(mu_t, {}, None, None)
lq_mu, _ = mu_dist_Q.log_prob(mu_t, {}, None, None)
lp_theta, _ = theta_dist_P.log_prob(theta_t, {"mu": mu_t}, None, None)
lq_theta, _ = theta_dist_Q.log_prob(theta_t, {}, None, None)
lp_x, _ = x_dist_P.log_prob(x_named, {"theta": theta_t}, None, None)

logA = (lp_mu - lq_mu).order(K_mu)
logB = (lp_x.sum(p1dim) - lq_theta.sum(p1dim)).order(K_theta)
logC = lp_theta.sum(p1dim).order(K_theta, K_mu)

log_r = logA[None, :] + logB[:, None] + logC
log_r = log_r - log_r.max()
r = log_r.exp()
col_sum = r.sum(0)
mu_p = mu_t.order(K_mu)
mu_hat_manual = ((mu_p * col_sum).sum() / col_sum.sum()).item()

s = Sample(problem=prob, sample=sample, groupvarname2Kdim=groupvarname2Kdim,
           sampler=PermutationSampler, reparam=False)
mu_hat_alan = s.moments("mu", MEAN).rename(None).item()

print("Using alan's OWN exact log-prob components, manual ratio aggregation:", mu_hat_manual)
print("alan's own Sample.moments('mu', mean):                              ", mu_hat_alan)
print("These disagree -- see README.md 'Correction' section.")
