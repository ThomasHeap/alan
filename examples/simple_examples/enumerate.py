"""
Exact marginalisation of a small discrete latent, via alan.Enumerate().

Ordinary K-sample importance sampling is unbiased for any Q, but only exact
as K -> infinity. Enumerate() is exact for any finite K, provided K equals
the variable's cardinality: rather than sampling from Q, particle i is
deterministically set to category i, and the group's contribution to the
log-joint is P's log-probability alone (Q isn't used at all -- there's
nothing to importance-sample, so nothing to approximate). See Enumerate's
docstring for the derivation of why this needs a different reduction than
ordinary K-sample importance weighting, not just "K-sampling with K set to
the cardinality".
"""
import torch as t
from alan import Normal, Bernoulli, Plate, BoundPlate, Problem, Data, Enumerate

p_true, mu0, mu1, sigma = 0.3, -2.0, 3.0, 1.0
obs = t.tensor(1.0)

P = Plate(
    z=Bernoulli(p_true),
    obs=Normal(lambda z: z * mu1 + (1 - z) * mu0, sigma),
)
P = BoundPlate(P, {})

Q = Plate(
    z=Enumerate(),
    obs=Data(),
)
Q = BoundPlate(Q, {})

prob = Problem(P, Q, {'obs': obs})

# K must equal z's cardinality (2, for a Bernoulli) -- this isn't a sample
# count to tune for accuracy, it's the exact size of the discrete support.
sample = prob.sample(K=2, reparam=False)
log_marginal = sample.elbo_nograd()
print(f"alan (exact, enumerated): log P(obs) = {log_marginal.item():.6f}")

# Closed form for comparison: P(obs) = (1-p)*N(obs;mu0,sigma) + p*N(obs;mu1,sigma)
import math


def normal_pdf(x, mu, sigma):
    return math.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * math.sqrt(2 * math.pi))


true_marginal = (1 - p_true) * normal_pdf(obs.item(), mu0, sigma) + p_true * normal_pdf(obs.item(), mu1, sigma)
print(f"closed form:               log P(obs) = {math.log(true_marginal):.6f}")

# Deterministic: unlike K-sample importance sampling, repeated calls agree exactly.
log_marginal_2 = prob.sample(K=2, reparam=False).elbo_nograd()
print(f"repeated call, same value? {log_marginal.item() == log_marginal_2.item()}")
