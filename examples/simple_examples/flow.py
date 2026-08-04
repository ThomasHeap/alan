"""
alan.Flow: a distribution built from a base Dist pushed through a chain of
invertible transforms, with log_prob computed exactly via change-of-
variables (no approximation -- same exactness class as any other Dist).

TorchDimDist (the machinery behind every ordinary alan distribution)
requires a real torch.distributions class with arg_constraints/event_dim
metadata for every argument; a transform-based distribution's parameters
(a base distribution object, a list of transforms) don't fit that shape.
Flow sidesteps TorchDimDist entirely rather than extending it: it
reimplements sample/log_prob directly via ordinary torchdim arithmetic
(which composes fine for elementwise ops -- see amortized_inference.py),
computing the log-det-Jacobian by hand for each transform in the chain.

This example approximates a log-normal-shaped posterior with a Flow Q
(base Normal + ExpTransform, giving support on the positive reals) --
recovering the closed-form Normal parameters of the true log-normal P
via ordinary VI.
"""
import torch as t
import torch.nn as nn
from alan import Normal, Plate, BoundPlate, Problem, Data, Flow, ExpTransform

mu_true, sigma_true, obs_sigma = 0.5, 0.8, 0.3

P = Plate(
    z=Flow(Normal(mu_true, sigma_true), [ExpTransform()]),
    obs=Normal('z', obs_sigma),
)
P = BoundPlate(P, {})

t.manual_seed(0)
obs_val = P.sample()['obs']

# Flow's own parameters are plain nn.Parameters (no OptParam integration in
# this first pass) -- register them the same way amortized_inference.py
# registers an encoder: plain `Q.some_name = some_parameter` attribute
# assignment. Apply any transform (like .exp(), for positivity) via a
# zero-argument lambda, not pre-computed -- Dist's machinery only recomputes
# fresh on every call for scope-lookups/lambdas, not for a value computed
# once before construction.
q_mu = nn.Parameter(t.tensor(0.0))
q_log_sigma = nn.Parameter(t.tensor(0.0))

Q = Plate(
    z=Flow(Normal(q_mu, lambda: q_log_sigma.exp()), [ExpTransform()]),
    obs=Data(),
)
Q = BoundPlate(Q, {})
Q.q_mu = q_mu
Q.q_log_sigma = q_log_sigma

prob = Problem(P, Q, {'obs': obs_val})
opt = t.optim.Adam(Q.parameters(), lr=0.03)

for i in range(500):
    opt.zero_grad()
    sample = prob.sample(K=10, reparam=True)
    elbo = sample.elbo_vi()
    (-elbo).backward()
    opt.step()
    if i % 100 == 0:
        print(f"Iter {i}. Elbo: {elbo:.3f}")

print(f"learned q_mu:    {q_mu.item():.3f}")
print(f"learned q_sigma: {q_log_sigma.exp().item():.3f}")
