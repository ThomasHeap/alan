"""
Exact discrete-state HMM inference, via alan.Enumerate() applied to a
Timeseries variable.

Q's Timeseries slot can already be Data() or a plain (non-Timeseries) Dist
instead of a full Timeseries object (a common mean-field approximation to a
genuinely sequential P). Enumerate() fits the same slot: rather than
approximating the state posterior with K-sample particles, it exactly
enumerates the discrete state space at every timestep. P's own transition
structure supplies all the cross-timestep coupling via the existing
chain_logmmexp reduction -- which already implements exactly the standard
HMM forward algorithm's alpha_t = alpha_{t-1} @ Transition_t recursion, so
this needs no new machinery, just K = the number of discrete states.
"""
import math
import torch as t
from alan import Normal, Bernoulli, Plate, BoundPlate, Problem, Data, Enumerate, Timeseries, mean

T_STEPS = 8
pi1 = 0.5                     # P(state_0 = 1)
trans01, trans11 = 0.3, 0.7   # P(state_t=1 | state_{t-1}=0 or 1)
mu0, mu1, sigma = -1.0, 2.0, 0.5

t.manual_seed(1)
states = [1 if t.rand(1).item() < pi1 else 0]
for _ in range(T_STEPS - 1):
    p = trans11 if states[-1] == 1 else trans01
    states.append(1 if t.rand(1).item() < p else 0)
obs = t.tensor([(mu1 if s == 1 else mu0) + sigma * t.randn(1).item() for s in states])

P = Plate(
    init=Bernoulli(pi1),
    T=Plate(
        z=Timeseries('init', Bernoulli(lambda prev: prev * trans11 + (1 - prev) * trans01)),
        obs=Normal(lambda z: z * mu1 + (1 - z) * mu0, sigma),
    ),
)
P = BoundPlate(P, {'T': T_STEPS})

Q = Plate(
    init=Enumerate(),  # K must equal the state cardinality (2, for Bernoulli)
    T=Plate(z=Enumerate(), obs=Data()),
)
Q = BoundPlate(Q, {'T': T_STEPS})

prob = Problem(P, Q, {'obs': obs.rename('T')})
sample = prob.sample(K=2, reparam=False)

print(f"true states:                {states}")
print(f"log P(obs) [exact, alan]:   {sample.elbo_nograd().item():.6f}")

# sample.moments gives the exact forward-backward SMOOTHED posterior --
# E[z_t] here is exactly P(z_t=1 | all obs), for every t simultaneously.
smoothed = sample.moments([('z', mean)])[0].rename(None)
print(f"P(z_t=1 | all obs), smoothed posterior:")
print(smoothed)
