"""
RESOLVED. Root cause of the hand-roll-vs-alan mismatch: a wrong mental model of
what "one K-dim shared across a plate" means, not a bug in alan.

I had assumed (call it Order A) that a plate-shared K-dim means each of the K
theta-particles is a single *vector-valued* hypothesis used identically across
every plate replicate -- i.e. for a fixed k_theta, all 6 elements use
theta_p[k_theta, :] together, so you sum the per-element log-densities *first*,
then logsumexp over k_theta once for the whole vector:

    log P_MP-ish(k_mu) = logsumexp_{k_theta} [ sum_i logdensity_i(k_theta, k_mu) ]

alan actually does something different and more powerful (Order B): each plate
element gets to *independently* pick which of the K theta-particles best
explains its own observation, via a per-element logsumexp over k_theta, and
*only then* are those (already-reduced) per-element log values summed across
the plate:

    log P_MP(k_mu) = sum_i logsumexp_{k_theta} [ logdensity_i(k_theta, k_mu) ]

These are NOT equal in general (logsumexp does not distribute over an inner
sum). Order B is a genuine additional bit of Rao-Blackwellization the plate
structure buys you -- each replicate resolves its own best particle rather
than the whole plate being forced into lockstep on one shared index -- and
it's what reduce_Ks/Plate.py's ".sum(new_platedim)" happening *after*
"reduce_Ks(...)" (which runs while the plate dim is still a live, broadcast
axis) actually implements. Confirmed directly: reduce_Ks on a toy 2-element-
plate x 3-particle tensor returns a result that still has the plate dim live
(order B), not summed away (order A) -- see the reduce_Ks() call at the
bottom of this file.

Both the ELBO (log P_MP(z), via elbo_nograd()) and the moment
(Sample.moments('mu', mean), via the same source-term-trick autodiff alan
uses internally) now match alan exactly under Order B -- to float32
precision, run this file to reproduce both.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import warnings
warnings.filterwarnings("ignore")

import math
import torch as t
from alan import PermutationSampler, mean as MEAN
from alan.Sample import Sample
from alan.reduce_Ks import reduce_Ks
from functorch.dim import Dim
from alan_compare import build_problem


def minimal_repro():
    """The whole discrepancy in five lines: reduce_Ks leaves the plate dim live."""
    K_dim, p_dim = Dim("K", 3), Dim("p", 2)
    t.manual_seed(0)
    lp = t.randn(2, 3)[p_dim, K_dim]
    result = reduce_Ks([lp], [K_dim])
    assert p_dim in result.dims, "reduce_Ks does NOT sum the plate -- order B confirmed"
    print("minimal repro: reduce_Ks(...) still has the plate dim live ->", result.dims)


def full_check():
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

    lp_mu, _ = mu_dist_P.log_prob(mu_t, {}, None, None)
    lq_mu, _ = mu_dist_Q.log_prob(mu_t, {}, None, None)
    lp_theta, _ = theta_dist_P.log_prob(theta_t, {"mu": mu_t}, None, None)
    lq_theta, _ = theta_dist_Q.log_prob(theta_t, {}, None, None)
    lp_x, _ = x_dist_P.log_prob(x_named, {"theta": theta_t}, None, None)

    # Order B: reduce K_theta PER PLATE ELEMENT, sum across the plate afterward.
    theta_term = (lp_theta - lq_theta - math.log(K)).order(p1dim, K_theta, K_mu)
    x_term = lp_x.order(p1dim, K_theta)
    per_elem_reduced = t.logsumexp(theta_term + x_term[:, :, None], dim=1)  # [plate, K_mu]
    p1_summed = per_elem_reduced.sum(0)                                     # [K_mu]
    logA = (lp_mu - lq_mu).order(K_mu) - math.log(K)

    log_PMP = t.logsumexp(logA + p1_summed, 0)

    mu_p = mu_t.order(K_mu).detach().requires_grad_(True)
    J = t.zeros((), requires_grad=True)
    log_PMP_J = t.logsumexp((logA + p1_summed) + J * mu_p, 0)
    mu_hat = t.autograd.grad(log_PMP_J, J)[0].item()

    s = Sample(problem=prob, sample=sample, groupvarname2Kdim=groupvarname2Kdim,
               sampler=PermutationSampler, reparam=False)

    print(f"log P_MP(z):  mine(order B)={log_PMP.item():.6f}  alan.elbo_nograd()={s.elbo_nograd().item():.6f}")
    print(f"E[mu]:        mine(order B)={mu_hat:.6f}  alan.moments()={s.moments('mu', MEAN).rename(None).item():.6f}")


if __name__ == "__main__":
    minimal_repro()
    print()
    full_check()
