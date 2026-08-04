"""
Real per-variable marginal ESS straight from alan's own Marginals.ess() (not
the suspect hand-rolled reimplementation -- see README's "Correction"
section). Confirms ESS(mu) stays a roughly constant ~0.6*K fraction across
K=10..3000, and ESS(theta) averages ~0.5*K but with real spread across the
6 plate replicates -- consistent with the "ESS is capped at K, not K^n"
claim, and a cleaner replacement for the earlier ad hoc joint-grid ESS check.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import warnings
warnings.filterwarnings("ignore")

import torch as t
from alan import PermutationSampler
from alan.utils import generic_dims, generic_order
from alan_compare import build_problem

if __name__ == "__main__":
    prob = build_problem()
    for K in [10, 30, 100, 300, 1000, 3000]:
        t.manual_seed(1)
        sample = prob.sample(K, reparam=False, sampler=PermutationSampler)
        ess = sample.marginals().ess()
        for gvn, e in ess.items():
            name = list(gvn)[0] if len(gvn) == 1 else tuple(gvn)
            e_flat = generic_order(e, generic_dims(e)).flatten()
            print(f"K={K:5d}  var={name!s:8s}  ESS min/mean/max = "
                  f"{e_flat.min().item():9.2f} / {e_flat.mean().item():9.2f} / {e_flat.max().item():9.2f}"
                  f"   (mean ESS/K: {e_flat.mean().item()/K:.4f})")
