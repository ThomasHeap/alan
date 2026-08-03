"""
Particle-Gibbs-style MCMC over the whole trajectory (init, ts), but using
alan's OWN dense (no intermediate resampling) K x K Timeseries machinery as
the SMC sweep -- not a hand-rolled bootstrap-PF-style resampling loop (that
already exists, validated, in experiments/particle_gibbs/). Each Gibbs
sweep:

  1. Builds Q with the current reference trajectory (both `init` and every
     `ts` timestep) bound in as `inputs` tensors.
     `Timeseries(..., ref_traj_key='ref_traj', ref_init_key='ref_init')`
     (src/alan/Timeseries.py) pins K-index 0 of both the initial state and
     every timestep's sample to the matching reference value. Unlike
     classical CSMC, this pin doesn't need to "protect" the reference from
     being pruned by intermediate resampling -- alan's forward pass
     (chain_logmmexp) never resamples between timesteps, so nothing is
     pruned until the final step.
  2. Runs alan's ordinary `problem.sample(K)` + `.importance_sample(1)` --
     the SAME machinery used for any other posterior sampling task in alan,
     unmodified -- to draw one new trajectory from the posterior implied by
     this pinned proposal pool.
  3. That new (init, ts) pair becomes the next reference.

Pinning `init` as well as `ts` matters for correctness, not just
completeness. It's also subtler than it looks: `init` is sampled by an
ordinary Dist, one plate level up from `ts`. An earlier version of this
file pinned `init`'s own Dist output directly (a `ref_val_key` on init's
own `Normal(...)`) -- which turned out NOT to survive the trip into
`Timeseries.sample`: sample_gdt's `sampler.resample_scope(...)` call
relabels/permutes a parent variable's K-dim into the child's own K-dim
(PermutationSampler's ordinary parent-particle mixing) BEFORE Timeseries
ever sees it, silently scrambling which physical value ends up at K-index
0. The fix has to happen on `prev_state` inside `Timeseries.sample`
itself, after that relabelling -- which is what `ref_init_key` does.
Empirically, the wrong (Dist-level) location showed up as a persistent
RMSE bias that didn't shrink with K; the right (Timeseries-level) location
fixes it (see csmc_validate.py's results).
"""
import warnings
warnings.filterwarnings("ignore")

import torch as t
from alan import Normal, Timeseries, Plate, BoundPlate, Problem, Data, PermutationSampler
from alan.utils import generic_dims, generic_order
from csmc_model import A, Q_VAR, R_VAR, P0, T_STEPS


def build_problem(y, ref_traj, ref_init, T=T_STEPS):
    P = Plate(
        init=Normal(0., P0 ** 0.5),
        T=Plate(
            ts=Timeseries('init', Normal(lambda prev: A * prev, Q_VAR ** 0.5)),
            obs=Normal('ts', R_VAR ** 0.5),
        ),
    )
    P = BoundPlate(P, {'T': T})

    pinned = ref_traj is not None
    q_inputs = {} if not pinned else {'ref_traj': ref_traj.rename('T'), 'ref_init': ref_init}
    Q = Plate(
        #ref_val_key pins init's own STORED sample -- what Timeseries.log_prob
        #actually reads via scope[self.init] -- so the reference pairing is
        #visible to the K x K cross-product, not just to Timeseries.sample's
        #own ephemeral, differently-relabelled prev_state copy.
        init=Normal(0., 1., ref_val_key='ref_init' if pinned else None),
        T=Plate(
            #ref_init_key ALSO pins that ephemeral prev_state copy, so the
            #OTHER (non-reference) K-1 timesteps' proposal draws are informed
            #by a reference-consistent parent too -- a proposal-quality nicety
            #on top of ref_val_key's correctness fix, not a substitute for it.
            ts=Timeseries('init', Normal(lambda prev: 0. * prev, 1.0),
                           ref_traj_key='ref_traj' if pinned else None,
                           ref_init_key='ref_init' if pinned else None),
            obs=Data(),
        ),
    )
    Q = BoundPlate(Q, {'T': T}, inputs=q_inputs)

    return Problem(P, Q, {'obs': y.rename('T')})


def _extract_ref(imp, problem):
    T_dim = problem.all_platedims['T']
    ts = imp.samples_flatdict['ts']
    init = imp.samples_flatdict['init']
    ref_traj = generic_order(ts, [imp.Ndim, T_dim])[0]      # [T]
    ref_init = generic_order(init, [imp.Ndim])[0]           # scalar
    return ref_traj, ref_init


def run_alan_csmc(y, n_iters, K, burn_in, seed, T=T_STEPS):
    """Chain driver. Returns [n_iters-burn_in, T] tensor of reference
    trajectories (post burn-in; init itself isn't returned)."""
    t.manual_seed(seed)

    problem0 = build_problem(y, None, None, T)
    sample0 = problem0.sample(K, reparam=False, sampler=PermutationSampler)
    ref_traj, ref_init = _extract_ref(sample0.importance_sample(1), problem0)

    chain = []
    for it in range(n_iters):
        problem = build_problem(y, ref_traj, ref_init, T)
        sample = problem.sample(K, reparam=False, sampler=PermutationSampler)
        ref_traj, ref_init = _extract_ref(sample.importance_sample(1), problem)
        if it >= burn_in:
            chain.append(ref_traj.clone())

    return t.stack(chain)


if __name__ == "__main__":
    from csmc_model import simulate, exact_posterior

    x_true, y = simulate()
    m_smooth, _ = exact_posterior(y)

    print("--- alan-native CSMC smoke test, K=10, 200 iterations, 50 burn-in ---")
    chain = run_alan_csmc(y, n_iters=200, K=10, burn_in=50, seed=0)
    rmse = (chain.mean(0) - m_smooth[1:]).pow(2).mean().sqrt().item()
    print(f"RMSE vs RTS smoother mean (ts only, t=1..T): {rmse:.4f}")
