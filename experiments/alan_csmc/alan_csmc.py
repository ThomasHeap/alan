"""
Particle-Gibbs-style MCMC over the Timeseries variable `ts`, but using
alan's OWN dense (no intermediate resampling) K x K Timeseries machinery as
the SMC sweep -- not a hand-rolled bootstrap-PF-style resampling loop (that
already exists, validated, in experiments/particle_gibbs/). Each Gibbs
sweep:

  1. Builds Q with the current reference trajectory bound in as an `inputs`
     tensor, and `Timeseries(..., ref_traj_key='ref_traj')` pins K-index 0
     of every timestep's sample to that reference value (see
     src/alan/Timeseries.py). Unlike classical CSMC, this pin doesn't need
     to "protect" the reference from being pruned by intermediate
     resampling -- alan's forward pass (chain_logmmexp) never resamples
     between timesteps, so nothing is pruned until the final step.
  2. Runs alan's ordinary `problem.sample(K)` + `.importance_sample(1)` --
     the SAME machinery used for any other posterior sampling task in alan,
     unmodified -- to draw one new trajectory from the posterior implied by
     this pinned proposal pool.
  3. That new trajectory becomes the next reference.

This only pins the Timeseries variable `ts`, not the separate `init` latent
(the state before the Timeseries starts) -- init is sampled/marginalised
the ordinary way. A fuller version would pin both; scoped out here to keep
the first test of the core mechanism simple.
"""
import warnings
warnings.filterwarnings("ignore")

import torch as t
from alan import Normal, Timeseries, Plate, BoundPlate, Problem, Data, PermutationSampler
from alan.utils import generic_dims, generic_order
from csmc_model import A, Q_VAR, R_VAR, P0, T_STEPS


def build_problem(y, ref_traj, T=T_STEPS):
    P = Plate(
        init=Normal(0., P0 ** 0.5),
        T=Plate(
            ts=Timeseries('init', Normal(lambda prev: A * prev, Q_VAR ** 0.5)),
            obs=Normal('ts', R_VAR ** 0.5),
        ),
    )
    P = BoundPlate(P, {'T': T})

    q_inputs = {} if ref_traj is None else {'ref_traj': ref_traj.rename('T')}
    Q = Plate(
        init=Normal(0., 1.),
        T=Plate(
            ts=Timeseries('init', Normal(lambda prev: 0. * prev, 1.0),
                           ref_traj_key='ref_traj' if ref_traj is not None else None),
            obs=Data(),
        ),
    )
    Q = BoundPlate(Q, {'T': T}, inputs=q_inputs)

    return Problem(P, Q, {'obs': y.rename('T')})


def _extract_ts(imp, problem):
    T_dim = problem.all_platedims['T']
    ts = imp.samples_flatdict['ts']
    return generic_order(ts, [imp.Ndim, T_dim])[0]  # [T]


def run_alan_csmc(y, n_iters, K, burn_in, seed, T=T_STEPS):
    """Chain driver. Returns [n_iters+1-burn_in, T] tensor of reference
    trajectories (post burn-in)."""
    t.manual_seed(seed)

    problem0 = build_problem(y, None, T)
    sample0 = problem0.sample(K, reparam=False, sampler=PermutationSampler)
    ref_traj = _extract_ts(sample0.importance_sample(1), problem0)

    chain = []
    for it in range(n_iters):
        problem = build_problem(y, ref_traj, T)
        sample = problem.sample(K, reparam=False, sampler=PermutationSampler)
        ref_traj = _extract_ts(sample.importance_sample(1), problem)
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
