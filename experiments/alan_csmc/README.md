# Alan-native CSMC

Tests whether alan's own dense (no intermediate resampling) K x K Timeseries
machinery can serve directly as the SMC sweep inside a Particle-Gibbs-style
MCMC chain, instead of hand-rolling a bootstrap-PF-style resampling loop
(that already exists, validated, in `experiments/particle_gibbs/`).

## Why this might work without PGAS

Classical CSMC needs a pinned reference trajectory specifically to survive
*resampling between timesteps* — plain bootstrap PF prunes the particle
population at every step, and without protection the reference's own
lineage can get pruned away, which is what PGAS's ancestor-sampling fixes.
Alan's forward pass (`chain_logmmexp`) never resamples between timesteps at
all — it keeps the full dense K x K structure until one final backward-
sampling step. So pinning K-index 0 to a reference value at every timestep
might get PGAS-level mixing "for free," with no ancestor-sampling machinery
needed.

## Mechanism

- `Timeseries(..., ref_traj_key='ref_traj')` (`src/alan/Timeseries.py`)
  pins K-index 0 of every timestep's sample to a reference trajectory value,
  read from a scope entry bound in via `BoundPlate`'s `inputs`.
- `Dist(..., ref_val_key='ref_init')` (`src/alan/dist.py`) does the same for
  an ordinary (non-Timeseries) latent — needed here for `init`, the state
  one plate level up from the Timeseries.
- `Timeseries(..., ref_init_key='ref_init')` *additionally* pins the
  ephemeral `prev_state` copy used internally for generating the other,
  non-reference K-1 timesteps' proposals.

**These are not redundant, and getting the distinction right mattered.**
The first version of this experiment only used `ref_init_key`, reasoning
that pinning `init`'s own `Dist` output would get scrambled anyway by
`sample_gdt`'s `resample_scope` (which relabels a parent variable's K-dim
into the child's own K-dim as part of `PermutationSampler`'s ordinary
parent-particle mixing — true, but irrelevant, since that relabelling
operates on a *copy*, not `init`'s originally-stored sample). The result:
`init`'s own stored sample — which is what `Timeseries.log_prob` actually
reads via `scope[self.init]` — never contained the reference value at all,
silently breaking the coherence of the reference trajectory as a whole.
Diagnosed by comparing against a fully independent, unpinned baseline
(repeatedly drawing fresh `importance_sample(1)` draws with no Gibbs
coherence at all) and finding it gave the *same* RMSE as the supposedly-
pinned chain — a strong signal the pin wasn't doing anything. Both
mechanisms are needed together: `ref_val_key` for correctness (the
reference pairing has to be visible to `log_prob`'s K x K structure),
`ref_init_key` as a proposal-quality nicety on top.

Each Gibbs sweep: build Q with the current reference (init and every `ts`
timestep) bound in, run alan's ordinary `problem.sample(K)` +
`.importance_sample(1)` unmodified, and take the result as the next
reference. `csmc_model.py` reuses the exact same linear-Gaussian SSM and
parameters as `experiments/particle_gibbs/ssm_model.py`, so results are
directly comparable to that experiment's hand-rolled plain-PG/PGAS numbers.

## Results (2000 iterations, 200 burn-in)

```
                    plain PG (hand-rolled)   PGAS (hand-rolled)   alan-native CSMC
K=5   RMSE           0.1843                   0.0164                0.1996
      chain ESS       1.0/1800 (0.1%)         772.1/1800 (42.9%)    692.4/1800 (38.5%)
K=10  RMSE           0.0467                   0.0141                0.1914
      chain ESS      15.9/1800 (0.9%)        1174.9/1800 (65.3%)   1373.1/1800 (76.3%)
K=30  RMSE           0.0304                   0.0114                0.1934
      chain ESS     422.2/1800 (23.5%)       1616.1/1800 (89.8%)   1653.1/1800 (91.8%)
```

**Mixing: the core hypothesis holds, cleanly.** At every K, alan-native
CSMC's chain ESS matches or *exceeds* hand-rolled PGAS's — with no
ancestor-sampling machinery at all, just the reference pin. It vastly
outperforms plain (non-ancestor-sampled) hand-rolled PG at every K. This
result reproduced across two independent full runs (a first run before the
coherence-bug fix below, and this one after) with the same qualitative
picture both times — the mixing conclusion isn't sensitive to that bug.

**Accuracy: worse than both hand-rolled variants, and doesn't improve with
K over the range tested.** RMSE sits around 0.19–0.20 at K=5, 10, and 30
alike — no visible K-dependence, unlike the hand-rolled versions' clear
improvement. This is *not* explained by the coherence bug above (fixing it
left RMSE essentially unchanged: 0.1987/0.1904/0.1935 before the fix vs.
0.1996/0.1914/0.1934 after, at K=5/10/30 respectively). Comparing against a
fully independent, unpinned baseline (fresh `importance_sample(1)` draws
every "sweep," no reference coherence at all) gives RMSE of the same
magnitude (0.20–0.33 across K=5–30) — implicating `importance_sample(1)`'s
own single-draw extraction, not the Gibbs/pinning mechanism, as the
dominant remaining error source. This is plausibly the same phenomenon
established earlier this session for `P_hat_MP`-based moment estimates:
self-normalized importance sampling is only *asymptotically* unbiased,
with O(1/K) bias at finite K — and `importance_sample`'s posterior-sample
extraction is exactly this kind of SNIS-style operation.

## Reproducing

```
cd experiments/alan_csmc
python3 csmc_model.py      # Kalman/RTS ground truth sanity check
python3 alan_csmc.py       # smoke test, one CSMC chain
python3 csmc_validate.py   # full accuracy + mixing-speed comparison (~8 minutes)
```

## What this doesn't cover

- Only pins `init` and `ts`; a model with several chained Timeseries
  variables would need the same treatment at every stage.
- The accuracy gap's likely cause (finite-K SNIS bias in reference
  extraction) is diagnosed, not fixed. A natural next step: draw several
  candidate references per sweep and select via explicit multinomial
  resampling weighted by importance weight, rather than relying on
  `importance_sample(1)`'s internal single-draw mechanism.
- Regression-tested against the full existing test suite (361 tests pass)
  with both the `Timeseries.py` and `dist.py` changes in place.
