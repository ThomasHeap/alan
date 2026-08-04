# alan

Software in development!! Not yet for external use!!

### Installation

Start by removing previous alan or alan_simplified
```
python -m pip uninstall alan
python -m pip uninstall alan_simplified
```
And then remove the corresponding repo folders before cloning the new repo.

To install, clone repo, navigate to repo root directory, and use,
```
pip install -e .
```

### Tests

To run tests, navigate to `tests/` and use `pytest`. To run them in parallel
(CI does this, ~3x faster): `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 pytest -n
auto` -- the thread pinning matters, since torch's own internal BLAS
threading otherwise oversubscribes the machine's cores together with
pytest-xdist's worker processes and makes the run slower, not faster.

### Docs

[Read the Docs](https://alan-ppl.readthedocs.io/en/latest/)

### Overall example:

See `examples/simple_examples/example.py`

### Amortized (VAE-style) inference:

There's no dedicated API for this -- an ordinary Python-function distribution
argument that closes over an `nn.Module` and a plain `Q.encoder = encoder`
attribute assignment (for `Q.parameters()` to pick up its weights) is
already enough. See `examples/simple_examples/amortized_inference.py`.

### Normalizing flows:

`alan.Flow(base_dist, transforms)` -- a distribution built by pushing a base
Dist's sample through a chain of invertible `Transform`s (`AffineTransform`,
`ExpTransform`, `SigmoidTransform`), with log_prob computed exactly via
change-of-variables. Bypasses `TorchDimDist` entirely (its `arg_constraints`
mechanism can't represent a transform-based distribution's arguments) rather
than extending it. Transform/base-distribution parameters are plain
`nn.Parameter`s, registered the same way as amortized inference (`Q.some_name
= some_parameter`), not via `OptParam`/`QEMParam`. See
`examples/simple_examples/flow.py`. Current scope (a first, minimal pass):
elementwise transforms only; not supported inside a Timeseries or a Group;
not yet wired into `sample_nonmp`, `importance_sample`, or
`predictive_ll`/`extend`.



### Meeting TODOs:
  * Docs:
     Friendly overview material (Pyro is a good example).
    - clarify `ExtendedImportanceSample.predictive_ll`.
      - How does it deal with the N samples?
      - How does it deal with the LL for the training data?
  * Think carefully about extended Plate errors.
    - e.g. if you try to extend a prior with plated parameters.
  * Timeseries:
    - tests (Kalman filter)
    - importance_sample
    - extend
  * MovieLens/bus experiments with new code, with VI / RWS / QEM==Natural RWS (see `example/example.py`):
    - small scale (usual subsampling)
    - large scale using `computation_strategy=Split(...)`


### Long-run TODOs:
  * Friendly error messages:
    - Marginals/moments make sense for variables on different plates if they're in the same heirarchy.
  * Enumeration:
    - Done for a single discrete variable per group -- see `Enumerate`
      (`examples/simple_examples/enumerate.py`). Works with moments,
      marginals, importance_sample, and sample_nonmp. Also works as a
      Timeseries variable's Q slot, giving exact discrete-state HMM
      inference (forward algorithm / forward-backward smoothing) --
      see `examples/simple_examples/enumerate_timeseries.py`.
    - Still to do: Enumerate combined with other variables in the same
      Group. Group's whole point is sharing ONE K-dim across otherwise-
      independent variables (not cross-producting their supports), so
      "jointly enumerate multiple discrete variables" doesn't have a
      natural meaning there. What WOULD fit Group's existing semantics --
      Enumerate() paired with an ordinarily-sampled Dist that shares its
      K-index despite being independent of it -- needs the group's log-prob
      formula split into an exact part (no -log(K)/Q correction) and an
      ordinary part (needs both), which is unclear enough value for the
      added complexity that it's deliberately not done.
  * A `Samples` class that aggregates over multiple `Sample` in a memory efficient way.
    - Acts like it contains a list of e.g. 10 `Sample`s, but doesn't actually.
    - Instead, it generates the `Sample`s as necessary by using frozen random seed.
   

### Ideas:
  * Remove checkpointing from reduce_Ks?
  * Numerically stable logsumexp_dims, by adding t.finfo(lp.dtype).eps before log
  * Check P: Timeseries, Q:Plate/Timeseries, but not P:Plate, Q:Timeseries, or P:Plate, Q:Data.
  * Check timeseries permutation is the right size.
  * Simplify sampler.resample_scope, by noting that:
    * all variables in scope are either parameters (no K's)
    * random variables drawn from a dist (unique K)
    * random variable drawn from a group (non-unique K).
    * can reason about these more easily if we e.g. pass in varname2groupvarname.
  * `importance_sample.dump` should output tensors with the `N` dimension first?
  * latent moments for `linear_gaussian_latents`
  * tests for mixture distributions.
  * remove the random init from most of the tests.
  * tests for extended_sample
    - TestProblem takes extended_platesizes and predicted_extended_moments as arguments.
    - some extended moments are exactly equal to importance sampled moments (i.e. unplated moments + first part of plated moments).
    - other extended moments aren't exactly equal.
    - instead, the extended moments are a function of `importance_sample` moments.
    - TestProblem takes 
    - predicted_extended_moments is a function that takes an importance sample, and returns mean + variance of moment.
    - should really compare moments to 
  * QEM distributions:
    - Categorical
    - Testing
