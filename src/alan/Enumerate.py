import torch.nn as nn

class Enumerate(nn.Module):
    """
    alan.Enumerate()

    Marks a discrete random variable in Q for exact marginalisation by
    enumeration, rather than K-sample importance sampling. Used exactly like
    Data() -- as the value for a variable name in Q, mirroring the same
    variable's distribution in P:

    .. code-block:: python

       P = Plate(
          z = Bernoulli(0.3),
          obs = Normal('z', 1.),
       )
       Q = Plate(
          z = Enumerate(),
          obs = Data(),
       )

    Unlike ordinary K-sample importance sampling (which is unbiased for any
    Q, but only exact as K -> infinity), enumeration is exact for any finite
    K, provided K equals the variable's cardinality (e.g. K=2 for a
    Bernoulli, K=len(probs) for a Categorical) -- Problem.sample(K=...) must
    be called with that K. Q's own distribution is not used at all for an
    enumerated variable (there's nothing to importance-sample, so nothing to
    approximate): particle i is deterministically assigned category i, and
    the group's contribution to the log-joint is P's log-probability alone
    -- ordinary logsumexp reduction over the K dimension then gives exactly
    log(sum_x P(x)), with zero variance.

    Also works as a Timeseries variable's Q slot (mirroring how that slot
    can already be Data() or a plain, non-Timeseries Dist instead of a full
    Timeseries object), giving EXACT discrete-state HMM inference: P's
    log_prob already returns the full [Kinit, Kcurr] transition
    log-probability matrix at each timestep, and the existing
    chain_logmmexp + logsumexp reduction already implements the standard
    forward algorithm's alpha_t = alpha_{t-1} @ Transition_t recursion, so
    this needs no new machinery -- just K = the number of discrete states.
    See examples/simple_examples/enumerate_timeseries.py; validated against
    a hand-rolled forward algorithm (elbo_nograd) and forward-backward
    smoother (sample.moments), including gradients back into P's
    parameters (a finite-difference check against the closed-form forward
    algorithm).

    Current scope: a single Enumerate() variable per group (not combined
    with other variables in a Group -- see the README's long-run TODOs for
    why that's a narrower, less clear-cut case than it might look).
    Supported by both Problem.sample (elbo_vi/elbo_rws/elbo_nograd,
    sample.moments(...), sample.marginals(), sample.importance_sample(...)
    -- all validated against closed-form/statistical ground truth,
    including joint queries with other variables, and with a Timeseries)
    and Problem.sample_nonmp (elbo_vi/elbo_rws/elbo_nograd; validated
    including the mixed case of an enumerated variable alongside an
    ordinarily K-sampled one in the same model).
    """
