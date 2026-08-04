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

    Current scope: a single Enumerate() variable per group (not combined
    with other variables), not inside a Timeseries or a Group, and only
    supported by the massively-parallel (Problem.sample) path -- not
    sample_nonmp (fails with a clear AssertionError). Everything built on
    Sample's shared reduction machinery works correctly for an enumerated
    variable, including joint queries with other variables:
    elbo_vi/elbo_rws/elbo_nograd, sample.moments(...), sample.marginals(),
    and sample.importance_sample(...) (validated: draws land within Monte
    Carlo noise of the true posterior).
    """
