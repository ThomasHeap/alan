import torch as t
from alan.utils import chain_logmmexp, logmmexp


def test_incremental_cumulative_matches_chain_logmmexp():
    """
    sample_Ks_timeseries's forward filtering pass used to recompute
    chain_logmmexp(lp[:t+1]) from scratch at every timestep t (O(T) calls,
    each over a growing prefix -> O(T^2) total matrix multiplies). It was
    replaced with a single incremental left-to-right scan (cumulative[t] =
    logmmexp(cumulative[t-1], lp[t])), which is O(T) total. Matrix
    multiplication is associative, so both must give the same prefix
    products, up to floating point rounding -- this pins that down so a
    future refactor can't silently reintroduce a numerical discrepancy.
    """
    t.manual_seed(0)
    T, K = 11, 6
    lp = t.randn(T, K, K)

    cumulative = [lp[0]]
    for t_idx in range(1, T):
        cumulative.append(logmmexp(cumulative[-1], lp[t_idx]))

    for t_idx in range(T):
        recomputed = chain_logmmexp(lp[:t_idx + 1])
        assert t.allclose(cumulative[t_idx], recomputed, atol=1e-4), (
            f"mismatch at t_idx={t_idx}: "
            f"max abs diff = {(cumulative[t_idx] - recomputed).abs().max().item()}"
        )
