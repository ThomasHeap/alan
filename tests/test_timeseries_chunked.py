import torch as t

from alan import Normal, Timeseries, Plate, BoundPlate, Problem, Data, mean
from alan.utils import generic_dims, generic_order

T = 12
A = 0.9
INIT_SCALE = 1.
TRANS_NOISE_SCALE = 0.1
OBS_NOISE_SCALE = 1.


def _build_problem(chunk_size):
    """
    AR(1)-plus-noisy-observation Timeseries problem, parameterised by
    `chunk_size` so the same model can be built chunked or unchunked.
    """
    P = Plate(
        init=Normal(0, INIT_SCALE),
        T=Plate(
            ts=Timeseries("init", Normal(lambda prev: A * prev, TRANS_NOISE_SCALE), chunk_size=chunk_size),
            obs=Normal('ts', OBS_NOISE_SCALE),
        ),
    )
    Q = Plate(
        init=Normal(0, 1),
        T=Plate(
            ts=Normal(0, 1),
            obs=Data(),
        ),
    )

    t.manual_seed(0)
    t1 = t.arange(T)[:, None]
    t2 = t.arange(T)[None, :]
    curr_t = t.min(t1, t2)
    diag_var = INIT_SCALE ** 2
    prior_cov = t.zeros(T, T)
    for i in range(T):
        diag_var = diag_var * A ** 2 + TRANS_NOISE_SCALE ** 2
        prior_cov[i, i] = diag_var
        future_covs = diag_var * A ** t.arange(T - i)
        prior_cov[i, i:] = future_covs
        prior_cov[i:, i] = future_covs
    true_dist = t.distributions.MultivariateNormal(t.zeros(T), prior_cov + OBS_NOISE_SCALE ** 2 * t.eye(T))
    data_ts = true_dist.sample()

    all_platesizes = {'T': T}
    P = BoundPlate(P, all_platesizes)
    Q = BoundPlate(Q, all_platesizes)
    data = {'obs': data_ts.refine_names('T')}
    return Problem(P, Q, data)


def test_chunked_log_prob_matches_unchunked():
    """
    Timeseries(..., chunk_size=k) computes log_prob in checkpointed chunks
    along T instead of in one shot, purely to bound peak autograd memory for
    long timeseries. It must produce numerically identical ELBOs and
    posterior moments to the unchunked (chunk_size=None) path, for both a
    chunk size that evenly divides T and one that leaves a remainder.
    """
    K, n_reps = 20, 4

    results = {}
    for chunk_size in [None, 4, 5]:
        elbos, moments = [], []
        for rep in range(n_reps):
            t.manual_seed(2000 + rep)
            problem = _build_problem(chunk_size)
            sample = problem.sample(K=K, reparam=False)
            elbos.append(sample.elbo_nograd().item())
            imp = sample.importance_sample(200)
            est = imp._moments('ts', mean)
            moments.append(generic_order(est, generic_dims(est)))
        results[chunk_size] = (t.tensor(elbos), t.stack(moments))

    base_elbo, base_mom = results[None]
    for chunk_size in [4, 5]:
        elbo, mom = results[chunk_size]
        assert t.allclose(elbo, base_elbo, atol=1e-4), (
            f"chunk_size={chunk_size}: elbo max abs diff = {(elbo - base_elbo).abs().max().item()}"
        )
        assert t.allclose(mom, base_mom, atol=1e-4), (
            f"chunk_size={chunk_size}: moment max abs diff = {(mom - base_mom).abs().max().item()}"
        )


def test_chunked_log_prob_checkpoint_recomputes_during_backward():
    """
    The whole point of chunking is that torch.utils.checkpoint recomputes
    each chunk's forward pass during backward instead of retaining all of
    them -- otherwise chunk_size would just be dead weight. Confirm that by
    counting calls to the transition distribution's log_prob: with 3 chunks,
    forward should trigger exactly 3 calls, and a subsequent backward should
    trigger 3 more (one recompute per chunk).
    """
    from functorch.dim import dims
    from alan.dist import Dist

    T_dim, K_dim, Kinit_dim = dims(3)
    T_dim.size, K_dim.size, Kinit_dim.size = 9, 5, 5

    trans = Normal(lambda prev: A * prev, TRANS_NOISE_SCALE)
    ts = Timeseries("init", trans, chunk_size=4)

    init = t.randn(5, requires_grad=True)[Kinit_dim]
    sample = t.randn(9, 5, requires_grad=True)[T_dim, K_dim]

    call_count = {'n': 0}
    orig_log_prob = Dist.log_prob

    def counting_log_prob(self, *args, **kwargs):
        call_count['n'] += 1
        return orig_log_prob(self, *args, **kwargs)

    Dist.log_prob = counting_log_prob
    try:
        lp, _ = ts.log_prob(sample, {'init': init}, T_dim, K_dim)
        calls_after_forward = call_count['n']

        total = generic_order(lp, generic_dims(lp)).sum()
        total.backward()
        calls_after_backward = call_count['n']
    finally:
        Dist.log_prob = orig_log_prob

    assert calls_after_forward == 3, f"expected 3 chunks -> 3 calls, got {calls_after_forward}"
    assert calls_after_backward == 6, (
        f"expected 3 more recomputed calls during backward (6 total), got {calls_after_backward}"
    )
