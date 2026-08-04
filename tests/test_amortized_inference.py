"""
Validates that amortized (VAE-style) inference works in alan, using only
existing, ordinary mechanisms -- no dedicated "amortized param" API exists,
or is needed:

  * Distribution arguments already accept an arbitrary Python function that
    closes over an nn.Module and references any name in scope (including a
    BoundPlate `inputs` tensor) -- exactly how e.g. the movielens example's
    `lambda z, x: z @ x` already works.
  * Gradients flow through that network into the ELBO via ordinary autograd.
  * The network's parameters are picked up by `Q.parameters()` for free via
    plain `Q.some_name = some_nn_module` attribute assignment, since
    BoundPlate is itself an nn.Module (standard PyTorch submodule
    registration) -- but ONLY if you remember to do that assignment; the
    encoder's gradients still compute either way, so a forgotten assignment
    is a silent "the encoder never trains" bug, not a loud error.

Ground truth: a linear-Gaussian VAE (fixed, known W, c, sigma), for which the
true amortized posterior q(z_m | x_m) is EXACTLY affine in x_m -- so a bare
nn.Linear encoder is the exactly-correct family, and training via ordinary
reparameterised VI should recover the closed-form affine map.

Only plain nn.Linear is exercised here. `x` inside the encoder lambda is
actually a functorch.dim tensor, not a plain torch.Tensor -- elementwise ops
and nn.Linear/nn.Sequential work via functorch.dim's operator dispatch, but
anything using `.shape`/`.view()`/`.reshape()` directly does not, since a
plate dimension isn't an ordinary positional axis there. See
examples/simple_examples/amortized_inference.py's docstring for the
workaround (`x.order(*x.dims)` / rewrap with `[dims]`), if you need that.
"""
import torch as t
import torch.nn as nn

from alan import Normal, Plate, BoundPlate, Problem, Data, OptParam


def _closed_form_encoder(W, c, sigma, d_z):
    """True q(z_m | x_m) = N(A x_m + b, Sigma) for this additive linear-Gaussian
    observation model -- Sigma is the same for every m (doesn't depend on x_m)."""
    prior_prec = t.eye(d_z)
    lik_prec = (W @ W.T) / sigma ** 2
    post_prec = prior_prec + lik_prec
    post_cov = t.linalg.inv(post_prec)
    A = post_cov @ W / sigma ** 2
    b = -A @ c
    return A, b, post_cov


def _build_problem(x, x_named, W, c, sigma, d_z, encoder, platesize):
    P = Plate(
        plate_1=Plate(
            z=Normal(t.zeros(d_z), t.ones(d_z)),
            obs=Normal(lambda z: z @ W + c, sigma),
        ),
    )
    P = BoundPlate(P, {'plate_1': platesize}, inputs={'x': x_named})

    Q = Plate(
        plate_1=Plate(
            z=Normal(lambda x: encoder(x), OptParam(t.zeros(d_z), transformation=t.exp)),
            obs=Data(),
        ),
    )
    Q = BoundPlate(Q, {'plate_1': platesize}, inputs={'x': x_named})

    return P, Q


def test_amortized_encoder_matches_closed_form_posterior():
    t.manual_seed(0)
    d_x, d_z, M = 4, 2, 200

    W = t.randn(d_z, d_x)
    c = t.randn(d_x)
    sigma = 0.5

    z_true = t.randn(M, d_z)
    x = z_true @ W + c + sigma * t.randn(M, d_x)
    x_named = x.rename('plate_1', None)

    A_true, b_true, post_cov = _closed_form_encoder(W, c, sigma, d_z)

    encoder = nn.Linear(d_x, d_z)
    P, Q = _build_problem(x, x_named, W, c, sigma, d_z, encoder, M)
    Q.encoder = encoder  # register so Q.parameters() includes the encoder's weights

    prob = Problem(P, Q, {'obs': x_named})
    opt = t.optim.Adam(Q.parameters(), lr=0.05)

    t.manual_seed(1)
    for _ in range(500):
        opt.zero_grad()
        sample = prob.sample(K=5, reparam=True)
        elbo = sample.elbo_vi()
        (-elbo).backward()
        opt.step()

    assert (encoder.weight.detach() - A_true).abs().max().item() < 0.1
    assert (encoder.bias.detach() - b_true).abs().max().item() < 0.1

    # learned (per-row, free) scale should be in the right ballpark of the
    # true (shared) posterior std -- loose tolerance, since each row is its
    # own free parameter fit from a single (K=5)-particle stochastic ELBO.
    learned_std = list(Q.opt_params().values())[0].rename(None)
    true_std = post_cov.diag().sqrt()
    assert (learned_std.mean(0) - true_std).abs().max().item() < 0.15


def test_amortized_encoder_gradients_flow_without_registration():
    """Gradients into the encoder compute regardless of whether it's
    registered on Q -- registration only controls whether the optimizer
    (via Q.parameters()) actually sees and updates those gradients. This is
    the sharp edge: a forgotten `Q.encoder = encoder` doesn't error, it just
    silently never trains the encoder."""
    t.manual_seed(0)
    d_x, d_z, M = 3, 2, 20

    W = t.randn(d_z, d_x)
    c = t.randn(d_x)
    sigma = 0.5
    x = t.randn(M, d_x)
    x_named = x.rename('plate_1', None)

    encoder = nn.Linear(d_x, d_z)
    P, Q = _build_problem(x, x_named, W, c, sigma, d_z, encoder, M)
    # deliberately NOT registered: Q.encoder = encoder

    assert not any(p is enc_p for p in Q.parameters() for enc_p in encoder.parameters())

    prob = Problem(P, Q, {'obs': x_named})
    sample = prob.sample(K=5, reparam=True)
    elbo = sample.elbo_vi()
    (-elbo).backward()

    assert encoder.weight.grad is not None  # gradient exists...
    assert not any(p is enc_p for p in Q.parameters() for enc_p in encoder.parameters())  # ...but optimizer would never see it
