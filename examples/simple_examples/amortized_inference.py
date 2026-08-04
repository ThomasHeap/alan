"""
Amortized (VAE-style) inference in alan.

There's no dedicated "amortized param" API -- it isn't needed. A distribution
argument can be an arbitrary Python function that closes over an nn.Module
and refers to any name in scope, including a BoundPlate `input` -- the same
mechanism already used for e.g. `lambda z, x: z @ x` in the movielens
example. And since BoundPlate is itself an nn.Module, a plain
`Q.encoder = encoder` attribute assignment registers the encoder's own
parameters with ordinary PyTorch submodule tracking, so `Q.parameters()`
picks them up for free -- no separate optimizer needed for the encoder.

The one sharp edge: the encoder's gradients compute whether or not you do
that assignment. Forgetting it doesn't error -- it just silently means the
optimizer never sees (and so never trains) the encoder. If a fit encoder
seems stuck at its random initialization, check this first.
"""
import torch as t
import torch.nn as nn
from alan import Normal, Plate, BoundPlate, Problem, Data, OptParam

d_x, d_z, M = 4, 2, 200

t.manual_seed(0)
W, c, sigma = t.randn(d_z, d_x), t.randn(d_x), 0.5
z_true = t.randn(M, d_z)
x = z_true @ W + c + sigma * t.randn(M, d_x)
x_named = x.rename('plate_1', None)

P = Plate(
    plate_1=Plate(
        z=Normal(t.zeros(d_z), t.ones(d_z)),
        obs=Normal(lambda z: z @ W + c, sigma),
    ),
)
P = BoundPlate(P, {'plate_1': M}, inputs={'x': x_named})

# The encoder maps each row of x directly to that row's z-location -- an
# ordinary nn.Module, applied elementwise over the plate dimension exactly
# like nn.Linear already broadcasts over any leading batch dimension.
encoder = nn.Linear(d_x, d_z)

Q = Plate(
    plate_1=Plate(
        z=Normal(lambda x: encoder(x), OptParam(t.zeros(d_z), transformation=t.exp)),
        obs=Data(),
    ),
)
Q = BoundPlate(Q, {'plate_1': M}, inputs={'x': x_named})
Q.encoder = encoder  # <-- the one thing to remember; see module docstring

prob = Problem(P, Q, {'obs': x_named})
opt = t.optim.Adam(Q.parameters(), lr=0.05)

for i in range(500):
    opt.zero_grad()
    sample = prob.sample(K=5, reparam=True)
    elbo = sample.elbo_vi()
    (-elbo).backward()
    opt.step()
    if i % 100 == 0:
        print(f"Iter {i}. Elbo: {elbo:.3f}")

print("Learned encoder weight:\n", encoder.weight.detach())
print("Learned encoder bias:\n", encoder.bias.detach())
