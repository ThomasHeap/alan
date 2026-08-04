import torch as t
import torch.nn as nn

from .utils import *
from .dist import Dist, _Dist


class Transform(nn.Module):
    """
    Base class for an invertible, elementwise transform usable in a Flow.

    Subclasses implement `forward`/`inverse`, each returning (value,
    log_det_jacobian): forward(x) -> (y=f(x), log|dy/dx|), and
    inverse(y) -> (x=f^-1(y), log|dx/dy|) (== -log|dy/dx| evaluated at that
    same x, by the inverse function theorem). Parameters (e.g. an
    AffineTransform's loc/scale) are plain nn.Parameters -- register a
    Transform (or a Flow containing it) on a BoundPlate via ordinary
    `Q.something = transform` attribute assignment for it to train (see
    alan's amortized-inference pattern; there's no OptParam integration
    here in this first pass).
    """
    def forward(self, x):
        raise NotImplementedError

    def inverse(self, y):
        raise NotImplementedError


class AffineTransform(Transform):
    """y = loc + scale*x. scale must be strictly positive."""
    def __init__(self, loc, scale):
        super().__init__()
        self.loc = loc
        self.scale = scale

    def forward(self, x):
        return self.loc + self.scale * x, self.scale.abs().log() if isinstance(self.scale, Tensor) else math.log(abs(self.scale))

    def inverse(self, y):
        log_scale = self.scale.abs().log() if isinstance(self.scale, Tensor) else math.log(abs(self.scale))
        return (y - self.loc) / self.scale, -log_scale


class ExpTransform(Transform):
    """y = exp(x); maps the real line to the positive reals."""
    def forward(self, x):
        return x.exp(), x

    def inverse(self, y):
        log_y = y.log()
        return log_y, -log_y


class SigmoidTransform(Transform):
    """y = sigmoid(x); maps the real line to (0, 1)."""
    def forward(self, x):
        y = x.sigmoid()
        return y, y.log() + (1 - y).log()

    def inverse(self, y):
        x = y.log() - (1 - y).log()  # logit(y)
        return x, -(y.log() + (1 - y).log())


class Flow(nn.Module):
    """
    alan.Flow(base_dist, transforms)

    A distribution built by sampling from `base_dist` (an alan Dist, e.g.
    `Normal(0., 1.)`) and pushing the sample through a chain of invertible
    `transforms`, applied in list order. log_prob is computed exactly via
    the change-of-variables formula (inverting the chain, accumulating each
    step's log-det-Jacobian) -- no approximation, same exactness class as
    any other Dist's log_prob.

    Duck-types the same interface Dist/Timeseries already implement
    (sample/log_prob/all_args/opt_qem_params/qem_dist), so it's usable
    anywhere those are, in either P or Q -- there's no dedicated "Flow slot"
    the way Enumerate needs one, since a Flow IS a genuine, normalisable
    distribution.

    Example -- a log-normal-ish Q for a strictly-positive latent:

    .. code-block:: python

       Q = Plate(
           z = Flow(Normal(OptParam(0.), OptParam(1., transformation=t.exp)), [ExpTransform()]),
           ...
       )

    Current scope (a first, minimal pass): elementwise transforms only (no
    multivariate transforms needing a non-trivial Jacobian determinant);
    transform parameters are plain nn.Parameters, not OptParam/QEMParam (no
    automatic registration -- register via `Q.some_name = transform`,
    mirroring alan's amortized-inference pattern); not supported inside a
    Timeseries or a Group; not yet wired into sample_nonmp,
    importance_sample, or predictive_ll/extend.
    """
    def __init__(self, base_dist, transforms):
        super().__init__()
        assert isinstance(base_dist, (Dist, _Dist))
        assert isinstance(transforms, (list, tuple)) and 0 < len(transforms)
        for tr in transforms:
            assert isinstance(tr, Transform)

        #base_dist arrives unfinalized (a _Dist) when constructed inline as
        #Flow(Normal(...), [...]), same as Timeseries.__init__'s trans.
        if isinstance(base_dist, _Dist):
            base_dist = base_dist.finalize(None)

        self.is_timeseries = False
        self.qem_dist = False
        self.base_dist = base_dist
        self.transforms = nn.ModuleList(transforms)
        self.all_args = base_dist.all_args

    @property
    def opt_qem_params(self):
        return {}

    @property
    def device(self):
        return self.base_dist.device

    def sample(self, scope, reparam: bool, active_platedims:list[Dim], K_dim:Dim, timeseries_perm):
        z = self.base_dist.sample(scope, reparam, active_platedims, K_dim, timeseries_perm)
        for transform in self.transforms:
            z, _ = transform.forward(z)
        return z

    def log_prob(self, sample, scope: dict, T_dim, K_dim):
        z = sample
        total_log_det = 0.
        for transform in reversed(self.transforms):
            z, log_det = transform.inverse(z)
            total_log_det = total_log_det + log_det

        lp_base, Kinit_dim = self.base_dist.log_prob(z, scope=scope, T_dim=T_dim, K_dim=K_dim)
        return lp_base + total_log_det, Kinit_dim
