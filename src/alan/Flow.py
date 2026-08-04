from typing import Optional

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
    (sample/log_prob/sample_extended/predictive_ll/all_args/opt_qem_params/
    qem_dist), so it's usable anywhere those are, in either P or Q, and
    through every downstream path that consumes them: elbo_vi/elbo_rws
    (massively-parallel and sample_nonmp alike), sample.moments,
    sample.importance_sample, and ImportanceSample.extend/predictive_ll.
    There's no dedicated "Flow slot" the way Enumerate needs one, since a
    Flow IS a genuine, normalisable distribution.

    Example -- a log-normal-ish Q for a strictly-positive latent. Flow's own
    parameters are plain nn.Parameters (no OptParam integration -- see
    below), registered the same way amortized_inference.py registers an
    encoder: plain `Q.some_name = some_parameter` attribute assignment, so
    `Q.parameters()` picks up its gradient. Apply any transform (like .exp(),
    for positivity) via a zero-argument lambda, not pre-computed -- Dist's
    machinery only recomputes fresh on every call for scope-lookups/lambdas,
    not for a value computed once before construction (a pre-computed
    `q_log_sigma.exp()` bakes a stale autograd graph node from construction
    time, breaking training after the first backward pass).

    .. code-block:: python

       q_mu, q_log_sigma = nn.Parameter(t.tensor(0.)), nn.Parameter(t.tensor(0.))
       Q = Plate(
           z = Flow(Normal(q_mu, lambda: q_log_sigma.exp()), [ExpTransform()]),
           ...
       )
       Q = BoundPlate(Q, ...)
       Q.q_mu, Q.q_log_sigma = q_mu, q_log_sigma

    Current scope (a first, minimal pass): elementwise transforms only (no
    multivariate transforms needing a non-trivial Jacobian determinant);
    transform/base-distribution parameters are plain nn.Parameters, not
    OptParam/QEMParam (OptParam/QEMParam need to know their own variable's
    name at construction time via `.finalize(varname)`, which Flow, like
    Timeseries, doesn't have -- both instead finalize their base_dist with
    `.finalize(None)`, which OptParam/QEMParam explicitly reject); not
    supported inside a Timeseries or a Group.

    Note: ImportanceSample.extend()/predictive_ll() have a separate,
    pre-existing limitation unrelated to Flow -- they don't handle a bare,
    top-level scalar Data() observation (no enclosing Plate at all), since
    corresponding_plates() assumes every sample carries torchdim .dims, but
    a genuinely 0-dim scalar tensor has none to carry. Confirmed identically
    reproducible with an ordinary, Flow-free Dist. Works fine for the
    realistic case of a Data() observation living inside a Plate.
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

    def sample_extended(
            self,
            sample,
            name:Optional[str],
            scope:dict,
            inputs_params:dict,
            original_platedims:dict[str, Dim],
            extended_platedims:dict[str, Dim],
            active_extended_platedims:list[Dim],
            Ndim:Dim,
            reparam:bool,
            original_data:Optional[dict]):
        #base_dist.sample_extended splices the *given* original-space sample
        #into a freshly-drawn extended-size tensor at the right plate
        #indices. sample (and original_data[name]) are here in z-space (the
        #Flow's own, transformed space), not base_dist's z0-space, so invert
        #first -- the splicing indices are unaffected by an elementwise,
        #monotonic transform, only the values are, so delegating the actual
        #plate-extension/splicing entirely to base_dist and transforming the
        #result forward again is exact, not an approximation.
        original_sample = sample if sample is not None else original_data[name]
        z0 = original_sample
        for transform in reversed(self.transforms):
            z0, _ = transform.inverse(z0)

        extended_z0 = self.base_dist.sample_extended(
            sample=z0,
            name=name,
            scope=scope,
            inputs_params=inputs_params,
            original_platedims=original_platedims,
            extended_platedims=extended_platedims,
            active_extended_platedims=active_extended_platedims,
            Ndim=Ndim,
            reparam=reparam,
            original_data=original_data,
        )

        extended_sample = extended_z0
        for transform in self.transforms:
            extended_sample, _ = transform.forward(extended_sample)
        return extended_sample

    def predictive_ll(
            self,
            sample:dict,
            name:Optional[str],
            scope:dict,
            inputs_params:dict,
            original_platedims:dict[str, Dim],
            extended_platedims:dict[str, Dim],
            original_data:dict,
            extended_data:dict):
        #Identical in structure to Dist.predictive_ll -- self.log_prob
        #already handles the change-of-variables correctly, so there's
        #nothing Flow-specific needed beyond calling it.
        original_ll, extended_ll = {}, {}

        if name in extended_data.keys():
            extended_ll[name], _ = self.log_prob(extended_data[name], scope, None, None)

            original_dims, extended_dims = corresponding_plates(original_platedims, extended_platedims, original_data[name], extended_data[name])

            original_idxs = [slice(0, dim.size) for dim in original_dims]
            original_ll[name] = generic_getitem(generic_order(extended_ll[name], extended_dims), original_idxs)
            original_ll[name] = generic_getitem(original_ll[name], original_dims)

        return original_ll, extended_ll
