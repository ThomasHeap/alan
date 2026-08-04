"""
Combines unnamed batch-dimensions with complex multivariate distribution (MultivariateNormal).

Note that the matrix algebra for computing the true posterior mean requires a bit of rearranging,
so e.g. 
prior_mean  is of shape [N, F], and goes into the probabilistic program, while
prior_mean_ is of shape [N, F, 1], and is used to compute the posterior mean
"""

import torch as t
from alan import Bernoulli, Beta, Plate, BoundPlate, Group, Problem, Data, mean, mean2, MultivariateNormal
from TestProblem import TestProblem

#Fixed seed so this test problem's random model (including Q's proposal
#covariance ap_cov, generated the same random way as the true posterior
#covariance) is deterministic across runs -- unseeded, an unlucky draw could
#make ap_cov a poor match for the true posterior, inflating importance-
#sampling variance well past what the test's fixed sigma margins assume.
#Module-level t.randn otherwise depends on whatever global RNG state exists
#at import time, which varies with test execution order -- e.g. under
#pytest-xdist.
t.manual_seed(19)

N = 3
F = 2
prior_mean = t.randn(N, F)
prior_mean_ = prior_mean[..., None]
A = t.randn(N, F, F)
prior_cov = A @ A.mT
prior_prec = t.inverse(prior_cov)

ap_mean = t.randn(N, F)
B = t.randn(N, F, F)
ap_cov = B@B.mT + 2*t.eye(F)

C = t.randn(N, F, F)
like_cov = C @ C.mT
like_prec = t.inverse(like_cov)

data = 1.5+t.randn(N, F)
data_ = data[..., None]
post_prec = prior_prec + like_prec
post_cov = t.inverse(post_prec)
post_mean_ = post_cov @ (prior_prec@prior_mean_ + like_prec@data_)
post_mean = post_mean_.squeeze(-1)

P = Plate(
    a = MultivariateNormal(prior_mean, prior_cov),
    d = MultivariateNormal('a', like_cov),
)

Q = Plate(
    a = MultivariateNormal(ap_mean, ap_cov),
    d = Data(),
)

all_platesizes = {}
P = BoundPlate(P, all_platesizes)
Q = BoundPlate(Q, all_platesizes)

data = {'d': data}

moments = [('a', mean)]
known_moments = {
    ('a', mean): post_mean,
}

tp = TestProblem(
    P, Q, data,
    moments, 
    known_moments=known_moments, 
    moment_K=1000000
)
