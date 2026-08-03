"""
Shared model definition: a small hierarchical Gaussian model with a closed-form
posterior, used to compare alan's MP-IS / Global-IS against HMC and the exact answer.

    mu ~ N(0, s0^2)
    theta_i | mu ~ N(mu, tau^2),  i = 1..P
    x_i | theta_i ~ N(theta_i, sigma^2)   (observed)

Because everything is linear-Gaussian, the joint posterior over (mu, theta_1..P)
is exactly multivariate normal and can be computed via the Hessian of the
negative log joint (constant everywhere, since it's an exact quadratic).
"""
import torch as t

t.manual_seed(0)

P_SIZE = 6
S0, TAU, SIGMA = 1.0, 1.0, 0.7

mu_true = 0.5
theta_true = mu_true + TAU * t.randn(P_SIZE)
x_data = theta_true + SIGMA * t.randn(P_SIZE)


def log_joint(z):
    """z = [mu, theta_1, ..., theta_P]"""
    mu = z[0]
    theta = z[1:]
    lp = -0.5 * (mu / S0) ** 2
    lp = lp - 0.5 * (((theta - mu) / TAU) ** 2).sum()
    lp = lp - 0.5 * (((x_data - theta) / SIGMA) ** 2).sum()
    return lp


def analytic_posterior():
    """Exact posterior mean + covariance via the Hessian of -log_joint (constant, since quadratic)."""
    z0 = t.zeros(P_SIZE + 1, requires_grad=True)
    lp = log_joint(z0)
    g = t.autograd.grad(lp, z0, create_graph=True)[0]
    H = t.stack([t.autograd.grad(g[i], z0, retain_graph=True)[0] for i in range(P_SIZE + 1)])
    precision = -H
    cov = t.inverse(precision)
    # Newton step from 0 (exact for a quadratic): mean = cov @ grad(lp)(0)
    z0b = t.zeros(P_SIZE + 1, requires_grad=True)
    lp0 = log_joint(z0b)
    grad0 = t.autograd.grad(lp0, z0b)[0]
    mean = cov @ grad0
    return mean.detach(), cov.detach()


if __name__ == "__main__":
    mean, cov = analytic_posterior()
    print("x_data:", x_data)
    print("analytic posterior mean [mu, theta_1..P]:", mean)
    print("analytic posterior sd   [mu, theta_1..P]:", cov.diag().sqrt())
