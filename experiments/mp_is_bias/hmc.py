"""A minimal, hand-rolled HMC sampler (leapfrog + Metropolis accept), used as an
independent ground-truth baseline -- no blackjax/numpyro dependency needed."""
import torch as t
from compare_model import log_joint, P_SIZE, analytic_posterior


def leapfrog(z, p, eps, L):
    z = z.clone().requires_grad_(True)
    lp = log_joint(z)
    g = t.autograd.grad(lp, z)[0]
    p = p + 0.5 * eps * g
    for i in range(L):
        z = (z + eps * p).detach().requires_grad_(True)
        lp = log_joint(z)
        g = t.autograd.grad(lp, z)[0]
        if i != L - 1:
            p = p + eps * g
    p = p + 0.5 * eps * g
    return z.detach(), p


def hmc(n_samples, burn_in, eps=0.15, L=20, seed=0):
    t.manual_seed(seed)
    z = t.zeros(P_SIZE + 1)
    samples = []
    n_accept = 0
    n_total = n_samples + burn_in
    for it in range(n_total):
        p0 = t.randn(P_SIZE + 1)
        z_new, p_new = leapfrog(z, p0, eps, L)
        cur_h = -log_joint(z) + 0.5 * (p0 ** 2).sum()
        new_h = -log_joint(z_new) + 0.5 * (p_new ** 2).sum()
        if t.log(t.rand(())) < (cur_h - new_h):
            z = z_new
            n_accept += 1
        if it >= burn_in:
            samples.append(z.clone())
    samples = t.stack(samples)
    accept_rate = n_accept / n_total
    return samples, accept_rate


if __name__ == "__main__":
    samples, acc = hmc(n_samples=20000, burn_in=2000)
    mean, cov = analytic_posterior()
    print(f"HMC accept rate: {acc:.2f}")
    print("HMC posterior mean:   ", samples.mean(0))
    print("analytic posterior mean:", mean)
    print("HMC posterior sd:     ", samples.std(0))
    print("analytic posterior sd:  ", cov.diag().sqrt())
    print("max abs mean error:", (samples.mean(0) - mean).abs().max().item())
