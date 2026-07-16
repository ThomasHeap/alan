import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import torch as t
from alan import Normal, Plate, BoundPlate, Group, Problem, Data, PermutationSampler, mean as MEAN

from compare_model import P_SIZE, S0, TAU, SIGMA, x_data, analytic_posterior

platesizes = {"p1": P_SIZE}
data = {"x": x_data.clone().rename("p1")}


def build_problem():
    P = Plate(
        mu=Normal(0.0, S0),
        p1=Plate(
            theta=Normal("mu", TAU),
            x=Normal("theta", SIGMA),
        ),
    )
    P = BoundPlate(P, platesizes)

    # Factorised proposal = independent standard normal per latent (ignores the
    # generative dependency structure), matching the "one-shot" IS methodology
    # in both alan papers' experiments (Q is deliberately a poor/simple proposal).
    Q = Plate(
        mu=Normal(0.0, 1.0),
        p1=Plate(
            theta=Normal(0.0, 1.0),
            x=Data(),
        ),
    )
    Q = BoundPlate(Q, platesizes)

    return Problem(P, Q, data)


def mp_is_moments(prob, K, seed):
    t.manual_seed(seed)
    sample = prob.sample(K, reparam=False, sampler=PermutationSampler)
    mu_mean = sample.moments("mu", MEAN)
    theta_mean = sample.moments("theta", MEAN)
    return mu_mean.rename(None).item(), theta_mean.rename(None)


def global_is_moments(prob, K, seed):
    t.manual_seed(seed)
    sample = prob.sample_nonmp(K, reparam=False)
    mu_mean = sample.moments("mu", MEAN)
    theta_mean = sample.moments("theta", MEAN)
    return mu_mean.rename(None).item(), theta_mean.rename(None)


if __name__ == "__main__":
    prob = build_problem()
    mean, cov = analytic_posterior()
    true_mu, true_theta = mean[0].item(), mean[1:]
    print("true mu:", true_mu)
    print("true theta:", true_theta)

    print("\n--- quick smoke test, K=5 ---")
    print("MP-IS:", mp_is_moments(prob, 5, seed=0))
    print("Global IS:", global_is_moments(prob, 5, seed=0))
