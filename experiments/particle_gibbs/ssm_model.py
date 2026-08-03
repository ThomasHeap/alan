"""
Toy linear-Gaussian state-space model (scalar AR(1) + noisy observation), used
to validate a Particle Gibbs prototype. Chosen specifically because it's a
genuinely SEQUENTIAL model (unlike the flat 2-level hierarchical model in
experiments/mp_is_bias/) -- Particle Gibbs / conditional SMC is a scheme about
resampling *across time steps* with a pinned reference trajectory, which only
means something when there's a sequence to resample across. It's also
linear-Gaussian, so the exact posterior (both filtering AND full-trajectory
smoothing) is available in closed form via the Kalman filter / RTS smoother --
no need for an approximate ground truth.

    x_0 ~ N(0, P0)
    x_t = A x_{t-1} + eps_t,   eps_t ~ N(0, Q),   t = 1..T
    y_t = x_t + eta_t,         eta_t ~ N(0, R),   t = 1..T   (observed)

Particle Gibbs targets p(x_0:T | y_1:T), the FULL joint smoothing posterior --
not just the per-t filtering marginals -- so the ground truth to validate
against is the RTS-smoothed mean/var at every t, not the forward-filtered one.
"""
import torch as t

A, Q_VAR, R_VAR, P0 = 0.9, 0.3, 0.5, 1.0
T_STEPS = 20


def simulate(T=T_STEPS, seed=0):
    g = t.Generator().manual_seed(seed)
    x = t.zeros(T + 1)
    x[0] = P0 ** 0.5 * t.randn(1, generator=g)
    for tt in range(1, T + 1):
        x[tt] = A * x[tt - 1] + Q_VAR ** 0.5 * t.randn(1, generator=g)
    y = x[1:] + R_VAR ** 0.5 * t.randn(T, generator=g)
    return x, y


def kalman_filter(y, T=T_STEPS):
    """Forward filter. Returns (m_filt, P_filt) at t=0..T, (m_pred, P_pred) at
    t=1..T, and the exact log marginal likelihood log p(y_1:T)."""
    m_filt = t.zeros(T + 1)
    P_filt = t.zeros(T + 1)
    m_pred = t.zeros(T + 1)   # index 0 unused
    P_pred = t.zeros(T + 1)
    m_filt[0], P_filt[0] = 0.0, P0

    log_lik = 0.0
    for tt in range(1, T + 1):
        m_pred[tt] = A * m_filt[tt - 1]
        P_pred[tt] = A ** 2 * P_filt[tt - 1] + Q_VAR

        S = P_pred[tt] + R_VAR
        innovation = y[tt - 1] - m_pred[tt]
        K_gain = P_pred[tt] / S

        m_filt[tt] = m_pred[tt] + K_gain * innovation
        P_filt[tt] = (1 - K_gain) * P_pred[tt]

        log_lik += (-0.5 * t.log(2 * t.pi * S) - 0.5 * innovation ** 2 / S).item()

    return m_filt, P_filt, m_pred, P_pred, log_lik


def rts_smoother(m_filt, P_filt, m_pred, P_pred, T=T_STEPS):
    """Backward RTS smoother. Returns exact E[x_t | y_1:T], Var[x_t | y_1:T] for t=0..T."""
    m_smooth = t.zeros(T + 1)
    P_smooth = t.zeros(T + 1)
    m_smooth[T], P_smooth[T] = m_filt[T], P_filt[T]

    for tt in range(T - 1, -1, -1):
        C = P_filt[tt] * A / P_pred[tt + 1]
        m_smooth[tt] = m_filt[tt] + C * (m_smooth[tt + 1] - m_pred[tt + 1])
        P_smooth[tt] = P_filt[tt] + C ** 2 * (P_smooth[tt + 1] - P_pred[tt + 1])

    return m_smooth, P_smooth


def exact_posterior(y, T=T_STEPS):
    m_filt, P_filt, m_pred, P_pred, log_lik = kalman_filter(y, T)
    m_smooth, P_smooth = rts_smoother(m_filt, P_filt, m_pred, P_pred, T)
    return m_smooth, P_smooth, log_lik


if __name__ == "__main__":
    x_true, y = simulate()
    m_smooth, P_smooth, log_lik = exact_posterior(y)
    print("true x:   ", x_true)
    print("smoothed mean:", m_smooth)
    print("smoothed sd:  ", P_smooth.sqrt())
    print("log p(y_1:T) =", log_lik)
    max_abs_err = (m_smooth - x_true).abs().max().item()
    print(f"max |smoothed_mean - true_x| = {max_abs_err:.4f}  (sanity check, not a validation --")
    print("the smoother recovers the POSTERIOR mean given y, not the true simulated x)")
