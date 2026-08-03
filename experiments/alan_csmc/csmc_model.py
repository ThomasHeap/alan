"""
Same toy linear-Gaussian state-space model and parameters as
experiments/particle_gibbs/ssm_model.py (A=0.9, Q_VAR=0.3, R_VAR=0.5, P0=1.0,
T=20) -- reused deliberately, so the results here are directly comparable to
that experiment's already-validated hand-rolled CSMC/PGAS numbers (RMSE and
chain-ESS tables in experiments/particle_gibbs/README.md). This file just
duplicates the closed-form Kalman filter / RTS smoother ground truth; see
ssm_model.py for the derivation/citation.
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
    m_filt, P_filt = t.zeros(T + 1), t.zeros(T + 1)
    m_pred, P_pred = t.zeros(T + 1), t.zeros(T + 1)
    m_filt[0], P_filt[0] = 0.0, P0

    for tt in range(1, T + 1):
        m_pred[tt] = A * m_filt[tt - 1]
        P_pred[tt] = A ** 2 * P_filt[tt - 1] + Q_VAR
        S = P_pred[tt] + R_VAR
        innovation = y[tt - 1] - m_pred[tt]
        K_gain = P_pred[tt] / S
        m_filt[tt] = m_pred[tt] + K_gain * innovation
        P_filt[tt] = (1 - K_gain) * P_pred[tt]

    return m_filt, P_filt, m_pred, P_pred


def rts_smoother(m_filt, P_filt, m_pred, P_pred, T=T_STEPS):
    m_smooth, P_smooth = t.zeros(T + 1), t.zeros(T + 1)
    m_smooth[T], P_smooth[T] = m_filt[T], P_filt[T]

    for tt in range(T - 1, -1, -1):
        C = P_filt[tt] * A / P_pred[tt + 1]
        m_smooth[tt] = m_filt[tt] + C * (m_smooth[tt + 1] - m_pred[tt + 1])
        P_smooth[tt] = P_filt[tt] + C ** 2 * (P_smooth[tt + 1] - P_pred[tt + 1])

    return m_smooth, P_smooth


def exact_posterior(y, T=T_STEPS):
    m_filt, P_filt, m_pred, P_pred = kalman_filter(y, T)
    return rts_smoother(m_filt, P_filt, m_pred, P_pred, T)
