"""Moment estimators: the raw one, and the shrinkage repair.

raw
    mu_hat  = sample mean
    S_hat   = sample covariance (unbiased, ddof = 1)

shrink_S
    Sigma_tilde = (1 - delta) S_hat + delta * (tr(S_hat)/N) I
    with delta the Ledoit-Wolf (2004) optimal intensity for the scaled-identity
    target.  This is the "well-conditioned estimator": every eigenvalue is
    lifted by delta * tr(S)/N, so the smallest one is bounded away from zero by
    construction.

A note on the mean
------------------
The paper also discusses shrinkage of the mean vector towards the cross
sectional mean.  It is not implemented as a separate estimator, on purpose.

With T = 60 the grand mean of the sample means is itself a coin flip: its
standard error is about 1.4e-3 per day against a signal of 2.6e-4, so in
roughly half the samples the estimated grand mean is *negative*.  Shrinking
towards a negative constant flips the sign of the whole portfolio, and the
"repair" produces a book that is short the minimum-variance portfolio.  That is
an artefact of the estimator, not a portfolio.

The equivalent and well-posed operation is shrinkage in *weight* space -- a
convex blend of the tangency portfolio and the minimum-variance portfolio,
which is what mean shrinkage accomplishes (Jagannathan & Ma 2003: constraining
weights and shrinking means are the same thing).  It is implemented in
``portfolio.blend`` and swept in e4.
"""

from __future__ import annotations

import numpy as np

from . import config as C


# --------------------------------------------------------------------------
# raw moments
# --------------------------------------------------------------------------
def sample_moments(returns: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sample mean and unbiased sample covariance of a T x N panel."""
    mu = returns.mean(axis=0)
    cov = np.cov(returns, rowvar=False, ddof=1)
    cov = np.atleast_2d(cov)
    return mu, cov


# --------------------------------------------------------------------------
# Ledoit-Wolf shrinkage intensity
# --------------------------------------------------------------------------
def ledoit_wolf_delta(returns: np.ndarray) -> float:
    """Optimal shrinkage intensity towards the scaled identity target.

    Implements Ledoit & Wolf (2004), "A well-conditioned estimator for
    large-dimensional covariance matrices", for the case F = m I with
    m = tr(S)/N.  The estimator of the optimal intensity is

        delta* = clip( (pi_hat - rho_hat) / (T * gamma_hat), 0, 1 )

        pi_hat    = sum_ij (1/T) sum_t (x_ti x_tj - s_ij)^2
        rho_hat   = sum_i  (1/T) sum_t (x_ti^2   - s_ii)^2
        gamma_hat = || S - m I ||_F^2

    Note the MLE convention S = X'X/T inside this function, as in the paper;
    the delta that comes out is then applied to whatever covariance you pass.
    """
    t, n = returns.shape
    x = returns - returns.mean(axis=0)
    s = x.T @ x / t                      # MLE covariance, as in the paper
    m = np.trace(s) / n

    gamma = float(np.sum((s - m * np.eye(n)) ** 2))
    if gamma <= 0.0:
        return 0.0

    x2 = x**2
    pi_mat = x2.T @ x2 / t - s**2        # elementwise
    pi_hat = float(pi_mat.sum())
    rho_hat = float(np.trace(pi_mat))

    delta = (pi_hat - rho_hat) / (t * gamma)
    return float(min(1.0, max(0.0, delta)))


def shrink_cov(cov: np.ndarray, delta: float) -> np.ndarray:
    """(1 - delta) S + delta * (tr(S)/N) I."""
    n = cov.shape[0]
    m = np.trace(cov) / n
    return (1.0 - delta) * cov + delta * m * np.eye(n)


# --------------------------------------------------------------------------
# convenience: the two named estimators
# --------------------------------------------------------------------------
def estimate(returns: np.ndarray, kind: str) -> dict:
    """Return a dict with keys mu, cov, delta for the requested estimator."""
    mu_hat, s_hat = sample_moments(returns)

    if kind == "raw":
        return {"mu": mu_hat, "cov": s_hat, "delta": 0.0}

    if kind == "shrink_S":
        delta = ledoit_wolf_delta(returns)
        return {"mu": mu_hat, "cov": shrink_cov(s_hat, delta), "delta": delta}

    raise ValueError(f"unknown estimator: {kind}")
