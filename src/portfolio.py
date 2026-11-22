"""Portfolio constructions.

Every constructor below returns the *direction* of the portfolio -- the raw
solution of the linear system -- and nothing else.  Scaling to a risk budget is
a separate, explicit step (``to_vol_target``), because mixing the two is what
produces the usual confusion about what "leverage" means.

    tangency(mu, cov)   x = Sigma^-1 mu
                        the maximum-Sharpe direction.  With estimated moments
                        it is the object the whole paper is about.

    gmv(cov)            x = Sigma^-1 1
                        the global minimum-variance direction.  It uses no mean
                        at all -- which makes it the clean control when we want
                        to isolate the effect of Sigma's conditioning.

    blend(w1, w2, d)    (1-d) w1 + d w2 after normalising both to unit length.
                        Shrinking the *portfolio* rather than the moments; the
                        d -> 1 end point is the minimum-variance portfolio.

    long_only_gmv(cov)  the same objective with w >= 0, 1'w = 1.  Solved by
                        accelerated projected gradient on the simplex.

Note the absence of a "force 1'w = 1" convention for the first two.  Dividing
by 1'Sigma^-1 mu looks harmless but that scalar wanders through zero, so the
resulting leverage is dominated by whichever sample happens to make it small.
Risk targeting is the well-posed alternative and it is what the experiments use.
"""

from __future__ import annotations

import numpy as np

from . import config as C


# --------------------------------------------------------------------------
# linear solves
# --------------------------------------------------------------------------
def solve_spd(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve a x = b, falling back to the pseudo-inverse if a is singular.

    We deliberately do NOT add a ridge here.  When the matrix is nearly
    singular the solve returns a huge answer -- that huge answer *is* the
    phenomenon under study, and hiding it behind a regulariser would defeat
    the purpose of the experiment.
    """
    try:
        return np.linalg.solve(a, b)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(a) @ b


# --------------------------------------------------------------------------
# portfolios
# --------------------------------------------------------------------------
def tangency(mu: np.ndarray, cov: np.ndarray) -> np.ndarray | None:
    """x = Sigma^-1 mu, unnormalised.  Scale it before you call it a portfolio."""
    x = solve_spd(cov, mu)
    if not np.all(np.isfinite(x)) or float(np.linalg.norm(x)) <= C.TINY:
        return None
    return x


def gmv(cov: np.ndarray) -> np.ndarray | None:
    """x = Sigma^-1 1, unnormalised."""
    n = cov.shape[0]
    x = solve_spd(cov, np.ones(n))
    if not np.all(np.isfinite(x)) or float(np.linalg.norm(x)) <= C.TINY:
        return None
    return x


def blend(w1: np.ndarray, w2: np.ndarray, delta: float) -> np.ndarray | None:
    """Convex blend of two portfolio directions, each first normalised to unit length.

    Normalising first matters: the two raw solutions have wildly different
    magnitudes (that is the whole point), so blending them unnormalised would
    just interpolate the scales rather than the directions.
    """
    if w1 is None or w2 is None:
        return None
    a = w1 / max(float(np.linalg.norm(w1)), C.TINY)
    b = w2 / max(float(np.linalg.norm(w2)), C.TINY)
    return (1.0 - delta) * a + delta * b


def _project_simplex(v: np.ndarray) -> np.ndarray:
    """Euclidean projection onto {w : w >= 0, sum(w) = 1}."""
    n = v.size
    u = np.sort(v)[::-1]
    css = np.cumsum(u)
    idx = np.nonzero(u * np.arange(1, n + 1) > (css - 1.0))[0]
    rho = int(idx[-1])
    theta = (css[rho] - 1.0) / (rho + 1.0)
    return np.maximum(v - theta, 0.0)


def long_only_gmv(cov: np.ndarray, n_iter: int = 30000, tol: float = 1e-14) -> np.ndarray:
    """min w' Sigma w  s.t.  1'w = 1,  w >= 0.

    Accelerated projected gradient (FISTA) with the step 1/lambda_max.  Plain
    projected gradient needs O(kappa) iterations and kappa here is 1e4, so the
    acceleration is not cosmetic.  Convergence is measured on the *iterate*, not
    on the objective: a flat objective with a wandering iterate would silently
    return nonsense.
    """
    n = cov.shape[0]
    step = 1.0 / float(np.linalg.eigvalsh(cov).max())
    w = np.full(n, 1.0 / n)
    y = w.copy()
    theta = 1.0
    for _ in range(n_iter):
        w_new = _project_simplex(y - step * (cov @ y))
        theta_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * theta * theta))
        y = w_new + ((theta - 1.0) / theta_new) * (w_new - w)
        if np.max(np.abs(w_new - w)) < tol:
            return w_new
        w, theta = w_new, theta_new
    return w


def equal_weight(n: int) -> np.ndarray:
    return np.full(n, 1.0 / n)


# --------------------------------------------------------------------------
# scaling convention
# --------------------------------------------------------------------------
def to_vol_target(w: np.ndarray, cov: np.ndarray, vol_target: float = C.VOL_TARGET):
    """Rescale w so that its annualised volatility under ``cov`` equals ``vol_target``.

    Returns None if the portfolio has zero risk under ``cov``.  The scaling
    factor is positive, so the *direction* of the portfolio -- which is the only
    thing the optimiser actually decides -- is untouched.  Sharpe ratios are
    invariant to this operation; only leverage changes.
    """
    daily_var = float(w @ cov @ w)
    if daily_var <= C.TINY:
        return None
    daily_target = vol_target / np.sqrt(C.TRADING_DAYS)
    return w * (daily_target / np.sqrt(daily_var))


# --------------------------------------------------------------------------
# named strategies used by the out-of-sample experiment
# --------------------------------------------------------------------------
STRATEGIES = (
    "oracle",
    "plug_in",
    "shrink_S",
    "gmv_sample",
    "gmv_shrunk",
    "gmv_long_only",
    "equal_weight",
)


def build(name: str, est: dict) -> np.ndarray | None:
    """Build a portfolio direction by strategy name.

    ``est`` is the dict returned by estimators.estimate (keys mu, cov), so the
    caller decides which estimator feeds which strategy.  Every strategy here
    sees only estimated moments -- nothing is allowed to peek at the truth.
    """
    if name in ("plug_in", "shrink_S"):
        return tangency(est["mu"], est["cov"])
    if name == "gmv_sample" or name == "gmv_shrunk":
        return gmv(est["cov"])
    if name == "gmv_long_only":
        return long_only_gmv(est["cov"])
    if name == "equal_weight":
        return equal_weight(est["mu"].size)
    raise ValueError(f"unknown strategy: {name}")
