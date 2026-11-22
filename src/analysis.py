"""Measurement instruments.

Everything here is a *measurement*: given a matrix or a weight vector, return a
number the paper can quote.  No experiment logic, no plotting.
"""

from __future__ import annotations

import numpy as np

from . import config as C


# --------------------------------------------------------------------------
# conditioning
# --------------------------------------------------------------------------
def eigen_spectrum(cov: np.ndarray) -> np.ndarray:
    """Eigenvalues in ascending order (symmetric matrix)."""
    return np.sort(np.linalg.eigvalsh(cov))


def condition_number(cov: np.ndarray) -> float:
    """kappa = lambda_max / lambda_min, with a floor on lambda_min.

    When T <= N the sample covariance is singular, lambda_min is numerically
    zero and kappa is infinite.  We return inf rather than a fake finite number.
    """
    eig = eigen_spectrum(cov)
    lo = float(eig[0])
    hi = float(eig[-1])
    if lo <= C.EIG_FLOOR:
        return float("inf")
    return hi / lo


def mp_condition_number(n_obs: int, n_assets: int) -> float:
    """Marchenko-Pastur reference for i.i.d. noise: ((1+sqrt(q))/(1-sqrt(q)))^2.

    This is what kappa would be if the true covariance were the identity and we
    only saw noise.  It is plotted as a reference curve, not used as a model:
    our market is factor-structured, so the sample spectrum is a *superposition*
    of a few signal eigenvalues and a noise bulk.
    """
    if n_obs <= n_assets:
        return float("inf")
    q = np.sqrt(n_assets / n_obs)
    return float(((1.0 + q) / (1.0 - q)) ** 2)


# --------------------------------------------------------------------------
# weight-vector diagnostics
# --------------------------------------------------------------------------
def gross_leverage(w: np.ndarray) -> float:
    """||w||_1 with 1'w = 1.  Equals 1 for a long-only portfolio."""
    return float(np.abs(w).sum())


def max_weight(w: np.ndarray) -> float:
    return float(np.abs(w).max())


def net_exposure(w: np.ndarray) -> float:
    return float(w.sum())


def effective_n(w: np.ndarray) -> float:
    """Participation ratio (sum|w|)^2 / sum(w^2): 1 = all in one asset."""
    a = np.abs(w)
    denom = float((a**2).sum())
    if denom <= C.TINY:
        return float("nan")
    return float(a.sum() ** 2 / denom)


def weight_corr(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) < C.TINY or np.std(b) < C.TINY:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def turnover(a: np.ndarray, b: np.ndarray) -> float:
    """Half the L1 distance: 0 = identical, 1 = completely different book."""
    return float(np.abs(a - b).sum() / 2.0)


def relative_error(approx: np.ndarray, exact: np.ndarray) -> float:
    """||approx - exact||_2 / ||exact||_2."""
    denom = float(np.linalg.norm(exact))
    if denom <= C.TINY:
        return float("nan")
    return float(np.linalg.norm(approx - exact) / denom)


# --------------------------------------------------------------------------
# eigen-decomposition of the solution
# --------------------------------------------------------------------------
def eigen_contributions(mu: np.ndarray, cov: np.ndarray) -> dict:
    """Decompose x = Sigma^-1 mu along the eigen-directions of Sigma.

        Sigma = V diag(lambda) V'   =>   x = sum_i (v_i' mu / lambda_i) v_i

    The coefficient c_i = (v_i' mu) / lambda_i is how much of the position sits
    in direction i.  A tiny lambda_i with a non-zero projection of mu produces a
    huge c_i -- that single line is the mechanism of the whole paper.
    """
    vals, vecs = np.linalg.eigh(cov)      # ascending
    proj = vecs.T @ mu                    # v_i' mu
    with np.errstate(divide="ignore", invalid="ignore"):
        coef = np.where(vals > C.EIG_FLOOR, proj / vals, np.nan)
    return {"eigenvalues": vals, "eigenvectors": vecs, "projection": proj, "coefficient": coef}


def coefficient_share(mu: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Share of ||x||^2 carried by each eigen-direction, sorted descending."""
    d = eigen_contributions(mu, cov)
    c = d["coefficient"]
    c = np.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0)
    energy = c**2
    total = float(energy.sum())
    if total <= C.TINY:
        return np.zeros_like(energy)
    return np.sort(energy / total)[::-1]


# --------------------------------------------------------------------------
# performance
# --------------------------------------------------------------------------
def population_sharpe(w: np.ndarray, mu: np.ndarray, cov: np.ndarray) -> float:
    """Annualised Sharpe ratio under the TRUE moments -- noise free.

    This is the number the theory talks about: it is a property of the weight
    vector, not of a particular realised path, so it does not need to be
    averaged over seeds.
    """
    ret = float(w @ mu)
    var = float(w @ cov @ w)
    if var <= C.TINY:
        return float("nan")
    return ret / np.sqrt(var) * np.sqrt(C.TRADING_DAYS)


def realized_sharpe(w: np.ndarray, returns: np.ndarray) -> float:
    """Annualised Sharpe ratio on a realised panel (T x N)."""
    p = returns @ w
    sd = float(p.std(ddof=1))
    if sd <= C.TINY:
        return float("nan")
    return float(p.mean() / sd * np.sqrt(C.TRADING_DAYS))


def portfolio_vol(w: np.ndarray, cov: np.ndarray) -> float:
    """Annualised volatility."""
    return float(np.sqrt(max(w @ cov @ w, 0.0)) * np.sqrt(C.TRADING_DAYS))


# --------------------------------------------------------------------------
# signal-to-noise of the mean estimate
# --------------------------------------------------------------------------
def mean_tstat(returns: np.ndarray) -> np.ndarray:
    """Per-asset t-statistic of the sample mean: mu_i / (s_i / sqrt(T))."""
    t = returns.shape[0]
    mu = returns.mean(axis=0)
    sd = returns.std(axis=0, ddof=1)
    sd = np.where(sd > C.TINY, sd, np.nan)
    return mu / (sd / np.sqrt(t))


def relative_input_error(mu_hat, cov_hat, mu_true, cov_true) -> dict:
    """Relative size of the two estimation errors, in the Frobenius/2-norm sense."""
    return {
        "mu": relative_error(mu_hat, mu_true),
        "cov": relative_error(cov_hat, cov_true),
    }


# --------------------------------------------------------------------------
# perturbation response
# --------------------------------------------------------------------------
def perturb_symmetric(cov: np.ndarray, size: float, rng: np.random.Generator) -> np.ndarray:
    """A random symmetric shock whose Frobenius norm is ``size`` x ||cov||_F."""
    n = cov.shape[0]
    a = rng.standard_normal((n, n))
    a = (a + a.T) / 2.0
    norm = float(np.linalg.norm(a))
    if norm <= C.TINY:
        return cov.copy()
    return cov + a * (size * float(np.linalg.norm(cov)) / norm)


def amplification(cov: np.ndarray, mu: np.ndarray, size: float, n_draws: int,
                  seed: int) -> dict:
    """Measure ||dx||/||x|| per unit ||dSigma||/||Sigma||.

    Theory (first-order perturbation of A x = mu, with A the matrix passed in):

        ||dx|| / ||x||  <=  kappa(A) * ||dSigma|| / ||Sigma||

    The constant is the condition number of the matrix actually inverted, i.e.
    of ``cov`` as passed in -- in the experiments that is the *sample* matrix
    S_hat, not the true Sigma.  Mixing the two up makes the bound look violated
    at large N.

    so the measured ratio should sit below kappa and, for a generic direction,
    not many orders of magnitude below it.
    """
    rng = np.random.default_rng(seed)
    x0 = solve(cov, mu)
    base = float(np.linalg.norm(x0))
    ratios = []
    for _ in range(n_draws):
        cov_p = perturb_symmetric(cov, size, rng)
        try:
            x1 = solve(cov_p, mu)
        except np.linalg.LinAlgError:
            continue
        rel = float(np.linalg.norm(x1 - x0) / base) if base > C.TINY else np.nan
        if np.isfinite(rel):
            ratios.append(rel / size)
    if not ratios:
        return {"amplification": float("nan"), "kappa": condition_number(cov), "n_draws": 0}
    return {
        "amplification": float(np.median(ratios)),
        "amplification_max": float(np.max(ratios)),
        "kappa": condition_number(cov),
        "n_draws": len(ratios),
    }


def solve(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.solve(a, b)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(a) @ b
