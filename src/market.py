"""The data generating process, and the replication machinery.

Why a simulated market instead of a downloaded price panel?

Because the paper's whole question is *how far the estimated portfolio is from
the true one*.  That question is only answerable if the true Sigma and the true
mu are known.  With real data they are not, and every claim about "error
amplification" would have to be indirect.  A calibrated factor model gives us
ground truth, exact control over N and T, and full reproducibility -- this is
exactly the device Ledoit & Wolf (2004) and DeMiguel et al. (2009) use when they
need to separate estimation error from model error.

The calibration is deliberately boring: it reproduces the stylised facts of
daily US equity returns and nothing more.  No return predictability, no regime
switching, no fat tails -- those would only add noise to the argument.

Every statistic in the paper is averaged over ``N_REPS`` independent
replications, each with its own market and its own estimation window.  That
matters: with T = 60 the plug-in portfolio's Sharpe moves by about +-0.15 from
one window to the next, which is the same size as the effects under study.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import config as C


@dataclass(frozen=True)
class Market:
    """A factor-structured market with known first and second moments."""

    mu: np.ndarray        # (N,)   daily expected excess returns
    cov: np.ndarray       # (N, N) daily covariance matrix
    loadings: np.ndarray  # (N, K) factor loadings
    idio_var: np.ndarray  # (N,)   diagonal of D
    cov_factor: np.ndarray  # (K, K)
    mu_factor: np.ndarray   # (K,)

    @property
    def n_assets(self) -> int:
        return int(self.mu.size)

    def subset(self, n: int) -> "Market":
        """The first ``n`` assets -- nested, so an N-sweep really is 'adding assets'."""
        return Market(
            mu=self.mu[:n].copy(),
            cov=self.cov[:n, :n].copy(),
            loadings=self.loadings[:n].copy(),
            idio_var=self.idio_var[:n].copy(),
            cov_factor=self.cov_factor.copy(),
            mu_factor=self.mu_factor.copy(),
        )


def build_market(n_assets: int = C.N_MAX, seed: int = C.SEED) -> Market:
    """Draw one market from the calibrated factor model."""
    rng = np.random.default_rng(seed)
    n = int(n_assets)
    k = C.K_FACTORS

    ann_vol = np.array([C.VOL_MARKET, *C.VOL_STYLE])
    ann_mu = np.array([C.MU_MARKET, *C.MU_STYLE])
    c1, c2 = C.CORR_MARKET_STYLE
    corr = np.array(
        [
            [1.0, c1, c2],
            [c1, 1.0, C.CORR_STYLE_STYLE],
            [c2, C.CORR_STYLE_STYLE, 1.0],
        ]
    )
    vol_d = ann_vol / np.sqrt(C.TRADING_DAYS)
    cov_f = np.outer(vol_d, vol_d) * corr
    mu_f = ann_mu / C.TRADING_DAYS

    beta = C.BETA_MEAN + C.BETA_STD * rng.standard_normal(n)
    style = C.STYLE_LOAD_STD * rng.standard_normal((n, k - 1))
    loadings = np.column_stack([beta, style])

    alpha = C.ALPHA_STD * rng.standard_normal(n) / C.TRADING_DAYS
    idio_vol = rng.uniform(*C.IDIO_VOL_RANGE, size=n) / np.sqrt(C.TRADING_DAYS)
    idio_var = idio_vol**2

    mu = loadings @ mu_f + alpha
    cov = loadings @ cov_f @ loadings.T + np.diag(idio_var)

    eig = np.linalg.eigvalsh(cov)
    if eig[0] <= 0:
        raise RuntimeError("calibration produced a non positive-definite Sigma")

    return Market(
        mu=mu,
        cov=cov,
        loadings=loadings,
        idio_var=idio_var,
        cov_factor=cov_f,
        mu_factor=mu_f,
    )


def sample_returns(market: Market, n_obs: int, rng: np.random.Generator) -> np.ndarray:
    """Draw ``n_obs`` i.i.d. daily returns from the market (shape: T x N)."""
    n = market.n_assets
    chol = np.linalg.cholesky(market.cov)
    z = rng.standard_normal((int(n_obs), n))
    return z @ chol.T + market.mu


@dataclass
class Context:
    """Everything an experiment needs: markets per replication, and samples."""

    n_reps: int = C.N_REPS
    seed: int = C.SEED
    _markets: dict = field(default_factory=dict)

    def market(self, rep: int = 0) -> Market:
        """The market used in replication ``rep`` (built once, then cached)."""
        if rep not in self._markets:
            self._markets[rep] = build_market(C.N_MAX, self.seed + 1000 * rep + 7)
        return self._markets[rep]

    def sample(self, rep: int, n_assets: int, n_obs: int, tag: int = 0) -> np.ndarray:
        """Deterministic panel: same (rep, n, T, tag) always gives the same data."""
        seed = (
            self.seed
            + 7919 * int(n_assets)
            + 104729 * int(n_obs)
            + 31 * int(tag)
            + 1000003 * int(rep)
        )
        return sample_returns(
            self.market(rep).subset(n_assets), n_obs, np.random.default_rng(seed)
        )

    def subset(self, rep: int, n_assets: int) -> Market:
        return self.market(rep).subset(n_assets)


def build_context(n_reps: int = C.N_REPS, seed: int = C.SEED) -> Context:
    return Context(n_reps=n_reps, seed=seed)
