"""Covariance ill-conditioning in portfolio optimisation.

A numerical study of why the Markowitz solution w ~ Sigma^-1 mu is unreliable
when Sigma has to be estimated, and what shrinkage does about it.
"""

__all__ = [
    "config",
    "market",
    "estimators",
    "portfolio",
    "analysis",
    "plotting",
    "checks",
    "report",
]
