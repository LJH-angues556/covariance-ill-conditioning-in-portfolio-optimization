"""Averaging across replications.

Each experiment appends one row per (replication, cell) and then collapses the
replications with ``group_mean``.  The mean is what the paper quotes; the
standard deviation across replications is kept so the figures can show a band,
and so that a difference smaller than the band is visibly not a result.
"""

from __future__ import annotations

import math

import numpy as np


def _finite(v) -> bool:
    return isinstance(v, (int, float, np.floating)) and math.isfinite(float(v))


def group_mean(rows: list[dict], group_cols: list[str], value_cols: list[str]) -> list[dict]:
    """Average ``value_cols`` over replications, grouped by ``group_cols``.

    Non-finite values (inf kappa for a singular matrix, nan Sharpe) are dropped
    before averaging, and the number of surviving replications is reported as
    ``n_valid`` -- so a cell where half the replications blew up cannot be
    mistaken for a stable number.
    """
    buckets: dict[tuple, dict[str, list[float]]] = {}
    for r in rows:
        key = tuple(r[c] for c in group_cols)
        b = buckets.setdefault(key, {c: [] for c in value_cols})
        for c in value_cols:
            v = r.get(c)
            if _finite(v):
                b[c].append(float(v))

    out: list[dict] = []
    for key, b in buckets.items():
        row = dict(zip(group_cols, key))
        n_valid = 0
        for c in value_cols:
            arr = np.asarray(b[c], dtype=float)
            if arr.size == 0:
                row[c] = float("nan")
                row[c + "_sd"] = float("nan")
            else:
                row[c] = float(arr.mean())
                row[c + "_sd"] = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
            n_valid = max(n_valid, int(arr.size))
        row["n_valid"] = n_valid
        out.append(row)

    out.sort(key=lambda r: tuple(r[c] for c in group_cols))
    return out


def lookup(rows: list[dict], **kwargs):
    """First row matching all keyword filters, or None."""
    for r in rows:
        if all(r.get(k) == v for k, v in kwargs.items()):
            return r
    return None


def value(rows: list[dict], field_name: str, default: float = float("nan"), **kwargs) -> float:
    r = lookup(rows, **kwargs)
    if r is None:
        return default
    v = r.get(field_name, default)
    return float(v) if _finite(v) else default
