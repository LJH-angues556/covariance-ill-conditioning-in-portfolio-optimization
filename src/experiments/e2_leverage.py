"""Experiment 2 -- the weight vector explodes.

Question
    If the ill-conditioning really matters, it should show up in the answer the
    optimiser returns, not only in the matrix.  Does gross leverage grow with N,
    and do the repairs bring it back?

Design
    A 2x2 factorial.  Two choices of covariance (raw sample vs Ledoit-Wolf) and
    two choices of objective (use the mean, or ignore it and use the minimum
    variance portfolio).  Four strategies, so the effect of each repair can be
    read off separately:

        plug_in       raw Sigma,   uses mu
        shrink_S      shrunk Sigma, uses mu
        gmv_sample    raw Sigma,   no mu
        gmv_shrunk    shrunk Sigma, no mu

Method
    Fix T = 60, sweep N.  Each portfolio is rescaled to a 15% annualised
    volatility target **using the covariance it believes in** (that is what an
    investor would do), and we record

        gross leverage ||w||_1        exposure per unit of NAV
        largest single position
        volatility actually suffered under the TRUE covariance

    The gap between the 15% the optimiser believes and the volatility it would
    actually suffer is the risk-underestimation half of the story.

Outputs
    fig3_leverage.png, e2_leverage.csv
"""

from __future__ import annotations

import numpy as np

from .. import config as C
from ..aggregate import group_mean, value
from ..analysis import (
    effective_n,
    gross_leverage,
    max_weight,
    population_sharpe,
    portfolio_vol,
)
from ..estimators import estimate
from ..plotting import GREY, MUSHR, ORANGE, SAMPLE, SHRINK, band, new_fig, save
from ..portfolio import gmv, tangency, to_vol_target
from ..tables import write_csv

FIELDS = ("gross_leverage", "max_weight", "min_weight", "effective_n",
          "true_vol", "pop_sharpe")

SERIES = (
    ("plug_in", SAMPLE, "s-", "raw $\\hat\\mu,\\hat S$  (plug-in)"),
    ("shrink_S", SHRINK, "^-", "raw $\\hat\\mu$, Ledoit-Wolf $\\Sigma$"),
    ("gmv_sample", GREY, "o--", "no $\\mu$, raw $\\hat S$"),
    ("gmv_shrunk", MUSHR, "D--", "no $\\mu$, Ledoit-Wolf $\\Sigma$"),
)


def _one(ctx, rep: int, n: int) -> list[dict]:
    sub = ctx.subset(rep, n)
    ret = ctx.sample(rep, n, C.T_TRAIN, tag=1)
    est_raw = estimate(ret, "raw")
    est_shr = estimate(ret, "shrink_S")

    variants = (
        ("plug_in", tangency(est_raw["mu"], est_raw["cov"]), est_raw["cov"]),
        ("shrink_S", tangency(est_shr["mu"], est_shr["cov"]), est_shr["cov"]),
        ("gmv_sample", gmv(est_raw["cov"]), est_raw["cov"]),
        ("gmv_shrunk", gmv(est_shr["cov"]), est_shr["cov"]),
    )

    out = []
    for name, w, cov in variants:
        if w is None:
            continue
        w_s = to_vol_target(w, cov)
        if w_s is None:
            continue
        out.append(
            {
                "rep": rep,
                "n_assets": n,
                "estimator": name,
                "delta_lw": est_shr["delta"],
                "gross_leverage": gross_leverage(w_s),
                "max_weight": max_weight(w_s),
                "min_weight": float(w_s.min()),
                "effective_n": effective_n(w_s),
                "true_vol": portfolio_vol(w_s, sub.cov),
                "pop_sharpe": population_sharpe(w_s, sub.mu, sub.cov),
            }
        )
    return out


def run(ctx, quick: bool = False) -> dict:
    n_grid = C.N_GRID if not quick else (5, 15, 30, 55)
    n_big = n_grid[-1]

    raw: list[dict] = []
    for rep in range(ctx.n_reps):
        for n in n_grid:
            raw.extend(_one(ctx, rep, n))

    rows = group_mean(raw, ["n_assets", "estimator"], list(FIELDS) + ["delta_lw"])
    write_csv(rows, "e2_leverage")

    # ------------------------------------------------------------------ fig 3
    fig, (ax1, ax2) = new_fig(7.4, 3.3, ncols=2)
    for name, color, style, label in SERIES:
        sub = [r for r in rows if r["estimator"] == name]
        if not sub:
            continue
        xs = [r["n_assets"] for r in sub]
        ys = [r["gross_leverage"] for r in sub]
        ax1.semilogy(xs, ys, style, color=color, label=label)
        band(ax1, xs, ys, [r["gross_leverage_sd"] for r in sub], color)

    ax1.set_title("(a) gross leverage at a 15% vol target")
    ax1.set_xlabel("number of assets N")
    ax1.set_ylabel("$\\sum_i |w_i|$  (exposure / NAV)")
    ax1.legend(frameon=False, fontsize=7)

    for name, color, style, _ in SERIES[:2]:
        sub = [r for r in rows if r["estimator"] == name]
        xs = [r["n_assets"] for r in sub]
        ax2.plot(xs, [r["true_vol"] for r in sub], style, color=color,
                 label=name.replace("_", " "))
    ax2.axhline(C.VOL_TARGET, color=ORANGE, lw=1.2, ls=":",
                label=f"believed {C.VOL_TARGET * 100:.0f}%")
    ax2.set_title("(b) volatility actually suffered")
    ax2.set_xlabel("number of assets N")
    ax2.set_ylabel("realised annualised vol")
    ax2.legend(frameon=False, fontsize=7)
    save(fig, "fig3_leverage")

    def g(name, field, n=n_big):
        return value(rows, field, n_assets=n, estimator=name)

    return {
        "rows": rows,
        "n_grid": list(n_grid),
        "n_big": n_big,
        "n_small": n_grid[0],
        "lev": {k: g(k, "gross_leverage") for k, *_ in SERIES},
        "lev_small": {k: g(k, "gross_leverage", n_grid[0]) for k, *_ in SERIES},
        "maxw": {k: g(k, "max_weight") for k, *_ in SERIES},
        "true_vol": {k: g(k, "true_vol") for k, *_ in SERIES},
        "sharpe": {k: g(k, "pop_sharpe") for k, *_ in SERIES},
        "delta_lw": value(rows, "delta_lw", n_assets=n_big, estimator="shrink_S"),
        "delta_lw_sd": value(rows, "delta_lw_sd", n_assets=n_big, estimator="shrink_S"),
        "reps": ctx.n_reps,
    }
