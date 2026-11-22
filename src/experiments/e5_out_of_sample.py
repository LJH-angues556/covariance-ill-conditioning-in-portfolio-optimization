"""Experiment 5 -- does any of this survive out of sample?

Question
    Everything so far is a statement about estimation error.  The only question
    that matters to an investor is whether the repairs actually deliver a better
    portfolio.  Two yardsticks:

      population Sharpe   w'mu / sqrt(w'Sigma w) under the TRUE moments.
                          Noise free, deterministic given the weight vector --
                          this is the number the theory talks about.
      realised Sharpe     mean/std of the portfolio's realised returns on an
                          independent 2000-day path.  Included only to show the
                          population number is not a fiction.

Strategies
    oracle          true mu, true Sigma  (an upper bound, not attainable)
    plug_in         raw sample moments, tangency
    shrink_S        Ledoit-Wolf Sigma, raw mu, tangency
    gmv_sample      raw Sigma, no mu at all
    gmv_shrunk      Ledoit-Wolf Sigma, no mu at all
    gmv_long_only   raw Sigma, no mu, w >= 0
    equal_weight    1/N

Outputs
    fig8_out_of_sample.png, e5_out_of_sample.csv, e5_T_over_N.csv
"""

from __future__ import annotations

import numpy as np

from .. import config as C
from ..aggregate import group_mean, value
from ..analysis import (
    gross_leverage,
    max_weight,
    population_sharpe,
    realized_sharpe,
)
from ..estimators import estimate, sample_moments
from ..plotting import BLUE, GREY, MUSHR, ORANGE, SAMPLE, SHRINK, TRUE, band, new_fig, save
from ..portfolio import build, tangency, to_vol_target
from ..tables import write_csv

STRATEGIES = (
    ("oracle", TRUE, "-"),
    ("plug_in", SAMPLE, "s-"),
    ("shrink_S", SHRINK, "^-"),
    ("gmv_sample", GREY, "o--"),
    ("gmv_shrunk", MUSHR, "D--"),
    ("gmv_long_only", BLUE, "v:"),
    ("equal_weight", ORANGE, "x-"),
)

FIELDS = ("pop_sharpe", "realized_sharpe", "gross_leverage", "max_weight", "min_weight")


def _weights(name: str, est_raw: dict, est_shr: dict, sub, ret):
    """Return a portfolio scaled to the common volatility target.

    Each strategy is scaled with the covariance *it* would use, so the 15%
    target is the risk the investor believes they are running -- which is why
    the plug-in book ends up far riskier than it thinks.
    """
    if name == "oracle":
        w = tangency(sub.mu, sub.cov)
        return to_vol_target(w, sub.cov) if w is not None else None
    if name == "equal_weight":
        _, s_hat = sample_moments(ret)
        return to_vol_target(build("equal_weight", {"mu": sub.mu}), s_hat)
    if name in ("plug_in", "gmv_sample", "gmv_long_only"):
        w = build(name, est_raw)
        return to_vol_target(w, est_raw["cov"]) if w is not None else None
    w = build(name, est_shr)
    return to_vol_target(w, est_shr["cov"]) if w is not None else None


def run(ctx, quick: bool = False) -> dict:
    n_grid = C.N_GRID if not quick else (5, 15, 30, 55)
    n_big = n_grid[-1]

    raw: list[dict] = []
    for rep in range(ctx.n_reps):
        for n in n_grid:
            sub = ctx.subset(rep, n)
            ret = ctx.sample(rep, n, C.T_TRAIN, tag=1)
            ret_test = ctx.sample(rep, n, C.T_TEST, tag=99)
            est_raw = estimate(ret, "raw")
            est_shr = estimate(ret, "shrink_S")

            for name, _color, _style in STRATEGIES:
                w = _weights(name, est_raw, est_shr, sub, ret)
                if w is None:
                    continue
                raw.append(
                    {
                        "rep": rep,
                        "n_assets": n,
                        "T": C.T_TRAIN,
                        "strategy": name,
                        "pop_sharpe": population_sharpe(w, sub.mu, sub.cov),
                        "realized_sharpe": realized_sharpe(w, ret_test),
                        "gross_leverage": gross_leverage(w),
                        "max_weight": max_weight(w),
                        "min_weight": float(w.min()),
                    }
                )

    rows = group_mean(raw, ["n_assets", "strategy"], list(FIELDS))
    write_csv(rows, "e5_out_of_sample")

    # ------------------------------------------------- T/N ratio at fixed N
    tn_raw: list[dict] = []
    for rep in range(ctx.n_reps):
        for t in (60, 120, 250, 600, 1200, 2400):
            sub = ctx.subset(rep, C.N_MAX)
            ret = ctx.sample(rep, C.N_MAX, t, tag=5)
            est_raw = estimate(ret, "raw")
            est_shr = estimate(ret, "shrink_S")
            w = _weights("plug_in", est_raw, est_shr, sub, ret)
            w_g = _weights("gmv_shrunk", est_raw, est_shr, sub, ret)
            tn_raw.append(
                {
                    "rep": rep,
                    "n_assets": C.N_MAX,
                    "T": t,
                    "T_over_N": t / C.N_MAX,
                    "sharpe_plug_in": population_sharpe(w, sub.mu, sub.cov),
                    "leverage_plug_in": gross_leverage(w),
                    "sharpe_gmv_shrunk": population_sharpe(w_g, sub.mu, sub.cov),
                    "sharpe_equal_weight": population_sharpe(
                        np.full(C.N_MAX, 1.0 / C.N_MAX), sub.mu, sub.cov
                    ),
                }
            )
    tn_rows = group_mean(tn_raw, ["n_assets", "T", "T_over_N"],
                         ["sharpe_plug_in", "leverage_plug_in",
                          "sharpe_gmv_shrunk", "sharpe_equal_weight"])
    write_csv(tn_rows, "e5_T_over_N")

    # ------------------------------------------------------------------ fig 8
    fig, (ax1, ax2) = new_fig(7.4, 3.3, ncols=2)
    for name, color, style in STRATEGIES:
        sub = [r for r in rows if r["strategy"] == name]
        if not sub:
            continue
        xs = [r["n_assets"] for r in sub]
        ys = [r["pop_sharpe"] for r in sub]
        ax1.plot(xs, ys, style, color=color, label=name)
        if name != "oracle":
            band(ax1, xs, ys, [r["pop_sharpe_sd"] for r in sub], color)

    ax1.axhline(0, color="black", lw=0.8)
    ax1.set_ylim(-0.12, 1.02)
    ax1.set_title("(a) population Sharpe (true moments)")
    ax1.set_xlabel("number of assets N")
    ax1.set_ylabel("annualised Sharpe")
    ax1.legend(frameon=False, fontsize=6.6, ncol=2, loc="upper left")

    ratio = [r["T_over_N"] for r in tn_rows]
    for field, color, label, style in (
        ("sharpe_plug_in", SAMPLE, "plug-in tangency", "s-"),
        ("sharpe_gmv_shrunk", MUSHR, "min-variance, Ledoit-Wolf $\\Sigma$", "D-"),
        ("sharpe_equal_weight", ORANGE, "1/N", "x-"),
    ):
        ax2.semilogx(ratio, [r[field] for r in tn_rows], style, color=color, label=label)
    ax2.axhline(0, color="black", lw=0.8)
    ax2.set_title(f"(b) how much data would be needed, N={C.N_MAX}")
    ax2.set_xlabel("T / N")
    ax2.set_ylabel("annualised Sharpe")
    ax2.legend(frameon=False, fontsize=7, loc="lower right")
    save(fig, "fig8_out_of_sample")

    def g(name, field, n=n_big):
        return value(rows, field, n_assets=n, strategy=name)

    return {
        "rows": rows,
        "tn_rows": tn_rows,
        "n_big": n_big,
        "n_small": n_grid[0],
        "pop": {name: g(name, "pop_sharpe") for name, *_ in STRATEGIES},
        "real": {name: g(name, "realized_sharpe") for name, *_ in STRATEGIES},
        "lev": {name: g(name, "gross_leverage") for name, *_ in STRATEGIES},
        "maxw": {name: g(name, "max_weight") for name, *_ in STRATEGIES},
        "pop_small": {name: g(name, "pop_sharpe", n_grid[0]) for name, *_ in STRATEGIES},
        "oracle_best_everywhere": all(
            g("oracle", "pop_sharpe", n) >= max(
                g(s, "pop_sharpe", n) for s, *_ in STRATEGIES if s != "oracle"
            ) - 1e-9
            for n in n_grid
        ),
    }
