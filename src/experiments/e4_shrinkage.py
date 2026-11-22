"""Experiment 4 -- the repair, and why the mean is the binding constraint.

Question
    Shrinkage is a bias-variance trade.  If we shrink too little nothing is
    fixed; if we shrink too much we throw away the signal.  Where is the
    optimum, and which of the two moments is the binding constraint?

Method
    Fix N = 55, T = 60, average over replications.

    Path 1 -- covariance shrinkage.  Sweep delta_Sigma in [0, 1] with the mean
    left raw.  The Ledoit-Wolf intensity is marked: it is the point that
    minimises E||Sigma_tilde - Sigma||_F^2, an excellent criterion for
    *estimating* a matrix and, as the figure shows, a poor one for deciding how
    much of it to *invert*.

    Path 2 -- mean shrinkage.  Fix Sigma at its Ledoit-Wolf value and blend the
    tangency direction towards the minimum-variance direction.  This is the
    well-posed version of "shrink mu towards the mean" (see estimators.py for
    why the moment-space version is not usable at T = 60).

    The reason the mean is the binding constraint is measured, not asserted:
    with T = 60 the t-statistic of a single mean is mu_i / (s_i / sqrt(T)).  If
    |t| is below 2 for essentially every asset, the cross-section of means is
    indistinguishable from a constant.

Outputs
    fig7_shrinkage.png, e4_shrinkage.csv, e4_intensity.csv
"""

from __future__ import annotations

import numpy as np

from .. import config as C
from ..aggregate import group_mean, value
from ..analysis import gross_leverage, mean_tstat, population_sharpe
from ..estimators import ledoit_wolf_delta, sample_moments, shrink_cov
from ..plotting import GREY, MUSHR, ORANGE, SAMPLE, SHRINK, band, new_fig, save
from ..portfolio import blend, gmv, tangency, to_vol_target
from ..tables import write_csv


def _stats(w, cov_used, sub):
    w_s = to_vol_target(w, cov_used)
    if w_s is None:
        return float("nan"), float("nan"), float("nan")
    return (
        gross_leverage(w_s),
        population_sharpe(w_s, sub.mu, sub.cov),
        float(np.abs(w_s).max()),
    )


def _sweeps(ctx, n_assets: int):
    raw: list[dict] = []
    tstats: list[np.ndarray] = []
    delta_lw: list[float] = []

    for rep in range(ctx.n_reps):
        sub = ctx.subset(rep, n_assets)
        ret = ctx.sample(rep, n_assets, C.T_TRAIN, tag=1)
        mu_hat, s_hat = sample_moments(ret)
        d_lw = ledoit_wolf_delta(ret)
        s_shr = shrink_cov(s_hat, d_lw)
        delta_lw.append(d_lw)
        tstats.append(mean_tstat(ret))

        for d in C.DELTA_GRID:
            cov_d = shrink_cov(s_hat, d)
            w = tangency(mu_hat, cov_d)
            if w is None:
                continue
            lev, sr, mx = _stats(w, cov_d, sub)
            raw.append({"rep": rep, "sweep": "delta_sigma", "value": d,
                        "gross_leverage": lev, "pop_sharpe": sr, "max_weight": mx})

        w_tan = tangency(mu_hat, s_shr)
        w_gmv = gmv(s_shr)
        for d in C.BLEND_GRID:
            w = blend(w_tan, w_gmv, d)
            if w is None:
                continue
            lev, sr, mx = _stats(w, s_shr, sub)
            raw.append({"rep": rep, "sweep": "delta_w", "value": d,
                        "gross_leverage": lev, "pop_sharpe": sr, "max_weight": mx})

    rows = group_mean(raw, ["sweep", "value"],
                      ["gross_leverage", "pop_sharpe", "max_weight"])
    t = np.concatenate([x[np.isfinite(x)] for x in tstats]) if tstats else np.array([])
    return rows, t, float(np.mean(delta_lw)), float(np.std(delta_lw))


def _intensity_table(ctx) -> list[dict]:
    """How the Ledoit-Wolf intensity depends on N and on T."""
    rows = []
    for n in C.N_GRID:
        vals = [ledoit_wolf_delta(ctx.sample(rep, n, C.T_TRAIN, tag=1))
                for rep in range(ctx.n_reps)]
        rows.append({"n_assets": n, "T": C.T_TRAIN,
                     "delta_lw": float(np.mean(vals)),
                     "delta_lw_sd": float(np.std(vals, ddof=1))})
    for t in (60, 120, 250, 600, 1200, 2400):
        vals = [ledoit_wolf_delta(ctx.sample(rep, C.N_MAX, t, tag=3))
                for rep in range(ctx.n_reps)]
        rows.append({"n_assets": C.N_MAX, "T": t,
                     "delta_lw": float(np.mean(vals)),
                     "delta_lw_sd": float(np.std(vals, ddof=1))})
    return rows


def run(ctx, quick: bool = False) -> dict:
    n_assets = C.N_MAX if not quick else 30
    rows, tstats, d_lw, d_lw_sd = _sweeps(ctx, n_assets)

    sig = [r for r in rows if r["sweep"] == "delta_sigma"]
    bln = [r for r in rows if r["sweep"] == "delta_w"]

    # ------------------------------------------------------------------ fig 7
    fig, axs = new_fig(7.4, 5.4, nrows=2, ncols=2)
    ax1, ax2, ax3, ax4 = axs.ravel()

    xs = [r["value"] for r in sig]
    ys = [r["gross_leverage"] for r in sig]
    ax1.semilogy(xs, ys, "o-", color=SHRINK, label="plug-in tangency")
    band(ax1, xs, ys, [r["gross_leverage_sd"] for r in sig], SHRINK)
    ax1.axvline(d_lw, color=ORANGE, ls=":", lw=1.2)
    ax1.annotate(f"Ledoit-Wolf $\\delta={d_lw:.2f}$", xy=(d_lw, 0.80),
                 xycoords=("data", "axes fraction"), fontsize=7,
                 color=ORANGE, ha="left", xytext=(6, 0), textcoords="offset points")
    ax1.set_title("(a) leverage vs $\\delta_\\Sigma$")
    ax1.set_xlabel("$\\delta_\\Sigma$")
    ax1.set_ylabel("$\\sum_i|w_i|$")

    ys = [r["pop_sharpe"] for r in sig]
    ax2.plot(xs, ys, "o-", color=SHRINK)
    band(ax2, xs, ys, [r["pop_sharpe_sd"] for r in sig], SHRINK)
    ax2.axhline(0.0, color="black", lw=0.8)
    ax2.axvline(d_lw, color=ORANGE, ls=":", lw=1.2)
    ax2.set_title("(b) Sharpe vs $\\delta_\\Sigma$")
    ax2.set_xlabel("$\\delta_\\Sigma$")
    ax2.set_ylabel("annualised Sharpe")

    xb = [r["value"] for r in bln]
    lev_b = [r["gross_leverage"] for r in bln]
    sr_b = [r["pop_sharpe"] for r in bln]
    ax3.semilogy(xb, lev_b, "s-", color=MUSHR, label="leverage")
    band(ax3, xb, lev_b, [r["gross_leverage_sd"] for r in bln], MUSHR)
    ax3.set_ylim(1.0, 20.0)     # honest: on a log axis starting at 1 the line is nearly flat
    ax3.set_xlabel("$\\delta_w$  (0 = tangency, 1 = min-variance)")
    ax3.set_ylabel("$\\sum_i|w_i|$", color=MUSHR)
    ax3.set_title("(c) blending away $\\mu$")
    ax3b = ax3.twinx()
    ax3b.plot(xb, sr_b, "o--", color=SAMPLE, label="Sharpe")
    ax3b.set_ylabel("annualised Sharpe", color=SAMPLE)
    ax3b.grid(False)

    ax4.hist(tstats, bins=24, color=GREY, alpha=0.75)
    for v in (2.0, -2.0):
        ax4.axvline(v, color=SAMPLE, ls="--", lw=1.0)
    ax4.set_title(f"(d) t-stat of $\\hat\\mu_i$, N={n_assets}, T={C.T_TRAIN}")
    ax4.set_xlabel("$t_i=\\hat\\mu_i/(s_i/\\sqrt{T})$")
    ax4.set_ylabel("count")
    save(fig, "fig7_shrinkage")

    intensity_rows = _intensity_table(ctx)
    write_csv(rows, "e4_shrinkage")
    write_csv(intensity_rows, "e4_intensity")

    nearest = min(sig, key=lambda r: abs(r["value"] - d_lw))
    return {
        "n_assets": n_assets,
        "rows": rows,
        "sig": sig,
        "blend": bln,
        "delta_lw": d_lw,
        "delta_lw_sd": d_lw_sd,
        "delta_lw_grid": nearest["value"],
        "intensity_table": intensity_rows,
        "tstat_max_abs": float(np.abs(tstats).max()),
        "tstat_frac_gt2": float((np.abs(tstats) > 2).mean()),
        "tstat_median_abs": float(np.median(np.abs(tstats))),
        "lev_at_lw": nearest["gross_leverage"],
        "lev_min": min(r["gross_leverage"] for r in sig),
        "sr_at_zero": value(sig, "pop_sharpe", value=0.0),
        "sr_best_sigma": max(r["pop_sharpe"] for r in sig),
        "sr_blend_end": bln[-1]["pop_sharpe"],
        "sr_blend_zero": bln[0]["pop_sharpe"],
        "lev_blend_zero": bln[0]["gross_leverage"],
        "lev_blend_end": bln[-1]["gross_leverage"],
    }
