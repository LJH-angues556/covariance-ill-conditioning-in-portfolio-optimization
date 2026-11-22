"""Experiment 1 -- where the ill-conditioning comes from.

Question
    The true covariance of the market is a perfectly ordinary, well-conditioned
    matrix.  The *estimated* one is not.  How bad is the gap, and how does it
    depend on N and T?

Method
    For each replication, draw a market and a T = 60 day panel.  For each
    cross-section size N estimate S_hat and record

        lambda_min, lambda_max, kappa = lambda_max / lambda_min

    for the true matrix, the sample matrix, and the Ledoit-Wolf shrinkage of the
    sample matrix.  Then fix N = 30 and sweep T to see kappa fall back towards 1
    as T grows.  Everything is averaged over replications.

Outputs
    fig1_spectrum.png, fig2_conditioning.png,
    e1_conditioning.csv, e1_conditioning_vs_T.csv
"""

from __future__ import annotations

import numpy as np

from .. import config as C
from ..aggregate import group_mean, value
from ..analysis import condition_number, eigen_spectrum, mp_condition_number
from ..estimators import ledoit_wolf_delta, sample_moments, shrink_cov
from ..plotting import GREY, ORANGE, SAMPLE, SHRINK, TRUE, band, new_fig, save
from ..tables import write_csv

FIELDS = (
    "delta_lw",
    "kappa_true",
    "kappa_sample",
    "kappa_shrunk",
    "lam_min_true",
    "lam_min_sample",
    "lam_min_shrunk",
    "lam_max_true",
    "lam_max_sample",
    "lam_max_shrunk",
)


def _n_sweep(ctx, n_grid) -> tuple[list[dict], list[np.ndarray]]:
    raw: list[dict] = []
    spectra: list[np.ndarray] = []
    for rep in range(ctx.n_reps):
        for n in n_grid:
            sub = ctx.subset(rep, n)
            ret = ctx.sample(rep, n, C.T_TRAIN, tag=1)
            _, s_hat = sample_moments(ret)
            delta = ledoit_wolf_delta(ret)
            s_shr = shrink_cov(s_hat, delta)

            e_t = eigen_spectrum(sub.cov)
            e_s = eigen_spectrum(s_hat)
            e_h = eigen_spectrum(s_shr)

            raw.append(
                {
                    "rep": rep,
                    "n_assets": n,
                    "T": C.T_TRAIN,
                    "delta_lw": delta,
                    "kappa_true": float(e_t[-1] / e_t[0]),
                    "kappa_sample": condition_number(s_hat),
                    "kappa_shrunk": float(e_h[-1] / e_h[0]),
                    "lam_min_true": float(e_t[0]),
                    "lam_min_sample": float(e_s[0]),
                    "lam_min_shrunk": float(e_h[0]),
                    "lam_max_true": float(e_t[-1]),
                    "lam_max_sample": float(e_s[-1]),
                    "lam_max_shrunk": float(e_h[-1]),
                }
            )
            if n == n_grid[-1] and rep == 0:
                spectra.append(e_t[::-1])
                spectra.append(e_s[::-1])
                spectra.append(e_h[::-1])
    return raw, spectra


def _t_sweep(ctx, n_assets: int, t_grid) -> list[dict]:
    raw: list[dict] = []
    for rep in range(ctx.n_reps):
        for t in t_grid:
            ret = ctx.sample(rep, n_assets, t, tag=2)
            _, s_hat = sample_moments(ret)
            raw.append(
                {
                    "rep": rep,
                    "n_assets": n_assets,
                    "T": t,
                    "T_over_N": t / n_assets,
                    "kappa_sample": condition_number(s_hat),
                    "kappa_mp": mp_condition_number(t, n_assets),
                }
            )
    return raw


def run(ctx, quick: bool = False) -> dict:
    n_grid = C.N_GRID if not quick else (5, 15, 30, 55)
    t_grid = C.T_GRID if not quick else (60, 120, 600, 2400)
    n_big = n_grid[-1]

    raw, spectra = _n_sweep(ctx, n_grid)
    rows = group_mean(raw, ["n_assets", "T"], list(FIELDS))
    t_rows = group_mean(_t_sweep(ctx, 30, t_grid), ["n_assets", "T", "T_over_N"],
                        ["kappa_sample", "kappa_mp"])

    write_csv(rows, "e1_conditioning")
    write_csv(t_rows, "e1_conditioning_vs_T")

    # ------------------------------------------------------------------ fig 1
    fig, (ax1, ax2) = new_fig(7.2, 3.2, ncols=2)
    idx = np.arange(1, n_big + 1)
    labels = ["true $\\Sigma$", "sample $\\hat S$ (T=60)", "Ledoit-Wolf"]
    colors = [TRUE, SAMPLE, SHRINK]
    styles = ["-", "-", "--"]
    for k, (lab, col, st) in enumerate(zip(labels, colors, styles)):
        ax1.semilogy(idx, spectra[k], st, color=col, label=lab)
    ax1.set_title(f"(a) eigenvalue spectrum, N={n_big}, T={C.T_TRAIN}")
    ax1.set_xlabel("index (descending)")
    ax1.set_ylabel("eigenvalue (daily variance)")
    ax1.legend(frameon=False, loc="lower left")

    ns = [r["n_assets"] for r in rows]
    for field, lab, col, mk in (
        ("kappa_true", "true $\\Sigma$", TRUE, "o-"),
        ("kappa_sample", "sample $\\hat S$", SAMPLE, "s-"),
        ("kappa_shrunk", "Ledoit-Wolf", SHRINK, "^-"),
    ):
        ax2.plot(ns, [r[field] for r in rows], mk, color=col, label=lab)
        band(ax2, ns, [r[field] for r in rows], [r[field + "_sd"] for r in rows], col)
    ax2.set_title("(b) condition number vs N")
    ax2.set_xlabel("number of assets N")
    ax2.set_ylabel("$\\kappa=\\lambda_{max}/\\lambda_{min}$")
    ax2.set_yscale("log")
    ax2.legend(frameon=False)
    save(fig, "fig1_spectrum")

    # ------------------------------------------------------------------ fig 2
    fig, (ax1, ax2) = new_fig(7.2, 3.2, ncols=2)
    ratio = [r["T_over_N"] for r in t_rows]
    ax1.loglog(ratio, [r["kappa_sample"] for r in t_rows], "o-", color=SAMPLE,
               label="sample $\\hat S$ (N=30)")
    band(ax1, ratio, [r["kappa_sample"] for r in t_rows],
         [r["kappa_sample_sd"] for r in t_rows], SAMPLE)
    ax1.loglog(ratio, [r["kappa_mp"] for r in t_rows], "--", color=GREY,
               label="Marchenko-Pastur (i.i.d.)")
    ax1.axvline(1.0, color=ORANGE, lw=1.0, ls=":")
    ax1.text(1.08, 0.95, "T = N", color=ORANGE, fontsize=8,
             transform=ax1.get_xaxis_transform(), va="top")
    ax1.set_title("(a) condition number vs T/N")
    ax1.set_xlabel("T / N")
    ax1.set_ylabel("$\\kappa$")
    ax1.legend(frameon=False, loc="lower left")

    e_t, e_s = spectra[0], spectra[1]
    ax2.loglog(e_t, e_s, "o", color=SAMPLE, ms=3.5, alpha=0.8, label="one replication")
    lims = [min(e_t.min(), e_s.min()) * 0.7, max(e_t.max(), e_s.max()) * 1.4]
    ax2.loglog(lims, lims, "--", color=GREY, lw=1.0, label="identity")
    ax2.set_title(f"(b) sorted spectra, N={n_big}: the small end collapses")
    ax2.set_xlabel("true eigenvalue (rank-matched)")
    ax2.set_ylabel("sample eigenvalue")
    ax2.legend(frameon=False, loc="upper left")
    save(fig, "fig2_conditioning")

    return {
        "n_sweep": rows,
        "t_sweep": t_rows,
        "n_big": n_big,
        "n_small": n_grid[0],
        "kappa_true": value(rows, "kappa_true", n_assets=n_big),
        "kappa_sample": value(rows, "kappa_sample", n_assets=n_big),
        "kappa_shrunk": value(rows, "kappa_shrunk", n_assets=n_big),
        "lam_min_true": value(rows, "lam_min_true", n_assets=n_big),
        "lam_min_sample": value(rows, "lam_min_sample", n_assets=n_big),
        "lam_min_shrunk": value(rows, "lam_min_shrunk", n_assets=n_big),
        "lam_max_true": value(rows, "lam_max_true", n_assets=n_big),
        "lam_max_sample": value(rows, "lam_max_sample", n_assets=n_big),
        "delta_lw": value(rows, "delta_lw", n_assets=n_big),
        "delta_lw_sd": value(rows, "delta_lw_sd", n_assets=n_big),
        "kappa_sample_n5": value(rows, "kappa_sample", n_assets=n_grid[0]),
        "kappa_true_n5": value(rows, "kappa_true", n_assets=n_grid[0]),
        "kappa_T60": value(t_rows, "kappa_sample", n_assets=30, T=60),
        "kappa_Tmax": value(t_rows, "kappa_sample", n_assets=30, T=t_grid[-1]),
        "T_max": t_grid[-1],
        "reps": ctx.n_reps,
    }
