"""Automatic verification of every quantitative claim the paper makes.

Each check is a named assertion with an expected range and a measured value.
``run_all.py`` prints the table and returns a non-zero exit code if any check
fails, which turns the whole program into a regression test: if you change the
calibration or the grids and a claim stops being true, the program tells you
instead of silently producing a paper with the wrong numbers.

A check whose inputs were not produced in this run (because of ``--only``) is
reported as *skipped*, not as a failure.
"""

from __future__ import annotations

import math
from typing import Any

Row = dict[str, Any]


def _finite(x) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x)


def _fmt(x) -> str:
    if isinstance(x, float):
        if math.isnan(x):
            return "nan"
        if math.isinf(x):
            return "inf" if x > 0 else "-inf"
        if x != 0 and (abs(x) >= 1e5 or abs(x) < 1e-3):
            return f"{x:.3e}"
        return f"{x:.4f}"
    return str(x)


def evaluate(m: dict) -> list[Row]:
    rows: list[Row] = []

    def add(name: str, expected: str, measured, passed: bool | None) -> None:
        rows.append(
            {
                "check": name,
                "expected": expected,
                "measured": measured,
                "status": "SKIP" if passed is None else ("PASS" if passed else "FAIL"),
            }
        )

    e1, e2, e3, e4, e5 = (m.get(k) for k in ("e1", "e2", "e3", "e4", "e5"))

    # ---------------------------------------------------------------- e1
    if e1:
        add("the true Sigma is well conditioned",
            "kappa(Sigma) < 500",
            _fmt(e1["kappa_true"]), e1["kappa_true"] < 500)

        ratio = e1["kappa_sample"] / e1["kappa_true"] if e1["kappa_true"] else float("inf")
        add("estimation destroys the conditioning",
            "kappa(S)/kappa(Sigma) > 20",
            _fmt(ratio), ratio > 20)

        lam = e1["lam_min_sample"] / e1["lam_min_true"]
        add("the smallest eigenvalue collapses",
            "lam_min(S)/lam_min(Sigma) < 0.25",
            _fmt(lam), lam < 0.25)

        growth = e1["kappa_sample"] / e1["kappa_sample_n5"]
        add("conditioning worsens with N",
            "kappa(55)/kappa(5) > 4",
            _fmt(growth), growth > 4)

        shrink = e1["kappa_Tmax"] / e1["kappa_T60"]
        add("conditioning recovers with T",
            f"kappa(T={e1['T_max']})/kappa(T=60) < 0.2",
            _fmt(shrink), shrink < 0.2)

    # ---------------------------------------------------------------- e2
    if e2:
        lo, hi = e2["lev_small"]["plug_in"], e2["lev"]["plug_in"]
        add("leverage explodes with N",
            f"lev({e2['n_big']})/lev({e2['n_small']}) > 8",
            f"{_fmt(hi)}/{_fmt(lo)} = {_fmt(hi / lo)}", hi / lo > 8)

        cut = 1.0 - e2["lev"]["shrink_S"] / hi
        add("Sigma shrinkage cuts leverage",
            "> 60% at N=55",
            f"-{cut * 100:.1f}%  (to {_fmt(e2['lev']['shrink_S'])})", cut > 0.60)

        add("dropping mu does not fix leverage",
            "GMV(raw) leverage within 2x of plug-in",
            f"{_fmt(e2['lev']['gmv_sample'])} vs {_fmt(hi)}",
            e2["lev"]["gmv_sample"] > 0.5 * hi)

        add("repaired leverage is sane",
            "GMV(Ledoit-Wolf) leverage < 10",
            _fmt(e2["lev"]["gmv_shrunk"]), e2["lev"]["gmv_shrunk"] < 10)

        add("the optimiser underestimates its own risk",
            "realised vol > 3x the 15% it believes",
            _fmt(e2["true_vol"]["plug_in"]), e2["true_vol"]["plug_in"] > 3 * 0.15)

    # ---------------------------------------------------------------- e3
    if e3:
        amp = e3["amplification"]
        worst = max((r["amp_over_kappa"] for r in amp if _finite(r["amp_over_kappa"])),
                    default=float("nan"))
        add("perturbation bound ||dx||/||x|| <= kappa*||dS||/||S||",
            "measured/kappa <= 1",
            _fmt(worst), worst <= 1.0)

        worst_kappa_row = max(amp, key=lambda r: r["kappa_sample"] if _finite(r["kappa_sample"]) else -1)
        add("a 1% error in Sigma moves the solution a lot",
            "amplification > 20 at the largest N",
            _fmt(worst_kappa_row["amplification"]),
            worst_kappa_row["amplification"] > 20)

        add("two samples give different portfolios",
            f"corr(w1,w2) < 0.5 at N={e3['n_big']}",
            _fmt(e3["corr_big"]), e3["corr_big"] < 0.5)

        gap = e3["sharpe_sigma_only"] - e3["sharpe_mu_only"]
        add("mu is the binding constraint",
            "Sharpe(Sigma-only) - Sharpe(mu-only) > 0.20",
            f"{_fmt(e3['sharpe_sigma_only'])} - {_fmt(e3['sharpe_mu_only'])}",
            gap > 0.20)

        add("the mean is ~an order of magnitude noisier than the covariance",
            "eps_mu / eps_Sigma > 10",
            f"{_fmt(e3['eps_mu'])} / {_fmt(e3['eps_cov'])} = "
            f"{_fmt(e3['eps_mu'] / e3['eps_cov'])}",
            e3["eps_mu"] / e3["eps_cov"] > 10)

        share = e3["mechanism"]["top3_share_hat"]
        share_t = e3["mechanism"]["top3_share_true"]
        add("plug-in energy sits in the noise directions",
            "top-3 share > 60% and > 1.5x the true share",
            f"{share * 100:.1f}% vs {share_t * 100:.1f}%",
            share > 0.60 and share > 1.5 * share_t)

    # ---------------------------------------------------------------- e4
    if e4:
        add("means are not distinguishable from zero",
            "median|t| < 1.0 and share(|t|>2) < 0.15",
            f"median|t|={e4['tstat_median_abs']:.2f}, "
            f"share={e4['tstat_frac_gt2'] * 100:.0f}%",
            e4["tstat_median_abs"] < 1.0 and e4["tstat_frac_gt2"] < 0.15)

        add("the Ledoit-Wolf intensity is far below what inversion needs",
            "delta_LW < 0.30",
            _fmt(e4["delta_lw"]), e4["delta_lw"] < 0.30)

        add("leverage keeps falling as delta_Sigma grows",
            "leverage at delta=0.99 < 50% of leverage at delta=0",
            f"{_fmt(e4['lev_min'])} vs {_fmt(value_at(e4['sig'], 0.0))}",
            e4["lev_min"] < 0.50 * value_at(e4["sig"], 0.0))

        add("blending away mu rescues the Sharpe",
            "Sharpe(delta_w=1) > Sharpe(delta_w=0) + 0.20",
            f"{_fmt(e4['sr_blend_end'])} vs {_fmt(e4['sr_blend_zero'])}",
            e4["sr_blend_end"] > e4["sr_blend_zero"] + 0.20)

    # ---------------------------------------------------------------- e5
    if e5:
        add("the oracle is an upper bound",
            "Sharpe(oracle) >= all, every N",
            str(e5["oracle_best_everywhere"]), e5["oracle_best_everywhere"])

        add("the plug-in loses to 1/N",
            f"SR(plug_in) < SR(1/N) at N={e5['n_big']}",
            f"{_fmt(e5['pop']['plug_in'])} vs {_fmt(e5['pop']['equal_weight'])}",
            e5["pop"]["plug_in"] < e5["pop"]["equal_weight"])

        add("dropping mu and shrinking Sigma recovers most of the gap",
            "SR(gmv_shrunk) > SR(plug_in) + 0.20",
            f"{_fmt(e5['pop']['gmv_shrunk'])} vs {_fmt(e5['pop']['plug_in'])}",
            e5["pop"]["gmv_shrunk"] > e5["pop"]["plug_in"] + 0.20)

        add("Sigma shrinkage alone barely helps the Sharpe",
            "SR(shrink_S) - SR(plug_in) < 0.10",
            f"{_fmt(e5['pop']['shrink_S'])} vs {_fmt(e5['pop']['plug_in'])}",
            e5["pop"]["shrink_S"] - e5["pop"]["plug_in"] < 0.10)

    return rows


def value_at(rows: list[dict], x: float, field: str = "gross_leverage") -> float:
    for r in rows:
        if abs(float(r["value"]) - x) < 1e-9:
            return float(r[field])
    return float("nan")


def summary(rows: list[Row]) -> dict:
    n_pass = sum(1 for r in rows if r["status"] == "PASS")
    n_fail = sum(1 for r in rows if r["status"] == "FAIL")
    n_skip = sum(1 for r in rows if r["status"] == "SKIP")
    return {"pass": n_pass, "fail": n_fail, "skip": n_skip, "total": len(rows)}
