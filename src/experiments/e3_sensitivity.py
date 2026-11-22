"""Experiment 3 -- the mechanism, the instability, and who is to blame.

(a) Mechanism.  In the eigenbasis of Sigma the solution is
        x = Sigma^-1 mu = sum_i (v_i' mu / lambda_i) v_i
    so the coefficient of direction i is (v_i' mu)/lambda_i.  If the sample
    matrix has a nearly-zero eigenvalue with a non-zero projection of mu on it,
    that single term dominates.  Plotting |c_i| against the eigenvalue index
    shows it directly.

(b) Instability, and the decomposition of its cause.  Draw two independent
    60-day panels from the SAME market and optimise on each.  If the answer were
    a property of the market, the two weight vectors would agree.

    Then split the blame by re-running the optimiser with one moment held at its
    true value:

        both        Sigma and mu estimated      (what an investor actually gets)
        Sigma only  mu replaced by the truth    (only Sigma is a problem)
        mu only     Sigma replaced by the truth (only mu is a problem)

    Whichever variant reproduces the instability is the binding constraint.

(c) Amplification.  Perturb Sigma by 1% (Frobenius norm) in random symmetric
    directions and measure the relative change in x.  First-order perturbation
    theory says

        ||dx||/||x||  <=  kappa(S_hat) * ||dSigma||/||Sigma||

    The constant is the condition number of the matrix that is actually
    inverted (S_hat), not of the true Sigma: the expansion contains
    ||S_hat^-1|| * ||S_hat||.  Using kappa(Sigma) instead would make the bound
    appear violated at large N, so the two must not be mixed up.

    so the measured amplification should sit below kappa and not far below it.

Outputs
    fig4_mechanism.png, fig5_instability.png, fig6_amplification.png,
    e3_amplification.csv, e3_two_sample.csv, e3_decomposition.csv
"""

from __future__ import annotations

import numpy as np

from .. import config as C
from ..aggregate import group_mean, value
from ..analysis import (
    amplification,
    coefficient_share,
    condition_number,
    eigen_contributions,
    population_sharpe,
    relative_error,
    solve,
    turnover,
    weight_corr,
)
from ..estimators import sample_moments
from ..plotting import (
    BLUE,
    GREY,
    MUSHR,
    ORANGE,
    SAMPLE,
    SHRINK,
    TRUE,
    band,
    new_fig,
    save,
)
from ..portfolio import tangency, to_vol_target
from ..tables import write_csv

VARIANT_COLORS = {"both": SAMPLE, "sigma_only": SHRINK, "mu_only": MUSHR}
VARIANT_LABELS = {
    "both": "both estimated",
    "sigma_only": "$\\Sigma$ only ($\\mu$ true)",
    "mu_only": "$\\mu$ only ($\\Sigma$ true)",
}


# --------------------------------------------------------------------------
# (a) mechanism
# --------------------------------------------------------------------------
def _mechanism(ctx, n_assets: int) -> dict:
    coef_t, coef_h, sh_t, sh_h = [], [], [], []
    for rep in range(ctx.n_reps):
        sub = ctx.subset(rep, n_assets)
        ret = ctx.sample(rep, n_assets, C.T_TRAIN, tag=1)
        mu_hat, s_hat = sample_moments(ret)

        d_t = eigen_contributions(sub.mu, sub.cov)
        d_h = eigen_contributions(mu_hat, s_hat)
        coef_t.append(np.abs(d_t["coefficient"]))
        coef_h.append(np.abs(d_h["coefficient"]))
        sh_t.append(coefficient_share(sub.mu, sub.cov))
        sh_h.append(coefficient_share(mu_hat, s_hat))

    c_t = np.nanmedian(np.vstack(coef_t), axis=0)
    c_h = np.nanmedian(np.vstack(coef_h), axis=0)
    s_t = np.nanmean(np.vstack(sh_t), axis=0)
    s_h = np.nanmean(np.vstack(sh_h), axis=0)

    return {
        "coef_true": c_t,
        "coef_hat": c_h,
        "share_true": s_t,
        "share_hat": s_h,
        "top1_share_true": float(s_t[0]),
        "top1_share_hat": float(s_h[0]),
        "top3_share_true": float(s_t[:3].sum()),
        "top3_share_hat": float(s_h[:3].sum()),
    }


def _fig_mechanism(m: dict, n_assets: int) -> None:
    fig, (ax1, ax2) = new_fig(7.2, 3.2, ncols=2)

    idx = np.arange(1, n_assets + 1)
    ax1.semilogy(idx, m["coef_true"], "-", color=TRUE, label="true $\\Sigma,\\mu$")
    ax1.semilogy(idx, m["coef_hat"], "-", color=SAMPLE, label="sample $\\hat S,\\hat\\mu$")
    ax1.set_title("(a) $|c_i|$ in $\\Sigma^{-1}\\mu=\\sum_i c_i v_i$")
    ax1.set_xlabel("eigen-direction (1 = smallest $\\lambda_i$)")
    ax1.set_ylabel("$|c_i|$  (median over replications)")
    ax1.legend(frameon=False)

    k = np.arange(1, n_assets + 1)
    ax2.plot(k, np.cumsum(m["share_true"]) * 100, "-", color=TRUE, label="true")
    ax2.plot(k, np.cumsum(m["share_hat"]) * 100, "-", color=SAMPLE, label="sample")
    ax2.axhline(90, color=ORANGE, ls=":", lw=1.0)
    ax2.set_title("(b) cumulative share of $\\|\\Sigma^{-1}\\mu\\|^2$")
    ax2.set_xlabel("number of eigen-directions used")
    ax2.set_ylabel("share (%)")
    ax2.set_ylim(0, 105)
    ax2.legend(frameon=False, loc="lower right")
    save(fig, "fig4_mechanism")


# --------------------------------------------------------------------------
# (b) instability and decomposition
# --------------------------------------------------------------------------
def _decomposition(ctx, n_grid) -> list[dict]:
    """Sharpe of the oracle and of the three estimation variants, per N."""
    raw: list[dict] = []
    for rep in range(ctx.n_reps):
        for n in n_grid:
            sub = ctx.subset(rep, n)
            mu_hat, s_hat = sample_moments(ctx.sample(rep, n, C.T_TRAIN, tag=1))

            variants = {
                "oracle": tangency(sub.mu, sub.cov),
                "both": tangency(mu_hat, s_hat),
                "sigma_only": tangency(sub.mu, s_hat),
                "mu_only": tangency(mu_hat, sub.cov),
            }
            for name, w in variants.items():
                if w is None:
                    continue
                cov_used = sub.cov if name in ("oracle", "mu_only") else s_hat
                w_s = to_vol_target(w, cov_used)
                if w_s is None:
                    continue
                raw.append(
                    {
                        "rep": rep,
                        "n_assets": n,
                        "variant": name,
                        "pop_sharpe": population_sharpe(w_s, sub.mu, sub.cov),
                    }
                )
    return raw


def _two_sample(ctx, n_assets: int, n_boot: int) -> dict:
    """Correlation of two independent estimates, and of the three variants."""
    sub_corr = {k: [] for k in ("both", "sigma_only", "mu_only")}
    head_corr, head_turn, head_err = [], [], []

    for rep in range(min(ctx.n_reps, C.N_REPS_HEAVY)):
        sub = ctx.subset(rep, n_assets)
        m1, s1 = sample_moments(ctx.sample(rep, n_assets, C.T_TRAIN, tag=1000))
        m2, s2 = sample_moments(ctx.sample(rep, n_assets, C.T_TRAIN, tag=2000))

        w1 = to_vol_target(tangency(m1, s1), s1)
        w2 = to_vol_target(tangency(m2, s2), s2)
        head_corr.append(weight_corr(w1, w2))
        head_turn.append(turnover(w1 / np.abs(w1).sum(), w2 / np.abs(w2).sum()))

        w_true = tangency(sub.mu, sub.cov)
        head_err.append(relative_error(w1, w_true))

        for i in range(n_boot):
            ma, sa = sample_moments(ctx.sample(rep, n_assets, C.T_TRAIN, tag=10000 + 2 * i))
            mb, sb = sample_moments(ctx.sample(rep, n_assets, C.T_TRAIN, tag=10001 + 2 * i))

            sub_corr["both"].append(
                weight_corr(to_vol_target(tangency(ma, sa), sa),
                            to_vol_target(tangency(mb, sb), sb))
            )
            sub_corr["sigma_only"].append(
                weight_corr(to_vol_target(tangency(sub.mu, sa), sa),
                            to_vol_target(tangency(sub.mu, sb), sb))
            )
            sub_corr["mu_only"].append(
                weight_corr(to_vol_target(tangency(ma, sub.cov), sub.cov),
                            to_vol_target(tangency(mb, sub.cov), sub.cov))
            )

    return {
        "corr_head": float(np.nanmean(head_corr)),
        "turnover_head": float(np.nanmean(head_turn)),
        "err_head": float(np.nanmean(head_err)),
        "corr": {k: np.asarray(v, dtype=float) for k, v in sub_corr.items()},
    }


def _fig_instability(ctx, n_assets: int, two: dict, decomp_rows: list[dict]) -> None:
    fig, (ax1, ax2) = new_fig(7.4, 3.3, ncols=2)

    # (a) the two books side by side
    m1, s1 = sample_moments(ctx.sample(0, n_assets, C.T_TRAIN, tag=1000))
    m2, s2 = sample_moments(ctx.sample(0, n_assets, C.T_TRAIN, tag=2000))
    w1 = to_vol_target(tangency(m1, s1), s1)
    w2 = to_vol_target(tangency(m2, s2), s2)
    order = np.argsort(-np.abs(w1))[:12]
    x = np.arange(len(order))
    ax1.bar(x - 0.2, w1[order], width=0.4, color=BLUE, label="sample 1")
    ax1.bar(x + 0.2, w2[order], width=0.4, color=SAMPLE, label="sample 2")
    ax1.axhline(0, color="black", lw=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(i) for i in order], fontsize=6.5)
    ax1.set_title(f"(a) top-12 positions, N={n_assets}, two 60-day panels")
    ax1.set_xlabel("asset index")
    ax1.set_ylabel("weight ($\\times$ NAV at 15% vol)")
    ax1.legend(frameon=False, fontsize=7.5)

    # (b) Sharpe of the three variants against N, labelled in place
    for name, color, va in (("oracle", TRUE, "bottom"), ("sigma_only", SHRINK, "bottom"),
                            ("mu_only", MUSHR, "bottom"), ("both", SAMPLE, "top")):
        sub = [r for r in decomp_rows if r["variant"] == name]
        xs = [r["n_assets"] for r in sub]
        ys = [r["pop_sharpe"] for r in sub]
        style = "--" if name == "oracle" else "-"
        ax2.plot(xs, ys, style, color=color)
        if name != "oracle":
            band(ax2, xs, ys, [r["pop_sharpe_sd"] for r in sub], color)
        label = "oracle (true $\\mu,\\Sigma$)" if name == "oracle" else VARIANT_LABELS[name]
        ax2.annotate(label, xy=(xs[-1], ys[-1]), xytext=(4, 0 if name != "both" else -10),
                     textcoords="offset points", fontsize=7, color=color,
                     va=va, ha="left")
    ax2.axhline(0, color="black", lw=0.8)
    ax2.set_xlim(min(xs) - 2, max(xs) + 18)
    ax2.set_title("(b) population Sharpe: who is to blame")
    ax2.set_xlabel("number of assets N")
    ax2.set_ylabel("annualised Sharpe")
    save(fig, "fig5_instability")


# --------------------------------------------------------------------------
# (c) amplification and the error bound
# --------------------------------------------------------------------------
def _amplification_table(ctx, n_grid) -> list[dict]:
    raw: list[dict] = []
    for rep in range(ctx.n_reps):
        for n in n_grid:
            sub = ctx.subset(rep, n)
            mu_hat, s_hat = sample_moments(ctx.sample(rep, n, C.T_TRAIN, tag=1))

            amp = amplification(s_hat, mu_hat, C.PERTURB_SIZE, C.N_PERTURB_DRAWS,
                                seed=C.SEED + 17 * n + 101 * rep)

            x_true = solve(sub.cov, sub.mu)
            x_hat = solve(s_hat, mu_hat)
            rel_x = relative_error(x_hat, x_true)
            rel_mu = relative_error(mu_hat, sub.mu)
            rel_cov = relative_error(s_hat, sub.cov)
            kappa = amp["kappa"]
            bound = kappa * (rel_mu + rel_cov) if np.isfinite(kappa) else float("inf")

            raw.append(
                {
                    "rep": rep,
                    "n_assets": n,
                    "T": C.T_TRAIN,
                    "kappa_sample": kappa,
                    "amplification": amp["amplification"],
                    "amp_over_kappa": amp["amplification"] / kappa if np.isfinite(kappa) else float("nan"),
                    "rel_error_mu": rel_mu,
                    "rel_error_cov": rel_cov,
                    "rel_error_x": rel_x,
                    "bound": bound,
                    "error_over_bound": rel_x / bound if bound > 0 else float("nan"),
                }
            )
    return group_mean(
        raw,
        ["n_assets", "T"],
        ["kappa_sample", "amplification", "amp_over_kappa",
         "rel_error_mu", "rel_error_cov", "rel_error_x", "bound", "error_over_bound"],
    )


def _fig_amplification(rows: list[dict]) -> None:
    fig, (ax1, ax2) = new_fig(7.2, 3.2, ncols=2)

    kappa = np.array([r["kappa_sample"] for r in rows], dtype=float)
    amp = np.array([r["amplification"] for r in rows], dtype=float)

    ax1.loglog(kappa, amp, "o-", color=SAMPLE, label="measured (median over draws)")
    lo, hi = np.nanmin(kappa) * 0.5, np.nanmax(kappa) * 2.0
    ax1.loglog([lo, hi], [lo, hi], "--", color=GREY, lw=1.0, label="$y=\\kappa$ (bound)")
    ax1.set_title("(a) 1% shock to $\\hat S$: amplification of $x=\\hat S^{-1}\\hat\\mu$")
    ax1.set_xlabel("$\\kappa(\\hat S)$")
    ax1.set_ylabel("$\\|\\delta x\\|/\\|x\\| \\;/\\; \\|\\delta S\\|/\\|S\\|$")
    ax1.legend(frameon=False, fontsize=7.5)

    err = np.array([r["rel_error_x"] for r in rows], dtype=float)
    bnd = np.array([r["bound"] for r in rows], dtype=float)
    ns = [r["n_assets"] for r in rows]
    ax2.loglog(bnd, err, "s-", color=SHRINK, label="plug-in error")
    lo2 = max(np.nanmin(bnd), 1e-3) * 0.5
    hi2 = np.nanmax(err) * 2.0
    ax2.loglog([lo2, hi2], [lo2, hi2], "--", color=GREY, lw=1.0,
               label="$y=\\kappa(\\hat S)(\\varepsilon_\\mu+\\varepsilon_\\Sigma)$")
    for nn, xx, yy in zip(ns, bnd, err):
        ax2.annotate(str(nn), (xx, yy), fontsize=6.5, color=GREY,
                     textcoords="offset points", xytext=(3, 3))
    ax2.set_title("(b) observed error vs first-order bound")
    ax2.set_xlabel("$\\kappa(\\hat S)(\\varepsilon_\\mu+\\varepsilon_\\Sigma)$")
    ax2.set_ylabel("$\\|\\hat x-x^\\star\\|/\\|x^\\star\\|$")
    ax2.legend(frameon=False, fontsize=7.5, loc="upper left")
    save(fig, "fig6_amplification")


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------
def run(ctx, quick: bool = False) -> dict:
    n_grid = C.N_GRID if not quick else (5, 15, 30, 55)
    n_big = n_grid[-1]
    n_boot = C.N_BOOTSTRAP if not quick else 12

    mech = _mechanism(ctx, n_big)
    _fig_mechanism(mech, n_big)

    decomp_raw = _decomposition(ctx, n_grid)
    decomp = group_mean(decomp_raw, ["n_assets", "variant"], ["pop_sharpe"])
    write_csv(decomp, "e3_decomposition")

    two = _two_sample(ctx, n_big, n_boot)
    _fig_instability(ctx, n_big, two, decomp)

    amp_rows = _amplification_table(ctx, n_grid)
    _fig_amplification(amp_rows)
    write_csv(amp_rows, "e3_amplification")

    two_rows = [
        {
            "n_assets": n_big,
            "corr_two_samples": two["corr_head"],
            "turnover_two_samples": two["turnover_head"],
            "rel_error_vs_truth": two["err_head"],
            "corr_both_mean": float(np.nanmean(two["corr"]["both"])),
            "corr_sigma_only_mean": float(np.nanmean(two["corr"]["sigma_only"])),
            "corr_mu_only_mean": float(np.nanmean(two["corr"]["mu_only"])),
            "corr_sigma_only_sd": float(np.nanstd(two["corr"]["sigma_only"])),
            "corr_mu_only_sd": float(np.nanstd(two["corr"]["mu_only"])),
        }
    ]
    write_csv(two_rows, "e3_two_sample")

    def dvariant(name, field="pop_sharpe", n=n_big):
        return value(decomp, field, n_assets=n, variant=name)

    return {
        "n_big": n_big,
        "mechanism": mech,
        "decomposition": decomp,
        "two_sample": two_rows,
        "corr_big": two["corr_head"],
        "turnover_big": two["turnover_head"],
        "corr_both": float(np.nanmean(two["corr"]["both"])),
        "corr_sigma_only": float(np.nanmean(two["corr"]["sigma_only"])),
        "corr_mu_only": float(np.nanmean(two["corr"]["mu_only"])),
        "sharpe_oracle": dvariant("oracle"),
        "sharpe_both": dvariant("both"),
        "sharpe_sigma_only": dvariant("sigma_only"),
        "sharpe_mu_only": dvariant("mu_only"),
        "sharpe_sigma_only_n30": dvariant("sigma_only", n=30),
        "sharpe_mu_only_n30": dvariant("mu_only", n=30),
        "eps_mu": value(amp_rows, "rel_error_mu", n_assets=n_big),
        "eps_cov": value(amp_rows, "rel_error_cov", n_assets=n_big),
        "amplification": amp_rows,
    }
