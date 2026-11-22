#!/usr/bin/env python3
"""Single entry point: run the experiments, verify the claims, write the report.

    python3 run_all.py                  # everything, ~1 minute, writes report.md
    python3 run_all.py --only e1,e3     # selected experiments (debugging)
    python3 run_all.py --quick          # coarser grids, smoke test
    python3 run_all.py --no-report      # skip report generation

Exit code is 0 only if every check passes, so this doubles as a regression test.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np

from src import config as C
from src import plotting
from src.checks import evaluate, summary
from src.experiments import (
    e1_conditioning,
    e2_leverage,
    e3_sensitivity,
    e4_shrinkage,
    e5_out_of_sample,
)
from src.market import build_context

EXPERIMENTS = (
    ("e1", "where the ill-conditioning comes from", e1_conditioning),
    ("e2", "the weight vector explodes", e2_leverage),
    ("e3", "mechanism, instability, amplification", e3_sensitivity),
    ("e4", "shrinkage: bias-variance and why mu dominates", e4_shrinkage),
    ("e5", "out of sample: does the repair pay", e5_out_of_sample),
)


class _Encoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, set):
            return sorted(o)
        return super().default(o)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="covariance ill-conditioning in portfolio optimisation")
    parser.add_argument("--only", default="", help="comma separated experiment keys, e.g. e1,e3")
    parser.add_argument("--quick", action="store_true", help="coarser grids, for a smoke test")
    parser.add_argument("--no-report", action="store_true", help="do not write outputs/report.md")
    args = parser.parse_args(argv)

    only = {s.strip() for s in args.only.split(",") if s.strip()}
    plotting.setup()

    print("=" * 78)
    print("Covariance ill-conditioning in portfolio optimisation")
    print(f"  seed={C.SEED}  N_max={C.N_MAX}  T_train={C.T_TRAIN}  T_test={C.T_TEST}")
    print("=" * 78)

    ctx = build_context()
    mkt = ctx.market(0)
    corr = mkt.cov / np.outer(np.sqrt(np.diag(mkt.cov)), np.sqrt(np.diag(mkt.cov)))
    print(f"market: {mkt.n_assets} assets, kappa(Sigma_true) = "
          f"{np.linalg.cond(mkt.cov):.1f}   (replication 0 of {ctx.n_reps})")
    print(f"  annualised mean return  : {mkt.mu.mean() * C.TRADING_DAYS * 100:.2f}%")
    print(f"  annualised vol          : {np.sqrt(np.diag(mkt.cov).mean() * C.TRADING_DAYS) * 100:.2f}%")
    print(f"  mean pairwise corr      : {corr.mean():.3f}")
    print()

    metrics: dict = {}
    for key, title, module in EXPERIMENTS:
        if only and key not in only:
            continue
        t0 = time.time()
        print(f"[{key}] {title} ...", flush=True)
        try:
            metrics[key] = module.run(ctx, quick=args.quick)
        except Exception as exc:  # noqa: BLE001
            print(f"    FAILED: {type(exc).__name__}: {exc}")
            raise
        elapsed = time.time() - t0
        metrics[key]["_seconds"] = round(elapsed, 2)
        print(f"    done in {elapsed:.1f}s", flush=True)

    # ------------------------------------------------------------- checks
    checks = evaluate(metrics)
    stat = summary(checks)

    print()
    print("-" * 78)
    print("verification")
    print("-" * 78)
    width = max((len(r["check"]) for r in checks), default=10)
    for r in checks:
        mark = {"PASS": "ok  ", "FAIL": "FAIL", "SKIP": "skip"}[r["status"]]
        print(f"  [{mark}] {r['check']:<{width}}  {r['measured']:<24} ({r['expected']})")
    print("-" * 78)
    print(f"  {stat['pass']} passed, {stat['fail']} failed, {stat['skip']} skipped")
    print("-" * 78)

    # ------------------------------------------------------------- outputs
    C.LOG_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "checks": checks,
        "summary": stat,
        "metrics": metrics,
        "config": {
            "seed": C.SEED,
            "n_max": C.N_MAX,
            "t_train": C.T_TRAIN,
            "t_test": C.T_TEST,
            "n_grid": list(C.N_GRID),
        },
    }
    with open(C.LOG_DIR / "metrics.json", "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, cls=_Encoder, ensure_ascii=False)
    print(f"\n  metrics -> {C.LOG_DIR / 'metrics.json'}")

    if not args.no_report:
        from src.report import write_report

        path = write_report(metrics, checks, stat)
        print(f"  report  -> {path}")

    print()
    return 0 if stat["fail"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
