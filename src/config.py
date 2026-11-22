"""Central configuration.

Every tunable number used by an experiment lives here, so that the experiment
scripts themselves contain no magic constants.  If you want to change the
market, the sample length or the grids, this is the only file you touch.

Units
-----
All returns, means and covariances are expressed in **daily** units.
Annualisation (x252 for means, x252 for variances, x sqrt(252) for Sharpe
ratios) happens only at the reporting / plotting boundary.
"""

from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------
# paths
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "outputs"
FIG_DIR = OUT_DIR / "figures"
TAB_DIR = OUT_DIR / "tables"
LOG_DIR = OUT_DIR / "logs"

# --------------------------------------------------------------------------
# reproducibility
# --------------------------------------------------------------------------
SEED = 20261028
TRADING_DAYS = 252

# --------------------------------------------------------------------------
# replication
# --------------------------------------------------------------------------
# Every statistic is averaged over independent replications of the WHOLE
# experiment: a fresh market drawn from the calibrated factor model, plus a
# fresh estimation window.  A single draw is not usable here -- the plug-in
# tangency portfolio's Sharpe moves by +-0.15 from one 60-day window to the
# next, which is the same order as the effects being measured.
N_REPS = 30
N_REPS_HEAVY = 8        # for the nested bootstrap in e3 (cost is quadratic)
N_BOOTSTRAP = 40        # sample pairs per replication in e3

# --------------------------------------------------------------------------
# market calibration (annualised, US large-cap equity scale)
# --------------------------------------------------------------------------
# The true data generating process is a 3-factor model:
#     r_t = B f_t + eps_t ,   eps_t ~ N(0, D) ,  D diagonal
#     mu_true = B mu_f + alpha
#     Sigma_true = B Sigma_f B' + D
# Calibration targets are the usual stylised facts of daily US equity data:
# market vol ~16%/yr, idiosyncratic vol 15-32%/yr, average pairwise correlation
# ~0.3, factor premia 1-7%/yr, small alpha dispersion.
N_MAX = 55          # largest cross-section used anywhere
T_TRAIN = 60        # estimation window (60 trading days ~ one quarter)
T_TEST = 2000       # out-of-sample evaluation path (~8 years)
K_FACTORS = 3       # market + 2 style factors

MU_MARKET = 0.065          # market factor premium, annualised
VOL_MARKET = 0.160         # market factor vol, annualised
VOL_STYLE = (0.050, 0.035)  # style factor vols, annualised
MU_STYLE = (0.020, 0.010)   # style factor premia, annualised
CORR_MARKET_STYLE = (0.10, -0.05)
CORR_STYLE_STYLE = 0.15
ALPHA_STD = 0.020          # cross-sectional dispersion of true alpha

BETA_MEAN = 1.00
BETA_STD = 0.25
STYLE_LOAD_STD = 0.40
IDIO_VOL_RANGE = (0.15, 0.32)   # annualised idiosyncratic vol, uniform

# --------------------------------------------------------------------------
# experiment grids
# --------------------------------------------------------------------------
N_GRID = tuple(range(5, 56, 5))                 # 5, 10, ..., 55
T_GRID = (40, 60, 90, 120, 180, 250, 400, 600, 1200, 2400)
DELTA_GRID = (0.00, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40,
              0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99)
BLEND_GRID = (0.00, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00)

PERTURB_SIZE = 0.01        # relative Frobenius norm of the covariance shock
N_PERTURB_DRAWS = 60       # random shock directions per (N, T) cell

# --------------------------------------------------------------------------
# leverage definition
# --------------------------------------------------------------------------
# Every portfolio is rescaled to the same annualised volatility target before
# leverage is measured.  Two reasons, both practical:
#   * "gross leverage" then has a fixed meaning -- gross exposure per unit of
#     NAV needed to run a 15%-vol book -- so it is comparable across N;
#   * the alternative convention (force 1'w = 1) divides by 1'Sigma^-1 mu, a
#     scalar that wanders through zero and makes the leverage curve pure noise.
# Sharpe ratios are scale invariant, so this choice cannot flatter anything.
VOL_TARGET = 0.15

# --------------------------------------------------------------------------
# numerical guards
# --------------------------------------------------------------------------
EIG_FLOOR = 1e-14          # below this an eigenvalue is treated as zero
TINY = 1e-14               # denominator guard

# --------------------------------------------------------------------------
# figure style
# --------------------------------------------------------------------------
FIG_DPI = 160
