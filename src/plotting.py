"""Figure style.

Two rules, both learned the hard way:
  1. matplotlib on this machine has no CJK font, so *every* label, tick and
     legend entry is in English.  Chinese captions belong in the paper text.
  2. every figure is written twice: .png for pasting into Word, .pdf for
     printing (vector, no blur when zoomed).
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from . import config as C  # noqa: E402

# palette ------------------------------------------------------------------
TRUE = "#1f4e79"     # the truth
SAMPLE = "#c0392b"   # the raw plug-in estimate
SHRINK = "#2e7d32"   # covariance shrinkage
MUSHR = "#7b3fa0"    # mean shrinkage
ORACLE = "#000000"
GREY = "#7f8c8d"
ORANGE = "#e67e22"
BLUE = "#2980b9"

_CYCLE = [SAMPLE, SHRINK, MUSHR, ORANGE, BLUE, GREY]


def setup() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": C.FIG_DPI,
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "lines.linewidth": 1.6,
            "lines.markersize": 4.5,
            "figure.autolayout": False,
        }
    )


def new_fig(width: float = 6.6, height: float = 3.4, nrows: int = 1, ncols: int = 1, **kw):
    fig, ax = plt.subplots(nrows, ncols, figsize=(width, height), **kw)
    return fig, ax


def save(fig, name: str) -> None:
    C.FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / f"{name}.png")
    fig.savefig(C.FIG_DIR / f"{name}.pdf")
    plt.close(fig)


def cycle(i: int) -> str:
    return _CYCLE[i % len(_CYCLE)]


def band(ax, x, mean, sd, color, alpha: float = 0.16) -> None:
    """Shade mean +- 1 sd across replications.

    Drawn on every curve that varies with a replication count, so that a
    difference smaller than the band is visibly not a result.
    """
    x = list(x)
    mean = list(mean)
    sd = list(sd)
    lo = [m - s for m, s in zip(mean, sd)]
    hi = [m + s for m, s in zip(mean, sd)]
    ax.fill_between(x, lo, hi, color=color, alpha=alpha, linewidth=0)
