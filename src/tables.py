"""CSV output helper (the tables the paper quotes)."""

from __future__ import annotations

import csv
import math

from . import config as C


def _fmt(v):
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, float):
        if math.isnan(v):
            return "nan"
        if math.isinf(v):
            return "inf" if v > 0 else "-inf"
        return f"{v:.6g}"
    return v


def write_csv(rows: list[dict], name: str) -> None:
    if not rows:
        return
    C.TAB_DIR.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    path = C.TAB_DIR / f"{name}.csv"
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: _fmt(r.get(k, "")) for k in keys})
