#!/usr/bin/env python
"""Paired analysis of E3: exact training versus angle transfer."""
from __future__ import annotations
import csv, json, math, sys
from pathlib import Path
import numpy as np

SRC = Path(sys.argv[1] if len(sys.argv) > 1 else "results/E3_training_vs_transfer")
ARMS = ["transfer_same_n", "transfer_small_n", "random_best", "init"]


def boot(d, reps=20000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(reps, len(d)))
    m = np.median(d[idx], axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def wilcoxon_pn(d):
    """Two-sided sign test p-value; exact, no SciPy dependency."""
    pos = int((d > 0).sum()); neg = int((d < 0).sum()); n = pos + neg
    if n == 0:
        return 1.0
    k = min(pos, neg)
    c = sum(math.comb(n, i) for i in range(k + 1))
    return min(1.0, 2.0 * c / (2 ** n))


rows = [r for r in csv.DictReader(open(SRC / "E3_raw.csv", encoding="utf-8"))
        if r["status"] == "ok" and r["obj_lc_trained"] not in ("", "nan")]
cells, out = {}, []
for r in rows:
    cells.setdefault((r["family"], int(r["n"])), []).append(r)

for (fam, n), rs in sorted(cells.items()):
    lc = np.array([float(r["obj_lc_trained"]) for r in rs])
    rec = dict(family=fam, n=n, seeds=len(rs),
               kmax_min=min(int(r["kmax"]) for r in rs), kmax_max=max(int(r["kmax"]) for r in rs))
    for arm in ARMS:
        base = np.array([float(r[f"obj_{arm}"]) for r in rs])
        rel = (lc - base) / np.maximum(np.abs(base), 1.0)          # paired, per instance
        lo, hi = boot(rel)
        rec[f"gain_vs_{arm}_median_pct"] = round(100 * float(np.median(rel)), 4)
        rec[f"gain_vs_{arm}_ci_pct"] = f"[{100*lo:.4f}, {100*hi:.4f}]"
        rec[f"gain_vs_{arm}_wins"] = f"{int((lc > base).sum())}/{len(rs)}"
        rec[f"gain_vs_{arm}_sign_p"] = round(wilcoxon_pn(lc - base), 5)
    out.append(rec)

with (SRC / "E3_paired_summary.csv").open("w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
    w.writeheader(); w.writerows(out)

print(f"{'family':<14}{'n':>5}{'seeds':>6}{'kmax':>9}   "
      f"{'vs same-n transfer':>28}{'vs small-n transfer':>28}{'vs random':>22}")
for r in out:
    print(f"{r['family']:<14}{r['n']:>5}{r['seeds']:>6}{str(r['kmax_min'])+'-'+str(r['kmax_max']):>9}   "
          f"{r['gain_vs_transfer_same_n_median_pct']:>10.3f}% {r['gain_vs_transfer_same_n_ci_pct']:>16} "
          f"{r['gain_vs_transfer_small_n_median_pct']:>10.3f}% {r['gain_vs_transfer_small_n_ci_pct']:>16} "
          f"{r['gain_vs_random_best_median_pct']:>10.3f}% {r['gain_vs_random_best_wins']:>8}")

# family-level pooled statement
print("\npooled by family (all sizes):")
for fam in sorted({r["family"] for r in rows}):
    rs = [r for r in rows if r["family"] == fam]
    lc = np.array([float(r["obj_lc_trained"]) for r in rs])
    for arm in ("transfer_same_n", "transfer_small_n"):
        base = np.array([float(r[f"obj_{arm}"]) for r in rs])
        rel = (lc - base) / np.maximum(np.abs(base), 1.0)
        lo, hi = boot(rel)
        print(f"  {fam:<14} vs {arm:<18} median {100*np.median(rel):8.3f}%  "
              f"CI [{100*lo:.3f}, {100*hi:.3f}]  wins {int((lc>base).sum())}/{len(rs)}  "
              f"sign p={wilcoxon_pn(lc-base):.5f}")
print(f"\nwrote {SRC/'E3_paired_summary.csv'}")
