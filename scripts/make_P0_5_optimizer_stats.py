from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class OptimizerStatRow:
    case: str
    optimizer_a: str
    optimizer_b: str
    paired_runs: int
    metric: str
    median_delta: float
    mean_delta: float
    ci95_low: float
    ci95_high: float
    wilcoxon_p: float
    effect_size_median_abs: float
    status: str
    notes: str


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator, reps: int = 2000) -> tuple[float, float]:
    if values.size == 0:
        return float("nan"), float("nan")
    samples = []
    for _ in range(reps):
        draw = rng.choice(values, size=values.size, replace=True)
        samples.append(float(np.median(draw)))
    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def paired_stats(df: pd.DataFrame, case: str, opt_a: str, opt_b: str, metric: str, rng) -> OptimizerStatRow:
    keys = ["case", "seed", "init_id"]
    a = df[(df["case"] == case) & (df["optimizer"] == opt_a) & (df["status"] == "ok")][keys + [metric]].rename(columns={metric: "a"})
    b = df[(df["case"] == case) & (df["optimizer"] == opt_b) & (df["status"] == "ok")][keys + [metric]].rename(columns={metric: "b"})
    merged = a.merge(b, on=keys, how="inner")
    if merged.empty:
        return OptimizerStatRow(case, opt_a, opt_b, 0, metric, float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), "NO_PAIRS", "no matched seed/init rows")
    delta = (merged["a"].to_numpy(dtype=float) - merged["b"].to_numpy(dtype=float))
    low, high = bootstrap_ci(delta[np.isfinite(delta)], rng)
    try:
        p = float(wilcoxon(delta).pvalue) if np.any(np.abs(delta) > 0) else 1.0
    except Exception:
        p = float("nan")
    denom = float(np.median(np.abs(merged["b"].to_numpy(dtype=float)))) or 1.0
    effect = float(np.median(delta) / max(denom, 1e-12))
    return OptimizerStatRow(
        case=case,
        optimizer_a=opt_a,
        optimizer_b=opt_b,
        paired_runs=int(len(merged)),
        metric=metric,
        median_delta=float(np.nanmedian(delta)),
        mean_delta=float(np.nanmean(delta)),
        ci95_low=low,
        ci95_high=high,
        wilcoxon_p=p,
        effect_size_median_abs=effect,
        status="ok",
        notes="paired by graph seed and initialization; rows, not raw optimizer steps, are the statistical unit",
    )


def write_md(rows: list[OptimizerStatRow], path: Path) -> None:
    lines = [
        "# P0-5 Optimizer Fairness and Statistical Analysis",
        "",
        "Statistics are paired by `(case, graph seed, initialization)`; the sample unit is an optimizer run, not a CSV row or training step.",
        "",
        "| case | comparison | metric | pairs | median delta | 95% CI | Wilcoxon p | effect |",
        "|---|---|---|---:|---:|---|---:|---:|",
    ]
    for r in rows:
        if r.status != "ok":
            continue
        lines.append(
            f"| {r.case} | {r.optimizer_a} - {r.optimizer_b} | {r.metric} | {r.paired_runs} | {r.median_delta:.4g} | "
            f"[{r.ci95_low:.4g}, {r.ci95_high:.4g}] | {r.wilcoxon_p:.3g} | {r.effect_size_median_abs:.4g} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=ROOT / "results" / "aaai27_final_curated_20260707" / "A2_training_quality_consolidated.csv")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "aaai27_required_experiments_20260710" / "P0_5_optimizer_stats")
    parser.add_argument("--cases", nargs="*", default=None)
    parser.add_argument("--metrics", nargs="*", default=["best_value", "normalized_improvement", "time_to_95pct_best"])
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.input)
    if args.cases:
        cases = args.cases
    else:
        cases = sorted(df["case"].dropna().unique().tolist())
    comparisons = [
        ("lc_adjoint_adam", "spsa"),
        ("lc_adjoint_amsgrad", "spsa"),
        ("lc_adjoint_adam", "random_search"),
        ("lbfgsb", "nelder_mead"),
        ("lbfgsb", "spsa"),
    ]
    rng = np.random.default_rng(91017)
    rows: list[OptimizerStatRow] = []
    for case in cases:
        for metric in args.metrics:
            if metric not in df.columns:
                continue
            for a, b in comparisons:
                rows.append(paired_stats(df, case, a, b, metric, rng))
    csv_path = args.out_dir / "P0_5_optimizer_stats.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
    write_md(rows, args.out_dir / "P0_5_optimizer_stats.md")
    print(f"WROTE {csv_path}")


if __name__ == "__main__":
    main()
