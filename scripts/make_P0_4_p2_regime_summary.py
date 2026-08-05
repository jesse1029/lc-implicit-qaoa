from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class P04Row:
    regime: str
    family: str
    p: int
    n_min: int
    n_max: int
    rows: int
    success_rows: int
    kmax_median: float
    kmax_max: float
    total_cone_states_median: float
    lc_obj_median_s: float
    lc_obj_mean_s: float
    lc_obj_iqr_s: float
    lc_obj_ci95_low_s: float
    lc_obj_ci95_high_s: float
    lc_grad_median_s: float
    lc_grad_mean_s: float
    lc_grad_iqr_s: float
    lc_peak_median_mb: float
    full_state_success_rows: int
    full_state_failure_rows: int
    notes: str


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator, reps: int = 2000) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    boot = [float(np.median(rng.choice(values, size=values.size, replace=True))) for _ in range(reps)]
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def classify_regime(row) -> str:
    kmax = float(row.get("kmax", np.nan))
    total = float(row.get("total_cone_states", np.nan))
    p = int(row.get("p", 0))
    if p < 2:
        return "diagnostic_p1_excluded"
    if np.isfinite(kmax) and np.isfinite(total) and kmax <= 14 and total <= 2.0e7:
        return "target_bounded_cone_kmax_le14"
    if np.isfinite(kmax) and np.isfinite(total) and kmax <= 24 and total <= 1.0e9:
        return "near_guardrail_kmax_le24"
    return "negative_or_out_of_regime"


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in [
        "n",
        "p",
        "kmax",
        "total_cone_states",
        "lc_obj_seconds",
        "lc_grad_seconds",
        "lc_peak_mb",
        "m",
        "fields",
    ]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    if "lc_obj_status" not in out.columns and "status" in out.columns:
        out["lc_obj_status"] = out["status"]
    if "lc_obj_seconds" not in out.columns and "seconds" in out.columns:
        out["lc_obj_seconds"] = out["seconds"]
    if "lc_grad_seconds" not in out.columns and "adjoint_seconds" in out.columns:
        out["lc_grad_seconds"] = out["adjoint_seconds"]
    if "lc_peak_mb" not in out.columns:
        if "peak_mb" in out.columns:
            out["lc_peak_mb"] = out["peak_mb"]
        elif "lc_peak_pool_mb" in out.columns:
            out["lc_peak_mb"] = out["lc_peak_pool_mb"]
        else:
            out["lc_peak_mb"] = np.nan
    if "full_precompute_status" not in out.columns:
        out["full_precompute_status"] = "NOT_RECORDED"
    out["regime"] = out.apply(classify_regime, axis=1)
    return out


def summarize(df: pd.DataFrame, rng) -> list[P04Row]:
    rows: list[P04Row] = []
    df = df[df["p"] >= 2].copy()
    for (regime, family, p), sub in df.groupby(["regime", "family", "p"], dropna=False):
        ok = sub[sub["lc_obj_status"].astype(str).str.lower().isin(["success", "ok"])]
        obj = ok["lc_obj_seconds"].to_numpy(dtype=float) if "lc_obj_seconds" in ok else np.asarray([])
        grad = ok["lc_grad_seconds"].to_numpy(dtype=float) if "lc_grad_seconds" in ok else np.asarray([])
        peak = ok["lc_peak_mb"].to_numpy(dtype=float) if "lc_peak_mb" in ok else np.asarray([])
        low, high = bootstrap_ci(obj, rng)
        full_status = sub["full_precompute_status"].astype(str).str.lower()
        full_success = int(full_status.isin(["success", "ok"]).sum())
        full_fail = int((full_status.str.contains("oom") | full_status.str.contains("failed") | full_status.str.contains("timeout")).sum())
        q75, q25 = (np.nanpercentile(obj, 75), np.nanpercentile(obj, 25)) if obj.size else (float("nan"), float("nan"))
        gq75, gq25 = (np.nanpercentile(grad, 75), np.nanpercentile(grad, 25)) if grad.size else (float("nan"), float("nan"))
        rows.append(
            P04Row(
                regime=str(regime),
                family=str(family),
                p=int(p),
                n_min=int(np.nanmin(sub["n"])),
                n_max=int(np.nanmax(sub["n"])),
                rows=int(len(sub)),
                success_rows=int(len(ok)),
                kmax_median=float(np.nanmedian(sub["kmax"])),
                kmax_max=float(np.nanmax(sub["kmax"])),
                total_cone_states_median=float(np.nanmedian(sub["total_cone_states"])),
                lc_obj_median_s=float(np.nanmedian(obj)) if obj.size else float("nan"),
                lc_obj_mean_s=float(np.nanmean(obj)) if obj.size else float("nan"),
                lc_obj_iqr_s=float(q75 - q25) if obj.size else float("nan"),
                lc_obj_ci95_low_s=low,
                lc_obj_ci95_high_s=high,
                lc_grad_median_s=float(np.nanmedian(grad)) if grad.size else float("nan"),
                lc_grad_mean_s=float(np.nanmean(grad)) if grad.size else float("nan"),
                lc_grad_iqr_s=float(gq75 - gq25) if grad.size else float("nan"),
                lc_peak_median_mb=float(np.nanmedian(peak)) if peak.size else float("nan"),
                full_state_success_rows=full_success,
                full_state_failure_rows=full_fail,
                notes="p>=2 only; regimes are pre-defined by kmax and total cone states, not by observed speed",
            )
        )
    return sorted(rows, key=lambda r: (r.regime, r.family, r.p))


def write_md(rows: list[P04Row], path: Path) -> None:
    lines = [
        "# P0-4 p>=2 Benchmark Regime Summary",
        "",
        "This table excludes p=1 from main comparisons. Target and near-guardrail regimes are defined before looking at speed: `target` uses kmax<=14 and sum_t 2^k_t<=2e7; `near_guardrail` uses kmax<=24 and sum_t 2^k_t<=1e9.",
        "",
        "| regime | family | p | n range | rows ok/all | obj median s [95% CI] | grad median s | peak MB | full-state fail/success |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r.regime} | {r.family} | {r.p} | {r.n_min}-{r.n_max} | {r.success_rows}/{r.rows} | "
            f"{r.lc_obj_median_s:.4g} [{r.lc_obj_ci95_low_s:.4g},{r.lc_obj_ci95_high_s:.4g}] | "
            f"{r.lc_grad_median_s:.4g} | {r.lc_peak_median_mb:.4g} | {r.full_state_failure_rows}/{r.full_state_success_rows} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="*", type=Path, default=[
        ROOT / "results" / "benchmark_suite_20260704_3090" / "A1_official_comparator_regime" / "A1_official_comparator_regime.csv",
        ROOT / "results" / "benchmark_suite_20260706" / "A1_official_comparator_regime.csv",
    ])
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "aaai27_required_experiments_20260710" / "P0_4_p2_regime_summary")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for p in args.inputs:
        if p.exists():
            df = pd.read_csv(p)
            df["source_path"] = str(p)
            frames.append(normalize(df))
    if not frames:
        raise FileNotFoundError(args.inputs)
    df = pd.concat(frames, ignore_index=True)
    # The follow-up sweep deliberately repeats several deterministic seeds on a
    # second host.  Those rows are timing replications, not independent graph
    # instances.  Preserve input priority and retain one row per graph key.
    graph_key = [c for c in ["family", "n", "p", "seed"] if c in df.columns]
    if len(graph_key) == 4:
        before = len(df)
        df = df.drop_duplicates(subset=graph_key, keep="first").copy()
        print(f"DEDUP {before - len(df)} repeated graph-seed rows by {graph_key}")
    rows = summarize(df, np.random.default_rng(91024))
    csv_path = args.out_dir / "P0_4_p2_regime_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
    write_md(rows, args.out_dir / "P0_4_p2_regime_summary.md")
    print(f"WROTE {csv_path}")


if __name__ == "__main__":
    main()
