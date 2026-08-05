from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class ScalingSummaryRow:
    source: str
    target: str
    rows: int
    intercept: float
    coef_log_total_cone_states: float
    coef_log_term_count: float
    coef_log_kmax: float
    r2: float
    adjusted_r2: float
    median_abs_pct_error: float
    notes: str


def first_existing(paths: list[Path]) -> Path:
    for p in paths:
        if p.exists():
            return p
    raise FileNotFoundError(paths)


def fit_model(df: pd.DataFrame, target_col: str, source: str) -> tuple[ScalingSummaryRow, pd.DataFrame]:
    data = df.copy()
    data = data[(data[target_col] > 0) & np.isfinite(data[target_col])]
    data = data[(data["total_cone_states"] > 0) & (data["term_count"] > 0) & (data["kmax"] > 0)]
    features = pd.DataFrame(
        {
            "log_total_cone_states": np.log(data["total_cone_states"].astype(float)),
            "log_term_count": np.log(data["term_count"].astype(float)),
            "log_kmax": np.log(data["kmax"].astype(float)),
        }
    )
    y = np.log(data[target_col].astype(float))
    model = LinearRegression().fit(features, y)
    pred_log = model.predict(features)
    pred = np.exp(pred_log)
    r2 = float(r2_score(y, pred_log))
    n = len(data)
    p = features.shape[1]
    adj = 1.0 - (1.0 - r2) * (n - 1) / max(n - p - 1, 1)
    ape = np.abs(pred - data[target_col].to_numpy(dtype=float)) / np.maximum(data[target_col].to_numpy(dtype=float), 1e-12)
    residual = data.copy()
    residual[f"predicted_{target_col}"] = pred
    residual[f"residual_log_{target_col}"] = y - pred_log
    summary = ScalingSummaryRow(
        source=source,
        target=target_col,
        rows=n,
        intercept=float(model.intercept_),
        coef_log_total_cone_states=float(model.coef_[0]),
        coef_log_term_count=float(model.coef_[1]),
        coef_log_kmax=float(model.coef_[2]),
        r2=r2,
        adjusted_r2=float(adj),
        median_abs_pct_error=float(np.median(ape)),
        notes="log-linear predictor over successful LC rows; failure rows excluded from fit but retained in source CSV",
    )
    return summary, residual


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "lc_obj_seconds" in out.columns:
        out["objective_seconds"] = out["lc_obj_seconds"]
    elif "seconds" in out.columns:
        out["objective_seconds"] = out["seconds"]
    if "lc_grad_seconds" in out.columns:
        out["gradient_seconds"] = out["lc_grad_seconds"]
    elif "adjoint_seconds" in out.columns:
        out["gradient_seconds"] = out["adjoint_seconds"]
    if "term_count" not in out.columns:
        if "m" in out.columns and "fields" in out.columns:
            out["term_count"] = out["m"].fillna(0) + out["fields"].fillna(0)
        elif "total_terms" in out.columns:
            out["term_count"] = out["total_terms"]
    if "kmax" not in out.columns and "state_qubits" in out.columns:
        out["kmax"] = out["state_qubits"]
    needed = ["total_cone_states", "term_count", "kmax"]
    for c in needed:
        if c not in out.columns:
            raise ValueError(f"missing column {c}")
    return out


def write_md(summary_rows: list[ScalingSummaryRow], path: Path) -> None:
    lines = [
        "# P1-2 Scaling-Law Validation",
        "",
        "Model: log(runtime) ~ log(sum_t 2^k_t) + log(term_count) + log(kmax).",
        "",
        "| source | target | rows | R2 | adj R2 | median abs pct error | coef log S | coef log terms | coef log kmax |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in summary_rows:
        lines.append(
            f"| {r.source} | {r.target} | {r.rows} | {r.r2:.4f} | {r.adjusted_r2:.4f} | {r.median_abs_pct_error:.3g} | "
            f"{r.coef_log_total_cone_states:.3g} | {r.coef_log_term_count:.3g} | {r.coef_log_kmax:.3g} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="*", type=Path, default=[
        ROOT / "results" / "benchmark_suite_20260704_3090" / "A1_official_comparator_regime" / "A1_official_comparator_regime.csv",
        ROOT / "results" / "benchmark_suite_20260706" / "A1_official_comparator_regime.csv",
        ROOT / "results" / "benchmark_suite_20260704" / "A3_cone_complexity_law" / "A3_cone_complexity_law_raw.csv",
    ])
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "aaai27_required_experiments_20260710" / "P1_2_scaling_law")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[ScalingSummaryRow] = []
    residual_frames = []
    for input_path in args.inputs:
        if not input_path.exists():
            continue
        df = normalize_columns(pd.read_csv(input_path))
        source = input_path.parent.name + "/" + input_path.name
        for target in ["objective_seconds", "gradient_seconds"]:
            if target not in df.columns:
                continue
            try:
                summary, residual = fit_model(df, target, source)
            except Exception as exc:
                summary_rows.append(ScalingSummaryRow(source, target, 0, float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), f"fit failed: {exc}"))
                continue
            summary_rows.append(summary)
            residual["source"] = source
            residual["target"] = target
            residual_frames.append(residual)
    csv_path = args.out_dir / "P1_2_scaling_law_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(summary_rows[0]).keys()))
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(asdict(row))
    if residual_frames:
        pd.concat(residual_frames, ignore_index=True).to_csv(args.out_dir / "P1_2_scaling_law_residuals.csv", index=False)
    write_md(summary_rows, args.out_dir / "P1_2_scaling_law.md")
    print(f"WROTE {csv_path}")


if __name__ == "__main__":
    main()
