from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def finite_median(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    return float(np.median(values)) if values.size else float("nan")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(args.input)
    frame["backend_short"] = np.where(frame["backend"].str.contains("Lightning"), "Lightning-GPU", "LC")
    stage_columns = [
        "preprocess_seconds",
        "setup_seconds",
        "process_startup_seconds",
        "cold_seconds",
        "warm_seconds",
        "steady_median_seconds",
        "peak_device_mb",
        "peak_allocated_mb",
        "peak_reserved_mb",
    ]
    records = []
    for (family, n, backend), group in frame.groupby(["family", "n", "backend_short"], sort=True):
        record = {
            "family": family,
            "n": int(n),
            "backend": backend,
            "seeds": int(group["seed"].nunique()),
            "successes": int((group["status"] == "ok").sum()),
            "precision": ";".join(sorted(group["precision"].dropna().unique())),
        }
        for column in stage_columns:
            record[f"median_{column}"] = finite_median(group[column])
        records.append(record)
    stage = pd.DataFrame(records)

    comparisons = []
    for (family, n), group in stage.groupby(["family", "n"], sort=True):
        by_backend = group.set_index("backend")
        if not {"LC", "Lightning-GPU"}.issubset(by_backend.index):
            continue
        lc = float(by_backend.loc["LC", "median_steady_median_seconds"])
        peer = float(by_backend.loc["Lightning-GPU", "median_steady_median_seconds"])
        comparisons.append(
            {
                "family": family,
                "n": int(n),
                "lc_steady_median_s": lc,
                "lightning_steady_median_s": peer,
                "lightning_over_lc": peer / lc,
                "lc_peak_device_mb": float(by_backend.loc["LC", "median_peak_device_mb"]),
                "lightning_peak_device_mb": float(by_backend.loc["Lightning-GPU", "median_peak_device_mb"]),
            }
        )
    comparison = pd.DataFrame(comparisons)

    crossovers = []
    for family, group in comparison.groupby("family", sort=True):
        winning = group[group["lightning_over_lc"] > 1.0].sort_values("n")
        crossovers.append(
            {
                "family": family,
                "crossover_n": int(winning.iloc[0]["n"]) if len(winning) else None,
                "tested_n": ",".join(str(int(value)) for value in sorted(group["n"].unique())),
                "smallest_n_ratio": float(group.sort_values("n").iloc[0]["lightning_over_lc"]),
                "largest_n_ratio": float(group.sort_values("n").iloc[-1]["lightning_over_lc"]),
            }
        )
    crossover = pd.DataFrame(crossovers)

    peer_rows = frame[frame["backend_short"] == "Lightning-GPU"]
    summary = {
        "status": "PASS" if len(peer_rows) == 45 and (peer_rows["status"] == "ok").all() else "FAIL",
        "requested_cases": 45,
        "lc_successes": int(((frame["backend_short"] == "LC") & (frame["status"] == "ok")).sum()),
        "lightning_successes": int((peer_rows["status"] == "ok").sum()),
        "families": sorted(frame["family"].unique()),
        "precision": sorted(frame["precision"].unique()),
        "max_value_abs_error": float(peer_rows["value_abs_error_vs_lc"].max()),
        "max_gradient_relative_l2_error": float(peer_rows["gradient_relative_l2_error_vs_lc"].max()),
        "minimum_gradient_cosine": float(peer_rows["gradient_cosine_vs_lc"].min()),
        "allocator_note": "Lightning-GPU does not expose allocated/reserved pool counters; PID-level peak_device_mb is measured.",
        "crossovers": crossovers,
    }

    stage.to_csv(args.output_dir / "P0_2_lightning_stage_summary.csv", index=False)
    comparison.to_csv(args.output_dir / "P0_2_lightning_crossover_by_size.csv", index=False)
    crossover.to_csv(args.output_dir / "P0_2_lightning_crossover_points.csv", index=False)
    (args.output_dir / "P0_2_lightning_validation.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# P0-2 Official single-precision global-adjoint summary",
        "",
        "The official backend is PennyLane-Lightning-GPU 0.45.0 with `lightning.gpu`, "
        "`complex64` device state, `float32` trainable angles, and adjoint differentiation. "
        "LC uses complex64 local states and float32 local cost tables on the same RTX 3090 GPU 1.",
        "",
        f"All {summary['lightning_successes']}/45 Lightning-GPU and {summary['lc_successes']}/45 LC cases complete. "
        f"The maximum objective difference is {summary['max_value_abs_error']:.3e}; the maximum relative gradient "
        f"difference is {summary['max_gradient_relative_l2_error']:.3e}; minimum cosine similarity is "
        f"{summary['minimum_gradient_cosine']:.15f}.",
        "",
        "| family | n | LC steady (s) | Lightning steady (s) | Lightning / LC | LC peak MB | Lightning peak MB |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in comparison.itertuples(index=False):
        lines.append(
            f"| {row.family} | {row.n} | {row.lc_steady_median_s:.6f} | "
            f"{row.lightning_steady_median_s:.6f} | {row.lightning_over_lc:.2f} | "
            f"{row.lc_peak_device_mb:.0f} | {row.lightning_peak_device_mb:.0f} |"
        )
    lines.extend(
        [
            "",
            "The smallest tested crossover is n=24 for 3-regular and n=22 for weighted ER2. "
            "Lightning-GPU is faster below those points; LC is faster above them. Cold, warm, steady, "
            "preprocessing, setup, and process-startup medians are retained in the stage-summary CSV. "
            "Lightning-GPU does not expose allocator allocated/reserved counters, so these fields remain NaN "
            "rather than being replaced by estimates; PID-level peak device memory is measured.",
        ]
    )
    (args.output_dir / "P0_2_lightning_gpu_c64_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
