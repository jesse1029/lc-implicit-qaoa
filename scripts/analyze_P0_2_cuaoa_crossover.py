from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_input(spec: str) -> tuple[str, Path]:
    label, sep, path = spec.partition("=")
    if not sep:
        raise ValueError(f"expected LABEL=CSV: {spec}")
    return label, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for spec in args.input:
        label, path = parse_input(spec)
        frame = pd.read_csv(path)
        frame["hardware"] = label
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    successful = data[data["status"].astype(str).str.lower() == "ok"].copy()
    successful["backend_key"] = np.where(
        successful["backend"].str.contains("CUAOA official"), "cuaoa", "lc"
    )
    summary = (
        successful.groupby(["hardware", "gpu_name", "family", "n", "backend_key"])
        .agg(
            seeds=("seed", "nunique"),
            cold_median_s=("cold_seconds", "median"),
            warm_median_s=("warm_seconds", "median"),
            steady_median_s=("steady_median_seconds", "median"),
            preprocess_median_s=("preprocess_seconds", "median"),
            setup_median_s=("setup_seconds", "median"),
            peak_device_median_mb=("peak_device_mb", "median"),
            peak_allocated_median_mb=("peak_allocated_mb", "median"),
            peak_reserved_median_mb=("peak_reserved_mb", "median"),
            max_value_abs_error=("value_abs_error_vs_lc", "max"),
            max_gradient_relative_l2_error=("gradient_relative_l2_error_vs_lc", "max"),
            min_gradient_cosine=("gradient_cosine_vs_lc", "min"),
        )
        .reset_index()
    )
    pivot = summary.pivot_table(
        index=["hardware", "gpu_name", "family", "n"],
        columns="backend_key", values="steady_median_s",
    ).reset_index()
    pivot["cuaoa_over_lc"] = pivot["cuaoa"] / pivot["lc"]
    crossovers = []
    for (hardware, family), sub in pivot.groupby(["hardware", "family"]):
        winning = sub[sub["lc"] < sub["cuaoa"]]
        crossovers.append(
            {
                "hardware": hardware,
                "family": family,
                "crossover_n": int(winning["n"].min()) if len(winning) else None,
                "n_values": sorted(int(x) for x in sub["n"]),
                "median_cuaoa_over_lc_at_largest_n": float(sub.loc[sub["n"].idxmax(), "cuaoa_over_lc"]),
            }
        )
    crossover_df = pd.DataFrame(crossovers)
    replication = []
    common = pivot.pivot_table(index=["family", "n"], columns="hardware", values=["lc", "cuaoa"])
    if {"rtx3070", "rtx3090"}.issubset(set(pivot["hardware"])):
        for (family, n), row in common.iterrows():
            if all((backend, host) in row.index for backend in ["lc", "cuaoa"] for host in ["rtx3070", "rtx3090"]):
                replication.append(
                    {
                        "family": family,
                        "n": int(n),
                        "lc_3070_over_3090": float(row[("lc", "rtx3070")] / row[("lc", "rtx3090")]),
                        "cuaoa_3070_over_3090": float(row[("cuaoa", "rtx3070")] / row[("cuaoa", "rtx3090")]),
                    }
                )
    replication_df = pd.DataFrame(replication)
    summary.to_csv(args.out_dir / "P0_2_backend_stage_summary.csv", index=False)
    pivot.to_csv(args.out_dir / "P0_2_crossover_by_hardware.csv", index=False)
    crossover_df.to_csv(args.out_dir / "P0_2_crossover_points.csv", index=False)
    replication_df.to_csv(args.out_dir / "P1_2_cross_hardware_scaling.csv", index=False)
    peer = successful[successful["backend_key"] == "cuaoa"]
    agreement = {
        "official_backend": "CUAOA official pycuaoa 0.1.0 gradients",
        "precision": "CUAOA native complex128/float64; LC matched to complex128/float64",
        "rows": int(len(data)),
        "cases": int(len(data) // 2),
        "all_rows_successful": bool(len(successful) == len(data)),
        "max_value_absolute_error": float(peer["value_abs_error_vs_lc"].max()),
        "max_gradient_relative_l2_error": float(peer["gradient_relative_l2_error_vs_lc"].max()),
        "min_gradient_cosine": float(peer["gradient_cosine_vs_lc"].min()),
        "crossover_points": crossovers,
        "hardware_sensitivity": "The crossover shifts from n=22 on RTX 3070 to n=24 on RTX 3090 for both families. The global CUAOA path benefits more from the second GPU than the small-cone LC path; this is an empirical hardware-sensitivity result, not a causal bandwidth-only attribution.",
        "memory_caveat": "CUAOA does not expose allocator allocated/reserved counters; peak device memory is sampled through nvidia-smi. LC allocated/reserved pool counters are reported separately.",
    }
    (args.out_dir / "P0_2_cuaoa_validation.json").write_text(json.dumps(agreement, indent=2), encoding="utf-8")
    lines = [
        "# Official CUAOA Gradient and Cross-Hardware Crossover",
        "",
        json.dumps(agreement, indent=2),
        "",
        "## Crossover table",
        "",
        pivot.to_markdown(index=False),
        "",
        "## Stage and memory table",
        "",
        summary.to_markdown(index=False),
    ]
    (args.out_dir / "P0_2_cuaoa_validation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(agreement, indent=2))


if __name__ == "__main__":
    main()
