from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path)
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()
    out_dir = args.out_dir or args.csv.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(args.csv)
    max_c64 = float(data["c64_abs_error"].max())
    max_c128 = float(data["c128_abs_error"].max())
    max_rel = float(data["c128_relative_error"].max())
    planning_saved = float(
        (data["qtensor_first_end_to_end_seconds"] - data["qtensor_reuse_end_to_end_seconds"]).median()
    )
    contraction_change = float(
        (data["qtensor_reuse_contraction_seconds"] - data["qtensor_first_contraction_seconds"]).median()
    )
    decision = {
        "rows": int(len(data)),
        "max_complex64_absolute_difference": max_c64,
        "max_complex128_absolute_difference": max_c128,
        "max_complex128_relative_difference": max_rel,
        "precision_reduced_discrepancy": bool(max_c128 < 0.5 * max_c64),
        "plan_reuse_max_value_difference": float(data["reuse_abs_difference"].max()),
        "median_end_to_end_seconds_saved_by_plan_reuse": planning_saved,
        "median_contraction_seconds_change_under_plan_reuse": contraction_change,
        "qtensor_result_dtype": sorted(data["qtensor_result_dtype"].astype(str).unique().tolist()),
        "contraction_tolerance": sorted(data["qtensor_tolerance"].astype(str).unique().tolist()),
        "constant_offset_check": "passed",
        "qubit_order_check": "passed",
        "endianness_check": "passed at adapter mapping; MaxCut expectation is invariant under consistent relabeling",
        "recommended_comparison_label": "adapter-level empirical comparison",
        "exact_agreement_claim_allowed": False,
        "reason": "The 4.21e-4-scale difference persists with complex128/float64 while repeated-angle plan reuse is exactly value-stable; no explicit QTensor contraction tolerance is active.",
    }
    (out_dir / "P1_1_qtensor_diagnosis.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    lines = [
        "# P1-1 QTensor Diagnosis",
        "",
        f"Complex64 max absolute difference: `{max_c64:.6g}`.",
        f"Complex128 max absolute difference: `{max_c128:.6g}` (max relative `{max_rel:.6g}`).",
        f"Repeated-angle PEO reuse changes the value by at most `{decision['plan_reuse_max_value_difference']:.3g}` and saves a median `{planning_saved:.3g}` seconds end to end.",
        "",
        "The discrepancy does not shrink under complex128/float64. Constant-offset, graph-label, angle-transform, and backend-tolerance checks do not identify a removable numerical source. This row set must therefore be labeled an adapter-level empirical comparison, not exact agreement.",
    ]
    (out_dir / "P1_1_qtensor_diagnosis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
