from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def add(checks, name, passed, evidence):
    checks.append({"name": name, "passed": bool(passed), "evidence": evidence})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    checks = []

    runtime = pd.read_csv(root / "P0_1_heldout_predictor" / "P0_1_runtime_metrics.csv")
    dispatch = pd.read_csv(root / "P0_1_heldout_predictor" / "P0_1_dispatch_metrics.csv")
    decision = pd.read_csv(root / "P0_1_heldout_predictor" / "P0_1_model_scope.csv")
    residual = pd.read_csv(root / "P0_1_heldout_predictor" / "P0_1_family_residuals.csv")
    add(checks, "P0-1 protocols present", set(runtime.protocol) == {"leave_one_family_out", "blocked_by_n", "grouped_by_seed"}, sorted(runtime.protocol.unique().tolist()))
    add(checks, "P0-1 hardware separated", set(runtime.dataset) == {"rtx3070", "rtx3090"}, sorted(runtime.dataset.unique().tolist()))
    add(checks, "P0-1 objective and gradient separated", set(runtime.target) == {"objective", "gradient"}, sorted(runtime.target.unique().tolist()))
    add(checks, "P0-1 claim decision follows 30 percent rule", decision.recommended_term.str.contains("descriptive").all(), decision.to_dict("records"))
    add(checks, "P0-1 runtime confidence intervals complete", len(runtime) == 12 and runtime[["mape_ci_low", "mape_ci_high", "r2_ci_low", "r2_ci_high", "spearman_ci_low", "spearman_ci_high"]].notna().all().all(), f"rows={len(runtime)}")
    add(checks, "P0-1 dispatch metrics complete", dispatch[["precision", "recall", "balanced_accuracy"]].notna().all().all(), f"rows={len(dispatch)}")
    add(checks, "P0-1 dispatch confidence intervals complete", dispatch.filter(regex="_ci_(low|high)$").notna().all().all(), f"rows={len(dispatch)}")
    add(checks, "P0-1 family residuals retained", len(residual) > 0 and residual.family.nunique() >= 4, {"rows": len(residual), "families": sorted(residual.family.unique().tolist())})

    family = pd.read_csv(root / "P0_3_family_summary" / "P0_3_family_summary.csv")
    macro = json.loads((root / "P0_3_family_summary" / "P0_3_family_macro.json").read_text(encoding="utf-8"))
    add(checks, "P0-3 four predefined families", len(family) == 4 and family.family.nunique() == 4, family.family.tolist())
    add(checks, "P0-3 target row accounting", macro["target_rows"] == 126 and macro["state_total_noncompletion"] == 88, macro)
    add(checks, "P0-3 row domination measured", abs(macro["three_regular_row_fraction"] - 92 / 126) < 1e-12, macro["three_regular_row_fraction"])

    qdiag = pd.read_csv(root / "P1_1_qtensor_precision" / "P1_1_qtensor_precision_diagnostic.csv")
    qdecision = json.loads((root / "P1_1_qtensor_precision" / "P1_1_qtensor_diagnosis.json").read_text(encoding="utf-8"))
    add(checks, "P1-1 all diagnostic rows completed", len(qdiag) == 3 and (qdiag.status == "ok").all(), qdiag.status.value_counts().to_dict())
    add(checks, "P1-1 complex128 executed", (qdiag.qtensor_result_dtype == "complex128").all(), qdiag.qtensor_result_dtype.tolist())
    add(checks, "P1-1 exact claim disabled when discrepancy persists", not qdecision["exact_agreement_claim_allowed"], qdecision["reason"])
    add(checks, "P1-1 plan reuse value stable", qdiag.reuse_abs_difference.max() <= 1e-12, float(qdiag.reuse_abs_difference.max()))

    official_paths = {
        "rtx3070": root / "P0_2_official_cuaoa_gradient_rtx3070" / "P0_2_official_cuaoa_gradient.csv",
        "rtx3090": root / "P1_2_crossover_rtx3090" / "P0_2_official_cuaoa_gradient.csv",
    }
    expected = {"rtx3070": 100, "rtx3090": 80}
    peers = []
    for host, path in official_paths.items():
        data = pd.read_csv(path)
        add(checks, f"P0-2 expected rows {host}", len(data) == expected[host], len(data))
        add(checks, f"P0-2 unique keys {host}", not data.duplicated(["family", "n", "p", "seed", "backend"]).any(), "family,n,p,seed,backend")
        add(checks, f"P0-2 all rows successful {host}", (data.status == "ok").all(), data.status.value_counts().to_dict())
        add(checks, f"P0-2 positive timings {host}", (data.steady_median_seconds > 0).all(), float(data.steady_median_seconds.min()))
        lc = data[data.backend.str.contains("LC local")]
        peer = data[data.backend.str.contains("CUAOA official")]
        add(checks, f"P0-2 LC allocator memory {host}", (lc.peak_allocated_mb > 0).all() and (lc.peak_reserved_mb > 0).all(), {"allocated_min": float(lc.peak_allocated_mb.min()), "reserved_min": float(lc.peak_reserved_mb.min())})
        add(checks, f"P0-2 CUAOA device memory {host}", (peer.peak_device_mb > 0).all(), float(peer.peak_device_mb.min()))
        peers.append(peer)
    peer = pd.concat(peers, ignore_index=True)
    add(checks, "P0-2 value agreement", peer.value_abs_error_vs_lc.max() < 1e-12, float(peer.value_abs_error_vs_lc.max()))
    add(checks, "P0-2 gradient agreement", peer.gradient_relative_l2_error_vs_lc.max() < 1e-12, float(peer.gradient_relative_l2_error_vs_lc.max()))
    add(checks, "P0-2 gradient cosine", peer.gradient_cosine_vs_lc.min() > 1 - 1e-12, float(peer.gradient_cosine_vs_lc.min()))

    lightning = pd.read_csv(root / "P0_2_lightning_gpu_c64_rtx3090" / "P0_2_lightning_gpu_adjoint_c64.csv")
    lightning_peer = lightning[lightning.backend.str.contains("Lightning")]
    lightning_lc = lightning[lightning.backend == "LC local adjoint"]
    add(checks, "P0-2 Lightning matched single-precision rows", len(lightning) == 90 and len(lightning_peer) == 45 and len(lightning_lc) == 45, {"rows": len(lightning), "peer": len(lightning_peer), "lc": len(lightning_lc)})
    add(checks, "P0-2 Lightning all rows successful", (lightning.status == "ok").all(), lightning.status.value_counts().to_dict())
    add(checks, "P0-2 Lightning precision matched", set(lightning.precision) == {"complex64/float32 native", "complex64/float32 matched"}, sorted(lightning.precision.unique().tolist()))
    add(checks, "P0-2 Lightning timing stages separated", lightning[["preprocess_seconds", "setup_seconds", "process_wall_seconds", "process_startup_seconds", "cold_seconds", "warm_seconds", "steady_median_seconds"]].notna().all().all(), "preprocess/setup/startup/cold/warm/steady")
    add(checks, "P0-2 Lightning device memory measured", (lightning_peer.peak_device_mb > 0).all(), float(lightning_peer.peak_device_mb.min()))
    add(checks, "P0-2 Lightning allocator counters explicitly unavailable", lightning_peer[["peak_allocated_mb", "peak_reserved_mb"]].isna().all().all(), "backend API does not expose allocator pools")
    add(checks, "P0-2 Lightning value agreement", lightning_peer.value_abs_error_vs_lc.max() < 1e-4, float(lightning_peer.value_abs_error_vs_lc.max()))
    add(checks, "P0-2 Lightning gradient agreement", lightning_peer.gradient_relative_l2_error_vs_lc.max() < 1e-4, float(lightning_peer.gradient_relative_l2_error_vs_lc.max()))
    add(checks, "P0-2 Lightning gradient cosine", lightning_peer.gradient_cosine_vs_lc.min() > 0.99999, float(lightning_peer.gradient_cosine_vs_lc.min()))
    lightning_cross = pd.read_csv(root / "P0_2_lightning_gpu_c64_rtx3090" / "P0_2_lightning_crossover_points.csv")
    lightning_observed = {row.family: int(row.crossover_n) for row in lightning_cross.itertuples()}
    add(checks, "P0-2 Lightning crossover", lightning_observed == {"3regular": 24, "weighted_qubo_er2": 22}, lightning_observed)

    crossover = pd.read_csv(root / "P0_2_P1_2_cuaoa_analysis" / "P0_2_crossover_points.csv")
    observed = {f"{r.hardware}:{r.family}": int(r.crossover_n) for r in crossover.itertuples()}
    expected_crossovers = {
        "rtx3070:3regular": 22,
        "rtx3070:weighted_qubo_er2": 22,
        "rtx3090:3regular": 24,
        "rtx3090:weighted_qubo_er2": 24,
    }
    add(checks, "P1-2 crossover replication", observed == expected_crossovers, observed)

    status = "PASS" if all(c["passed"] for c in checks) else "FAIL"
    result = {"status": status, "checks": checks, "failed": [c for c in checks if not c["passed"]]}
    (root / "FOLLOWUP_VALIDATION.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"status": status, "checks": len(checks), "failed": len(result["failed"])}, indent=2))
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
