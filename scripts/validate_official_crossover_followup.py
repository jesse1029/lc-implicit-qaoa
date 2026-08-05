from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    pairs = pd.read_csv(args.analysis_dir / "paired_seed_ratios.csv")
    summary = pd.read_csv(args.analysis_dir / "paired_crossover_summary.csv")
    seed_breaks = pd.read_csv(args.analysis_dir / "break_even_by_seed.csv")
    break_summary = pd.read_csv(args.analysis_dir / "break_even_summary.csv")
    dispatch = pd.read_csv(args.analysis_dir / "external_dispatcher_cells.csv")
    dispatch_metrics = pd.read_csv(args.analysis_dir / "external_dispatcher_metrics.csv")
    metadata = json.loads(
        (args.analysis_dir / "external_dispatcher_metadata.json").read_text(encoding="utf-8")
    )

    checks: list[dict] = []

    def check(name: str, condition: bool, evidence) -> None:
        checks.append({"name": name, "passed": bool(condition), "evidence": evidence})

    recomputed_ratio = pairs.peer_steady_median_seconds / pairs.lc_steady_median_seconds
    check(
        "paired ratios recompute",
        np.allclose(recomputed_ratio, pairs.peer_over_lc, rtol=1e-12, atol=1e-12),
        float(np.max(np.abs(recomputed_ratio - pairs.peer_over_lc))),
    )
    check("all pairs use same seed", not pairs.duplicated(["dataset", "family", "n", "p", "seed"]).any(), len(pairs))
    check("five seeds per crossover cell", summary.paired_seeds.eq(5).all(), summary.paired_seeds.value_counts().to_dict())
    check("bootstrap intervals ordered", (summary.bootstrap_ci_low <= summary.bootstrap_ci_high).all(), len(summary))
    expected_decision = np.where(
        summary.bootstrap_ci_low > 1,
        "lc_faster",
        np.where(summary.bootstrap_ci_high < 1, "peer_faster", "tie_or_unstable"),
    )
    check("decision follows interval", np.array_equal(expected_decision, summary.decision), summary.decision.value_counts().to_dict())

    for q in (1, 5, 10, 25, 50, 100, 250):
        lc = (
            pairs.lc_preprocess_seconds
            + pairs.lc_setup_seconds
            + pairs.lc_cold_seconds
            + (q - 1) * pairs.lc_steady_median_seconds
        )
        peer = (
            pairs.peer_preprocess_seconds
            + pairs.peer_setup_seconds
            + pairs.peer_cold_seconds
            + (q - 1) * pairs.peer_steady_median_seconds
        )
        check(f"T({q}) LC formula", np.allclose(lc, seed_breaks[f"lc_T{q}"]), float(np.max(np.abs(lc - seed_breaks[f"lc_T{q}"]))))
        check(f"T({q}) peer formula", np.allclose(peer, seed_breaks[f"peer_T{q}"]), float(np.max(np.abs(peer - seed_breaks[f"peer_T{q}"]))))
    check("100-step flag consistent", np.array_equal(seed_breaks.optimizer_100_break_even, seed_breaks.lc_T100 < seed_breaks.peer_T100), int(seed_breaks.optimizer_100_break_even.sum()))
    check("all 23 requested break-even cells retained", len(break_summary) == 23, len(break_summary))

    metric_checks = []
    for (dataset, rule), stored in dispatch_metrics.groupby(["dataset", "rule"]):
        stored = stored.iloc[0]
        cell = dispatch[dispatch.dataset == dataset]
        actual = cell.actual_choose_lc.to_numpy()
        column = (
            "predicted_choose_lc"
            if rule == "frozen_logistic_threshold_0.5"
            else "predefined_target_choose_lc"
        )
        pred = cell[column].to_numpy()
        matrix = confusion_matrix(actual, pred, labels=[0, 1])
        values = {
            "accuracy": accuracy_score(actual, pred),
            "balanced_accuracy": balanced_accuracy_score(actual, pred),
            "precision_choose_lc": precision_score(actual, pred, zero_division=0),
            "recall_choose_lc": recall_score(actual, pred, zero_division=0),
            "tn": matrix[0, 0], "fp": matrix[0, 1], "fn": matrix[1, 0], "tp": matrix[1, 1],
        }
        metric_checks.append(all(np.isclose(float(stored[k]), float(v)) for k, v in values.items()))
    check("dispatcher metrics recompute", all(metric_checks), dispatch_metrics.to_dict(orient="records"))
    check("dispatcher threshold frozen", metadata["fixed_probability_threshold"] == 0.5 and not metadata["threshold_retuned_on_official_cells"], metadata)
    check("external dispatcher includes all pairs", len(dispatch) == len(pairs) == 115, len(dispatch))

    selected = summary[
        (
            (summary.dataset == "lightning_rtx3090_c64")
            & (summary.family == "weighted_qubo_er2")
            & (summary.n == 22)
        )
        | (
            (summary.dataset == "cuaoa_rtx3070_c128")
            & (summary.family == "weighted_qubo_er2")
            & (summary.n == 22)
        )
        | (
            (summary.dataset == "cuaoa_rtx3090_c128")
            & (summary.family == "weighted_qubo_er2")
            & (summary.n == 24)
        )
        | (
            (summary.dataset == "cuaoa_rtx3090_c128")
            & (summary.family == "3regular")
            & (summary.n == 24)
        )
    ]
    check("four requested cells present", len(selected) == 4, selected[["dataset", "family", "n"]].to_dict(orient="records"))

    failed = [item for item in checks if not item["passed"]]
    payload = {
        "status": "PASS" if not failed else "FAIL",
        "checks": checks,
        "failed": [item["name"] for item in failed],
        "artifact_sha256": {
            path.name: sha256(path)
            for path in sorted(args.analysis_dir.iterdir())
            if path.is_file()
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"status": payload["status"], "checks": len(checks), "failed": payload["failed"]}, indent=2))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
