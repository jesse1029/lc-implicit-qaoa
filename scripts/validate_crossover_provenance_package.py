from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    provenance_dir = args.root / "P0_1_cuaoa_provenance_rtx3070"
    analysis_dir = args.root / "official_crossover_analysis"

    provenance = json.loads(
        (provenance_dir / "CUAOA_PROVENANCE_VALIDATION.json").read_text(encoding="utf-8")
    )
    analysis = json.loads(
        (analysis_dir / "VALIDATION_SUMMARY.json").read_text(encoding="utf-8")
    )
    enriched = pd.read_csv(provenance_dir / "P0_1_cuaoa_provenance_gradient.csv")
    paired = pd.read_csv(analysis_dir / "paired_crossover_summary.csv")
    break_even = pd.read_csv(analysis_dir / "break_even_summary.csv")
    dispatcher = pd.read_csv(analysis_dir / "external_dispatcher_metrics.csv")
    scope = (args.root / "P1_2_FEASIBILITY_FRONTIER_DECISION.md").read_text(
        encoding="utf-8"
    )

    checks: list[dict] = []

    def check(name: str, condition: bool, evidence) -> None:
        checks.append({"name": name, "passed": bool(condition), "evidence": evidence})

    check("CUAOA provenance validation passes", provenance["status"] == "PASS", provenance.get("failed"))
    check("paired analysis validation passes", analysis["status"] == "PASS", analysis.get("failed"))
    check("all provenance checks pass", all(item["passed"] for item in provenance["checks"]), len(provenance["checks"]))
    check("all analysis checks pass", all(item["passed"] for item in analysis["checks"]), len(analysis["checks"]))
    check("fresh CUAOA rows complete", len(enriched) == 60 and enriched.status.eq("ok").all(), len(enriched))
    peer_labels = enriched[enriched.backend.str.contains("CUAOA", regex=False)].backend.unique().tolist()
    check(
        "CUAOA label discloses patch",
        peer_labels == ["official CUAOA 0.1.0 codebase with documented device-information compatibility patch"],
        peer_labels,
    )
    check("paired cells complete", len(paired) == 23 and paired.paired_seeds.eq(5).all(), len(paired))
    check("break-even grid complete", len(break_even) == 23, len(break_even))
    check(
        "dispatcher threshold not tuned",
        set(dispatcher.rule) == {"frozen_logistic_threshold_0.5", "predefined_target_k14_S2e7"},
        sorted(dispatcher.rule.unique()),
    )
    check(
        "dispatcher negative result retained",
        dispatcher.balanced_accuracy.max() < 0.60,
        float(dispatcher.balanced_accuracy.max()),
    )
    check(
        "conditional P1-2 closed",
        "NOT_RUN_EXPLAINED" in scope
        and "configured matched state-plus-cost reference" in scope
        and "universal frontier" in scope,
        "scoped reference frontier",
    )
    completion = args.root / "EXPERIMENT_COMPLETION_REPORT.md"
    check("completion report present", completion.exists() and completion.stat().st_size > 3000, completion.stat().st_size)

    failed = [item for item in checks if not item["passed"]]
    artifacts = {
        str(path.relative_to(args.root)).replace("\\", "/"): {
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in sorted(args.root.rglob("*"))
        if path.is_file() and path.resolve() != args.out.resolve()
    }
    payload = {
        "status": "PASS" if not failed else "FAIL",
        "checks": checks,
        "failed": [item["name"] for item in failed],
        "artifacts": artifacts,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"status": payload["status"], "checks": len(checks), "artifacts": len(artifacts), "failed": payload["failed"]}, indent=2))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
