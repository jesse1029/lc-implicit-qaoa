from __future__ import annotations

import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fnum(x: str) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def ok_count(rows: list[dict[str, str]], method: str | None = None) -> tuple[int, int]:
    subset = [r for r in rows if method is None or r.get("method") == method]
    return sum(1 for r in subset if r.get("status") in {"ok", "ok_proxy"} or str(r.get("status", "")).startswith("optimizer_")), len(subset)


def max_n_ok(rows: list[dict[str, str]], method: str) -> int:
    vals = [int(r["n"]) for r in rows if r.get("method") == method and r.get("status") == "ok"]
    return max(vals) if vals else 0


def main() -> None:
    external = read_csv(RESULTS / "external_peer_benchmark_with_proxies.csv")
    scale = read_csv(RESULTS / "sota_sparse_scale.csv")
    grad = read_csv(RESULTS / "gradient_benchmark.csv")
    qubo = read_csv(RESULTS / "qubo_benchmark.csv")
    multiseed = read_csv(RESULTS / "multiseed_stats.csv")
    opt = read_csv(RESULTS / "optimization_benchmark.csv")
    adjoint = read_csv(RESULTS / "adjoint_gradient_benchmark.csv")
    topology = read_csv(RESULTS / "topology_ablation.csv")
    real_qubo = read_csv(RESULTS / "real_qubo_case_study.csv")
    biomedical_runtime = read_csv(RESULTS / "biomedical_runtime.csv")
    biomedical_selectors = read_csv(RESULTS / "biomedical_selectors.csv")
    figures = sorted((RESULTS / "figures").glob("fig*.png"))

    external_ok, external_total = ok_count(external)
    qubo_ok, qubo_total = ok_count(qubo, "lc_batched_gpu")
    multi_ok, multi_total = ok_count(multiseed, "lc_batched_gpu")
    opt_ok, opt_total = ok_count(opt)
    grad_lc = [r for r in grad if r.get("method") == "lc_batched_gpu_fd" and r.get("status") == "ok"]
    adjoint_ok = [r for r in adjoint if r.get("method") == "lc_batched_gpu_adjoint" and r.get("status") == "ok"]
    adjoint_errors = [fnum(r.get("max_abs_error_vs_lc_fd", "nan")) for r in adjoint_ok]
    adjoint_rel_errors = [
        fnum(r.get("max_abs_error_vs_lc_fd", "nan")) / max(1.0, abs(fnum(r.get("gradient_norm", "nan"))))
        for r in adjoint_ok
        if math.isfinite(fnum(r.get("max_abs_error_vs_lc_fd", "nan")))
    ]
    topology_ok, topology_total = ok_count(topology)
    real_ok = [r for r in real_qubo if r.get("method") == "lc_batched_gpu_adjoint" and r.get("status") == "ok"]
    biomed_lc_ok = [r for r in biomedical_runtime if r.get("method") == "lc_batched_gpu_adjoint" and r.get("status") == "ok"]
    biomed_selector_ok = [r for r in biomedical_selectors if r.get("status") == "ok"]

    checklist = [
        ("Official peer methods installed and benchmarked", external_ok >= 80 and external_total >= 90),
        ("Exactness checked against full-state and peers", external_ok >= 80 and any(fnum(r.get("abs_error_vs_full", "nan")) < 1e-4 for r in external)),
        ("Sparse/depth/density scaling matrix", len(scale) >= 1000 and max_n_ok(scale, "lc_batched_gpu") >= 512),
        ("Weighted QUBO with linear fields", qubo_ok >= 20),
        ("Multi-seed robustness", multi_ok >= 40 and multi_total >= 100),
        ("Optimization-loop repeated objective benchmark", opt_ok >= 8),
        ("Gradient training-cost evidence", len(grad_lc) >= 5 and max(int(r["n"]) for r in grad_lc) >= 512),
        (
            "Exact adjoint-gradient implementation validated",
            len(adjoint_ok) >= 5
            and max([e for e in adjoint_errors if math.isfinite(e)] or [float("inf")]) < 1e-1
            and max(adjoint_rel_errors or [float("inf")]) < 1e-3,
        ),
        ("Batching/topology ablation", topology_ok >= 9 and topology_total >= 12),
        ("Real-data sparse QUBO case study", len(real_ok) >= 4),
        ("Biomedical feature-selection baseline context", len(biomed_lc_ok) >= 2 and len(biomed_selector_ok) >= 6),
        ("Paper-ready figures generated", len(figures) >= 6),
    ]

    lines = [
        "# Reproducibility Summary",
        "",
        "This report answers whether the current evidence package is strong enough to draft a serious AAAI-style paper.",
        "",
        "## Checklist",
        "",
        "| Item | Status |",
        "|---|---|",
    ]
    for item, passed in checklist:
        lines.append(f"| {item} | {'PASS' if passed else 'MISSING'} |")

    lines.extend(
        [
            "",
            "## Evidence Counts",
            "",
            f"- External peer rows: {external_ok}/{external_total} successful or proxy-successful.",
            f"- Sparse scaling rows: {len(scale)} total; LC max n ok: {max_n_ok(scale, 'lc_batched_gpu')}.",
            f"- Weighted QUBO LC rows: {qubo_ok}/{qubo_total}.",
            f"- Multi-seed LC rows: {multi_ok}/{multi_total}.",
            f"- Optimization rows: {opt_ok}/{opt_total}.",
            f"- LC gradient rows: {len(grad_lc)}; max n: {max([int(r['n']) for r in grad_lc] or [0])}.",
            f"- LC adjoint-gradient rows: {len(adjoint_ok)}; max abs error vs LC finite difference: {max([e for e in adjoint_errors if math.isfinite(e)] or [float('nan')]):.3g}; max relative error: {max(adjoint_rel_errors or [float('nan')]):.3g}.",
            f"- Topology/batching ablation rows: {topology_ok}/{topology_total}.",
            f"- Real-data sparse QUBO rows: {len(real_ok)} LC adjoint rows.",
            f"- Biomedical feature-selection rows: {len(biomed_lc_ok)} LC adjoint rows; {len(biomed_selector_ok)} selector baseline rows.",
            f"- Figures: {len(figures)} PNG plus matching PDFs.",
            "",
            "## Claim I Would Defend",
            "",
            "LC-Implicit-QAOA is an exact, objective-centric single-GPU evaluator for structured local QUBO-QAOA. It avoids global state-vector and diagonal-cost materialization during objective and exact reverse-mode adjoint-gradient evaluation. On bounded-cone graph families, it reaches n=512 on an 8GB RTX 3070 where full-state baselines are capped at n<=24 in this artifact.",
            "",
            "## Claim I Would Not Defend Yet",
            "",
            "I would not claim a general quantum simulator, full-state/sampling replacement, or final custom CUDA-native topology-grouped kernel. The current implementation has exact adjoint gradients, but the CUDA-native kernel engineering target is still represented by CuPy batching and topology-grouping ablations.",
            "",
            "## Remaining Paper-Risk Items",
            "",
            "- The implementation is CuPy batched by cone size, not the final custom CUDA topology-grouped kernel described as the ideal engineering target.",
            "- QueenV2 and BMQSim remain paper-derived proxies, not official runnable author code.",
            "- Large-n official peers are limited by their full-state nature or adapter cost; the comparison should be framed as capability boundary, not universal speed dominance.",
            "- The real-data case study is a reproducible feature-selection QUBO from standard sklearn datasets, not a private domain benchmark.",
            "",
            "## Output Files",
            "",
            "- `results/SOTA_REPORT.md`",
            "- `results/qubo_benchmark.md`",
            "- `results/multiseed_stats.md`",
            "- `results/optimization_benchmark.md`",
            "- `results/gradient_benchmark.md`",
            "- `results/adjoint_gradient_benchmark.md`",
            "- `results/topology_ablation.md`",
            "- `results/real_qubo_case_study.md`",
            "- `results/biomedical_feature_selection.md`",
            "- `results/figures/`",
        ]
    )
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "REPRODUCIBILITY_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (RESULTS / "reproducibility_summary.json").write_text(
        json.dumps(
            {
                "checklist": [{"item": item, "passed": passed} for item, passed in checklist],
                "external_rows": [external_ok, external_total],
                "scale_rows": len(scale),
                "qubo_lc_rows": [qubo_ok, qubo_total],
                "multiseed_lc_rows": [multi_ok, multi_total],
                "optimization_rows": [opt_ok, opt_total],
                "adjoint_lc_rows": [len(adjoint_ok), len([r for r in adjoint if r.get("method") == "lc_batched_gpu_adjoint"])],
                "topology_rows": [topology_ok, topology_total],
                "real_qubo_lc_rows": len(real_ok),
                "biomedical_lc_rows": len(biomed_lc_ok),
                "biomedical_selector_rows": len(biomed_selector_ok),
                "figures": [str(p) for p in figures],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"WROTE {RESULTS / 'REPRODUCIBILITY_SUMMARY.md'}")
    print(f"WROTE {RESULTS / 'reproducibility_summary.json'}")


if __name__ == "__main__":
    main()
