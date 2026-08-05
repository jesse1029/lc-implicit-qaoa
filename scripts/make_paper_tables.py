from __future__ import annotations

import csv
import math
import statistics
from collections import defaultdict
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


def latex_escape(text: str) -> str:
    return (
        str(text)
        .replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("%", "\\%")
        .replace("&", "\\&")
    )


def fmt(x: float, digits: int = 3) -> str:
    if not math.isfinite(x):
        return "--"
    if abs(x) >= 1000 or (abs(x) > 0 and abs(x) < 0.001):
        return f"{x:.{digits}e}"
    return f"{x:.{digits}g}"


def table_external(external: list[dict[str, str]]) -> str:
    by_method: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in external:
        by_method[row["method"]].append(row)
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Small-instance peer validation.}",
        "\\begin{tabular}{lrrr}",
        "\\toprule",
        "Method & OK & Max error & Median time (s)\\\\",
        "\\midrule",
    ]
    for method in sorted(by_method):
        subset = by_method[method]
        ok = [r for r in subset if r.get("status") in {"ok", "ok_proxy"}]
        errors = [fnum(r.get("abs_error_vs_full", "nan")) for r in ok]
        secs = sorted(fnum(r.get("seconds", "nan")) for r in ok if math.isfinite(fnum(r.get("seconds", "nan"))))
        med = statistics.median(secs) if secs else float("nan")
        lines.append(f"{latex_escape(method)} & {len(ok)}/{len(subset)} & {fmt(max(errors or [float('nan')]))} & {fmt(med)}\\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    return "\n".join(lines)


def table_scaling(scale: list[dict[str, str]]) -> str:
    keys = [
        ("3regular", "1"),
        ("3regular", "2"),
        ("er_deg2", "2"),
        ("er_deg3", "2"),
        ("modular_sparse", "1"),
        ("scale_free_a1", "1"),
    ]
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Representative exact LC scaling reach on the RTX 3070.}",
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Family & p & OK & Max n & Time at max n (s)\\\\",
        "\\midrule",
    ]
    for family, p in keys:
        subset = [r for r in scale if r.get("family") == family and r.get("p") == p and r.get("method") == "lc_batched_gpu"]
        ok = [r for r in subset if r.get("status") == "ok"]
        max_n = max([int(r["n"]) for r in ok] or [0])
        row = next((r for r in ok if int(r["n"]) == max_n), None)
        lines.append(
            f"{latex_escape(family)} & {p} & {len(ok)}/{len(subset)} & {max_n} & {fmt(fnum(row['seconds']) if row else float('nan'))}\\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    return "\n".join(lines)


def table_qubo(qubo: list[dict[str, str]]) -> str:
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Weighted QUBO with linear-field validation.}",
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Family & p & LC OK & Max n & Max full-check error\\\\",
        "\\midrule",
    ]
    for family, p in sorted({(r["family"], r["p"]) for r in qubo if r.get("method") == "lc_batched_gpu"}):
        subset = [r for r in qubo if r["family"] == family and r["p"] == p and r["method"] == "lc_batched_gpu"]
        ok = [r for r in subset if r["status"] == "ok"]
        errors = [fnum(r["abs_error_vs_full"]) for r in ok if math.isfinite(fnum(r["abs_error_vs_full"]))]
        lines.append(
            f"{latex_escape(family)} & {p} & {len(ok)}/{len(subset)} & {max([int(r['n']) for r in ok] or [0])} & {fmt(max(errors or [float('nan')]))}\\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    return "\n".join(lines)


def table_training(opt: list[dict[str, str]]) -> str:
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Repeated objective-evaluation training-loop cost.}",
        "\\begin{tabular}{llrrr}",
        "\\toprule",
        "Case & Method & Evals & s/eval & Improvement\\\\",
        "\\midrule",
    ]
    for row in opt:
        if row.get("mode") != "trajectory" or row.get("status") != "ok":
            continue
        lines.append(
            f"{latex_escape(row['case'])} & {latex_escape(row['method'])} & {row['evals']} & {fmt(fnum(row['seconds_per_eval']))} & {fmt(fnum(row['improvement']))}\\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    return "\n".join(lines)


def table_adjoint(adjoint: list[dict[str, str]]) -> str:
    rows = [r for r in adjoint if r.get("method") == "lc_batched_gpu_adjoint" and r.get("status") == "ok"]
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Adjoint-gradient validation for the exact LC objective.}",
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Case & n & Time (s) & Err vs LC FD & Err vs full FD\\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(
            f"{latex_escape(row['case'])} & {row['n']} & {fmt(fnum(row['seconds']))} & "
            f"{fmt(fnum(row['max_abs_error_vs_lc_fd']))} & {fmt(fnum(row['max_abs_error_vs_full_fd']))}\\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    return "\n".join(lines)


def table_topology(topology: list[dict[str, str]]) -> str:
    rows = [r for r in topology if r.get("mode") in {"naive_per_cone", "size_batched", "topology_grouped"} and r.get("status") == "ok"]
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Batching and topology-grouping ablation.}",
        "\\begin{tabular}{llrrr}",
        "\\toprule",
        "Case & Mode & Groups & Time (s) & Speedup\\\\",
        "\\midrule",
    ]
    for row in rows:
        groups = row["n_topology_groups"] if row["mode"] == "topology_grouped" else row["n_size_groups"]
        lines.append(
            f"{latex_escape(row['case'])} & {latex_escape(row['mode'])} & {groups} & "
            f"{fmt(fnum(row['seconds']))} & {fmt(fnum(row['speedup_vs_naive']))}\\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    return "\n".join(lines)


def table_real_qubo(real_qubo: list[dict[str, str]]) -> str:
    rows = [r for r in real_qubo if r.get("method") == "lc_batched_gpu_adjoint"]
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Real-data sparse feature-selection QUBO case study.}",
        "\\begin{tabular}{llrrrr}",
        "\\toprule",
        "Dataset & Variant & n & p & Obj (s) & Grad (s)\\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(
            f"{latex_escape(row['dataset'])} & {latex_escape(row['variant'])} & {row['n']} & {row['p']} & "
            f"{fmt(fnum(row['objective_seconds']))} & {fmt(fnum(row['gradient_seconds']))}\\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    return "\n".join(lines)


def table_biomedical(selectors: list[dict[str, str]]) -> str:
    rows = [
        r
        for r in selectors
        if r.get("status") == "ok"
        and r.get("dataset") == "breast_cancer_wisconsin"
        and r.get("variant", "").startswith("top20")
    ]
    if not rows:
        rows = [r for r in selectors if r.get("status") == "ok"][:8]
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Biomedical feature-selection QUBO baseline context.}",
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Selector & Budget & QUBO score & CV mean & Time (s)\\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(
            f"{latex_escape(row['selector'])} & {row['budget']} & {fmt(fnum(row['qubo_score']))} & "
            f"{fmt(fnum(row['cv_mean']))} & {fmt(fnum(row['seconds']))}\\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    return "\n".join(lines)


def main() -> None:
    external = read_csv(RESULTS / "external_peer_benchmark_with_proxies.csv")
    scale = read_csv(RESULTS / "sota_sparse_scale.csv")
    qubo = read_csv(RESULTS / "qubo_benchmark.csv")
    opt = read_csv(RESULTS / "optimization_benchmark.csv")
    adjoint = read_csv(RESULTS / "adjoint_gradient_benchmark.csv")
    topology = read_csv(RESULTS / "topology_ablation.csv")
    real_qubo = read_csv(RESULTS / "real_qubo_case_study.csv")
    biomedical_selectors = read_csv(RESULTS / "biomedical_selectors.csv")
    parts = [
        "% Auto-generated tables for the LC-Implicit-QAOA AAAI draft.",
        "\\usepackage{booktabs}",
        "",
        table_external(external),
        table_scaling(scale),
    ]
    if qubo:
        parts.append(table_qubo(qubo))
    if opt:
        parts.append(table_training(opt))
    if adjoint:
        parts.append(table_adjoint(adjoint))
    if topology:
        parts.append(table_topology(topology))
    if real_qubo:
        parts.append(table_real_qubo(real_qubo))
    if biomedical_selectors:
        parts.append(table_biomedical(biomedical_selectors))
    out = RESULTS / "paper_tables.tex"
    out.write_text("\n".join(parts), encoding="utf-8")
    print(f"WROTE {out}")


if __name__ == "__main__":
    main()
