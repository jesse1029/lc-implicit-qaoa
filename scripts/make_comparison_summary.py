from __future__ import annotations

import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fnum(x: str) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def fmt(x: float) -> str:
    if isinstance(x, str):
        if x == "":
            return ""
        x = fnum(x)
    if not math.isfinite(x):
        return "NA"
    if abs(x) >= 1000 or (abs(x) > 0 and abs(x) < 0.001):
        return f"{x:.3e}"
    return f"{x:.3g}"


def median(values: list[float]) -> float:
    values = [v for v in values if math.isfinite(v)]
    return statistics.median(values) if values else float("nan")


def method_rows(rows: list[dict[str, str]], method: str) -> dict[tuple, dict[str, str]]:
    out = {}
    for row in rows:
        if row.get("method") == method and row.get("status") in {"ok", "ok_proxy"}:
            key = (row.get("family"), row.get("n"), row.get("p"))
            out[key] = row
    return out


def external_peer_summary(external: list[dict[str, str]]) -> list[dict[str, object]]:
    lc = method_rows(external, "lc_batched_gpu")
    summaries = []
    for method in sorted({r["method"] for r in external if r["method"] != "lc_batched_gpu"}):
        peer = method_rows(external, method)
        keys = sorted(set(lc) & set(peer))
        ratios = []
        peer_times = []
        lc_times = []
        errors = []
        for key in keys:
            t_peer = fnum(peer[key]["seconds"])
            t_lc = fnum(lc[key]["seconds"])
            if t_peer > 0 and t_lc > 0:
                ratios.append(t_peer / t_lc)
                peer_times.append(t_peer)
                lc_times.append(t_lc)
            errors.append(fnum(lc[key].get("abs_error_vs_full", "nan")))
        if keys:
            summaries.append(
                {
                    "comparison": f"{method} vs lc_batched_gpu",
                    "matched_cases": len(keys),
                    "peer_median_s": median(peer_times),
                    "lc_median_s": median(lc_times),
                    "median_speedup_peer_over_lc": median(ratios),
                    "max_lc_error_vs_full": max([e for e in errors if math.isfinite(e)] or [float("nan")]),
                }
            )
    return summaries


def fullstate_summary(scale: list[dict[str, str]]) -> list[dict[str, object]]:
    by_case: dict[tuple, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in scale:
        if row.get("status") == "ok":
            key = (row["family"], row["n"], row["p"])
            by_case[key][row["method"]] = row
    summaries = []
    speedups = []
    memory = []
    for key, methods in sorted(by_case.items()):
        if "full_precompute_gpu" not in methods or "lc_batched_gpu" not in methods:
            continue
        full = methods["full_precompute_gpu"]
        lc = methods["lc_batched_gpu"]
        full_t = fnum(full["seconds"])
        lc_t = fnum(lc["seconds"])
        full_m = fnum(full["peak_pool_mb"])
        lc_m = fnum(lc["peak_pool_mb"])
        row = {
            "family": key[0],
            "n": int(key[1]),
            "p": int(key[2]),
            "full_s": full_t,
            "lc_s": lc_t,
            "speedup_vs_full": full_t / lc_t if lc_t > 0 else float("nan"),
            "full_peak_mb": full_m,
            "lc_peak_mb": lc_m,
            "memory_reduction_vs_full": full_m / lc_m if lc_m > 0 else float("nan"),
            "lc_error_vs_full": fnum(lc.get("abs_error_vs_full", "nan")),
        }
        summaries.append(row)
        speedups.append(row["speedup_vs_full"])
        memory.append(row["memory_reduction_vs_full"])
    summaries.append(
        {
            "family": "AGGREGATE_MEDIAN",
            "n": "",
            "p": "",
            "full_s": "",
            "lc_s": "",
            "speedup_vs_full": median(speedups),
            "full_peak_mb": "",
            "lc_peak_mb": "",
            "memory_reduction_vs_full": median(memory),
            "lc_error_vs_full": "",
        }
    )
    return summaries


def target_regime_summary(rows: list[dict[str, object]]) -> dict[str, float]:
    target = [
        r
        for r in rows
        if r.get("family") in {"3regular", "er_deg2", "modular_sparse"}
        and str(r.get("n")) == "24"
        and str(r.get("p")) in {"1", "2"}
    ]
    return {
        "cases": len(target),
        "median_speedup": median([float(r["speedup_vs_full"]) for r in target]),
        "min_speedup": min([float(r["speedup_vs_full"]) for r in target] or [float("nan")]),
        "max_speedup": max([float(r["speedup_vs_full"]) for r in target] or [float("nan")]),
        "median_memory_reduction": median([float(r["memory_reduction_vs_full"]) for r in target]),
        "min_memory_reduction": min([float(r["memory_reduction_vs_full"]) for r in target] or [float("nan")]),
        "max_memory_reduction": max([float(r["memory_reduction_vs_full"]) for r in target] or [float("nan")]),
    }


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    external = read_csv(RESULTS / "external_peer_benchmark_with_proxies.csv")
    scale = read_csv(RESULTS / "sota_sparse_scale.csv")
    peers = external_peer_summary(external)
    full = fullstate_summary(scale)
    target = target_regime_summary([r for r in full if r.get("family") != "AGGREGATE_MEDIAN"])
    write_csv(peers, RESULTS / "comparison_peer_speedups.csv")
    write_csv(full, RESULTS / "comparison_fullstate_ratios.csv")

    lines = [
        "# Comparison Summary",
        "",
        "All ratios are from the RTX 3070 artifact run. Values above 1 mean LC-Implicit-QAOA is faster or uses less GPU memory than the comparator.",
        "",
        "## Official/Proxy Peer Small-Instance Speed",
        "",
        "| Comparator | Cases | Comparator median s | LC median s | Median speedup | Max LC error vs full |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in peers:
        lines.append(
            f"| {row['comparison'].replace(' vs lc_batched_gpu', '')} | {row['matched_cases']} | "
            f"{fmt(row['peer_median_s'])} | {fmt(row['lc_median_s'])} | "
            f"{fmt(row['median_speedup_peer_over_lc'])}x | {fmt(row['max_lc_error_vs_full'])} |"
        )

    lines.extend(
        [
            "",
            "## QOKit/CUAOA-Class Full-State Diagonal Baseline",
            "",
            "This compares against `full_precompute_gpu`, the internal GPU baseline that materializes the global state and global diagonal cost table. It represents the memory model used by QOKit/CUAOA-style full-state diagonal evaluation, not official QOKit GPU timing on this host.",
            "",
            "### Target Sparse/Local Regime",
            "",
            "This is the regime claimed in the paper: bounded-degree or modular sparse graphs at p=1/2.",
            "",
            f"- Cases: {int(target['cases'])}.",
            f"- Median speedup vs full-state diagonal baseline: {fmt(target['median_speedup'])}x.",
            f"- Speedup range: {fmt(target['min_speedup'])}x to {fmt(target['max_speedup'])}x.",
            f"- Median GPU-memory reduction: {fmt(target['median_memory_reduction'])}x.",
            f"- GPU-memory reduction range: {fmt(target['min_memory_reduction'])}x to {fmt(target['max_memory_reduction'])}x.",
            "",
            "| Family | n | p | Full s | LC s | Speedup | Full MB | LC MB | Memory reduction | Error |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in full:
        lines.append(
            f"| {row['family']} | {row['n']} | {row['p']} | {fmt(row['full_s'])} | {fmt(row['lc_s'])} | "
            f"{fmt(row['speedup_vs_full'])}x | {fmt(row['full_peak_mb'])} | {fmt(row['lc_peak_mb'])} | "
            f"{fmt(row['memory_reduction_vs_full'])}x | {fmt(row['lc_error_vs_full'])} |"
        )

    lines.extend(
        [
            "",
            "## Capability Boundary",
            "",
            "- On the tested 8GB RTX 3070 artifact, full-state reference rows are capped at n<=24.",
            "- LC reaches n=512 for 3-regular p=1/p=2, ER degree-2 p=1, ER degree-3 p=1, ER degree-4 p=1, and modular-sparse p=1 when local cones stay within the exact-work guardrails.",
            "- For tiny n, monolithic full-state kernels or CUAOA can be as fast or faster; the main advantage is memory reach and sparse-regime scaling.",
        ]
    )
    (RESULTS / "comparison_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"WROTE {RESULTS / 'comparison_summary.md'}")
    print(f"WROTE {RESULTS / 'comparison_peer_speedups.csv'}")
    print(f"WROTE {RESULTS / 'comparison_fullstate_ratios.csv'}")


if __name__ == "__main__":
    main()
