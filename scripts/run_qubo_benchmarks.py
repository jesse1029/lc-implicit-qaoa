from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from lcqaoa.graphs import WeightedGraph, weighted_modular_qubo_graph, weighted_qubo_graph
from lcqaoa.lightcone import extract_lightcones, lightcone_expectation
from lcqaoa.qaoa import full_state_expectation
from run_benchmarks import params_for
from run_sota_sparse_scale import degree_stats, state_vector_mib


@dataclass
class QuboRow:
    family: str
    n: int
    m: int
    fields: int
    p: int
    avg_degree: float
    max_degree: int
    kmax: int
    total_cone_states: int
    method: str
    status: str
    value: float
    seconds: float
    backend: str
    peak_pool_mb: float
    abs_error_vs_full: float
    notes: str


FAMILY_OFFSETS = {
    "qubo_er_deg2": 1201,
    "qubo_er_deg3": 1202,
    "qubo_modular_sparse": 1203,
    "qubo_modular_dense": 1204,
}


def graph_for_qubo(family: str, n: int, seed: int) -> WeightedGraph:
    if family == "qubo_er_deg2":
        return weighted_qubo_graph(n, min(0.40, 2.0 / max(2, n)), seed=seed)
    if family == "qubo_er_deg3":
        return weighted_qubo_graph(n, min(0.40, 3.0 / max(2, n)), seed=seed)
    if family == "qubo_modular_sparse":
        return weighted_modular_qubo_graph(n, modules=max(4, n // 16), p_in=0.18, p_out=0.002, seed=seed)
    if family == "qubo_modular_dense":
        return weighted_modular_qubo_graph(n, modules=max(4, n // 24), p_in=0.32, p_out=0.012, seed=seed)
    raise ValueError(family)


def cone_stats(graph: WeightedGraph, p: int) -> tuple[int, int]:
    cones = extract_lightcones(graph, p)
    if not cones:
        return 0, 0
    return max(c.k for c in cones), sum(1 << c.k for c in cones)


def skipped(
    family: str,
    graph: WeightedGraph,
    p: int,
    avg_degree: float,
    max_degree: int,
    kmax: int,
    total_cone_states: int,
    method: str,
    status: str,
    notes: str,
) -> QuboRow:
    return QuboRow(
        family=family,
        n=graph.n,
        m=graph.m,
        fields=len(graph.fields),
        p=p,
        avg_degree=avg_degree,
        max_degree=max_degree,
        kmax=kmax,
        total_cone_states=total_cone_states,
        method=method,
        status=status,
        value=float("nan"),
        seconds=0.0,
        backend="skipped",
        peak_pool_mb=0.0,
        abs_error_vs_full=float("nan"),
        notes=notes,
    )


def make_row(
    family: str,
    graph: WeightedGraph,
    p: int,
    avg_degree: float,
    max_degree: int,
    kmax: int,
    total_cone_states: int,
    method: str,
    stats,
    ref_value: float,
    notes: str,
) -> QuboRow:
    err = (
        abs(stats.value - ref_value)
        if math.isfinite(stats.value) and math.isfinite(ref_value) and stats.status == "ok"
        else float("nan")
    )
    return QuboRow(
        family=family,
        n=graph.n,
        m=graph.m,
        fields=len(graph.fields),
        p=p,
        avg_degree=avg_degree,
        max_degree=max_degree,
        kmax=kmax,
        total_cone_states=total_cone_states,
        method=method,
        status=stats.status,
        value=stats.value,
        seconds=stats.seconds,
        backend=stats.backend,
        peak_pool_mb=stats.peak_pool_bytes / 1024**2,
        abs_error_vs_full=err,
        notes=notes,
    )


def run_case(
    family: str,
    n: int,
    p: int,
    *,
    max_k: int,
    full_cap: int,
    max_total_cone_states: int,
    max_batch_states: int,
) -> list[QuboRow]:
    graph = graph_for_qubo(family, n, seed=80000 + n * 43 + p * 127 + FAMILY_OFFSETS[family])
    gammas, betas = params_for(p)
    avg_degree, max_degree = degree_stats(graph)
    kmax, total_cone_states = cone_stats(graph, p)
    print(
        "QUBO "
        f"family={family} n={n} m={graph.m} fields={len(graph.fields)} p={p} "
        f"kmax={kmax} total_cone_states={total_cone_states}",
        flush=True,
    )
    rows: list[QuboRow] = []
    ref_value = float("nan")
    if n <= full_cap:
        try:
            full = full_state_expectation(graph, gammas, betas, method="precompute", prefer_gpu=True, max_qubits=full_cap)
            ref_value = full.value if full.status == "ok" else float("nan")
            rows.append(
                make_row(
                    family,
                    graph,
                    p,
                    avg_degree,
                    max_degree,
                    kmax,
                    total_cone_states,
                    "full_precompute_gpu",
                    full,
                    ref_value,
                    "QUBO full-state reference with materialized diagonal",
                )
            )
        except Exception as exc:
            rows.append(
                skipped(
                    family,
                    graph,
                    p,
                    avg_degree,
                    max_degree,
                    kmax,
                    total_cone_states,
                    "full_precompute_gpu",
                    f"failed:{type(exc).__name__}",
                    str(exc)[:180],
                )
            )
        try:
            implicit = full_state_expectation(graph, gammas, betas, method="implicit", prefer_gpu=True, max_qubits=full_cap)
            rows.append(
                make_row(
                    family,
                    graph,
                    p,
                    avg_degree,
                    max_degree,
                    kmax,
                    total_cone_states,
                    "full_implicit_gpu",
                    implicit,
                    ref_value,
                    "QUBO full-state implicit diagonal baseline",
                )
            )
        except Exception as exc:
            rows.append(
                skipped(
                    family,
                    graph,
                    p,
                    avg_degree,
                    max_degree,
                    kmax,
                    total_cone_states,
                    "full_implicit_gpu",
                    f"failed:{type(exc).__name__}",
                    str(exc)[:180],
                )
            )
    else:
        projected = state_vector_mib(n)
        rows.append(
            skipped(
                family,
                graph,
                p,
                avg_degree,
                max_degree,
                kmax,
                total_cone_states,
                "full_precompute_gpu",
                f"skipped_over_{full_cap}_qubits",
                f"full-state cap; projected state+cost ~= {projected:.3g} MiB",
            )
        )
        rows.append(
            skipped(
                family,
                graph,
                p,
                avg_degree,
                max_degree,
                kmax,
                total_cone_states,
                "full_implicit_gpu",
                f"skipped_over_{full_cap}_qubits",
                "implicit diagonal still requires the global state vector",
            )
        )

    if kmax > max_k:
        rows.append(
            skipped(
                family,
                graph,
                p,
                avg_degree,
                max_degree,
                kmax,
                total_cone_states,
                "lc_batched_gpu",
                f"skipped_kmax_{kmax}_over_{max_k}",
                "QUBO light-cone degeneration boundary",
            )
        )
    elif total_cone_states > max_total_cone_states:
        rows.append(
            skipped(
                family,
                graph,
                p,
                avg_degree,
                max_degree,
                kmax,
                total_cone_states,
                "lc_batched_gpu",
                f"skipped_total_cone_states_{total_cone_states}",
                f"total-cone work cap {max_total_cone_states}",
            )
        )
    else:
        try:
            lc = lightcone_expectation(
                graph,
                gammas,
                betas,
                p=p,
                prefer_gpu=True,
                max_k=max_k,
                max_batch_states=max_batch_states,
            )
            rows.append(
                make_row(
                    family,
                    graph,
                    p,
                    avg_degree,
                    max_degree,
                    kmax,
                    total_cone_states,
                    "lc_batched_gpu",
                    lc,
                    ref_value,
                    "Exact weighted QUBO LC objective including linear fields",
                )
            )
        except Exception as exc:
            rows.append(
                skipped(
                    family,
                    graph,
                    p,
                    avg_degree,
                    max_degree,
                    kmax,
                    total_cone_states,
                    "lc_batched_gpu",
                    f"failed:{type(exc).__name__}",
                    str(exc)[:180],
                )
            )
    return rows


def write_markdown(rows: list[QuboRow], path: Path) -> None:
    lines = [
        "# Weighted QUBO Benchmark",
        "",
        "This benchmark exercises the actual QUBO path: weighted quadratic terms plus linear fields.",
        "",
        "## Aggregate",
        "",
        "| Family | p | LC ok | Max n ok | Max k ok | Max full-check error |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for family, p in sorted({(r.family, r.p) for r in rows if r.method == "lc_batched_gpu"}):
        subset = [r for r in rows if r.family == family and r.p == p and r.method == "lc_batched_gpu"]
        ok = [r for r in subset if r.status == "ok"]
        errors = [r.abs_error_vs_full for r in ok if math.isfinite(r.abs_error_vs_full)]
        lines.append(
            f"| {family} | {p} | {len(ok)}/{len(subset)} | {max([r.n for r in ok] or [0])} | "
            f"{max([r.kmax for r in ok] or [0])} | {max(errors or [float('nan')]):.3g} |"
        )
    lines.extend(
        [
            "",
            "## Raw Rows",
            "",
            "| Family | n | m | fields | p | avg deg | max deg | kmax | total cone states | Method | Status | Value | Time s | Peak MB | Error | Notes |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---|",
        ]
    )
    for r in rows:
        lines.append(
            f"| {r.family} | {r.n} | {r.m} | {r.fields} | {r.p} | {r.avg_degree:.2f} | "
            f"{r.max_degree} | {r.kmax} | {r.total_cone_states} | {r.method} | {r.status} | "
            f"{r.value:.7g} | {r.seconds:.4g} | {r.peak_pool_mb:.3g} | {r.abs_error_vs_full:.3g} | {r.notes} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "qubo_benchmark.csv")
    parser.add_argument("--markdown", type=Path, default=ROOT / "results" / "qubo_benchmark.md")
    parser.add_argument("--families", nargs="*", default=[
        "qubo_er_deg2",
        "qubo_er_deg3",
        "qubo_modular_sparse",
        "qubo_modular_dense",
    ])
    parser.add_argument("--ns", nargs="*", type=int, default=[12, 16, 20, 24, 32, 48, 64, 96, 128, 192, 256])
    parser.add_argument("--ps", nargs="*", type=int, default=[1, 2])
    parser.add_argument("--max-k", type=int, default=24)
    parser.add_argument("--full-cap", type=int, default=24)
    parser.add_argument("--max-total-cone-states", type=int, default=1 << 30)
    parser.add_argument("--max-batch-states", type=int, default=1 << 21)
    args = parser.parse_args()

    rows: list[QuboRow] = []
    for family in args.families:
        for n in args.ns:
            for p in args.ps:
                rows.extend(
                    run_case(
                        family,
                        n,
                        p,
                        max_k=args.max_k,
                        full_cap=args.full_cap,
                        max_total_cone_states=args.max_total_cone_states,
                        max_batch_states=args.max_batch_states,
                    )
                )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
    write_markdown(rows, args.markdown)
    print(f"WROTE {args.out}")
    print(f"WROTE {args.markdown}")


if __name__ == "__main__":
    main()
