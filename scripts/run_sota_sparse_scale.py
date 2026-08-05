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

from lcqaoa.graphs import (
    WeightedGraph,
    erdos_renyi_graph,
    modular_graph,
    random_regular_graph,
    scale_free_graph,
)
from lcqaoa.lightcone import extract_lightcones, lightcone_expectation
from lcqaoa.proxies import bmqsim_proxy_quantized_expectation, queen_proxy_fused_expectation
from lcqaoa.qaoa import full_state_expectation
from run_benchmarks import params_for


FAMILY_SEED_OFFSETS = {
    "3regular": 101,
    "er_deg2": 202,
    "er_deg3": 303,
    "er_deg4": 404,
    "er_dense": 505,
    "modular_sparse": 606,
    "modular_dense": 707,
    "scale_free_a1": 808,
    "scale_free_a2": 909,
}


@dataclass
class SotaScaleRow:
    family: str
    n: int
    m: int
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
    kmax_or_qubits: int
    peak_pool_mb: float
    abs_error_vs_full: float
    speedup_vs_full: float
    memory_ratio_vs_full: float
    notes: str


def graph_for_scale(family: str, n: int, seed: int) -> WeightedGraph:
    if family == "3regular":
        return random_regular_graph(n, 3, seed=seed)
    if family == "er_deg2":
        return erdos_renyi_graph(n, min(0.45, 2.0 / max(2, n)), seed=seed)
    if family == "er_deg3":
        return erdos_renyi_graph(n, min(0.45, 3.0 / max(2, n)), seed=seed)
    if family == "er_deg4":
        return erdos_renyi_graph(n, min(0.45, 4.0 / max(2, n)), seed=seed)
    if family == "er_dense":
        return erdos_renyi_graph(n, 0.35, seed=seed)
    if family == "modular_sparse":
        modules = max(4, n // 16)
        return modular_graph(n, modules=modules, p_in=0.22, p_out=0.0025, seed=seed)
    if family == "modular_dense":
        modules = max(4, n // 24)
        return modular_graph(n, modules=modules, p_in=0.35, p_out=0.015, seed=seed)
    if family == "scale_free_a1":
        return scale_free_graph(n, attachment=1, seed=seed)
    if family == "scale_free_a2":
        return scale_free_graph(n, attachment=2, seed=seed)
    raise ValueError(f"unknown family {family}")


def family_ns(family: str) -> list[int]:
    if family == "er_dense":
        return [12, 16, 20, 24, 32, 48]
    return [24, 32, 48, 64, 96, 128, 192, 256, 384, 512]


def degree_stats(graph: WeightedGraph) -> tuple[float, int]:
    deg = [0 for _ in range(graph.n)]
    for i, j, _ in graph.edges:
        deg[i] += 1
        deg[j] += 1
    return (sum(deg) / graph.n if graph.n else 0.0, max(deg) if deg else 0)


def cone_stats(graph: WeightedGraph, p: int) -> tuple[int, int]:
    cones = extract_lightcones(graph, p)
    if not cones:
        return 0, 0
    kmax = max(cone.k for cone in cones)
    total = sum(1 << cone.k for cone in cones)
    return kmax, total


def state_vector_mib(n: int, *, with_cost_float32: bool = True) -> float:
    # complex64 state = 8 bytes/state; optional float32 cost table = 4 bytes/state.
    bytes_per_state = 8 + (4 if with_cost_float32 else 0)
    if n > 70:
        return float("inf")
    return (bytes_per_state * (1 << n)) / 1024**2


def row_from_eval(
    *,
    family: str,
    graph: WeightedGraph,
    p: int,
    avg_degree: float,
    max_degree: int,
    kmax: int,
    total_cone_states: int,
    method: str,
    eval_stats,
    ref_value: float,
    ref_seconds: float,
    ref_peak_mb: float,
    notes: str,
) -> SotaScaleRow:
    err = (
        abs(float(eval_stats.value) - ref_value)
        if math.isfinite(ref_value) and math.isfinite(float(eval_stats.value)) and eval_stats.status.startswith("ok")
        else float("nan")
    )
    speedup = (
        ref_seconds / float(eval_stats.seconds)
        if ref_seconds > 0.0 and float(eval_stats.seconds) > 0.0 and eval_stats.status.startswith("ok")
        else float("nan")
    )
    mem_ratio = (
        ref_peak_mb / (float(eval_stats.peak_pool_bytes) / 1024**2)
        if ref_peak_mb > 0.0 and float(eval_stats.peak_pool_bytes) > 0.0 and eval_stats.status.startswith("ok")
        else float("nan")
    )
    return SotaScaleRow(
        family=family,
        n=graph.n,
        m=graph.m,
        p=p,
        avg_degree=avg_degree,
        max_degree=max_degree,
        kmax=kmax,
        total_cone_states=total_cone_states,
        method=method,
        status=eval_stats.status,
        value=float(eval_stats.value),
        seconds=float(eval_stats.seconds),
        backend=eval_stats.backend,
        kmax_or_qubits=int(eval_stats.state_qubits),
        peak_pool_mb=float(eval_stats.peak_pool_bytes) / 1024**2,
        abs_error_vs_full=err,
        speedup_vs_full=speedup,
        memory_ratio_vs_full=mem_ratio,
        notes=notes,
    )


def skipped_row(
    *,
    family: str,
    graph: WeightedGraph,
    p: int,
    avg_degree: float,
    max_degree: int,
    kmax: int,
    total_cone_states: int,
    method: str,
    status: str,
    kmax_or_qubits: int,
    notes: str,
) -> SotaScaleRow:
    return SotaScaleRow(
        family=family,
        n=graph.n,
        m=graph.m,
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
        kmax_or_qubits=kmax_or_qubits,
        peak_pool_mb=0.0,
        abs_error_vs_full=float("nan"),
        speedup_vs_full=float("nan"),
        memory_ratio_vs_full=float("nan"),
        notes=notes,
    )


def run_case(
    family: str,
    n: int,
    p: int,
    *,
    max_k: int,
    max_total_cone_states: int,
    max_batch_states: int,
    full_cap: int,
    proxy_cap: int,
    include_naive: bool,
) -> list[SotaScaleRow]:
    seed = 70000 + n * 31 + p * 101 + FAMILY_SEED_OFFSETS.get(family, 0)
    graph = graph_for_scale(family, n, seed=seed)
    gammas, betas = params_for(p)
    avg_degree, max_degree = degree_stats(graph)
    kmax, total_cone_states = cone_stats(graph, p)
    print(
        "SOTA_SCALE "
        f"family={family} n={graph.n} m={graph.m} p={p} "
        f"avg_degree={avg_degree:.2f} max_degree={max_degree} "
        f"kmax={kmax} total_cone_states={total_cone_states}",
        flush=True,
    )

    rows: list[SotaScaleRow] = []
    ref_value = float("nan")
    ref_seconds = 0.0
    ref_peak_mb = 0.0

    if graph.n <= full_cap:
        full = full_state_expectation(
            graph,
            gammas,
            betas,
            method="precompute",
            prefer_gpu=True,
            max_qubits=full_cap,
        )
        ref_value = full.value if full.status == "ok" else float("nan")
        ref_seconds = full.seconds if full.status == "ok" else 0.0
        ref_peak_mb = full.peak_pool_bytes / 1024**2 if full.status == "ok" else 0.0
        rows.append(
            row_from_eval(
                family=family,
                graph=graph,
                p=p,
                avg_degree=avg_degree,
                max_degree=max_degree,
                kmax=kmax,
                total_cone_states=total_cone_states,
                method="full_precompute_gpu",
                eval_stats=full,
                ref_value=ref_value,
                ref_seconds=ref_seconds,
                ref_peak_mb=ref_peak_mb,
                notes="materialized global state plus diagonal cost table",
            )
        )

        implicit = full_state_expectation(
            graph,
            gammas,
            betas,
            method="implicit",
            prefer_gpu=True,
            max_qubits=full_cap,
        )
        rows.append(
            row_from_eval(
                family=family,
                graph=graph,
                p=p,
                avg_degree=avg_degree,
                max_degree=max_degree,
                kmax=kmax,
                total_cone_states=total_cone_states,
                method="full_implicit_gpu",
                eval_stats=implicit,
                ref_value=ref_value,
                ref_seconds=ref_seconds,
                ref_peak_mb=ref_peak_mb,
                notes="full state without persistent global cost table",
            )
        )
    else:
        projected = state_vector_mib(graph.n)
        rows.append(
            skipped_row(
                family=family,
                graph=graph,
                p=p,
                avg_degree=avg_degree,
                max_degree=max_degree,
                kmax=kmax,
                total_cone_states=total_cone_states,
                method="full_precompute_gpu",
                status=f"skipped_over_{full_cap}_qubits",
                kmax_or_qubits=graph.n,
                notes=f"full-state cap on 8GB GPU; projected complex64+float32 global arrays ~= {projected:.3g} MiB",
            )
        )
        rows.append(
            skipped_row(
                family=family,
                graph=graph,
                p=p,
                avg_degree=avg_degree,
                max_degree=max_degree,
                kmax=kmax,
                total_cone_states=total_cone_states,
                method="full_implicit_gpu",
                status=f"skipped_over_{full_cap}_qubits",
                kmax_or_qubits=graph.n,
                notes="full-state implicit still materializes the global state vector",
            )
        )

    if graph.n <= proxy_cap:
        queen = queen_proxy_fused_expectation(
            graph,
            gammas,
            betas,
            prefer_gpu=True,
            fusion_width=4,
            max_qubits=proxy_cap,
        )
        rows.append(
            row_from_eval(
                family=family,
                graph=graph,
                p=p,
                avg_degree=avg_degree,
                max_degree=max_degree,
                kmax=kmax,
                total_cone_states=total_cone_states,
                method="queen_proxy_fused_gpu",
                eval_stats=queen,
                ref_value=ref_value,
                ref_seconds=ref_seconds,
                ref_peak_mb=ref_peak_mb,
                notes="paper-derived QueenV2-style exact full-state fused-mixer proxy; not official QueenV2",
            )
        )
        bmq = bmqsim_proxy_quantized_expectation(
            graph,
            gammas,
            betas,
            prefer_gpu=True,
            quant_bits=8,
            block_states=1 << 16,
            max_qubits=proxy_cap,
        )
        rows.append(
            row_from_eval(
                family=family,
                graph=graph,
                p=p,
                avg_degree=avg_degree,
                max_degree=max_degree,
                kmax=kmax,
                total_cone_states=total_cone_states,
                method="bmqsim_proxy_quantized_gpu",
                eval_stats=bmq,
                ref_value=ref_value,
                ref_seconds=ref_seconds,
                ref_peak_mb=ref_peak_mb,
                notes="paper-derived BMQSim-style 8-bit block quantization proxy; approximate and not official BMQSim",
            )
        )
    else:
        for method in ("queen_proxy_fused_gpu", "bmqsim_proxy_quantized_gpu"):
            rows.append(
                skipped_row(
                    family=family,
                    graph=graph,
                    p=p,
                    avg_degree=avg_degree,
                    max_degree=max_degree,
                    kmax=kmax,
                    total_cone_states=total_cone_states,
                    method=method,
                    status=f"skipped_over_{proxy_cap}_qubits",
                    kmax_or_qubits=graph.n,
                    notes="proxy is still a global full-state route",
                )
            )

    if kmax > max_k:
        rows.append(
            skipped_row(
                family=family,
                graph=graph,
                p=p,
                avg_degree=avg_degree,
                max_degree=max_degree,
                kmax=kmax,
                total_cone_states=total_cone_states,
                method="lc_batched_gpu",
                status=f"skipped_kmax_{kmax}_over_{max_k}",
                kmax_or_qubits=kmax,
                notes="explicit light-cone degeneration boundary",
            )
        )
    elif total_cone_states > max_total_cone_states:
        rows.append(
            skipped_row(
                family=family,
                graph=graph,
                p=p,
                avg_degree=avg_degree,
                max_degree=max_degree,
                kmax=kmax,
                total_cone_states=total_cone_states,
                method="lc_batched_gpu",
                status=f"skipped_total_cone_states_{total_cone_states}",
                kmax_or_qubits=kmax,
                notes=f"total-cone work cap {max_total_cone_states}",
            )
        )
    else:
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
            row_from_eval(
                family=family,
                graph=graph,
                p=p,
                avg_degree=avg_degree,
                max_degree=max_degree,
                kmax=kmax,
                total_cone_states=total_cone_states,
                method="lc_batched_gpu",
                eval_stats=lc,
                ref_value=ref_value,
                ref_seconds=ref_seconds,
                ref_peak_mb=ref_peak_mb,
                notes=f"exact LC-Implicit-QAOA; max_batch_states={max_batch_states}",
            )
        )
        if include_naive and graph.n <= 32 and p <= 2 and total_cone_states <= (1 << 25):
            naive = lightcone_expectation(
                graph,
                gammas,
                betas,
                p=p,
                prefer_gpu=True,
                max_k=max_k,
                max_batch_states=max_batch_states,
                naive=True,
            )
            rows.append(
                row_from_eval(
                    family=family,
                    graph=graph,
                    p=p,
                    avg_degree=avg_degree,
                    max_degree=max_degree,
                    kmax=kmax,
                    total_cone_states=total_cone_states,
                    method="lc_naive_gpu",
                    eval_stats=naive,
                    ref_value=ref_value,
                    ref_seconds=ref_seconds,
                    ref_peak_mb=ref_peak_mb,
                    notes="one-light-cone-at-a-time ablation for launch/batching overhead",
                )
            )

    return rows


def aggregate_lines(rows: list[SotaScaleRow]) -> list[str]:
    lines = [
        "## Aggregate",
        "",
        "| Family | p | LC ok cases | Max n ok | Max k ok | Fastest/slowest ok time s | Degeneration statuses |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    keys = sorted({(r.family, r.p) for r in rows if r.method == "lc_batched_gpu"})
    for family, p in keys:
        subset = [r for r in rows if r.family == family and r.p == p and r.method == "lc_batched_gpu"]
        oks = [r for r in subset if r.status == "ok"]
        skips = [r.status for r in subset if r.status != "ok"]
        if oks:
            max_n = max(r.n for r in oks)
            max_k = max(r.kmax for r in oks)
            min_t = min(r.seconds for r in oks)
            max_t = max(r.seconds for r in oks)
            time_cell = f"{min_t:.4g}/{max_t:.4g}"
        else:
            max_n = 0
            max_k = 0
            time_cell = "nan"
        skip_summary = ", ".join(sorted(set(skips))[:4])
        if len(set(skips)) > 4:
            skip_summary += ", ..."
        lines.append(
            f"| {family} | {p} | {len(oks)}/{len(subset)} | {max_n} | {max_k} | {time_cell} | {skip_summary} |"
        )
    return lines


def write_markdown(rows: list[SotaScaleRow], path: Path) -> None:
    lines = [
        "# SOTA Sparse Scaling Matrix",
        "",
        "This table separates full-state caps, exact LC-Implicit-QAOA rows, and paper-derived full-state proxies.",
        "The important negative result is the explicit degeneration boundary when `kmax` or total cone work grows.",
        "",
    ]
    lines.extend(aggregate_lines(rows))
    lines.extend(
        [
            "",
            "## Raw Rows",
            "",
            "| Family | n | m | p | avg deg | max deg | kmax | total cone states | Method | Status | Value | Time s | Peak/Est MB | Error | Speedup vs full | Mem ratio vs full | Notes |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for r in rows:
        lines.append(
            f"| {r.family} | {r.n} | {r.m} | {r.p} | {r.avg_degree:.2f} | {r.max_degree} | "
            f"{r.kmax} | {r.total_cone_states} | {r.method} | {r.status} | {r.value:.7g} | "
            f"{r.seconds:.4g} | {r.peak_pool_mb:.3g} | {r.abs_error_vs_full:.3g} | "
            f"{r.speedup_vs_full:.3g} | {r.memory_ratio_vs_full:.3g} | {r.notes} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "sota_sparse_scale.csv")
    parser.add_argument("--markdown", type=Path, default=ROOT / "results" / "sota_sparse_scale.md")
    parser.add_argument("--max-k", type=int, default=24)
    parser.add_argument("--max-total-cone-states", type=int, default=1 << 30)
    parser.add_argument("--max-batch-states", type=int, default=1 << 21)
    parser.add_argument("--full-cap", type=int, default=24)
    parser.add_argument("--proxy-cap", type=int, default=24)
    parser.add_argument("--families", nargs="*", default=[
        "3regular",
        "er_deg2",
        "er_deg3",
        "er_deg4",
        "modular_sparse",
        "modular_dense",
        "scale_free_a1",
        "scale_free_a2",
        "er_dense",
    ])
    parser.add_argument("--ps", nargs="*", type=int, default=[1, 2, 3])
    parser.add_argument("--include-naive", action="store_true")
    args = parser.parse_args()

    rows: list[SotaScaleRow] = []
    for family in args.families:
        for n in family_ns(family):
            for p in args.ps:
                rows.extend(
                    run_case(
                        family,
                        n,
                        p,
                        max_k=args.max_k,
                        max_total_cone_states=args.max_total_cone_states,
                        max_batch_states=args.max_batch_states,
                        full_cap=args.full_cap,
                        proxy_cap=args.proxy_cap,
                        include_naive=args.include_naive,
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
