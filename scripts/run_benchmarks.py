from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lcqaoa.graphs import (
    WeightedGraph,
    erdos_renyi_graph,
    modular_graph,
    random_regular_graph,
    scale_free_graph,
)
from lcqaoa.lightcone import extract_lightcones, lightcone_expectation
from lcqaoa.proxies import bmqsim_proxy_quantized_expectation, queen_proxy_fused_expectation
from lcqaoa.qaoa import full_state_expectation, finite_difference_gradient


@dataclass
class Row:
    family: str
    n: int
    m: int
    p: int
    method: str
    status: str
    value: float
    seconds: float
    backend: str
    kmax_or_qubits: int
    peak_pool_mb: float
    abs_error_vs_full: float
    grad_seconds: float
    notes: str


def qokit_cpu_expectation(graph: WeightedGraph, gammas: list[float], betas: list[float]):
    if graph.objective != "maxcut":
        raise ValueError("QOKit adapter is only wired for MaxCut")
    import time
    import networkx as nx
    from qokit.maxcut import get_maxcut_terms
    from qokit.fur.python.qaoa_simulator import QAOAFURXSimulator

    G = nx.Graph()
    G.add_nodes_from(range(graph.n))
    G.add_weighted_edges_from(graph.edges)
    terms = get_maxcut_terms(G)
    sim = QAOAFURXSimulator(graph.n, terms=terms)
    t0 = time.perf_counter()
    result = sim.simulate_qaoa([2.0 * g for g in gammas], betas)
    value = float(sim.get_expectation(result))
    return value, time.perf_counter() - t0


def graph_for(family: str, n: int, seed: int) -> WeightedGraph:
    if family == "3regular":
        return random_regular_graph(n, 3, seed=seed)
    if family == "er_sparse":
        return erdos_renyi_graph(n, min(0.12, 3.0 / max(2, n)), seed=seed)
    if family == "er_dense":
        return erdos_renyi_graph(n, 0.35, seed=seed)
    if family == "modular":
        return modular_graph(n, modules=4, p_in=0.28, p_out=0.015, seed=seed)
    if family == "scale_free":
        return scale_free_graph(n, attachment=2, seed=seed)
    raise ValueError(f"unknown family {family}")


def params_for(p: int) -> tuple[list[float], list[float]]:
    gammas = [0.20 + 0.05 * i for i in range(p)]
    betas = [0.32 - 0.035 * i for i in range(p)]
    return gammas, betas


def run_one(graph: WeightedGraph, family: str, p: int, gradient: bool) -> list[Row]:
    gammas, betas = params_for(p)
    rows: list[Row] = []
    cones = extract_lightcones(graph, p)
    kmax = max(c.k for c in cones) if cones else 0
    total_cone_states = sum(1 << c.k for c in cones)

    full_ref = full_state_expectation(
        graph,
        gammas,
        betas,
        method="precompute",
        prefer_gpu=True,
        max_qubits=24,
    )
    ref_value = full_ref.value if full_ref.status == "ok" else float("nan")
    rows.append(
        Row(
            family,
            graph.n,
            graph.m,
            p,
            "full_precompute_gpu",
            full_ref.status,
            full_ref.value,
            full_ref.seconds,
            full_ref.backend,
            full_ref.state_qubits,
            full_ref.peak_pool_bytes / 1024**2,
            0.0 if full_ref.status == "ok" else float("nan"),
            float("nan"),
            "QOKit/CUAOA-style materialized diagonal full-state baseline",
        )
    )

    queen = queen_proxy_fused_expectation(
        graph,
        gammas,
        betas,
        prefer_gpu=True,
        fusion_width=4,
        max_qubits=24,
    )
    rows.append(
        Row(
            family,
            graph.n,
            graph.m,
            p,
            "queen_proxy_fused_gpu",
            queen.status,
            queen.value,
            queen.seconds,
            queen.backend,
            queen.state_qubits,
            queen.peak_pool_bytes / 1024**2,
            abs(queen.value - ref_value) if math.isfinite(ref_value) and queen.status == "ok" else float("nan"),
            float("nan"),
            "paper-derived QueenV2-style proxy: exact full-state, fused adjacent X-mixer blocks; not official QueenV2",
        )
    )

    bmq = bmqsim_proxy_quantized_expectation(
        graph,
        gammas,
        betas,
        prefer_gpu=True,
        quant_bits=8,
        block_states=1 << 16,
        max_qubits=24,
    )
    rows.append(
        Row(
            family,
            graph.n,
            graph.m,
            p,
            "bmqsim_proxy_quantized_gpu",
            bmq.status,
            bmq.value,
            bmq.seconds,
            bmq.backend,
            bmq.state_qubits,
            bmq.peak_pool_bytes / 1024**2,
            abs(bmq.value - ref_value) if math.isfinite(ref_value) and bmq.status.startswith("ok") else float("nan"),
            float("nan"),
            "paper-derived BMQSim-style proxy: block-wise 8-bit lossy state checkpoint; peak MB is compressed-payload estimate, not CuPy pool; not official BMQSim",
        )
    )

    if graph.objective == "maxcut" and graph.n <= 18 and p <= 2:
        try:
            qokit_value, qokit_seconds = qokit_cpu_expectation(graph, gammas, betas)
            rows.append(
                Row(
                    family,
                    graph.n,
                    graph.m,
                    p,
                    "qokit_cpu_external",
                    "ok",
                    qokit_value,
                    qokit_seconds,
                    "numpy",
                    graph.n,
                    0.0,
                    abs(qokit_value - ref_value) if math.isfinite(ref_value) else float("nan"),
                    float("nan"),
                    "external QOKit FUR CPU simulator; gammas doubled to match phase convention",
                )
            )
        except Exception as exc:
            rows.append(
                Row(
                    family,
                    graph.n,
                    graph.m,
                    p,
                    "qokit_cpu_external",
                    f"failed:{type(exc).__name__}",
                    float("nan"),
                    0.0,
                    "numpy",
                    graph.n,
                    0.0,
                    float("nan"),
                    float("nan"),
                    str(exc)[:180],
                )
            )
    elif graph.objective == "maxcut":
        rows.append(
            Row(
                family,
                graph.n,
                graph.m,
                p,
                "qokit_cpu_external",
                "skipped_external_cpu_cap",
                float("nan"),
                0.0,
                "numpy",
                graph.n,
                0.0,
                float("nan"),
                float("nan"),
                "external QOKit CPU row capped at n<=18,p<=2",
            )
        )

    full_impl = full_state_expectation(
        graph,
        gammas,
        betas,
        method="implicit",
        prefer_gpu=True,
        max_qubits=24,
    )
    rows.append(
        Row(
            family,
            graph.n,
            graph.m,
            p,
            "full_implicit_gpu",
            full_impl.status,
            full_impl.value,
            full_impl.seconds,
            full_impl.backend,
            full_impl.state_qubits,
            full_impl.peak_pool_bytes / 1024**2,
            abs(full_impl.value - ref_value) if math.isfinite(ref_value) and full_impl.status == "ok" else float("nan"),
            float("nan"),
            "state-vector route without persistent cost vector",
        )
    )

    if total_cone_states > (1 << 27):
        lc = None
        lc_status = f"skipped_total_cone_states_{total_cone_states}"
    else:
        lc = lightcone_expectation(
            graph,
            gammas,
            betas,
            p=p,
            prefer_gpu=True,
            max_k=24,
            max_batch_states=1 << 21,
        )
        lc_status = lc.status
    grad_seconds = float("nan")
    if gradient and lc is not None and lc.status == "ok" and graph.n <= 32 and p <= 2:
        def fn(g, b):
            return lightcone_expectation(
                graph,
                g,
                b,
                p=len(g),
                prefer_gpu=True,
                max_k=24,
                max_batch_states=1 << 21,
            )

        _, grad_seconds = finite_difference_gradient(fn, gammas, betas, eps=1e-3)
    rows.append(
        Row(
            family,
            graph.n,
            graph.m,
            p,
            "lc_batched_gpu",
            lc_status,
            lc.value if lc is not None else float("nan"),
            lc.seconds if lc is not None else 0.0,
            lc.backend if lc is not None else "cupy",
            kmax,
            lc.peak_pool_bytes / 1024**2 if lc is not None else 0.0,
            abs(lc.value - ref_value) if lc is not None and math.isfinite(ref_value) and lc.status == "ok" else float("nan"),
            grad_seconds,
            f"LC-Implicit-QAOA exact batched local evaluator; total_cone_states={total_cone_states}",
        )
    )

    if graph.n <= 32 and p <= 2 and total_cone_states <= (1 << 25):
        naive = lightcone_expectation(
            graph,
            gammas,
            betas,
            p=p,
            prefer_gpu=True,
            max_k=24,
            max_batch_states=1 << 21,
            naive=True,
        )
        rows.append(
            Row(
                family,
                graph.n,
                graph.m,
                p,
                "lc_naive_gpu",
                naive.status,
                naive.value,
                naive.seconds,
                naive.backend,
                kmax,
                naive.peak_pool_bytes / 1024**2,
                abs(naive.value - ref_value) if math.isfinite(ref_value) and naive.status == "ok" else float("nan"),
                float("nan"),
                "QTensor-style one-cone-at-a-time overhead baseline",
            )
        )
    return rows


def write_markdown(rows: list[Row], path: Path) -> None:
    lines = [
        "# LC-Implicit-QAOA Benchmark",
        "",
        "All timings are single-process wall-clock measurements on the active host.",
        "`peak_pool_mb` is CuPy memory-pool allocation, not total process RSS.",
        "Rows ending in `_proxy_*` are paper-derived local reimplementations, not official author code.",
        "",
        "| Family | n | m | p | Method | Status | Value | Time s | k/qubits | Peak/Est MB | Error vs full | Grad s | Notes |",
        "|---|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r.family} | {r.n} | {r.m} | {r.p} | {r.method} | {r.status} | "
            f"{r.value:.6g} | {r.seconds:.4g} | {r.kmax_or_qubits} | {r.peak_pool_mb:.1f} | "
            f"{r.abs_error_vs_full:.3g} | {r.grad_seconds:.4g} | {r.notes} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("results/benchmark.csv"))
    parser.add_argument("--markdown", type=Path, default=Path("results/benchmark.md"))
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--gradient", action="store_true")
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    if args.quick:
        families = ["3regular", "er_sparse", "er_dense", "modular"]
        ns = [12, 16, 20, 24]
        ps = [1, 2]
    else:
        families = ["3regular", "er_sparse", "er_dense", "modular", "scale_free"]
        ns = [12, 16, 20, 24, 28, 32, 40, 56, 72]
        ps = [1, 2, 3]

    rows: list[Row] = []
    for family in families:
        for n in ns:
            for p in ps:
                graph = graph_for(family, n, seed=10000 + n * 13 + p)
                print(f"RUN family={family} n={n} m={graph.m} p={p}", flush=True)
                rows.extend(run_one(graph, family, p, gradient=args.gradient))

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
