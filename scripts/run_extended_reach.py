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

from lcqaoa.graphs import erdos_renyi_graph, modular_graph, random_regular_graph, scale_free_graph
from lcqaoa.lightcone import extract_lightcones, lightcone_expectation, lightcone_gradient_adjoint
from run_benchmarks import params_for


@dataclass
class ExtendedReachRow:
    family: str
    n: int
    m: int
    p: int
    avg_degree: float
    max_degree: int
    kmax: int
    total_cone_states: int
    task: str
    status: str
    seconds: float
    peak_pool_mb: float
    value: float
    grad_norm: float
    projected_full_precompute_mb: float
    notes: str


def graph_for(family: str, n: int, seed: int):
    if family == "3regular":
        return random_regular_graph(n, 3, seed=seed)
    if family == "er_deg2":
        return erdos_renyi_graph(n, min(0.45, 2.0 / max(2, n)), seed=seed)
    if family == "modular_sparse":
        modules = max(4, n // 16)
        return modular_graph(n, modules=modules, p_in=0.22, p_out=0.0025, seed=seed)
    if family == "scale_free_a1":
        return scale_free_graph(n, attachment=1, seed=seed)
    raise ValueError(family)


def degree_stats(graph) -> tuple[float, int]:
    deg = [0 for _ in range(graph.n)]
    for i, j, _ in graph.edges:
        deg[int(i)] += 1
        deg[int(j)] += 1
    return (sum(deg) / graph.n if graph.n else 0.0, max(deg) if deg else 0)


def cone_stats(graph, p: int) -> tuple[int, int]:
    cones = extract_lightcones(graph, p)
    if not cones:
        return 0, 0
    return max(c.k for c in cones), sum(1 << c.k for c in cones)


def projected_full_precompute_mb(n: int) -> float:
    if n > 70:
        return float("inf")
    return ((8 + 4) * (1 << n)) / 1024**2


def skipped_row(
    *,
    family: str,
    graph,
    p: int,
    avg_degree: float,
    max_degree: int,
    kmax: int,
    total_cone_states: int,
    task: str,
    status: str,
    notes: str,
) -> ExtendedReachRow:
    return ExtendedReachRow(
        family=family,
        n=graph.n,
        m=graph.m,
        p=p,
        avg_degree=avg_degree,
        max_degree=max_degree,
        kmax=kmax,
        total_cone_states=total_cone_states,
        task=task,
        status=status,
        seconds=0.0,
        peak_pool_mb=0.0,
        value=float("nan"),
        grad_norm=float("nan"),
        projected_full_precompute_mb=projected_full_precompute_mb(graph.n),
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
    run_adjoint: bool,
) -> list[ExtendedReachRow]:
    seed = 260000 + n * 17 + p * 101 + {"3regular": 1, "er_deg2": 2, "modular_sparse": 3, "scale_free_a1": 4}[family]
    graph = graph_for(family, n, seed=seed)
    gammas, betas = params_for(p)
    avg_degree, max_degree = degree_stats(graph)
    kmax, total_cone_states = cone_stats(graph, p)
    print(
        f"EXTENDED_REACH family={family} n={n} p={p} m={graph.m} "
        f"kmax={kmax} total_cone_states={total_cone_states}",
        flush=True,
    )
    rows: list[ExtendedReachRow] = []
    if kmax > max_k:
        for task in ("objective", "adjoint") if run_adjoint else ("objective",):
            rows.append(
                skipped_row(
                    family=family,
                    graph=graph,
                    p=p,
                    avg_degree=avg_degree,
                    max_degree=max_degree,
                    kmax=kmax,
                    total_cone_states=total_cone_states,
                    task=task,
                    status=f"skipped_kmax_{kmax}_over_{max_k}",
                    notes="explicit light-cone degeneration boundary",
                )
            )
        return rows
    if total_cone_states > max_total_cone_states:
        for task in ("objective", "adjoint") if run_adjoint else ("objective",):
            rows.append(
                skipped_row(
                    family=family,
                    graph=graph,
                    p=p,
                    avg_degree=avg_degree,
                    max_degree=max_degree,
                    kmax=kmax,
                    total_cone_states=total_cone_states,
                    task=task,
                    status=f"skipped_total_cone_states_{total_cone_states}",
                    notes=f"total-cone work cap {max_total_cone_states}",
                )
            )
        return rows

    obj = lightcone_expectation(
        graph,
        gammas,
        betas,
        p=p,
        prefer_gpu=True,
        max_k=max_k,
        max_batch_states=max_batch_states,
    )
    rows.append(
        ExtendedReachRow(
            family=family,
            n=graph.n,
            m=graph.m,
            p=p,
            avg_degree=avg_degree,
            max_degree=max_degree,
            kmax=kmax,
            total_cone_states=total_cone_states,
            task="objective",
            status=obj.status,
            seconds=float(obj.seconds),
            peak_pool_mb=float(obj.peak_pool_bytes) / 1024**2,
            value=float(obj.value),
            grad_norm=float("nan"),
            projected_full_precompute_mb=projected_full_precompute_mb(graph.n),
            notes="exact LC objective reach row; no full-state reference at this n",
        )
    )

    if run_adjoint:
        adj = lightcone_gradient_adjoint(
            graph,
            gammas,
            betas,
            p=p,
            prefer_gpu=True,
            max_k=max_k,
            max_batch_states=max_batch_states,
        )
        grad_norm = float("nan")
        if adj.gradient is not None:
            import numpy as np

            grad_norm = float(np.linalg.norm(adj.gradient))
        rows.append(
            ExtendedReachRow(
                family=family,
                n=graph.n,
                m=graph.m,
                p=p,
                avg_degree=avg_degree,
                max_degree=max_degree,
                kmax=kmax,
                total_cone_states=total_cone_states,
                task="adjoint",
                status=adj.status,
                seconds=float(adj.seconds),
                peak_pool_mb=float(adj.peak_pool_bytes) / 1024**2,
                value=float(adj.value),
                grad_norm=grad_norm,
                projected_full_precompute_mb=projected_full_precompute_mb(graph.n),
                notes="exact LC adjoint-gradient reach row; gradient formula validated separately against finite differences",
            )
        )
    return rows


def write_markdown(rows: list[ExtendedReachRow], path: Path) -> None:
    lines = [
        "# Extended LC Reach",
        "",
        "These rows stress the exact LC objective and adjoint-gradient evaluator beyond the 512-variable main scaling table.",
        "They do not use full-state references at large n; full-state memory is reported as a projection for context.",
        "",
        "| Family | n | p | m | kmax | total cone states | Task | Status | Seconds | Peak MB | Projected full MB | Notes |",
        "|---|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---|",
    ]
    for r in rows:
        projected = "inf" if math.isinf(r.projected_full_precompute_mb) else f"{r.projected_full_precompute_mb:.3g}"
        lines.append(
            f"| {r.family} | {r.n} | {r.p} | {r.m} | {r.kmax} | {r.total_cone_states} | "
            f"{r.task} | {r.status} | {r.seconds:.4g} | {r.peak_pool_mb:.4g} | {projected} | {r.notes} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "extended_reach.csv")
    parser.add_argument("--markdown", type=Path, default=ROOT / "results" / "extended_reach.md")
    parser.add_argument("--max-k", type=int, default=24)
    parser.add_argument("--max-total-cone-states", type=int, default=1 << 31)
    parser.add_argument("--max-batch-states", type=int, default=1 << 21)
    parser.add_argument("--run-adjoint", action="store_true")
    args = parser.parse_args()

    cases = []
    for n in [512, 1024, 2048, 4096, 8192, 16384]:
        cases.append(("3regular", n, 2))
    for n in [512, 1024, 2048, 4096]:
        cases.append(("er_deg2", n, 1))
    for n in [512, 1024, 2048]:
        cases.append(("er_deg2", n, 2))
    for n in [512, 1024, 2048, 4096]:
        cases.append(("modular_sparse", n, 1))
    for n in [512, 1024, 2048]:
        cases.append(("scale_free_a1", n, 1))

    rows: list[ExtendedReachRow] = []
    for family, n, p in cases:
        rows.extend(
            run_case(
                family,
                n,
                p,
                max_k=args.max_k,
                max_total_cone_states=args.max_total_cone_states,
                max_batch_states=args.max_batch_states,
                run_adjoint=args.run_adjoint,
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
