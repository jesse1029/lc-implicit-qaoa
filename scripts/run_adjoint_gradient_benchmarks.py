from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from lcqaoa.graphs import weighted_modular_qubo_graph, weighted_qubo_graph
from lcqaoa.lightcone import extract_lightcones, lightcone_expectation, lightcone_gradient_adjoint
from lcqaoa.qaoa import finite_difference_gradient, full_state_expectation
from run_benchmarks import params_for
from run_sota_sparse_scale import FAMILY_SEED_OFFSETS, degree_stats, graph_for_scale


@dataclass
class AdjointGradientRow:
    case: str
    objective: str
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
    objective_value: float
    seconds: float
    gradient_norm: float
    max_abs_error_vs_lc_fd: float
    max_abs_error_vs_full_fd: float
    value_error_vs_lc: float
    peak_pool_mb: float
    notes: str


def cone_stats(graph, p: int) -> tuple[int, int]:
    cones = extract_lightcones(graph, p)
    if not cones:
        return 0, 0
    return max(c.k for c in cones), sum(1 << c.k for c in cones)


def case_specs() -> list[tuple[str, str, int, int]]:
    return [
        ("maxcut", "3regular", 24, 2),
        ("maxcut", "3regular", 128, 2),
        ("maxcut", "3regular", 512, 2),
        ("maxcut", "modular_sparse", 128, 1),
        ("maxcut", "er_deg2", 128, 2),
        ("qubo", "qubo_er_deg2", 24, 2),
        ("qubo", "qubo_er_deg2", 96, 2),
        ("qubo", "qubo_modular_sparse", 128, 1),
    ]


def make_graph(objective: str, family: str, n: int, p: int):
    seed = 120000 + n * 37 + p * 101 + FAMILY_SEED_OFFSETS.get(family.replace("qubo_", ""), 0)
    if objective == "maxcut":
        return graph_for_scale(family, n, seed=seed)
    if family == "qubo_er_deg2":
        return weighted_qubo_graph(n, min(1.0, 2.0 / max(n - 1, 1)), seed=seed)
    if family == "qubo_modular_sparse":
        return weighted_modular_qubo_graph(n, modules=8 if n >= 64 else 4, p_in=0.08, p_out=0.002, seed=seed)
    raise ValueError(f"unknown QUBO family: {family}")


def skipped_row(
    case: str,
    objective: str,
    family: str,
    graph,
    p: int,
    avg_degree: float,
    max_degree: int,
    kmax: int,
    total_cone_states: int,
    method: str,
    status: str,
    notes: str,
) -> AdjointGradientRow:
    return AdjointGradientRow(
        case=case,
        objective=objective,
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
        objective_value=float("nan"),
        seconds=0.0,
        gradient_norm=float("nan"),
        max_abs_error_vs_lc_fd=float("nan"),
        max_abs_error_vs_full_fd=float("nan"),
        value_error_vs_lc=float("nan"),
        peak_pool_mb=0.0,
        notes=notes,
    )


def run_case(
    objective: str,
    family: str,
    n: int,
    p: int,
    *,
    max_k: int,
    max_total_cone_states: int,
    max_batch_states: int,
    full_cap: int,
    eps: float,
) -> list[AdjointGradientRow]:
    graph = make_graph(objective, family, n, p)
    gammas, betas = params_for(p)
    avg_degree, max_degree = degree_stats(graph)
    kmax, total_cone_states = cone_stats(graph, p)
    case = f"{family}_n{n}_p{p}"
    print(
        "ADJOINT_GRAD "
        f"case={case} objective={objective} m={graph.m} kmax={kmax} total_cone_states={total_cone_states}",
        flush=True,
    )
    rows: list[AdjointGradientRow] = []

    full_fd: np.ndarray | None = None
    lc_fd: np.ndarray | None = None
    lc_value = float("nan")

    if graph.n <= full_cap:
        def full_fn(g, b):
            return full_state_expectation(graph, g, b, method="precompute", prefer_gpu=True, max_qubits=full_cap)

        full_base = full_fn(gammas, betas)
        full_fd, full_seconds = finite_difference_gradient(full_fn, gammas, betas, eps=eps)
        rows.append(
            AdjointGradientRow(
                case=case,
                objective=objective,
                family=family,
                n=graph.n,
                m=graph.m,
                p=p,
                avg_degree=avg_degree,
                max_degree=max_degree,
                kmax=kmax,
                total_cone_states=total_cone_states,
                method="full_precompute_gpu_fd",
                status=full_base.status,
                objective_value=full_base.value,
                seconds=full_seconds,
                gradient_norm=float(np.linalg.norm(full_fd)),
                max_abs_error_vs_lc_fd=float("nan"),
                max_abs_error_vs_full_fd=0.0,
                value_error_vs_lc=float("nan"),
                peak_pool_mb=full_base.peak_pool_bytes / 1024**2,
                notes="central finite difference over global full-state objective",
            )
        )
    else:
        rows.append(
            skipped_row(
                case,
                objective,
                family,
                graph,
                p,
                avg_degree,
                max_degree,
                kmax,
                total_cone_states,
                "full_precompute_gpu_fd",
                f"skipped_over_{full_cap}_qubits",
                "global full-state gradient exceeds configured artifact cap",
            )
        )

    if kmax > max_k:
        for method in ("lc_batched_gpu_fd", "lc_batched_gpu_adjoint"):
            rows.append(
                skipped_row(
                    case,
                    objective,
                    family,
                    graph,
                    p,
                    avg_degree,
                    max_degree,
                    kmax,
                    total_cone_states,
                    method,
                    f"skipped_kmax_{kmax}_over_{max_k}",
                    "exact light-cone degeneration boundary",
                )
            )
        return rows
    if total_cone_states > max_total_cone_states:
        for method in ("lc_batched_gpu_fd", "lc_batched_gpu_adjoint"):
            rows.append(
                skipped_row(
                    case,
                    objective,
                    family,
                    graph,
                    p,
                    avg_degree,
                    max_degree,
                    kmax,
                    total_cone_states,
                    method,
                    f"skipped_total_cone_states_{total_cone_states}",
                    f"total-cone work cap {max_total_cone_states}",
                )
            )
        return rows

    def lc_fn(g, b):
        return lightcone_expectation(
            graph,
            g,
            b,
            p=len(g),
            prefer_gpu=True,
            max_k=max_k,
            max_batch_states=max_batch_states,
        )

    lc_base = lc_fn(gammas, betas)
    lc_value = lc_base.value
    lc_fd, lc_fd_seconds = finite_difference_gradient(lc_fn, gammas, betas, eps=eps)
    rows.append(
        AdjointGradientRow(
            case=case,
            objective=objective,
            family=family,
            n=graph.n,
            m=graph.m,
            p=p,
            avg_degree=avg_degree,
            max_degree=max_degree,
            kmax=kmax,
            total_cone_states=total_cone_states,
            method="lc_batched_gpu_fd",
            status=lc_base.status,
            objective_value=lc_base.value,
            seconds=lc_fd_seconds,
            gradient_norm=float(np.linalg.norm(lc_fd)),
            max_abs_error_vs_lc_fd=0.0,
            max_abs_error_vs_full_fd=float(np.max(np.abs(lc_fd - full_fd))) if full_fd is not None else float("nan"),
            value_error_vs_lc=0.0,
            peak_pool_mb=lc_base.peak_pool_bytes / 1024**2,
            notes="central finite difference over exact LC objective",
        )
    )

    adj = lightcone_gradient_adjoint(
        graph,
        gammas,
        betas,
        p=p,
        prefer_gpu=True,
        max_k=max_k,
        max_batch_states=max_batch_states,
    )
    grad = adj.gradient
    rows.append(
        AdjointGradientRow(
            case=case,
            objective=objective,
            family=family,
            n=graph.n,
            m=graph.m,
            p=p,
            avg_degree=avg_degree,
            max_degree=max_degree,
            kmax=kmax,
            total_cone_states=total_cone_states,
            method="lc_batched_gpu_adjoint",
            status=adj.status,
            objective_value=adj.value,
            seconds=adj.seconds,
            gradient_norm=float(np.linalg.norm(grad)) if grad is not None else float("nan"),
            max_abs_error_vs_lc_fd=float(np.max(np.abs(grad - lc_fd))) if grad is not None and lc_fd is not None else float("nan"),
            max_abs_error_vs_full_fd=float(np.max(np.abs(grad - full_fd))) if grad is not None and full_fd is not None else float("nan"),
            value_error_vs_lc=abs(adj.value - lc_value) if math.isfinite(lc_value) else float("nan"),
            peak_pool_mb=adj.peak_pool_bytes / 1024**2,
            notes="single reverse-mode adjoint pass over exact local light-cone simulations",
        )
    )
    return rows


def write_markdown(rows: list[AdjointGradientRow], path: Path) -> None:
    lines = [
        "# Adjoint Gradient Benchmark",
        "",
        "This benchmark validates the exact reverse-mode adjoint gradient for LC-Implicit-QAOA.",
        "Errors are measured against LC central finite difference, and against full-state central finite difference when the full-state case fits the configured cap.",
        "",
        "| Case | Objective | n | m | p | kmax | Method | Status | Seconds | Grad norm | Err vs LC FD | Err vs full FD | Peak MB | Notes |",
        "|---|---|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r.case} | {r.objective} | {r.n} | {r.m} | {r.p} | {r.kmax} | {r.method} | {r.status} | "
            f"{r.seconds:.4g} | {r.gradient_norm:.4g} | {r.max_abs_error_vs_lc_fd:.3g} | "
            f"{r.max_abs_error_vs_full_fd:.3g} | {r.peak_pool_mb:.3g} | {r.notes} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "adjoint_gradient_benchmark.csv")
    parser.add_argument("--markdown", type=Path, default=ROOT / "results" / "adjoint_gradient_benchmark.md")
    parser.add_argument("--max-k", type=int, default=24)
    parser.add_argument("--max-total-cone-states", type=int, default=1 << 30)
    parser.add_argument("--max-batch-states", type=int, default=1 << 21)
    parser.add_argument("--full-cap", type=int, default=24)
    parser.add_argument("--eps", type=float, default=1e-3)
    args = parser.parse_args()

    rows: list[AdjointGradientRow] = []
    for objective, family, n, p in case_specs():
        rows.extend(
            run_case(
                objective,
                family,
                n,
                p,
                max_k=args.max_k,
                max_total_cone_states=args.max_total_cone_states,
                max_batch_states=args.max_batch_states,
                full_cap=args.full_cap,
                eps=args.eps,
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
