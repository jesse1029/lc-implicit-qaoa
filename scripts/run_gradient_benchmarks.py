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

from lcqaoa.lightcone import extract_lightcones, lightcone_expectation
from lcqaoa.qaoa import finite_difference_gradient, full_state_expectation
from run_benchmarks import params_for
from run_sota_sparse_scale import FAMILY_SEED_OFFSETS, degree_stats, graph_for_scale


@dataclass
class GradientRow:
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
    objective_seconds: float
    gradient_seconds: float
    gradient_norm: float
    max_abs_gradient_error_vs_full: float
    backend: str
    peak_pool_mb: float
    notes: str


def gradient_case_specs() -> list[tuple[str, int, int]]:
    return [
        ("3regular", 24, 2),
        ("3regular", 24, 3),
        ("3regular", 128, 2),
        ("3regular", 512, 2),
        ("er_deg2", 128, 2),
        ("er_deg3", 24, 2),
        ("er_deg4", 24, 2),
        ("modular_sparse", 128, 1),
        ("modular_sparse", 256, 2),
        ("scale_free_a1", 64, 1),
        ("er_dense", 20, 1),
    ]


def cone_stats(graph, p: int) -> tuple[int, int]:
    cones = extract_lightcones(graph, p)
    if not cones:
        return 0, 0
    return max(c.k for c in cones), sum(1 << c.k for c in cones)


def skipped_row(
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
) -> GradientRow:
    return GradientRow(
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
        objective_seconds=0.0,
        gradient_seconds=0.0,
        gradient_norm=float("nan"),
        max_abs_gradient_error_vs_full=float("nan"),
        backend="skipped",
        peak_pool_mb=0.0,
        notes=notes,
    )


def run_one_case(
    family: str,
    n: int,
    p: int,
    *,
    max_k: int,
    max_total_cone_states: int,
    max_batch_states: int,
    full_cap: int,
    eps: float,
) -> list[GradientRow]:
    graph = graph_for_scale(family, n, seed=90000 + n * 41 + p * 113 + FAMILY_SEED_OFFSETS.get(family, 0))
    gammas, betas = params_for(p)
    avg_degree, max_degree = degree_stats(graph)
    kmax, total_cone_states = cone_stats(graph, p)
    print(
        "GRAD "
        f"family={family} n={graph.n} m={graph.m} p={p} "
        f"kmax={kmax} total_cone_states={total_cone_states}",
        flush=True,
    )

    rows: list[GradientRow] = []
    full_grad: np.ndarray | None = None
    full_value = float("nan")

    if graph.n <= full_cap:
        def full_fn(g, b):
            return full_state_expectation(
                graph,
                g,
                b,
                method="precompute",
                prefer_gpu=True,
                max_qubits=full_cap,
            )

        base = full_fn(gammas, betas)
        grad, grad_seconds = finite_difference_gradient(full_fn, gammas, betas, eps=eps)
        full_grad = grad
        full_value = base.value
        rows.append(
            GradientRow(
                family=family,
                n=graph.n,
                m=graph.m,
                p=p,
                avg_degree=avg_degree,
                max_degree=max_degree,
                kmax=kmax,
                total_cone_states=total_cone_states,
                method="full_precompute_gpu_fd",
                status=base.status,
                objective_value=base.value,
                objective_seconds=base.seconds,
                gradient_seconds=grad_seconds,
                gradient_norm=float(np.linalg.norm(grad)),
                max_abs_gradient_error_vs_full=0.0,
                backend=base.backend,
                peak_pool_mb=base.peak_pool_bytes / 1024**2,
                notes="central finite difference over global full-state objective",
            )
        )
    else:
        rows.append(
            skipped_row(
                family,
                graph,
                p,
                avg_degree,
                max_degree,
                kmax,
                total_cone_states,
                "full_precompute_gpu_fd",
                f"skipped_over_{full_cap}_qubits",
                "global state-vector gradient is infeasible under the configured 8GB cap",
            )
        )

    if kmax > max_k:
        rows.append(
            skipped_row(
                family,
                graph,
                p,
                avg_degree,
                max_degree,
                kmax,
                total_cone_states,
                "lc_batched_gpu_fd",
                f"skipped_kmax_{kmax}_over_{max_k}",
                "gradient inherits the exact light-cone degeneration boundary",
            )
        )
        return rows
    if total_cone_states > max_total_cone_states:
        rows.append(
            skipped_row(
                family,
                graph,
                p,
                avg_degree,
                max_degree,
                kmax,
                total_cone_states,
                "lc_batched_gpu_fd",
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

    base = lc_fn(gammas, betas)
    grad, grad_seconds = finite_difference_gradient(lc_fn, gammas, betas, eps=eps)
    if full_grad is not None:
        grad_err = float(np.max(np.abs(grad - full_grad)))
    else:
        grad_err = float("nan")
    rows.append(
        GradientRow(
            family=family,
            n=graph.n,
            m=graph.m,
            p=p,
            avg_degree=avg_degree,
            max_degree=max_degree,
            kmax=kmax,
            total_cone_states=total_cone_states,
            method="lc_batched_gpu_fd",
            status=base.status,
            objective_value=base.value,
            objective_seconds=base.seconds,
            gradient_seconds=grad_seconds,
            gradient_norm=float(np.linalg.norm(grad)),
            max_abs_gradient_error_vs_full=grad_err,
            backend=base.backend,
            peak_pool_mb=base.peak_pool_bytes / 1024**2,
            notes=(
                "central finite difference over exact LC objective; "
                f"objective_error_vs_full={abs(base.value - full_value):.3g}"
                if math.isfinite(full_value)
                else "central finite difference over exact LC objective; no full-state gradient at this n"
            ),
        )
    )
    return rows


def write_markdown(rows: list[GradientRow], path: Path) -> None:
    lines = [
        "# Gradient Benchmark",
        "",
        "This benchmark measures training-time gradient cost using central finite differences over the exact objective.",
        "It is intentionally conservative: no fused analytic LC gradient is claimed here.",
        "",
        "| Family | n | m | p | kmax | Method | Status | Obj value | Obj s | Grad s | Grad norm | Grad err vs full | Peak MB | Notes |",
        "|---|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r.family} | {r.n} | {r.m} | {r.p} | {r.kmax} | {r.method} | {r.status} | "
            f"{r.objective_value:.7g} | {r.objective_seconds:.4g} | {r.gradient_seconds:.4g} | "
            f"{r.gradient_norm:.4g} | {r.max_abs_gradient_error_vs_full:.3g} | "
            f"{r.peak_pool_mb:.3g} | {r.notes} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "gradient_benchmark.csv")
    parser.add_argument("--markdown", type=Path, default=ROOT / "results" / "gradient_benchmark.md")
    parser.add_argument("--max-k", type=int, default=24)
    parser.add_argument("--max-total-cone-states", type=int, default=1 << 30)
    parser.add_argument("--max-batch-states", type=int, default=1 << 21)
    parser.add_argument("--full-cap", type=int, default=24)
    parser.add_argument("--eps", type=float, default=1e-3)
    args = parser.parse_args()

    rows: list[GradientRow] = []
    for family, n, p in gradient_case_specs():
        rows.extend(
            run_one_case(
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
