from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from lcqaoa.graphs import WeightedGraph
from lcqaoa.lightcone import extract_lightcones, lightcone_expectation, lightcone_gradient_adjoint
from lcqaoa.qaoa import full_state_expectation
from run_benchmarks import params_for
from run_qubo_benchmarks import graph_for_qubo
from run_real_qubo_case_study import feature_selection_qubo, load_datasets
from run_sota_sparse_scale import FAMILY_SEED_OFFSETS, graph_for_scale


@dataclass
class A4Row:
    case: str
    objective: str
    family: str
    n: int
    m: int
    fields: int
    p: int
    kmax: int
    total_cone_states: int
    dtype_label: str
    reference: str
    eps: float
    status: str
    adjoint_value: float
    reference_value: float
    value_abs_diff: float
    adjoint_seconds: float
    fd_seconds: float
    objective_calls_fd: int
    rel_l2_error: float
    max_abs_error: float
    cosine_similarity: float
    adjoint_grad_norm: float
    reference_grad_norm: float
    one_step_predicted_delta: float
    one_step_actual_delta: float
    peak_pool_mb: float
    notes: str


DTYPES = {
    "c64_f32": (np.complex64, np.float32),
    "c128_f64": (np.complex128, np.float64),
}


def pack_params(p: int) -> np.ndarray:
    gammas, betas = params_for(p)
    return np.asarray(gammas + betas, dtype=np.float64)


def unpack_params(x: np.ndarray, p: int) -> tuple[list[float], list[float]]:
    return x[:p].astype(float).tolist(), x[p:].astype(float).tolist()


def cone_stats(graph: WeightedGraph, p: int) -> tuple[int, int]:
    cones = extract_lightcones(graph, p)
    if not cones:
        return 0, 0
    return max(c.k for c in cones), sum(1 << c.k for c in cones)


def graph_case(case: str) -> tuple[str, str, str, WeightedGraph, int]:
    if case == "maxcut_3regular_n24_p2":
        n, p = 24, 2
        return case, "maxcut", "3regular", graph_for_scale("3regular", n, seed=110000 + n * 47 + p * 151 + FAMILY_SEED_OFFSETS["3regular"]), p
    if case == "maxcut_3regular_n128_p2":
        n, p = 128, 2
        return case, "maxcut", "3regular", graph_for_scale("3regular", n, seed=110000 + n * 47 + p * 151 + FAMILY_SEED_OFFSETS["3regular"]), p
    if case == "maxcut_3regular_n512_p2":
        n, p = 512, 2
        return case, "maxcut", "3regular", graph_for_scale("3regular", n, seed=110000 + n * 47 + p * 151 + FAMILY_SEED_OFFSETS["3regular"]), p
    if case == "qubo_er_deg2_n96_p2":
        n, p = 96, 2
        return case, "qubo", "qubo_er_deg2", graph_for_qubo("qubo_er_deg2", n, seed=110000 + n * 47 + p * 151), p
    if case == "qubo_modular_sparse_n128_p1":
        n, p = 128, 1
        return case, "qubo", "qubo_modular_sparse", graph_for_qubo("qubo_modular_sparse", n, seed=110000 + n * 47 + p * 151), p
    if case == "wdbc_top20_p1":
        datasets = load_datasets()
        graph = feature_selection_qubo(*datasets["breast_cancer"], top_features=20, top_degree=3)
        return case, "qubo", "wdbc_top20_deg3", graph, 1
    if case == "wdbc_top20_p2":
        datasets = load_datasets()
        graph = feature_selection_qubo(*datasets["breast_cancer"], top_features=20, top_degree=3)
        return case, "qubo", "wdbc_top20_deg3", graph, 2
    raise ValueError(case)


def evaluate_lc(graph: WeightedGraph, x: np.ndarray, p: int, complex_dtype, float_dtype, max_k: int, max_batch_states: int):
    gammas, betas = unpack_params(x, p)
    return lightcone_expectation(
        graph,
        gammas,
        betas,
        p=p,
        prefer_gpu=True,
        max_k=max_k,
        max_batch_states=max_batch_states,
        complex_dtype=complex_dtype,
        float_dtype=float_dtype,
    )


def evaluate_full(graph: WeightedGraph, x: np.ndarray, p: int, complex_dtype, float_dtype, full_cap: int):
    gammas, betas = unpack_params(x, p)
    return full_state_expectation(
        graph,
        gammas,
        betas,
        method="precompute",
        prefer_gpu=True,
        complex_dtype=complex_dtype,
        float_dtype=float_dtype,
        max_qubits=full_cap,
    )


def central_fd_gradient(eval_fn, x: np.ndarray, eps: float) -> tuple[np.ndarray, float, float, int, str]:
    grad = np.zeros_like(x, dtype=np.float64)
    calls = 0
    peak_mb = 0.0
    status = "ok"
    t0 = time.perf_counter()
    for i in range(x.size):
        plus = x.copy()
        minus = x.copy()
        plus[i] += eps
        minus[i] -= eps
        sp = eval_fn(plus)
        sm = eval_fn(minus)
        calls += 2
        peak_mb = max(peak_mb, sp.peak_pool_bytes / 1024**2, sm.peak_pool_bytes / 1024**2)
        if sp.status != "ok" or sm.status != "ok":
            status = f"failed_fd:{sp.status}/{sm.status}"
            return grad, time.perf_counter() - t0, peak_mb, calls, status
        grad[i] = (float(sp.value) - float(sm.value)) / (2.0 * eps)
    return grad, time.perf_counter() - t0, peak_mb, calls, status


def grad_metrics(g_adj: np.ndarray, g_ref: np.ndarray) -> tuple[float, float, float]:
    diff = g_adj - g_ref
    ref_norm = float(np.linalg.norm(g_ref))
    adj_norm = float(np.linalg.norm(g_adj))
    rel = float(np.linalg.norm(diff) / max(ref_norm, 1e-12))
    max_abs = float(np.max(np.abs(diff))) if diff.size else 0.0
    denom = max(ref_norm * adj_norm, 1e-12)
    cosine = float(np.dot(g_adj, g_ref) / denom)
    return rel, max_abs, cosine


def one_step_check(eval_fn, x: np.ndarray, value0: float, g_adj: np.ndarray, g_ref: np.ndarray, step_size: float) -> tuple[float, float]:
    norm = float(np.linalg.norm(g_adj))
    if norm < 1e-12:
        return 0.0, 0.0
    direction = g_adj / norm
    predicted = float(step_size * np.dot(g_ref, direction))
    stats = eval_fn(x + step_size * direction)
    actual = float(stats.value - value0) if stats.status == "ok" else float("nan")
    return predicted, actual


def run_case(case: str, eps_values: list[float], dtype_labels: list[str], args) -> list[A4Row]:
    case_name, objective, family, graph, p = graph_case(case)
    x0 = pack_params(p)
    kmax, total = cone_stats(graph, p)
    rows: list[A4Row] = []
    for dtype_label in dtype_labels:
        complex_dtype, float_dtype = DTYPES[dtype_label]
        gammas, betas = unpack_params(x0, p)
        adj = lightcone_gradient_adjoint(
            graph,
            gammas,
            betas,
            p=p,
            prefer_gpu=True,
            max_k=args.max_k,
            max_batch_states=args.max_batch_states,
            complex_dtype=complex_dtype,
            float_dtype=float_dtype,
        )
        if adj.status != "ok" or adj.gradient is None:
            rows.append(
                A4Row(
                    case_name,
                    objective,
                    family,
                    graph.n,
                    graph.m,
                    len(graph.fields),
                    p,
                    kmax,
                    total,
                    dtype_label,
                    "lc_fd",
                    float("nan"),
                    adj.status,
                    float(adj.value),
                    float("nan"),
                    float("nan"),
                    float(adj.seconds),
                    0.0,
                    0,
                    float("nan"),
                    float("nan"),
                    float("nan"),
                    float("nan"),
                    float("nan"),
                    float("nan"),
                    float("nan"),
                    float(adj.peak_pool_bytes) / 1024**2,
                    "adjoint did not complete",
                )
            )
            continue
        eval_lc = lambda x: evaluate_lc(graph, x, p, complex_dtype, float_dtype, args.max_k, args.max_batch_states)
        references = [("lc_fd", eval_lc)]
        if graph.n <= args.full_cap:
            references.append(("full_fd", lambda x, g=graph: evaluate_full(g, x, p, complex_dtype, float_dtype, args.full_cap)))
        for reference, eval_fn in references:
            base_stats = eval_fn(x0)
            for eps in eps_values:
                fd_grad, fd_seconds, fd_peak_mb, calls, fd_status = central_fd_gradient(eval_fn, x0, eps)
                status = fd_status if base_stats.status == "ok" else f"base_failed:{base_stats.status}"
                rel, max_abs, cosine = grad_metrics(adj.gradient, fd_grad) if status == "ok" else (float("nan"), float("nan"), float("nan"))
                predicted, actual = (
                    one_step_check(eval_fn, x0, float(base_stats.value), adj.gradient, fd_grad, args.one_step)
                    if status == "ok"
                    else (float("nan"), float("nan"))
                )
                rows.append(
                    A4Row(
                        case_name,
                        objective,
                        family,
                        graph.n,
                        graph.m,
                        len(graph.fields),
                        p,
                        kmax,
                        total,
                        dtype_label,
                        reference,
                        eps,
                        status,
                        float(adj.value),
                        float(base_stats.value),
                        abs(float(adj.value) - float(base_stats.value)) if status == "ok" else float("nan"),
                        float(adj.seconds),
                        fd_seconds,
                        calls,
                        rel,
                        max_abs,
                        cosine,
                        float(np.linalg.norm(adj.gradient)),
                        float(np.linalg.norm(fd_grad)) if status == "ok" else float("nan"),
                        predicted,
                        actual,
                        max(float(adj.peak_pool_bytes) / 1024**2, fd_peak_mb),
                        "central FD uses 4p objective calls for 2p QAOA parameters",
                    )
                )
                print(f"A4 {case_name} {dtype_label} {reference} eps={eps:g} rel={rel:.3g} cos={cosine:.6g} status={status}", flush=True)
    return rows


def write_markdown(rows: list[A4Row], path: Path) -> None:
    lines = [
        "# A4 Gradient Soundness and Stability",
        "",
        "Central finite differences use 4p objective evaluations for 2p QAOA parameters. Forward differences would require 2p additional evaluations plus a base objective.",
        "",
        "| Case | dtype | reference | best eps | rel L2 | max abs | cosine | adj s | FD s | calls | one-step predicted | one-step actual |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    grouped: dict[tuple[str, str, str], list[A4Row]] = {}
    for row in rows:
        if row.status == "ok":
            grouped.setdefault((row.case, row.dtype_label, row.reference), []).append(row)
    for key, group in sorted(grouped.items()):
        best = min(group, key=lambda r: r.rel_l2_error)
        lines.append(
            f"| {best.case} | {best.dtype_label} | {best.reference} | {best.eps:g} | {best.rel_l2_error:.3g} | "
            f"{best.max_abs_error:.3g} | {best.cosine_similarity:.8g} | {best.adjoint_seconds:.4g} | "
            f"{best.fd_seconds:.4g} | {best.objective_calls_fd} | {best.one_step_predicted_delta:.4g} | {best.one_step_actual_delta:.4g} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "benchmark_suite_20260704" / "A4_gradient_soundness")
    parser.add_argument("--cases", nargs="+", default=[
        "maxcut_3regular_n24_p2",
        "maxcut_3regular_n128_p2",
        "maxcut_3regular_n512_p2",
        "qubo_er_deg2_n96_p2",
        "qubo_modular_sparse_n128_p1",
        "wdbc_top20_p1",
        "wdbc_top20_p2",
    ])
    parser.add_argument("--eps", nargs="+", type=float, default=[1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5])
    parser.add_argument("--dtype-labels", nargs="+", default=["c64_f32", "c128_f64"])
    parser.add_argument("--max-k", type=int, default=24)
    parser.add_argument("--max-batch-states", type=int, default=1 << 21)
    parser.add_argument("--full-cap", type=int, default=24)
    parser.add_argument("--one-step", type=float, default=1e-2)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[A4Row] = []
    for case in args.cases:
        rows.extend(run_case(case, args.eps, args.dtype_labels, args))
    csv_path = args.out_dir / "A4_gradient_soundness.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
    write_markdown(rows, args.out_dir / "A4_gradient_soundness.md")
    print(f"WROTE {csv_path}")
    print(f"WROTE {args.out_dir / 'A4_gradient_soundness.md'}")


if __name__ == "__main__":
    main()
