from __future__ import annotations

import argparse
import csv
import gc
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from lcqaoa.backend import get_backend
from lcqaoa.qaoa import cost_table, full_state_expectation
from run_benchmarks import params_for
from run_sota_sparse_scale import FAMILY_SEED_OFFSETS, graph_for_scale


@dataclass
class A7Row:
    n: int
    variant: str
    complex_dtype: str
    float_dtype: str
    theoretical_state_mb: float
    theoretical_cost_mb: float
    status: str
    seconds: float
    peak_gpu_pool_mb: float
    host_rss_mb: float
    value: float
    notes: str


def rss_mb() -> float:
    try:
        import psutil

        return psutil.Process(os.getpid()).memory_info().rss / 1024**2
    except Exception:
        return float("nan")


def dtype_name(dtype) -> str:
    return np.dtype(dtype).name


def theoretical_mb(n: int, dtype) -> float:
    return (np.dtype(dtype).itemsize * (1 << n)) / 1024**2


def cleanup(backend) -> None:
    try:
        backend.sync()
    except Exception:
        pass
    gc.collect()
    try:
        backend.free_memory_pool()
    except Exception:
        pass


def allocate_state(n: int, complex_dtype) -> tuple[str, float, float, float]:
    backend = get_backend(True)
    xp = backend.xp
    cleanup(backend)
    t0 = time.perf_counter()
    status = "ok"
    try:
        psi = xp.empty(1 << n, dtype=complex_dtype)
        psi.fill(1.0 / math.sqrt(1 << n))
        backend.sync()
        del psi
    except Exception as exc:
        status = f"failed:{type(exc).__name__}"
    seconds = time.perf_counter() - t0
    peak = backend.memory_pool_bytes() / 1024**2
    host = rss_mb()
    cleanup(backend)
    return status, seconds, peak, host


def allocate_state_cost(n: int, complex_dtype, float_dtype) -> tuple[str, float, float, float]:
    backend = get_backend(True)
    xp = backend.xp
    cleanup(backend)
    graph = graph_for_scale("3regular", n if n % 2 == 0 else n + 1, seed=170000 + n)
    # If n is odd, use a simple pathless empty graph for allocation-equivalent cost
    # table size; random 3-regular requires n*3 even.
    if graph.n != n:
        from lcqaoa.graphs import WeightedGraph

        graph = WeightedGraph(n=n, edges=((0, 1, 1.0),), objective="maxcut")
    t0 = time.perf_counter()
    status = "ok"
    try:
        psi = xp.empty(1 << n, dtype=complex_dtype)
        psi.fill(1.0 / math.sqrt(1 << n))
        cost = cost_table(n, graph.edges, graph.fields, graph.objective, xp, float_dtype)
        backend.sync()
        del psi, cost
    except Exception as exc:
        status = f"failed:{type(exc).__name__}"
    seconds = time.perf_counter() - t0
    peak = backend.memory_pool_bytes() / 1024**2
    host = rss_mb()
    cleanup(backend)
    return status, seconds, peak, host


def run_objective(n: int, complex_dtype, float_dtype) -> tuple[str, float, float, float, float]:
    graph_n = n if n % 2 == 0 else n + 1
    graph = graph_for_scale("3regular", graph_n, seed=110000 + graph_n * 47 + 2 * 151 + FAMILY_SEED_OFFSETS["3regular"])
    if graph.n != n:
        from lcqaoa.graphs import WeightedGraph

        graph = WeightedGraph(n=n, edges=tuple((i, i + 1, 1.0) for i in range(n - 1)), objective="maxcut")
    gammas, betas = params_for(2)
    t0 = time.perf_counter()
    try:
        stats = full_state_expectation(
            graph,
            gammas,
            betas,
            method="precompute",
            prefer_gpu=True,
            complex_dtype=complex_dtype,
            float_dtype=float_dtype,
            max_qubits=None,
        )
        status = stats.status
        seconds = stats.seconds
        peak = stats.peak_pool_bytes / 1024**2
        value = stats.value
    except Exception as exc:
        status = f"failed:{type(exc).__name__}"
        seconds = time.perf_counter() - t0
        peak = 0.0
        value = float("nan")
    host = rss_mb()
    try:
        get_backend(True).free_memory_pool()
    except Exception:
        pass
    return status, seconds, peak, host, value


def row(n: int, variant: str, complex_dtype, float_dtype, status: str, seconds: float, peak: float, host: float, value: float, notes: str) -> A7Row:
    cost_mb = 0.0 if float_dtype is None else theoretical_mb(n, float_dtype)
    return A7Row(
        n=n,
        variant=variant,
        complex_dtype=dtype_name(complex_dtype),
        float_dtype="" if float_dtype is None else dtype_name(float_dtype),
        theoretical_state_mb=theoretical_mb(n, complex_dtype),
        theoretical_cost_mb=cost_mb,
        status=status,
        seconds=seconds,
        peak_gpu_pool_mb=peak,
        host_rss_mb=host,
        value=value,
        notes=notes,
    )


def write_markdown(rows: list[A7Row], path: Path) -> None:
    lines = [
        "# A7 Full-state Memory Wall",
        "",
        "Measured on the CUDA-visible GPU. Rows separate allocation-only probes from a one-evaluation full-state precompute route.",
        "",
        "| n | variant | dtypes | theory state MB | theory cost MB | status | seconds | peak GPU MB | host RSS MB |",
        "|---:|---|---|---:|---:|---|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r.n} | {r.variant} | {r.complex_dtype}/{r.float_dtype} | {r.theoretical_state_mb:.4g} | "
            f"{r.theoretical_cost_mb:.4g} | {r.status} | {r.seconds:.4g} | {r.peak_gpu_pool_mb:.4g} | {r.host_rss_mb:.4g} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "benchmark_suite_20260704" / "A7_memory_wall")
    parser.add_argument("--n-min", type=int, default=20)
    parser.add_argument("--n-max", type=int, default=30)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[A7Row] = []
    for n in range(args.n_min, args.n_max + 1):
        for complex_dtype in [np.complex64, np.complex128]:
            status, seconds, peak, host = allocate_state(n, complex_dtype)
            rows.append(row(n, "state_only", complex_dtype, None, status, seconds, peak, host, float("nan"), "allocate and fill statevector"))
        for float_dtype in [np.float32, np.float64]:
            status, seconds, peak, host = allocate_state_cost(n, np.complex64, float_dtype)
            rows.append(row(n, "state_plus_cost", np.complex64, float_dtype, status, seconds, peak, host, float("nan"), "allocate statevector plus diagonal cost table"))
        for float_dtype in [np.float32, np.float64]:
            status, seconds, peak, host, value = run_objective(n, np.complex64, float_dtype)
            rows.append(row(n, "one_qaoa_eval_precompute", np.complex64, float_dtype, status, seconds, peak, host, value, "full-state precompute objective evaluation p=2"))
        print(f"A7 n={n} done", flush=True)
    csv_path = args.out_dir / "A7_memory_wall.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow(asdict(r))
    write_markdown(rows, args.out_dir / "A7_memory_wall.md")
    print(f"WROTE {csv_path}")


if __name__ == "__main__":
    main()
