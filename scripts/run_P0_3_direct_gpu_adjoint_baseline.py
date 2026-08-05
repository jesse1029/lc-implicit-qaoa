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

from lcqaoa.backend import get_backend
from lcqaoa.graphs import erdos_renyi_graph, random_regular_graph, weighted_modular_qubo_graph, weighted_qubo_graph
from lcqaoa.lightcone import lightcone_gradient_adjoint
from lcqaoa.qaoa import apply_mixer_batched_inplace, apply_mixer_inplace, cost_table, expectation_from_state, full_state_expectation
from benchmark_common import cone_metrics, params_for_depth


@dataclass
class P03Row:
    family: str
    n: int
    p: int
    seed: int
    precision: str
    method: str
    stage: str
    status: str
    cold_seconds: float
    warm_seconds: float
    steady_median_seconds: float
    peak_reserved_mb: float
    peak_allocated_mb: float
    value: float
    grad_norm: float
    agreement_abs_error: float
    kmax: int
    total_cone_states: int
    notes: str


def make_graph(family: str, n: int, seed: int):
    if family == "3regular":
        return random_regular_graph(n, 3, seed=seed)
    if family == "er2":
        return erdos_renyi_graph(n, min(0.45, 2.0 / max(2, n)), seed=seed)
    if family == "weighted_qubo_er2":
        return weighted_qubo_graph(n, min(0.45, 2.0 / max(2, n)), seed=seed, field_scale=0.7)
    if family == "modular_sparse":
        return weighted_modular_qubo_graph(n, modules=max(4, n // 12), p_in=0.14, p_out=0.0015, seed=seed)
    raise ValueError(family)


def x_sum_state(psi, k: int, xp):
    out = xp.zeros_like(psi)
    for q in range(k):
        step = 1 << q
        block = step << 1
        src = psi.reshape(-1, block)
        dst = out.reshape(-1, block)
        dst[:, :step] += src[:, step:block]
        dst[:, step:block] += src[:, :step]
    return out


def pool_stats(xp, gpu: bool) -> tuple[int, int]:
    if not gpu:
        return 0, 0
    pool = xp.get_default_memory_pool()
    return int(pool.used_bytes()), int(pool.total_bytes())


def global_gpu_adjoint(graph, gammas, betas, *, prefer_gpu: bool, complex_dtype, float_dtype) -> tuple[str, float, float, float, float, np.ndarray | None, str]:
    backend = get_backend(prefer_gpu=prefer_gpu)
    xp = backend.xp
    gpu = backend.gpu
    backend.free_memory_pool()
    t0 = time.perf_counter()
    peak_used, peak_total = 0, 0
    try:
        p = len(gammas)
        nstates = 1 << graph.n
        psi = xp.empty(nstates, dtype=complex_dtype)
        psi.fill(1.0 / math.sqrt(nstates))
        cost = cost_table(graph.n, graph.edges, graph.fields, graph.objective, xp, float_dtype)
        used, total = pool_stats(xp, gpu)
        peak_used, peak_total = max(peak_used, used), max(peak_total, total)
        after_costs = []
        after_layers = []
        for gamma, beta in zip(gammas, betas):
            psi *= xp.exp((-1j * float(gamma)) * cost)
            after_costs.append(psi.copy())
            apply_mixer_inplace(psi, graph.n, beta, xp)
            after_layers.append(psi.copy())
            used, total = pool_stats(xp, gpu)
            peak_used, peak_total = max(peak_used, used), max(peak_total, total)
        value = expectation_from_state(psi, cost, xp)
        grad_gamma = xp.zeros(p, dtype=xp.float64)
        grad_beta = xp.zeros(p, dtype=xp.float64)
        adjoint = cost * psi
        used, total = pool_stats(xp, gpu)
        peak_used, peak_total = max(peak_used, used), max(peak_total, total)
        for layer in range(p - 1, -1, -1):
            xsum = x_sum_state(after_layers[layer], graph.n, xp)
            d_beta = -1j * xsum
            grad_beta[layer] = 2.0 * xp.real(xp.sum(xp.conj(adjoint) * d_beta))
            apply_mixer_inplace(adjoint, graph.n, -float(betas[layer]), xp)
            d_gamma = -1j * cost * after_costs[layer]
            grad_gamma[layer] = 2.0 * xp.real(xp.sum(xp.conj(adjoint) * d_gamma))
            adjoint *= xp.exp((1j * float(gammas[layer])) * cost)
            used, total = pool_stats(xp, gpu)
            peak_used, peak_total = max(peak_used, used), max(peak_total, total)
        grad = xp.concatenate([grad_gamma, grad_beta])
        if gpu:
            xp.cuda.Stream.null.synchronize()
        value_f = float(value)
        grad_np = grad.get() if hasattr(grad, "get") else np.asarray(grad)
        return "ok", time.perf_counter() - t0, peak_total / 1024**2, peak_used / 1024**2, value_f, grad_np, backend.name
    except Exception as exc:
        if gpu:
            try:
                xp.cuda.Stream.null.synchronize()
            except Exception:
                pass
        used, total = pool_stats(xp, gpu)
        return f"failed:{type(exc).__name__}", time.perf_counter() - t0, total / 1024**2, used / 1024**2, float("nan"), None, str(exc)[:240]


def time_call(fn, repeats: int) -> tuple[str, float, float, float, float, float, np.ndarray | None, str]:
    status, cold_s, cold_res, cold_alloc, value, grad, notes = fn()
    if status != "ok":
        return status, cold_s, float("nan"), float("nan"), cold_res, cold_alloc, value, grad, notes
    status2, warm_s, warm_res, warm_alloc, value2, grad2, notes2 = fn()
    steady = []
    peak_res = max(cold_res, warm_res)
    peak_alloc = max(cold_alloc, warm_alloc)
    last_value, last_grad = value2, grad2
    for _ in range(repeats):
        s, sec, res, alloc, val, g, n = fn()
        if s != "ok":
            return s, cold_s, warm_s, float("nan"), max(peak_res, res), max(peak_alloc, alloc), val, g, n
        steady.append(sec)
        peak_res = max(peak_res, res)
        peak_alloc = max(peak_alloc, alloc)
        last_value, last_grad = val, g
    return "ok", cold_s, warm_s, float(np.median(steady)) if steady else warm_s, peak_res, peak_alloc, last_value, last_grad, notes2


def run_case(family: str, n: int, p: int, seed_id: int, args) -> list[P03Row]:
    seed = 330000 + seed_id * 997 + n * 37 + p * 131
    graph = make_graph(family, n, seed)
    gammas, betas = params_for_depth(p, seed=seed_id)
    cmet = cone_metrics(graph, p)
    precision = args.precision
    complex_dtype = np.complex64 if precision == "c64_f32" else np.complex128
    float_dtype = np.float32 if precision == "c64_f32" else np.float64
    rows: list[P03Row] = []

    lc_grad_ref = None
    lc_value_ref = float("nan")

    def lc_fn():
        t0 = time.perf_counter()
        stats = lightcone_gradient_adjoint(
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
        sec = time.perf_counter() - t0
        return (
            stats.status,
            sec,
            stats.peak_pool_bytes / 1024**2,
            stats.peak_pool_bytes / 1024**2,
            float(stats.value),
            np.asarray(stats.gradient) if stats.gradient is not None else None,
            stats.backend,
        )

    s, cold, warm, steady, peak_res, peak_alloc, value, grad, notes = time_call(lc_fn, args.repeats)
    if grad is not None:
        lc_grad_ref = grad
        lc_value_ref = value
    rows.append(
        P03Row(family, n, p, seed_id, precision, "LC local adjoint", "value+gradient", s, cold, warm, steady, peak_res, peak_alloc, value, float(np.linalg.norm(grad)) if grad is not None else float("nan"), 0.0, int(cmet["kmax"]), int(cmet["total_cone_states"]), notes)
    )

    if graph.n <= args.global_max_n:
        def global_fn():
            return global_gpu_adjoint(graph, gammas, betas, prefer_gpu=True, complex_dtype=complex_dtype, float_dtype=float_dtype)

        s, cold, warm, steady, peak_res, peak_alloc, value, grad, notes = time_call(global_fn, args.repeats)
        err = abs(value - lc_value_ref) if math.isfinite(value) and math.isfinite(lc_value_ref) else float("nan")
        rows.append(
            P03Row(family, n, p, seed_id, precision, "global-state GPU adjoint", "value+gradient", s, cold, warm, steady, peak_res, peak_alloc, value, float(np.linalg.norm(grad)) if grad is not None else float("nan"), err, int(cmet["kmax"]), int(cmet["total_cone_states"]), notes)
        )
    else:
        rows.append(
            P03Row(family, n, p, seed_id, precision, "global-state GPU adjoint", "value+gradient", "NOT_RUN_EXPLAINED", 0.0, 0.0, 0.0, 0.0, 0.0, float("nan"), float("nan"), float("nan"), int(cmet["kmax"]), int(cmet["total_cone_states"]), f"guarded above n={args.global_max_n}")
        )

    if graph.n <= args.global_max_n:
        def value_fn():
            t0 = time.perf_counter()
            try:
                stats = full_state_expectation(graph, gammas, betas, method="precompute", prefer_gpu=True, complex_dtype=complex_dtype, float_dtype=float_dtype)
                return stats.status, time.perf_counter() - t0, stats.peak_pool_bytes / 1024**2, stats.peak_pool_bytes / 1024**2, float(stats.value), None, stats.backend
            except Exception as exc:
                return f"failed:{type(exc).__name__}", time.perf_counter() - t0, 0.0, 0.0, float("nan"), None, str(exc)[:240]

        s, cold, warm, steady, peak_res, peak_alloc, value, grad, notes = time_call(value_fn, args.repeats)
        err = abs(value - lc_value_ref) if math.isfinite(value) and math.isfinite(lc_value_ref) else float("nan")
        rows.append(
            P03Row(family, n, p, seed_id, precision, "global-state GPU value", "value-only", s, cold, warm, steady, peak_res, peak_alloc, value, float("nan"), err, int(cmet["kmax"]), int(cmet["total_cone_states"]), notes)
        )
    return rows


def write_md(rows: list[P03Row], path: Path) -> None:
    lines = [
        "# P0-3 Direct GPU Adjoint Baseline",
        "",
        "The stable global baseline here is an independent dense global-state GPU adjoint. Official CUAOA/CUDA-Q value-only status is kept in the A1/P0 failure-diagnostic tables and is not interpreted as an official gradient comparison unless a public gradient API is detected.",
        "",
        "| family | n | p | method | rows | success | median steady s | median peak MB | max agreement err |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|",
    ]
    keys = sorted({(r.family, r.n, r.p, r.method) for r in rows})
    for key in keys:
        sub = [r for r in rows if (r.family, r.n, r.p, r.method) == key]
        ok = [r for r in sub if r.status == "ok"]
        med = lambda vals: float(np.nanmedian(vals)) if vals else float("nan")
        lines.append(
            f"| {key[0]} | {key[1]} | {key[2]} | {key[3]} | {len(sub)} | {len(ok)} | "
            f"{med([r.steady_median_seconds for r in ok]):.4g} | {med([r.peak_reserved_mb for r in ok]):.4g} | "
            f"{np.nanmax([r.agreement_abs_error for r in ok]) if ok else float('nan'):.3e} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "aaai27_required_experiments_20260710" / "P0_3_direct_gpu_adjoint_baseline")
    parser.add_argument("--families", nargs="*", default=["3regular", "er2", "weighted_qubo_er2", "modular_sparse"])
    parser.add_argument("--ns", nargs="*", type=int, default=[18, 20, 22, 24, 26, 28])
    parser.add_argument("--ps", nargs="*", type=int, default=[1, 2, 3])
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--precision", choices=["c64_f32", "c128_f64"], default="c64_f32")
    parser.add_argument("--max-k", type=int, default=28)
    parser.add_argument("--max-batch-states", type=int, default=1 << 21)
    parser.add_argument("--global-max-n", type=int, default=28)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if args.quick:
        args.families = ["3regular"]
        args.ns = [18, 20]
        args.ps = [1, 2]
        args.seeds = 1
        args.repeats = 1
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[P03Row] = []
    csv_path = args.out_dir / "P0_3_direct_gpu_adjoint_baseline.csv"
    for family in args.families:
        for n in args.ns:
            if family == "3regular" and (n * 3) % 2 != 0:
                continue
            for p in args.ps:
                for seed in range(args.seeds):
                    print(f"P0-3 family={family} n={n} p={p} seed={seed}", flush=True)
                    rows.extend(run_case(family, n, p, seed, args))
                    with csv_path.open("w", newline="", encoding="utf-8") as f:
                        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
                        writer.writeheader()
                        for row in rows:
                            writer.writerow(asdict(row))
                    write_md(rows, args.out_dir / "P0_3_direct_gpu_adjoint_baseline.md")
    print(f"WROTE {csv_path}")


if __name__ == "__main__":
    main()
