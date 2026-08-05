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
from run_sota_sparse_scale import FAMILY_SEED_OFFSETS, degree_stats, graph_for_scale


@dataclass
class StepRow:
    case: str
    objective: str
    family: str
    n: int
    m: int
    fields: int
    p: int
    kmax: int
    total_cone_states: int
    optimizer: str
    step: int
    status: str
    value: float
    best_value: float
    seconds_step: float
    seconds_elapsed: float
    objective_calls_step: int
    objective_calls_total: int
    gradient_norm: float
    peak_pool_mb: float
    full_reference_value: float
    abs_error_vs_full_reference: float
    notes: str


@dataclass
class SummaryRow:
    case: str
    objective: str
    family: str
    n: int
    m: int
    fields: int
    p: int
    kmax: int
    total_cone_states: int
    optimizer: str
    status: str
    steps: int
    objective_calls: int
    seconds_total: float
    seconds_per_step: float
    initial_value: float
    final_value: float
    best_value: float
    improvement_final: float
    improvement_best: float
    max_abs_error_vs_full_reference: float
    peak_pool_mb: float
    notes: str


class ObjectiveCounter:
    def __init__(
        self,
        graph: WeightedGraph,
        p: int,
        *,
        max_k: int,
        max_batch_states: int,
        prefer_gpu: bool,
        full_cap: int,
        mode: str,
    ) -> None:
        self.graph = graph
        self.p = p
        self.max_k = max_k
        self.max_batch_states = max_batch_states
        self.prefer_gpu = prefer_gpu
        self.full_cap = full_cap
        self.mode = mode
        self.calls = 0
        self.peak_pool_mb = 0.0

    def eval(self, x: np.ndarray) -> tuple[float, float, str]:
        gammas, betas = unpack_params(x, self.p)
        if self.mode == "lc":
            stats = lightcone_expectation(
                self.graph,
                gammas,
                betas,
                p=self.p,
                prefer_gpu=self.prefer_gpu,
                max_k=self.max_k,
                max_batch_states=self.max_batch_states,
            )
        elif self.mode == "full":
            stats = full_state_expectation(
                self.graph,
                gammas,
                betas,
                method="precompute",
                prefer_gpu=self.prefer_gpu,
                max_qubits=self.full_cap,
            )
        else:
            raise ValueError(self.mode)
        self.calls += 1
        self.peak_pool_mb = max(self.peak_pool_mb, stats.peak_pool_bytes / 1024**2)
        if stats.status != "ok":
            raise RuntimeError(stats.status)
        return float(stats.value), float(stats.seconds), stats.status


def unpack_params(x: np.ndarray, p: int) -> tuple[list[float], list[float]]:
    return x[:p].astype(float).tolist(), x[p:].astype(float).tolist()


def pack_initial(p: int) -> np.ndarray:
    gammas, betas = params_for(p)
    return np.asarray(gammas + betas, dtype=np.float64)


def wrap_angles(x: np.ndarray) -> np.ndarray:
    return ((x + math.pi) % (2.0 * math.pi)) - math.pi


def graph_for_case(objective: str, family: str, n: int, p: int) -> WeightedGraph:
    if objective == "qubo":
        return graph_for_qubo(family, n, seed=110000 + n * 47 + p * 151)
    return graph_for_scale(family, n, seed=110000 + n * 47 + p * 151 + FAMILY_SEED_OFFSETS.get(family, 0))


def case_specs() -> list[tuple[str, str, str, int, int]]:
    return [
        ("maxcut_3regular_n512_p2", "maxcut", "3regular", 512, 2),
        ("qubo_er_deg2_n96_p2", "qubo", "qubo_er_deg2", 96, 2),
        ("maxcut_3regular_n24_p2", "maxcut", "3regular", 24, 2),
    ]


def cone_stats(graph: WeightedGraph, p: int) -> tuple[int, int]:
    cones = extract_lightcones(graph, p)
    if not cones:
        return 0, 0
    return max(c.k for c in cones), sum(1 << c.k for c in cones)


def full_reference(graph: WeightedGraph, x: np.ndarray, p: int, full_cap: int, prefer_gpu: bool) -> float:
    if graph.n > full_cap:
        return float("nan")
    gammas, betas = unpack_params(x, p)
    stats = full_state_expectation(graph, gammas, betas, method="precompute", prefer_gpu=prefer_gpu, max_qubits=full_cap)
    if stats.status != "ok":
        return float("nan")
    return float(stats.value)


def finite_difference(counter: ObjectiveCounter, x: np.ndarray, eps: float) -> tuple[np.ndarray, float, int, float]:
    grad = np.zeros_like(x, dtype=np.float64)
    calls_before = counter.calls
    peak_before = counter.peak_pool_mb
    t0 = time.perf_counter()
    for i in range(x.size):
        plus = x.copy()
        minus = x.copy()
        plus[i] += eps
        minus[i] -= eps
        vp, _, _ = counter.eval(plus)
        vm, _, _ = counter.eval(minus)
        grad[i] = (vp - vm) / (2.0 * eps)
    return grad, time.perf_counter() - t0, counter.calls - calls_before, max(0.0, counter.peak_pool_mb - peak_before)


def adam_update(x: np.ndarray, grad: np.ndarray, step: int, state: dict, lr: float) -> np.ndarray:
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    if "m" not in state:
        state["m"] = np.zeros_like(x)
        state["v"] = np.zeros_like(x)
    state["m"] = beta1 * state["m"] + (1.0 - beta1) * grad
    state["v"] = beta2 * state["v"] + (1.0 - beta2) * (grad * grad)
    m_hat = state["m"] / (1.0 - beta1 ** step)
    v_hat = state["v"] / (1.0 - beta2 ** step)
    return wrap_angles(x + lr * m_hat / (np.sqrt(v_hat) + eps))


def row_common(case: str, objective: str, family: str, graph: WeightedGraph, p: int, kmax: int, total: int) -> dict:
    return {
        "case": case,
        "objective": objective,
        "family": family,
        "n": graph.n,
        "m": graph.m,
        "fields": len(graph.fields),
        "p": p,
        "kmax": kmax,
        "total_cone_states": total,
    }


def run_adjoint_adam(
    case: str,
    objective: str,
    family: str,
    graph: WeightedGraph,
    p: int,
    *,
    steps: int,
    lr: float,
    max_k: int,
    max_batch_states: int,
    full_cap: int,
    prefer_gpu: bool,
) -> tuple[list[StepRow], SummaryRow]:
    kmax, total = cone_stats(graph, p)
    common = row_common(case, objective, family, graph, p, kmax, total)
    x = pack_initial(p)
    state: dict = {}
    rows: list[StepRow] = []
    best = float("-inf")
    first = float("nan")
    peak = 0.0
    max_full_err = 0.0
    status = "ok"
    t_start = time.perf_counter()
    for step in range(steps + 1):
        t0 = time.perf_counter()
        gammas, betas = unpack_params(x, p)
        adj = lightcone_gradient_adjoint(
            graph,
            gammas,
            betas,
            p=p,
            prefer_gpu=prefer_gpu,
            max_k=max_k,
            max_batch_states=max_batch_states,
        )
        seconds_step = time.perf_counter() - t0
        if adj.status != "ok" or adj.gradient is None:
            status = adj.status
            value = float("nan")
            grad = np.zeros_like(x)
        else:
            value = float(adj.value)
            grad = np.asarray(adj.gradient, dtype=np.float64)
        if step == 0:
            first = value
        best = max(best, value) if math.isfinite(value) else best
        peak = max(peak, adj.peak_pool_bytes / 1024**2)
        ref = full_reference(graph, x, p, full_cap, prefer_gpu)
        err = abs(value - ref) if math.isfinite(value) and math.isfinite(ref) else float("nan")
        if math.isfinite(err):
            max_full_err = max(max_full_err, err)
        rows.append(
            StepRow(
                **common,
                optimizer="lc_adjoint_adam",
                step=step,
                status=adj.status,
                value=value,
                best_value=best,
                seconds_step=seconds_step,
                seconds_elapsed=time.perf_counter() - t_start,
                objective_calls_step=1,
                objective_calls_total=step + 1,
                gradient_norm=float(np.linalg.norm(grad)) if adj.gradient is not None else float("nan"),
                peak_pool_mb=peak,
                full_reference_value=ref,
                abs_error_vs_full_reference=err,
                notes="Adam ascent using LC adjoint value and gradient",
            )
        )
        if step < steps and adj.status == "ok" and adj.gradient is not None:
            x = adam_update(x, grad, step + 1, state, lr)
    final = rows[-1].value
    return rows, SummaryRow(
        **common,
        optimizer="lc_adjoint_adam",
        status=status,
        steps=len(rows),
        objective_calls=len(rows),
        seconds_total=time.perf_counter() - t_start,
        seconds_per_step=(time.perf_counter() - t_start) / max(1, len(rows)),
        initial_value=first,
        final_value=final,
        best_value=best,
        improvement_final=final - first if math.isfinite(final) and math.isfinite(first) else float("nan"),
        improvement_best=best - first if math.isfinite(best) and math.isfinite(first) else float("nan"),
        max_abs_error_vs_full_reference=max_full_err if max_full_err > 0 else float("nan"),
        peak_pool_mb=peak,
        notes=f"Adam lr={lr}; full reference evaluated only when n<=full_cap",
    )


def run_fd_adam(
    case: str,
    objective: str,
    family: str,
    graph: WeightedGraph,
    p: int,
    *,
    optimizer_name: str,
    counter_mode: str,
    steps: int,
    lr: float,
    eps: float,
    max_k: int,
    max_batch_states: int,
    full_cap: int,
    prefer_gpu: bool,
) -> tuple[list[StepRow], SummaryRow]:
    kmax, total = cone_stats(graph, p)
    common = row_common(case, objective, family, graph, p, kmax, total)
    counter = ObjectiveCounter(graph, p, max_k=max_k, max_batch_states=max_batch_states, prefer_gpu=prefer_gpu, full_cap=full_cap, mode=counter_mode)
    x = pack_initial(p)
    state: dict = {}
    rows: list[StepRow] = []
    best = float("-inf")
    first = float("nan")
    peak = 0.0
    max_full_err = 0.0
    status = "ok"
    t_start = time.perf_counter()
    for step in range(steps + 1):
        calls_before = counter.calls
        t0 = time.perf_counter()
        try:
            value, _, _ = counter.eval(x)
            grad, _, _, _ = finite_difference(counter, x, eps)
        except Exception as exc:
            status = f"failed:{type(exc).__name__}"
            value = float("nan")
            grad = np.zeros_like(x)
        seconds_step = time.perf_counter() - t0
        if step == 0:
            first = value
        best = max(best, value) if math.isfinite(value) else best
        peak = max(peak, counter.peak_pool_mb)
        ref = full_reference(graph, x, p, full_cap, prefer_gpu)
        err = abs(value - ref) if math.isfinite(value) and math.isfinite(ref) else float("nan")
        if math.isfinite(err):
            max_full_err = max(max_full_err, err)
        rows.append(
            StepRow(
                **common,
                optimizer=optimizer_name,
                step=step,
                status=status,
                value=value,
                best_value=best,
                seconds_step=seconds_step,
                seconds_elapsed=time.perf_counter() - t_start,
                objective_calls_step=counter.calls - calls_before,
                objective_calls_total=counter.calls,
                gradient_norm=float(np.linalg.norm(grad)) if math.isfinite(value) else float("nan"),
                peak_pool_mb=peak,
                full_reference_value=ref,
                abs_error_vs_full_reference=err,
                notes=f"Adam ascent using central finite-difference gradient eps={eps} over {counter_mode} objective",
            )
        )
        if step < steps and status == "ok":
            x = adam_update(x, grad, step + 1, state, lr)
    final = rows[-1].value
    return rows, SummaryRow(
        **common,
        optimizer=optimizer_name,
        status=status,
        steps=len(rows),
        objective_calls=counter.calls,
        seconds_total=time.perf_counter() - t_start,
        seconds_per_step=(time.perf_counter() - t_start) / max(1, len(rows)),
        initial_value=first,
        final_value=final,
        best_value=best,
        improvement_final=final - first if math.isfinite(final) and math.isfinite(first) else float("nan"),
        improvement_best=best - first if math.isfinite(best) and math.isfinite(first) else float("nan"),
        max_abs_error_vs_full_reference=max_full_err if max_full_err > 0 else float("nan"),
        peak_pool_mb=peak,
        notes=f"Adam lr={lr}; finite-difference eps={eps}; mode={counter_mode}",
    )


def run_nelder_mead(
    case: str,
    objective: str,
    family: str,
    graph: WeightedGraph,
    p: int,
    *,
    maxfev: int,
    max_k: int,
    max_batch_states: int,
    full_cap: int,
    prefer_gpu: bool,
) -> SummaryRow:
    from scipy.optimize import minimize

    kmax, total = cone_stats(graph, p)
    common = row_common(case, objective, family, graph, p, kmax, total)
    counter = ObjectiveCounter(graph, p, max_k=max_k, max_batch_states=max_batch_states, prefer_gpu=prefer_gpu, full_cap=full_cap, mode="lc")
    x0 = pack_initial(p)
    initial, _, _ = counter.eval(x0)
    best = initial
    t0 = time.perf_counter()

    def obj(x: np.ndarray) -> float:
        nonlocal best
        value, _, _ = counter.eval(wrap_angles(np.asarray(x, dtype=np.float64)))
        best = max(best, value)
        return -value

    status = "ok"
    notes = f"Nelder-Mead on LC objective with maxfev={maxfev}"
    try:
        res = minimize(obj, x0, method="Nelder-Mead", options={"maxfev": maxfev, "xatol": 1e-3, "fatol": 1e-3, "disp": False})
        if not res.success:
            status = "optimizer_" + str(res.message).replace(" ", "_")[:80]
    except Exception as exc:
        status = f"failed:{type(exc).__name__}"
        notes = str(exc)[:180]
    seconds = time.perf_counter() - t0
    final = -float(res.fun) if "res" in locals() and hasattr(res, "fun") else float("nan")
    return SummaryRow(
        **common,
        optimizer="lc_nelder_mead",
        status=status,
        steps=0,
        objective_calls=counter.calls,
        seconds_total=seconds,
        seconds_per_step=seconds / max(1, counter.calls),
        initial_value=initial,
        final_value=final,
        best_value=best,
        improvement_final=final - initial if math.isfinite(final) else float("nan"),
        improvement_best=best - initial,
        max_abs_error_vs_full_reference=float("nan"),
        peak_pool_mb=counter.peak_pool_mb,
        notes=notes,
    )


def write_markdown(summary_rows: list[SummaryRow], path: Path) -> None:
    lines = [
        "# P0 Adjoint Optimizer Benchmark",
        "",
        "This benchmark connects the LC adjoint gradient to an actual Adam optimizer.",
        "All rows use the same initial QAOA angles per case. The full-state reference is evaluated only for the n=24 case where the global route fits the configured cap.",
        "",
        "| Case | Optimizer | Status | Steps | Obj calls | Seconds | Initial | Final | Best | Best improvement | Peak MB | Full ref max err | Notes |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in summary_rows:
        lines.append(
            f"| {r.case} | {r.optimizer} | {r.status} | {r.steps} | {r.objective_calls} | {r.seconds_total:.4g} | "
            f"{r.initial_value:.7g} | {r.final_value:.7g} | {r.best_value:.7g} | {r.improvement_best:.4g} | "
            f"{r.peak_pool_mb:.4g} | {r.max_abs_error_vs_full_reference:.3g} | {r.notes} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "p0_adjoint_optimizer")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--fd-steps", type=int, default=100)
    parser.add_argument("--nelder-maxfev", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--fd-lr", type=float, default=0.02)
    parser.add_argument("--eps", type=float, default=1e-3)
    parser.add_argument("--max-k", type=int, default=24)
    parser.add_argument("--max-batch-states", type=int, default=1 << 21)
    parser.add_argument("--full-cap", type=int, default=24)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    prefer_gpu = not args.cpu
    args.out_dir.mkdir(parents=True, exist_ok=True)
    step_rows: list[StepRow] = []
    summary_rows: list[SummaryRow] = []

    for case, objective, family, n, p in case_specs():
        graph = graph_for_case(objective, family, n, p)
        kmax, total = cone_stats(graph, p)
        print(f"P0 case={case} objective={objective} n={n} p={p} m={graph.m} fields={len(graph.fields)} kmax={kmax} total={total}", flush=True)
        rows, summary = run_adjoint_adam(
            case,
            objective,
            family,
            graph,
            p,
            steps=args.steps,
            lr=args.lr,
            max_k=args.max_k,
            max_batch_states=args.max_batch_states,
            full_cap=args.full_cap,
            prefer_gpu=prefer_gpu,
        )
        step_rows.extend(rows)
        summary_rows.append(summary)

        rows, summary = run_fd_adam(
            case,
            objective,
            family,
            graph,
            p,
            optimizer_name="lc_fd_adam",
            counter_mode="lc",
            steps=args.fd_steps,
            lr=args.fd_lr,
            eps=args.eps,
            max_k=args.max_k,
            max_batch_states=args.max_batch_states,
            full_cap=args.full_cap,
            prefer_gpu=prefer_gpu,
        )
        step_rows.extend(rows)
        summary_rows.append(summary)

        if graph.n <= args.full_cap:
            rows, summary = run_fd_adam(
                case,
                objective,
                family,
                graph,
                p,
                optimizer_name="full_fd_adam",
                counter_mode="full",
                steps=args.fd_steps,
                lr=args.fd_lr,
                eps=args.eps,
                max_k=args.max_k,
                max_batch_states=args.max_batch_states,
                full_cap=args.full_cap,
                prefer_gpu=prefer_gpu,
            )
            step_rows.extend(rows)
            summary_rows.append(summary)

        summary_rows.append(
            run_nelder_mead(
                case,
                objective,
                family,
                graph,
                p,
                maxfev=args.nelder_maxfev,
                max_k=args.max_k,
                max_batch_states=args.max_batch_states,
                full_cap=args.full_cap,
                prefer_gpu=prefer_gpu,
            )
        )

    step_csv = args.out_dir / "p0_adjoint_optimizer_steps.csv"
    summary_csv = args.out_dir / "p0_adjoint_optimizer_summary.csv"
    summary_md = args.out_dir / "p0_adjoint_optimizer_summary.md"
    with step_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(step_rows[0]).keys()))
        writer.writeheader()
        for row in step_rows:
            writer.writerow(asdict(row))
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(summary_rows[0]).keys()))
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(asdict(row))
    write_markdown(summary_rows, summary_md)
    print(f"WROTE {step_csv}")
    print(f"WROTE {summary_csv}")
    print(f"WROTE {summary_md}")


if __name__ == "__main__":
    main()
