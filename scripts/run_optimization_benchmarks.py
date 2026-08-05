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
from lcqaoa.lightcone import extract_lightcones, lightcone_expectation
from lcqaoa.qaoa import full_state_expectation
from run_benchmarks import params_for
from run_qubo_benchmarks import graph_for_qubo
from run_sota_sparse_scale import FAMILY_SEED_OFFSETS, cone_stats, degree_stats, graph_for_scale


@dataclass
class OptRow:
    case: str
    objective: str
    family: str
    n: int
    m: int
    fields: int
    p: int
    kmax: int
    total_cone_states: int
    method: str
    mode: str
    status: str
    evals: int
    seconds: float
    seconds_per_eval: float
    initial_value: float
    best_value: float
    improvement: float
    backend: str
    peak_pool_mb: float
    notes: str


def graph_for_case(family: str, n: int, p: int, objective: str) -> WeightedGraph:
    if objective == "qubo":
        return graph_for_qubo(family, n, seed=110000 + n * 47 + p * 151)
    return graph_for_scale(family, n, seed=110000 + n * 47 + p * 151 + FAMILY_SEED_OFFSETS.get(family, 0))


def cases() -> list[tuple[str, str, str, int, int]]:
    return [
        ("maxcut_3regular_n24_p2", "maxcut", "3regular", 24, 2),
        ("maxcut_3regular_n128_p2", "maxcut", "3regular", 128, 2),
        ("maxcut_3regular_n512_p2", "maxcut", "3regular", 512, 2),
        ("maxcut_modular_sparse_n128_p1", "maxcut", "modular_sparse", 128, 1),
        ("qubo_er_deg2_n24_p2", "qubo", "qubo_er_deg2", 24, 2),
        ("qubo_modular_sparse_n128_p1", "qubo", "qubo_modular_sparse", 128, 1),
    ]


def unpack_params(x: np.ndarray, p: int) -> tuple[list[float], list[float]]:
    return x[:p].astype(float).tolist(), x[p:].astype(float).tolist()


class Evaluator:
    def __init__(self, graph: WeightedGraph, p: int, method: str, max_k: int, max_batch_states: int, full_cap: int):
        self.graph = graph
        self.p = p
        self.method = method
        self.max_k = max_k
        self.max_batch_states = max_batch_states
        self.full_cap = full_cap
        self.evals = 0
        self.peak_pool_mb = 0.0
        self.backend = ""

    def eval(self, x: np.ndarray) -> float:
        gammas, betas = unpack_params(x, self.p)
        if self.method == "full_precompute_gpu":
            stats = full_state_expectation(
                self.graph,
                gammas,
                betas,
                method="precompute",
                prefer_gpu=True,
                max_qubits=self.full_cap,
            )
        elif self.method == "lc_batched_gpu":
            stats = lightcone_expectation(
                self.graph,
                gammas,
                betas,
                p=self.p,
                prefer_gpu=True,
                max_k=self.max_k,
                max_batch_states=self.max_batch_states,
            )
        else:
            raise ValueError(self.method)
        if stats.status != "ok":
            raise RuntimeError(stats.status)
        self.evals += 1
        self.peak_pool_mb = max(self.peak_pool_mb, stats.peak_pool_bytes / 1024**2)
        self.backend = stats.backend
        return float(stats.value)


def trajectory_params(p: int, count: int, seed: int) -> list[np.ndarray]:
    gammas, betas = params_for(p)
    base = np.asarray(gammas + betas, dtype=np.float64)
    rng = np.random.default_rng(seed)
    out = [base]
    for _ in range(count - 1):
        noise = rng.normal(0.0, 0.08, size=base.shape)
        out.append(np.clip(base + noise, -1.0, 1.0))
    return out


def run_trajectory(
    case_name: str,
    objective: str,
    family: str,
    graph: WeightedGraph,
    p: int,
    evaluator: Evaluator,
    count: int,
) -> OptRow:
    xs = trajectory_params(p, count=count, seed=130000 + graph.n * 17 + p)
    values: list[float] = []
    t0 = time.perf_counter()
    status = "ok"
    notes = "fixed parameter trajectory; measures repeated objective-evaluation training cost"
    try:
        for x in xs:
            values.append(evaluator.eval(x))
    except Exception as exc:
        status = f"failed:{type(exc).__name__}"
        notes = str(exc)[:180]
    seconds = time.perf_counter() - t0
    initial = values[0] if values else float("nan")
    best = max(values) if values else float("nan")
    return OptRow(
        case=case_name,
        objective=objective,
        family=family,
        n=graph.n,
        m=graph.m,
        fields=len(graph.fields),
        p=p,
        kmax=max(c.k for c in extract_lightcones(graph, p)),
        total_cone_states=sum(1 << c.k for c in extract_lightcones(graph, p)),
        method=evaluator.method,
        mode="trajectory",
        status=status,
        evals=evaluator.evals,
        seconds=seconds,
        seconds_per_eval=seconds / evaluator.evals if evaluator.evals else float("nan"),
        initial_value=initial,
        best_value=best,
        improvement=best - initial if math.isfinite(best) and math.isfinite(initial) else float("nan"),
        backend=evaluator.backend,
        peak_pool_mb=evaluator.peak_pool_mb,
        notes=notes,
    )


def run_minimize(
    case_name: str,
    objective: str,
    family: str,
    graph: WeightedGraph,
    p: int,
    evaluator: Evaluator,
    maxiter: int,
) -> OptRow:
    try:
        from scipy.optimize import minimize
    except Exception as exc:
        return OptRow(
            case_name,
            objective,
            family,
            graph.n,
            graph.m,
            len(graph.fields),
            p,
            0,
            0,
            evaluator.method,
            "nelder_mead",
            "skipped_missing_scipy",
            0,
            0.0,
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            "",
            0.0,
            str(exc)[:180],
        )
    gammas, betas = params_for(p)
    base = np.asarray(gammas + betas, dtype=np.float64)
    best_seen = {"value": float("-inf")}
    t0 = time.perf_counter()
    initial = evaluator.eval(base)
    best_seen["value"] = initial

    def obj(x: np.ndarray) -> float:
        value = evaluator.eval(np.asarray(x, dtype=np.float64))
        if value > best_seen["value"]:
            best_seen["value"] = value
        return -value

    status = "ok"
    notes = f"scipy Nelder-Mead maxiter={maxiter}; objective is maximized as -expectation"
    try:
        result = minimize(
            obj,
            base,
            method="Nelder-Mead",
            options={"maxiter": maxiter, "xatol": 1e-3, "fatol": 1e-3, "disp": False},
        )
        if not result.success:
            status = f"optimizer_{str(result.message).replace(' ', '_')[:80]}"
    except Exception as exc:
        status = f"failed:{type(exc).__name__}"
        notes = str(exc)[:180]
    seconds = time.perf_counter() - t0
    cones = extract_lightcones(graph, p)
    return OptRow(
        case=case_name,
        objective=objective,
        family=family,
        n=graph.n,
        m=graph.m,
        fields=len(graph.fields),
        p=p,
        kmax=max(c.k for c in cones),
        total_cone_states=sum(1 << c.k for c in cones),
        method=evaluator.method,
        mode="nelder_mead",
        status=status,
        evals=evaluator.evals,
        seconds=seconds,
        seconds_per_eval=seconds / evaluator.evals if evaluator.evals else float("nan"),
        initial_value=initial,
        best_value=best_seen["value"],
        improvement=best_seen["value"] - initial,
        backend=evaluator.backend,
        peak_pool_mb=evaluator.peak_pool_mb,
        notes=notes,
    )


def skipped_row(case_name: str, objective: str, family: str, graph: WeightedGraph, p: int, method: str, status: str) -> OptRow:
    cones = extract_lightcones(graph, p)
    return OptRow(
        case_name,
        objective,
        family,
        graph.n,
        graph.m,
        len(graph.fields),
        p,
        max(c.k for c in cones),
        sum(1 << c.k for c in cones),
        method,
        "trajectory",
        status,
        0,
        0.0,
        float("nan"),
        float("nan"),
        float("nan"),
        float("nan"),
        "skipped",
        0.0,
        "guardrail skip",
    )


def run_case(
    case_name: str,
    objective: str,
    family: str,
    n: int,
    p: int,
    *,
    full_cap: int,
    max_k: int,
    max_batch_states: int,
    trajectory_evals: int,
    optimizer_iters: int,
) -> list[OptRow]:
    graph = graph_for_case(family, n, p, objective)
    kmax, total_cone_states = cone_stats(graph, p)
    print(f"OPT case={case_name} kmax={kmax} total={total_cone_states}", flush=True)
    rows: list[OptRow] = []
    methods = ["lc_batched_gpu"]
    if n <= full_cap:
        methods.insert(0, "full_precompute_gpu")
    else:
        rows.append(skipped_row(case_name, objective, family, graph, p, "full_precompute_gpu", f"skipped_over_{full_cap}_qubits"))

    for method in methods:
        if method == "lc_batched_gpu" and kmax > max_k:
            rows.append(skipped_row(case_name, objective, family, graph, p, method, f"skipped_kmax_{kmax}_over_{max_k}"))
            continue
        ev = Evaluator(graph, p, method, max_k, max_batch_states, full_cap)
        rows.append(run_trajectory(case_name, objective, family, graph, p, ev, trajectory_evals))
        if n <= 128:
            ev2 = Evaluator(graph, p, method, max_k, max_batch_states, full_cap)
            rows.append(run_minimize(case_name, objective, family, graph, p, ev2, optimizer_iters))
    return rows


def write_markdown(rows: list[OptRow], path: Path) -> None:
    lines = [
        "# Optimization-Loop Benchmark",
        "",
        "This benchmark measures repeated objective evaluation and small Nelder-Mead training loops.",
        "",
        "| Case | Method | Mode | Status | Evals | Total s | s/eval | Initial | Best | Improvement | Peak MB | Notes |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r.case} | {r.method} | {r.mode} | {r.status} | {r.evals} | {r.seconds:.4g} | "
            f"{r.seconds_per_eval:.4g} | {r.initial_value:.7g} | {r.best_value:.7g} | "
            f"{r.improvement:.4g} | {r.peak_pool_mb:.3g} | {r.notes} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "optimization_benchmark.csv")
    parser.add_argument("--markdown", type=Path, default=ROOT / "results" / "optimization_benchmark.md")
    parser.add_argument("--full-cap", type=int, default=24)
    parser.add_argument("--max-k", type=int, default=24)
    parser.add_argument("--max-batch-states", type=int, default=1 << 21)
    parser.add_argument("--trajectory-evals", type=int, default=64)
    parser.add_argument("--optimizer-iters", type=int, default=25)
    args = parser.parse_args()

    rows: list[OptRow] = []
    for spec in cases():
        rows.extend(
            run_case(
                *spec,
                full_cap=args.full_cap,
                max_k=args.max_k,
                max_batch_states=args.max_batch_states,
                trajectory_evals=args.trajectory_evals,
                optimizer_iters=args.optimizer_iters,
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
