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
from lcqaoa.graphs import WeightedGraph
from lcqaoa.lightcone import extract_lightcones, lightcone_expectation
from lcqaoa.qaoa import apply_cost_precomputed, apply_mixer_inplace, cost_table, expectation_from_state
from run_benchmarks import params_for
from run_optimization_benchmarks import graph_for_case, trajectory_params


@dataclass
class AmortizedRow:
    case: str
    method: str
    n: int
    m: int
    p: int
    kmax: int
    total_cone_states: int
    status: str
    evals: int
    setup_seconds: float
    eval_seconds: float
    seconds_per_eval: float
    initial_value: float
    best_value: float
    improvement: float
    peak_pool_mb: float
    persistent_cost_mb: float
    backend: str
    notes: str


class CachedFullStateEvaluator:
    def __init__(self, graph: WeightedGraph, *, prefer_gpu: bool = True) -> None:
        self.graph = graph
        self.backend = get_backend(prefer_gpu)
        self.xp = self.backend.xp
        self.backend.free_memory_pool()
        self.cost = None
        self.setup_seconds = 0.0
        self.peak_pool_mb = 0.0
        self.persistent_cost_mb = 0.0

    def setup(self) -> None:
        t0 = time.perf_counter()
        self.cost = cost_table(
            self.graph.n,
            self.graph.edges,
            self.graph.fields,
            self.graph.objective,
            self.xp,
            np.float32,
        )
        self.backend.sync()
        self.setup_seconds = time.perf_counter() - t0
        self.persistent_cost_mb = float(self.cost.nbytes) / 1024**2
        self.peak_pool_mb = max(self.peak_pool_mb, self.backend.memory_pool_bytes() / 1024**2)

    def eval(self, gammas: list[float], betas: list[float]) -> float:
        if self.cost is None:
            raise RuntimeError("cached cost table is not initialized")
        nstates = 1 << self.graph.n
        psi = self.xp.empty(nstates, dtype=np.complex64)
        psi.fill(1.0 / math.sqrt(nstates))
        for gamma, beta in zip(gammas, betas):
            apply_cost_precomputed(psi, self.cost, gamma, self.xp)
            apply_mixer_inplace(psi, self.graph.n, beta, self.xp)
        value = expectation_from_state(psi, self.cost, self.xp)
        self.backend.sync()
        self.peak_pool_mb = max(self.peak_pool_mb, self.backend.memory_pool_bytes() / 1024**2)
        return float(value)


def unpack_params(x: np.ndarray, p: int) -> tuple[list[float], list[float]]:
    return x[:p].astype(float).tolist(), x[p:].astype(float).tolist()


def cone_stats(graph: WeightedGraph, p: int) -> tuple[int, int]:
    cones = extract_lightcones(graph, p)
    return max(c.k for c in cones), sum(1 << c.k for c in cones)


def run_cached_full(case: str, graph: WeightedGraph, p: int, count: int) -> AmortizedRow:
    kmax, total = cone_stats(graph, p)
    evaluator = CachedFullStateEvaluator(graph, prefer_gpu=True)
    xs = trajectory_params(p, count=count, seed=130000 + graph.n * 17 + p)
    values: list[float] = []
    status = "ok"
    notes = "global diagonal cost table cached once; eval time excludes setup"
    try:
        evaluator.setup()
        t0 = time.perf_counter()
        for x in xs:
            gammas, betas = unpack_params(x, p)
            values.append(evaluator.eval(gammas, betas))
        eval_seconds = time.perf_counter() - t0
    except Exception as exc:
        status = f"failed:{type(exc).__name__}"
        notes = str(exc)[:180]
        eval_seconds = 0.0
    initial = values[0] if values else float("nan")
    best = max(values) if values else float("nan")
    return AmortizedRow(
        case=case,
        method="full_precompute_gpu_amortized",
        n=graph.n,
        m=graph.m,
        p=p,
        kmax=kmax,
        total_cone_states=total,
        status=status,
        evals=len(values),
        setup_seconds=evaluator.setup_seconds,
        eval_seconds=eval_seconds,
        seconds_per_eval=eval_seconds / len(values) if values else float("nan"),
        initial_value=initial,
        best_value=best,
        improvement=best - initial if math.isfinite(best) and math.isfinite(initial) else float("nan"),
        peak_pool_mb=evaluator.peak_pool_mb,
        persistent_cost_mb=evaluator.persistent_cost_mb,
        backend=evaluator.backend.name,
        notes=notes,
    )


def run_lc(case: str, graph: WeightedGraph, p: int, count: int) -> AmortizedRow:
    kmax, total = cone_stats(graph, p)
    xs = trajectory_params(p, count=count, seed=130000 + graph.n * 17 + p)
    values: list[float] = []
    peak = 0.0
    backend = ""
    status = "ok"
    notes = "LC trajectory with no global state or global diagonal table"
    t0 = time.perf_counter()
    try:
        for x in xs:
            gammas, betas = unpack_params(x, p)
            stats = lightcone_expectation(
                graph,
                gammas,
                betas,
                p=p,
                prefer_gpu=True,
                max_k=24,
                max_batch_states=1 << 21,
            )
            if stats.status != "ok":
                raise RuntimeError(stats.status)
            values.append(float(stats.value))
            peak = max(peak, stats.peak_pool_bytes / 1024**2)
            backend = stats.backend
    except Exception as exc:
        status = f"failed:{type(exc).__name__}"
        notes = str(exc)[:180]
    seconds = time.perf_counter() - t0
    initial = values[0] if values else float("nan")
    best = max(values) if values else float("nan")
    return AmortizedRow(
        case=case,
        method="lc_batched_gpu",
        n=graph.n,
        m=graph.m,
        p=p,
        kmax=kmax,
        total_cone_states=total,
        status=status,
        evals=len(values),
        setup_seconds=0.0,
        eval_seconds=seconds,
        seconds_per_eval=seconds / len(values) if values else float("nan"),
        initial_value=initial,
        best_value=best,
        improvement=best - initial if math.isfinite(best) and math.isfinite(initial) else float("nan"),
        peak_pool_mb=peak,
        persistent_cost_mb=0.0,
        backend=backend,
        notes=notes,
    )


def write_outputs(rows: list[AmortizedRow], out_dir: Path) -> None:
    csv_path = out_dir / "p1_amortized_fullstate.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
    md_lines = [
        "# P1 Amortized Full-state Baseline",
        "",
        "The full-state amortized row builds the global diagonal cost table once and reports repeated trajectory evaluations separately from setup.",
        "",
        "| Case | Method | Status | Setup s | Eval s | s/eval | Peak MB | Persistent cost MB | Best gain |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        md_lines.append(
            f"| {row.case} | {row.method} | {row.status} | {row.setup_seconds:.4g} | {row.eval_seconds:.4g} | "
            f"{row.seconds_per_eval:.4g} | {row.peak_pool_mb:.4g} | {row.persistent_cost_mb:.4g} | {row.improvement:.5g} |"
        )
    (out_dir / "p1_amortized_fullstate.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"wrote {csv_path}")
    print(f"wrote {out_dir / 'p1_amortized_fullstate.md'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "p1_amortized_fullstate")
    parser.add_argument("--count", type=int, default=64)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cases = [
        ("maxcut_3regular_n24_p2", "maxcut", "3regular", 24, 2),
        ("maxcut_3regular_n26_p2", "maxcut", "3regular", 26, 2),
    ]
    rows: list[AmortizedRow] = []
    for case, objective, family, n, p in cases:
        graph = graph_for_case(family, n, p, objective)
        rows.append(run_cached_full(case, graph, p, args.count))
        rows.append(run_lc(case, graph, p, args.count))
        print(f"{case}: cached_full={rows[-2].seconds_per_eval:.4g}s/eval, lc={rows[-1].seconds_per_eval:.4g}s/eval")
    write_outputs(rows, args.out_dir)


if __name__ == "__main__":
    main()
