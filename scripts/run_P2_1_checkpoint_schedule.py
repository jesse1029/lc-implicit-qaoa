from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from lcqaoa.backend import get_backend
from lcqaoa.graphs import random_regular_graph, weighted_modular_qubo_graph, weighted_qubo_graph
from lcqaoa.lightcone import (
    LightConeProblem,
    _split_batches,
    _x_sum_batched,
    extract_lightcones,
)
from lcqaoa.qaoa import apply_mixer_batched_inplace, cost_table, term_table
from benchmark_common import cone_metrics, params_for_depth


@dataclass
class CheckpointRow:
    case: str
    family: str
    n: int
    p: int
    seed: int
    schedule: str
    checkpoint_interval: int
    kmax: int
    total_cone_states: int
    n_batches: int
    objective_value: float
    gradient_norm: float
    seconds: float
    peak_pool_mb: float
    predicted_active_state_mb: float
    predicted_cache_mb: float
    predicted_total_mb: float
    measured_over_predicted: float
    rel_grad_l2_error_vs_cache_all: float
    max_grad_abs_error_vs_cache_all: float
    status: str


def make_graph(family: str, n: int, seed: int):
    if family == "3regular":
        return random_regular_graph(n, 3, seed=seed)
    if family == "weighted_qubo_er2":
        return weighted_qubo_graph(n, min(0.40, 2.0 / max(2, n)), seed=seed, field_scale=0.7)
    if family == "qubo_modular_sparse":
        return weighted_modular_qubo_graph(n, modules=max(4, n // 16), p_in=0.12, p_out=0.0015, seed=seed)
    raise ValueError(family)


def case_specs() -> list[tuple[str, int, int]]:
    return [
        ("3regular", 24, 2),
        ("3regular", 128, 2),
        ("3regular", 512, 2),
        ("3regular", 24, 3),
        ("weighted_qubo_er2", 96, 2),
        ("qubo_modular_sparse", 128, 1),
    ]


def _prepare_batch(batch: list[LightConeProblem], objective: str, xp, complex_dtype, float_dtype):
    k = batch[0].k
    nstates = 1 << k
    bsz = len(batch)
    psi0 = xp.empty((bsz, nstates), dtype=complex_dtype)
    psi0.fill(1.0 / math.sqrt(nstates))
    cost = xp.empty((bsz, nstates), dtype=float_dtype)
    terms = xp.empty((bsz, nstates), dtype=float_dtype)
    for row, problem in enumerate(batch):
        cost[row, :] = cost_table(k, problem.edges, problem.fields, objective, xp, float_dtype)
        terms[row, :] = term_table(k, problem.term_kind, problem.local_term_nodes, problem.weight, objective, xp, float_dtype)
    return psi0, cost, terms


def _forward_from_initial(psi0, cost, gammas, betas, xp, upto_layer: int | None = None):
    psi = psi0.copy()
    after_costs = []
    after_layers = []
    last = len(gammas) if upto_layer is None else upto_layer + 1
    for layer in range(last):
        psi *= xp.exp((-1j * float(gammas[layer])) * cost)
        after_costs.append(psi.copy())
        apply_mixer_batched_inplace(psi, psi.shape[1].bit_length() - 1, float(betas[layer]), xp)
        after_layers.append(psi.copy())
    return psi, after_costs, after_layers


def _state_for_layer(schedule: str, layer: int, psi0, cost, gammas, betas, xp, checkpoints, interval: int):
    if schedule == "cache_all":
        return checkpoints["after_costs"][layer], checkpoints["after_layers"][layer]
    if schedule == "recompute_all":
        _, after_costs, after_layers = _forward_from_initial(psi0, cost, gammas, betas, xp, upto_layer=layer)
        return after_costs[layer], after_layers[layer]
    # Fixed schedules store sparse after-layer checkpoints.
    start = max([idx for idx in checkpoints["after_layers"].keys() if idx < layer], default=-1)
    if start < 0:
        psi = psi0.copy()
        begin = 0
    else:
        psi = checkpoints["after_layers"][start].copy()
        begin = start + 1
    after_cost = None
    after_layer = None
    k = psi.shape[1].bit_length() - 1
    for idx in range(begin, layer + 1):
        psi *= xp.exp((-1j * float(gammas[idx])) * cost)
        after_cost = psi.copy()
        apply_mixer_batched_inplace(psi, k, float(betas[idx]), xp)
        after_layer = psi.copy()
    return after_cost, after_layer


def _batch_gradient(batch, gammas, betas, objective, xp, complex_dtype, float_dtype, schedule: str, interval: int):
    p = len(gammas)
    k = batch[0].k
    psi0, cost, terms = _prepare_batch(batch, objective, xp, complex_dtype, float_dtype)
    checkpoints = {"after_costs": {}, "after_layers": {}}
    if schedule == "cache_all":
        psi, after_costs, after_layers = _forward_from_initial(psi0, cost, gammas, betas, xp)
        checkpoints = {"after_costs": after_costs, "after_layers": after_layers}
    else:
        psi = psi0.copy()
        for layer in range(p):
            psi *= xp.exp((-1j * float(gammas[layer])) * cost)
            if schedule != "recompute_all" and layer % max(1, interval) == 0:
                checkpoints["after_costs"][layer] = psi.copy()
            apply_mixer_batched_inplace(psi, k, float(betas[layer]), xp)
            if schedule != "recompute_all" and layer % max(1, interval) == 0:
                checkpoints["after_layers"][layer] = psi.copy()
    values = xp.sum((xp.abs(psi) ** 2) * terms, axis=1)
    total = xp.sum(values)
    grad_gamma = xp.zeros(p, dtype=xp.float64)
    grad_beta = xp.zeros(p, dtype=xp.float64)
    adjoint = terms * psi
    for layer in range(p - 1, -1, -1):
        after_cost, after_layer = _state_for_layer(schedule, layer, psi0, cost, gammas, betas, xp, checkpoints, interval)
        x_sum = _x_sum_batched(after_layer, k, xp)
        d_beta_state = -1j * x_sum
        grad_beta[layer] = 2.0 * xp.real(xp.sum(xp.conj(adjoint) * d_beta_state))
        apply_mixer_batched_inplace(adjoint, k, -float(betas[layer]), xp)
        d_gamma_state = -1j * cost * after_cost
        grad_gamma[layer] = 2.0 * xp.real(xp.sum(xp.conj(adjoint) * d_gamma_state))
        adjoint *= xp.exp((1j * float(gammas[layer])) * cost)
    grad = xp.concatenate([grad_gamma, grad_beta])
    if hasattr(total, "get"):
        return float(total.get()), grad.get()
    return float(total), np.asarray(grad)


def run_schedule(graph, gammas, betas, p: int, schedule: str, interval: int, args):
    backend = get_backend(True)
    xp = backend.xp
    backend.free_memory_pool()
    problems = extract_lightcones(graph, p)
    groups: dict[int, list[LightConeProblem]] = defaultdict(list)
    for prob in problems:
        groups[prob.k].append(prob)
    batches: list[list[LightConeProblem]] = []
    for k in sorted(groups):
        batches.extend(_split_batches(groups[k], args.max_batch_states))
    total = 0.0
    grad = np.zeros(2 * p, dtype=np.float64)
    t0 = time.perf_counter()
    for batch in batches:
        val, g = _batch_gradient(batch, gammas, betas, graph.objective, xp, np.complex64, np.float32, schedule, interval)
        total += val
        grad += g
    backend.sync()
    return total, grad, time.perf_counter() - t0, backend.memory_pool_bytes(), len(batches), batches


def predicted_bytes(batches, p: int, schedule: str, interval: int) -> tuple[int, int, int]:
    state_bytes = 8
    float_bytes = 4
    max_state = max(len(b) * (1 << b[0].k) for b in batches) if batches else 0
    active = max_state * (2 * state_bytes + 2 * float_bytes)
    if schedule == "cache_all":
        cached_states = 2 * p
    elif schedule == "recompute_all":
        cached_states = 0
    else:
        cached_states = 2 * max(1, math.ceil(p / max(1, interval)))
    cache = max_state * cached_states * state_bytes
    temp = max_state * 2 * state_bytes
    return active, cache, active + cache + temp


def run_case(family: str, n: int, p: int, seed_id: int, args) -> list[CheckpointRow]:
    seed = 920000 + 1009 * seed_id + 37 * n + 131 * p
    graph = make_graph(family, n, seed)
    gammas, betas = params_for_depth(p, seed=seed_id)
    cmet = cone_metrics(graph, p)
    schedules = [
        ("cache_all", 1),
        ("recompute_all", p),
        ("fixed_interval", 2),
        ("sparse_fixed_interval", max(2, p)),
    ]
    case = f"{family}_n{n}_p{p}"
    ref_grad = None
    rows: list[CheckpointRow] = []
    for schedule, interval in schedules:
        print(f"P2-1 case={case} seed={seed_id} schedule={schedule}", flush=True)
        try:
            val, grad, sec, pool, nbatches, batches = run_schedule(graph, gammas, betas, p, schedule, interval, args)
            status = "ok"
        except Exception as exc:
            val, grad, sec, pool, nbatches, batches = float("nan"), np.full(2 * p, np.nan), 0.0, 0, 0, []
            status = f"failed:{type(exc).__name__}:{str(exc)[:120]}"
        if schedule == "cache_all" and status == "ok":
            ref_grad = grad.copy()
        if ref_grad is not None and status == "ok":
            diff = grad - ref_grad
            rel = float(np.linalg.norm(diff) / max(np.linalg.norm(ref_grad), 1e-30))
            max_abs = float(np.max(np.abs(diff)))
        else:
            rel, max_abs = float("nan"), float("nan")
        active, cache, total_pred = predicted_bytes(batches, p, schedule, interval)
        rows.append(
            CheckpointRow(
                case=case,
                family=family,
                n=n,
                p=p,
                seed=seed_id,
                schedule=schedule,
                checkpoint_interval=interval,
                kmax=int(cmet["kmax"]),
                total_cone_states=int(cmet["total_cone_states"]),
                n_batches=nbatches,
                objective_value=val,
                gradient_norm=float(np.linalg.norm(grad)) if status == "ok" else float("nan"),
                seconds=sec,
                peak_pool_mb=pool / 1024**2,
                predicted_active_state_mb=active / 1024**2,
                predicted_cache_mb=cache / 1024**2,
                predicted_total_mb=total_pred / 1024**2,
                measured_over_predicted=(pool / max(total_pred, 1)),
                rel_grad_l2_error_vs_cache_all=rel,
                max_grad_abs_error_vs_cache_all=max_abs,
                status=status,
            )
        )
    return rows


def write_md(rows: list[CheckpointRow], path: Path) -> None:
    lines = [
        "# P2-1 Checkpoint Schedule Ablation",
        "",
        "The variants compute the same LC adjoint gradient but trade cached forward states for recomputation.",
        "",
        "| Case | Schedule | Seconds | Peak MB | Pred MB | measured/pred | rel grad err |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r.case} seed{r.seed} | {r.schedule} | {r.seconds:.4g} | {r.peak_pool_mb:.4g} | "
            f"{r.predicted_total_mb:.4g} | {r.measured_over_predicted:.3g} | {r.rel_grad_l2_error_vs_cache_all:.3g} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "aaai27_required_experiments_20260710" / "P2_1_checkpoint_schedule")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--max-batch-states", type=int, default=1 << 19)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if args.quick:
        args.seeds = 1
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[CheckpointRow] = []
    specs = case_specs()[:2] if args.quick else case_specs()
    csv_path = args.out_dir / "P2_1_checkpoint_schedule.csv"
    for family, n, p in specs:
        for seed_id in range(args.seeds):
            rows.extend(run_case(family, n, p, seed_id, args))
            with csv_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
                writer.writeheader()
                for row in rows:
                    writer.writerow(asdict(row))
            write_md(rows, args.out_dir / "P2_1_checkpoint_schedule.md")
    print(f"WROTE {csv_path}")


if __name__ == "__main__":
    main()
