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

from lcqaoa.graphs import random_regular_graph, weighted_qubo_graph
from lcqaoa.lightcone import lightcone_expectation, lightcone_gradient_adjoint
from lcqaoa.qaoa import apply_cost_precomputed, apply_mixer_inplace, cost_table
from benchmark_common import cone_metrics, pack_params, params_for_depth, unpack_params, wrap_angles


@dataclass
class A8Row:
    case: str
    objective: str
    n: int
    p: int
    seed: int
    kmax: int
    training_steps: int
    train_status: str
    lc_expected_initial: float
    lc_expected_final: float
    full_expected_final: float
    sampled_mean: float
    sampled_best: float
    exact_optimum: float
    sample_count: int
    final_gap_to_optimum: float
    best_sample_gap_to_optimum: float
    notes: str


def make_graph(case: str, n: int, seed: int):
    if case == "3regular":
        return random_regular_graph(n, 3, seed=seed)
    if case == "weighted_sparse_qubo":
        return weighted_qubo_graph(n, min(0.40, 2.0 / max(2, n)), seed=seed, field_scale=0.7)
    raise ValueError(case)


def qubo_value_from_index(idx: int, graph) -> float:
    total = 0.0
    if graph.objective == "maxcut":
        for i, j, w in graph.edges:
            total += float(w) * (((idx >> int(i)) ^ (idx >> int(j))) & 1)
    else:
        for i, j, w in graph.edges:
            total += float(w) * (((idx >> int(i)) & 1) & ((idx >> int(j)) & 1))
        for i, w in graph.fields:
            total += float(w) * ((idx >> int(i)) & 1)
    return total


def full_state_probs(graph, gammas: list[float], betas: list[float]):
    from lcqaoa.backend import get_backend

    backend = get_backend(True)
    xp = backend.xp
    backend.free_memory_pool()
    nstates = 1 << graph.n
    psi = xp.empty(nstates, dtype=np.complex64)
    psi.fill(1.0 / math.sqrt(nstates))
    cost = cost_table(graph.n, graph.edges, graph.fields, graph.objective, xp, np.float32)
    for gamma, beta in zip(gammas, betas):
        apply_cost_precomputed(psi, cost, gamma, xp)
        apply_mixer_inplace(psi, graph.n, beta, xp)
    probs = xp.abs(psi) ** 2
    exp_val = float(xp.sum(probs * cost).get())
    probs_np = probs.get()
    backend.free_memory_pool()
    return probs_np / probs_np.sum(), exp_val


def brute_force_optimum(graph) -> float:
    best = -float("inf")
    for idx in range(1 << graph.n):
        best = max(best, qubo_value_from_index(idx, graph))
    return best


def train_lc(graph, p: int, seed: int, steps: int, lr: float, args):
    gammas, betas = params_for_depth(p, seed=seed)
    x = pack_params(gammas, betas)
    first = None
    status = "ok"
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    for step in range(steps):
        gammas, betas = unpack_params(x, p)
        stats = lightcone_gradient_adjoint(graph, gammas, betas, p=p, prefer_gpu=True, max_k=args.max_k, max_batch_states=args.max_batch_states)
        if stats.status != "ok" or stats.gradient is None:
            status = stats.status
            break
        if first is None:
            first = float(stats.value)
        grad = np.asarray(stats.gradient, dtype=np.float64)
        m = 0.9 * m + 0.1 * grad
        v = 0.999 * v + 0.001 * (grad * grad)
        mh = m / (1.0 - 0.9 ** (step + 1))
        vh = v / (1.0 - 0.999 ** (step + 1))
        x = wrap_angles(x + lr * mh / (np.sqrt(vh) + 1e-8))
    gammas, betas = unpack_params(x, p)
    final = lightcone_expectation(graph, gammas, betas, p=p, prefer_gpu=True, max_k=args.max_k, max_batch_states=args.max_batch_states)
    if first is None:
        first = float("nan")
    return gammas, betas, first, float(final.value), status if status != "ok" else final.status


def run_case(case: str, n: int, p: int, seed_id: int, args) -> A8Row:
    seed = 330000 + 997 * seed_id + 37 * n + 131 * p
    graph = make_graph(case, n, seed)
    cmet = cone_metrics(graph, p)
    gammas, betas, initial, final_lc, train_status = train_lc(graph, p, seed_id, args.steps, args.lr, args)
    probs, full_expected = full_state_probs(graph, gammas, betas)
    rng = np.random.default_rng(seed)
    idxs = rng.choice(np.arange(1 << graph.n), size=args.samples, p=probs)
    vals = np.asarray([qubo_value_from_index(int(i), graph) for i in idxs], dtype=np.float64)
    optimum = brute_force_optimum(graph)
    return A8Row(
        case=case,
        objective=graph.objective,
        n=graph.n,
        p=p,
        seed=seed_id,
        kmax=int(cmet["kmax"]),
        training_steps=args.steps,
        train_status=train_status,
        lc_expected_initial=initial,
        lc_expected_final=final_lc,
        full_expected_final=full_expected,
        sampled_mean=float(vals.mean()),
        sampled_best=float(vals.max()),
        exact_optimum=optimum,
        sample_count=args.samples,
        final_gap_to_optimum=optimum - final_lc,
        best_sample_gap_to_optimum=optimum - float(vals.max()),
        notes="LC trains parameters; full-state sampler used only for n<=24 handoff validation",
    )


def write_md(rows: list[A8Row], path: Path) -> None:
    lines = [
        "# A8 Sampling / Final-Solution Handoff",
        "",
        "| Case | n | p | seed | LC expected | full expected | sampled mean | sampled best | optimum | best gap |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r.case} | {r.n} | {r.p} | {r.seed} | {r.lc_expected_final:.6g} | {r.full_expected_final:.6g} | "
            f"{r.sampled_mean:.6g} | {r.sampled_best:.6g} | {r.exact_optimum:.6g} | {r.best_sample_gap_to_optimum:.6g} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "benchmark_suite_20260704" / "A8_sampling_handoff")
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--samples", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--max-k", type=int, default=24)
    parser.add_argument("--max-batch-states", type=int, default=1 << 21)
    parser.add_argument("--seeds", type=int, default=5)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[A8Row] = []
    for case in ["3regular", "weighted_sparse_qubo"]:
        for p in [1, 2]:
            for seed_id in range(args.seeds):
                print(f"A8 case={case} n=24 p={p} seed={seed_id}", flush=True)
                rows.append(run_case(case, 24, p, seed_id, args))
    csv_path = args.out_dir / "A8_sampling_handoff.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
    write_md(rows, args.out_dir / "A8_sampling_handoff.md")
    print(f"WROTE {csv_path}")


if __name__ == "__main__":
    main()
