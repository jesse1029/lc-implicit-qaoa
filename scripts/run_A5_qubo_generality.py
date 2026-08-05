from __future__ import annotations

import argparse
import csv
import math
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from lcqaoa.graphs import WeightedGraph, weighted_modular_qubo_graph, weighted_qubo_graph
from lcqaoa.lightcone import lightcone_expectation, lightcone_gradient_adjoint
from lcqaoa.qaoa import full_state_expectation
from run_real_qubo_case_study import feature_selection_qubo, load_datasets
from benchmark_common import cone_metrics, graph_metrics, params_for_depth


@dataclass
class A5Row:
    family: str
    n: int
    p: int
    seed: int
    weight_model: str
    field_edge_ratio: float
    objective: str
    m: int
    fields: int
    mean_degree: float
    max_degree: int
    kmax: int
    total_cone_states: int
    lc_status: str
    lc_seconds: float
    lc_grad_status: str
    lc_grad_seconds: float
    lc_peak_mb: float
    full_status: str
    full_seconds: float
    full_peak_mb: float
    abs_error_vs_full: float
    greedy_score: float
    local_search_score: float
    notes: str


def signed_sparse_qubo(n: int, seed: int, model: str, field_ratio: float) -> WeightedGraph:
    rng = random.Random(seed)
    edge_prob = min(0.40, 2.0 / max(2, n))
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < edge_prob:
                if model == "uniform":
                    w = rng.uniform(-1.0, 1.0)
                elif model == "normal":
                    w = rng.gauss(0.0, 0.6)
                elif model == "heavy_tail":
                    w = rng.choice([-1.0, 1.0]) * min(3.0, 0.2 / max(rng.random(), 1e-3))
                else:
                    raise ValueError(model)
                edges.append((i, j, float(w)))
    if not edges and n >= 2:
        edges.append((0, 1, 1.0))
    fields = tuple((i, rng.uniform(-field_ratio, field_ratio)) for i in range(n))
    return WeightedGraph(n=n, edges=tuple(edges), fields=fields, objective="qubo")


def budget_penalty_qubo(n: int, seed: int, dense: bool) -> WeightedGraph:
    rng = random.Random(seed)
    edges = []
    fields = []
    # Base relevance fields.
    for i in range(n):
        fields.append((i, rng.uniform(-0.5, 0.5)))
    if dense:
        # Expansion of lambda(sum x_i - k)^2 produces all-to-all positive couplings.
        lam = 0.08
        for i in range(n):
            for j in range(i + 1, n):
                edges.append((i, j, 2.0 * lam))
    else:
        # Local partition surrogate: budget only within small adjacent blocks.
        lam = 0.08
        block = 4
        for start in range(0, n, block):
            stop = min(n, start + block)
            for i in range(start, stop):
                for j in range(i + 1, stop):
                    edges.append((i, j, 2.0 * lam))
    return WeightedGraph(n=n, edges=tuple(edges), fields=tuple(fields), objective="qubo")


def feature_qubo(name: str, top_features: int, top_degree: int) -> WeightedGraph:
    datasets = load_datasets()
    if name not in datasets:
        raise ValueError(name)
    return feature_selection_qubo(*datasets[name], top_features=top_features, top_degree=top_degree)


def graph_for(family: str, n: int, seed: int, weight_model: str, ratio: float) -> WeightedGraph:
    if family == "signed_sparse":
        return signed_sparse_qubo(n, seed, weight_model, ratio)
    if family == "field_dominant":
        return signed_sparse_qubo(n, seed, "uniform", ratio)
    if family == "modular_community":
        return weighted_modular_qubo_graph(n, modules=max(4, n // 16), p_in=0.12, p_out=0.0015, seed=seed)
    if family == "budget_dense":
        return budget_penalty_qubo(n, seed, dense=True)
    if family == "budget_local":
        return budget_penalty_qubo(n, seed, dense=False)
    if family == "wdbc_feature":
        return feature_qubo("breast_cancer", min(n, 30), 3)
    if family == "digits_feature":
        return feature_qubo("digits", min(n, 30), 3)
    return weighted_qubo_graph(n, min(0.40, 2.0 / max(2, n)), seed=seed)


def qubo_score(graph: WeightedGraph, bits: np.ndarray) -> float:
    total = 0.0
    for i, j, w in graph.edges:
        total += float(w) * float(bits[int(i)] * bits[int(j)])
    for i, w in graph.fields:
        total += float(w) * float(bits[int(i)])
    return total


def greedy_local_scores(graph: WeightedGraph, seed: int, steps: int = 1000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    bits = np.zeros(graph.n, dtype=np.int8)
    # Greedy insert if it helps.
    improved = True
    while improved:
        improved = False
        base = qubo_score(graph, bits)
        best_i, best_v = -1, base
        for i in range(graph.n):
            cand = bits.copy()
            cand[i] = 1 - cand[i]
            val = qubo_score(graph, cand)
            if val > best_v:
                best_i, best_v = i, val
        if best_i >= 0:
            bits[best_i] = 1 - bits[best_i]
            improved = True
    greedy = qubo_score(graph, bits)
    bits = rng.integers(0, 2, size=graph.n, dtype=np.int8)
    best = qubo_score(graph, bits)
    temp = 1.0
    for _ in range(steps):
        i = int(rng.integers(0, graph.n))
        cand = bits.copy()
        cand[i] = 1 - cand[i]
        val = qubo_score(graph, cand)
        if val > best or rng.random() < math.exp(min(0.0, (val - best) / max(temp, 1e-8))):
            bits = cand
            best = max(best, val)
        temp *= 0.995
    return greedy, best


def run_case(family: str, n: int, p: int, seed_id: int, weight_model: str, ratio: float, args) -> A5Row:
    seed = 280000 + 1009 * seed_id + 37 * n + 131 * p
    graph = graph_for(family, n, seed, weight_model, ratio)
    gammas, betas = params_for_depth(p, seed=seed_id)
    gmet = graph_metrics(graph)
    if family == "budget_dense" and p >= 1:
        # Dense cardinality penalties are a negative family. Every nonconstant
        # term reaches the whole graph at p>=1, so avoid spending minutes
        # rediscovering this by per-edge BFS on a dense graph.
        term_count = graph.m + len(graph.fields)
        cmet = {
            "term_count": term_count,
            "kmax": graph.n,
            "k_median": float(graph.n),
            "k_p95": float(graph.n),
            "total_cone_states": int(term_count) * (1 << graph.n),
            "max_batch_state_elements": int(term_count) * (1 << graph.n),
        }
    else:
        cmet = cone_metrics(graph, p)
    lc_status = "NOT_RUN_EXPLAINED"
    lc_seconds = 0.0
    lc_grad_status = "NOT_RUN_EXPLAINED"
    lc_grad_seconds = 0.0
    lc_peak = 0.0
    lc_value = float("nan")
    notes = []
    if cmet["kmax"] <= args.max_k and cmet["total_cone_states"] <= args.max_total_cone_states:
        try:
            lc = lightcone_expectation(graph, gammas, betas, p=p, prefer_gpu=True, max_k=args.max_k, max_batch_states=args.max_batch_states)
            lc_status, lc_seconds, lc_peak, lc_value = lc.status, lc.seconds, lc.peak_pool_bytes / 1024**2, lc.value
        except Exception as exc:
            lc_status = f"failed:{type(exc).__name__}"
            notes.append(str(exc)[:160])
        try:
            grad = lightcone_gradient_adjoint(graph, gammas, betas, p=p, prefer_gpu=True, max_k=args.max_k, max_batch_states=args.max_batch_states)
            lc_grad_status, lc_grad_seconds = grad.status, grad.seconds
            lc_peak = max(lc_peak, grad.peak_pool_bytes / 1024**2)
        except Exception as exc:
            lc_grad_status = f"failed:{type(exc).__name__}"
            notes.append(str(exc)[:160])
    else:
        notes.append("LC guardrail triggered by cone growth")
    full_status, full_seconds, full_peak, full_value = "NOT_RUN_EXPLAINED", 0.0, 0.0, float("nan")
    if n <= args.full_cap:
        try:
            full = full_state_expectation(graph, gammas, betas, method="precompute", prefer_gpu=True, max_qubits=None)
            full_status, full_seconds, full_peak, full_value = full.status, full.seconds, full.peak_pool_bytes / 1024**2, full.value
        except Exception as exc:
            full_status = f"failed:{type(exc).__name__}"
    if graph.m > args.classical_max_edges:
        greedy, local = float("nan"), float("nan")
        notes.append(f"classical greedy/local-search skipped for m={graph.m} over cap {args.classical_max_edges}")
    else:
        steps = min(args.local_search_steps, max(50, args.classical_work_cap // max(graph.m, 1)))
        greedy, local = greedy_local_scores(graph, seed, steps=steps)
    err = abs(lc_value - full_value) if math.isfinite(lc_value) and math.isfinite(full_value) else float("nan")
    if family == "budget_dense":
        notes.append("naive cardinality penalty creates dense couplings")
    if family == "budget_local":
        notes.append("block-local budget surrogate preserves locality")
    return A5Row(
        family=family,
        n=graph.n,
        p=p,
        seed=seed_id,
        weight_model=weight_model,
        field_edge_ratio=ratio,
        objective=graph.objective,
        m=graph.m,
        fields=len(graph.fields),
        mean_degree=gmet["mean_degree"],
        max_degree=int(gmet["max_degree"]),
        kmax=int(cmet["kmax"]),
        total_cone_states=int(cmet["total_cone_states"]),
        lc_status=lc_status,
        lc_seconds=lc_seconds,
        lc_grad_status=lc_grad_status,
        lc_grad_seconds=lc_grad_seconds,
        lc_peak_mb=lc_peak,
        full_status=full_status,
        full_seconds=full_seconds,
        full_peak_mb=full_peak,
        abs_error_vs_full=err,
        greedy_score=greedy,
        local_search_score=local,
        notes="; ".join(notes),
    )


def write_md(rows: list[A5Row], path: Path) -> None:
    lines = [
        "# A5 QUBO Generality Beyond MaxCut",
        "",
        "This benchmark keeps the same LC evaluator and varies fields, signed weights, modularity, and constraint-penalty locality.",
        "",
        "| Family | p | rows | LC success | max n success | median kmax success | full-check max error | note |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for family, p in sorted({(r.family, r.p) for r in rows}):
        sub = [r for r in rows if r.family == family and r.p == p]
        ok = [r for r in sub if r.lc_status == "ok"]
        errs = [r.abs_error_vs_full for r in ok if math.isfinite(r.abs_error_vs_full)]
        note = "; ".join(sorted(set(r.notes for r in sub if r.notes)))[:140]
        lines.append(
            f"| {family} | {p} | {len(sub)} | {len(ok)} | {max([r.n for r in ok] or [0])} | "
            f"{float(np.median([r.kmax for r in ok])) if ok else float('nan'):.3g} | {max(errs or [float('nan')]):.3g} | {note} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def flush_csv(rows: list[A5Row], path: Path) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "benchmark_suite_20260704" / "A5_qubo_generality")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--max-k", type=int, default=24)
    parser.add_argument("--max-total-cone-states", type=int, default=1 << 31)
    parser.add_argument("--max-batch-states", type=int, default=1 << 19)
    parser.add_argument("--full-cap", type=int, default=24)
    parser.add_argument("--local-search-steps", type=int, default=1000)
    parser.add_argument("--classical-work-cap", type=int, default=200000)
    parser.add_argument("--classical-max-edges", type=int, default=5000)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    families = ["signed_sparse", "field_dominant", "modular_community", "budget_dense", "budget_local", "wdbc_feature", "digits_feature"]
    ns = [24, 48, 64, 96, 128, 256]
    ps = [1, 2]
    models = ["uniform", "normal", "heavy_tail"]
    ratios = [0.1, 1.0, 10.0]
    if args.quick:
        families, ns, ps, models, ratios = families[:4], [24, 64], [1], ["uniform"], [1.0]
        args.seeds = min(args.seeds, 2)
    rows: list[A5Row] = []
    csv_path = args.out_dir / "A5_qubo_generality.csv"
    for family in families:
        for n in ns:
            if family in {"wdbc_feature", "digits_feature"} and n not in {24, 48}:
                continue
            for p in ps:
                for seed_id in range(args.seeds):
                    weight_models = models if family == "signed_sparse" else ["uniform"]
                    field_ratios = ratios if family == "field_dominant" else [1.0]
                    for model in weight_models:
                        for ratio in field_ratios:
                            print(f"A5 family={family} n={n} p={p} seed={seed_id} model={model} ratio={ratio}", flush=True)
                            rows.append(run_case(family, n, p, seed_id, model, ratio, args))
                            flush_csv(rows, csv_path)
                            write_md(rows, args.out_dir / "A5_qubo_generality.md")
    flush_csv(rows, csv_path)
    write_md(rows, args.out_dir / "A5_qubo_generality.md")
    print(f"WROTE {csv_path}")


if __name__ == "__main__":
    main()
