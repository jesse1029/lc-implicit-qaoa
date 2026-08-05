from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from lcqaoa.graphs import weighted_modular_qubo_graph, weighted_qubo_graph
from lcqaoa.lightcone import extract_lightcones, lightcone_expectation, lightcone_topology_signature
from run_benchmarks import params_for
from run_sota_sparse_scale import FAMILY_SEED_OFFSETS, degree_stats, graph_for_scale


@dataclass
class TopologyAblationRow:
    case: str
    objective: str
    family: str
    n: int
    m: int
    p: int
    avg_degree: float
    max_degree: int
    kmax: int
    n_cones: int
    n_size_groups: int
    n_topology_groups: int
    largest_size_group: int
    largest_topology_group: int
    mode: str
    status: str
    value: float
    seconds: float
    speedup_vs_naive: float
    abs_error_vs_size_batched: float
    peak_pool_mb: float
    notes: str


def case_specs() -> list[tuple[str, str, int, int]]:
    return [
        ("maxcut", "3regular", 24, 2),
        ("maxcut", "3regular", 128, 2),
        ("maxcut", "3regular", 512, 2),
        ("maxcut", "modular_sparse", 128, 1),
        ("maxcut", "scale_free_a1", 64, 1),
        ("qubo", "qubo_er_deg2", 96, 2),
        ("qubo", "qubo_modular_sparse", 128, 1),
    ]


def make_graph(objective: str, family: str, n: int, p: int):
    seed = 140000 + n * 29 + p * 131 + FAMILY_SEED_OFFSETS.get(family.replace("qubo_", ""), 0)
    if objective == "maxcut":
        return graph_for_scale(family, n, seed=seed)
    if family == "qubo_er_deg2":
        return weighted_qubo_graph(n, min(1.0, 2.0 / max(n - 1, 1)), seed=seed)
    if family == "qubo_modular_sparse":
        return weighted_modular_qubo_graph(n, modules=8 if n >= 64 else 4, p_in=0.08, p_out=0.002, seed=seed)
    raise ValueError(f"unknown QUBO family: {family}")


def group_stats(graph, p: int) -> tuple[int, int, int, int, int, int]:
    cones = extract_lightcones(graph, p)
    if not cones:
        return 0, 0, 0, 0, 0, 0
    size_counts = Counter(c.k for c in cones)
    topo_counts = Counter(lightcone_topology_signature(c) for c in cones)
    return (
        max(c.k for c in cones),
        len(cones),
        len(size_counts),
        len(topo_counts),
        max(size_counts.values()),
        max(topo_counts.values()),
    )


def run_case(
    objective: str,
    family: str,
    n: int,
    p: int,
    *,
    max_k: int,
    max_batch_states: int,
) -> list[TopologyAblationRow]:
    graph = make_graph(objective, family, n, p)
    gammas, betas = params_for(p)
    avg_degree, max_degree = degree_stats(graph)
    kmax, n_cones, n_size_groups, n_topology_groups, largest_size_group, largest_topology_group = group_stats(graph, p)
    case = f"{family}_n{n}_p{p}"
    print(
        "TOPO_ABLATION "
        f"case={case} objective={objective} cones={n_cones} kmax={kmax} "
        f"size_groups={n_size_groups} topology_groups={n_topology_groups}",
        flush=True,
    )
    rows: list[TopologyAblationRow] = []
    modes = [
        ("naive_per_cone", True, "size"),
        ("size_batched", False, "size"),
        ("topology_grouped", False, "topology"),
    ]
    naive_seconds = float("nan")
    size_value = float("nan")
    for mode, naive, group_by in modes:
        stat = lightcone_expectation(
            graph,
            gammas,
            betas,
            p=p,
            prefer_gpu=True,
            max_k=max_k,
            max_batch_states=max_batch_states,
            naive=naive,
            group_by=group_by,
        )
        if mode == "naive_per_cone" and stat.status == "ok":
            naive_seconds = stat.seconds
        if mode == "size_batched" and stat.status == "ok":
            size_value = stat.value
        rows.append(
            TopologyAblationRow(
                case=case,
                objective=objective,
                family=family,
                n=graph.n,
                m=graph.m,
                p=p,
                avg_degree=avg_degree,
                max_degree=max_degree,
                kmax=kmax,
                n_cones=n_cones,
                n_size_groups=n_size_groups,
                n_topology_groups=n_topology_groups,
                largest_size_group=largest_size_group,
                largest_topology_group=largest_topology_group,
                mode=mode,
                status=stat.status,
                value=stat.value,
                seconds=stat.seconds,
                speedup_vs_naive=(naive_seconds / stat.seconds) if stat.status == "ok" and stat.seconds > 0 and naive_seconds == naive_seconds else float("nan"),
                abs_error_vs_size_batched=abs(stat.value - size_value) if stat.status == "ok" and size_value == size_value else float("nan"),
                peak_pool_mb=stat.peak_pool_bytes / 1024**2,
                notes=(
                    "one kernel-sized launch pattern per cone"
                    if mode == "naive_per_cone"
                    else "groups all cones with same local state size"
                    if mode == "size_batched"
                    else "groups cones by local edge/field topology signature"
                ),
            )
        )
    return rows


def write_markdown(rows: list[TopologyAblationRow], path: Path) -> None:
    lines = [
        "# Topology/Batching Ablation",
        "",
        "This ablation separates naive per-cone evaluation, size-batched LC evaluation, and topology-signature grouped LC evaluation.",
        "",
        "| Case | n | p | kmax | Cones | Size groups | Topology groups | Mode | Status | Seconds | Speedup vs naive | Err vs size | Peak MB |",
        "|---|---:|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r.case} | {r.n} | {r.p} | {r.kmax} | {r.n_cones} | {r.n_size_groups} | "
            f"{r.n_topology_groups} | {r.mode} | {r.status} | {r.seconds:.4g} | "
            f"{r.speedup_vs_naive:.3g} | {r.abs_error_vs_size_batched:.3g} | {r.peak_pool_mb:.3g} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "topology_ablation.csv")
    parser.add_argument("--markdown", type=Path, default=ROOT / "results" / "topology_ablation.md")
    parser.add_argument("--max-k", type=int, default=24)
    parser.add_argument("--max-batch-states", type=int, default=1 << 21)
    args = parser.parse_args()

    rows: list[TopologyAblationRow] = []
    for objective, family, n, p in case_specs():
        rows.extend(run_case(objective, family, n, p, max_k=args.max_k, max_batch_states=args.max_batch_states))

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
