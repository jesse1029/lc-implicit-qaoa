from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from lcqaoa.graphs import weighted_modular_qubo_graph, weighted_qubo_graph
from lcqaoa.lightcone import extract_lightcones, lightcone_expectation, lightcone_topology_signature
from run_sota_sparse_scale import graph_for_scale
from benchmark_common import cone_metrics, graph_metrics, params_for_depth


@dataclass
class A6Row:
    case: str
    objective: str
    family: str
    n: int
    p: int
    m: int
    kmax: int
    total_cone_states: int
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
    speedup_vs_size: float
    abs_error_vs_size: float
    peak_mb: float
    kernel_launch_proxy: int
    fragmentation_ratio: float
    notes: str


def make_graph(objective: str, family: str, n: int, seed: int):
    if objective == "maxcut":
        return graph_for_scale(family, n, seed=seed)
    if family == "qubo_er_deg2":
        return weighted_qubo_graph(n, min(0.40, 2.0 / max(2, n)), seed=seed, field_scale=0.7)
    if family == "qubo_modular_sparse":
        return weighted_modular_qubo_graph(n, modules=max(4, n // 16), p_in=0.12, p_out=0.0015, seed=seed)
    raise ValueError(family)


def case_specs() -> list[tuple[str, str, int, int]]:
    return [
        ("maxcut", "3regular", 24, 2),
        ("maxcut", "3regular", 128, 2),
        ("maxcut", "3regular", 512, 2),
        ("maxcut", "3regular", 24, 3),
        ("maxcut", "er_deg2", 128, 2),
        ("maxcut", "modular_sparse", 128, 1),
        ("maxcut", "scale_free_a1", 64, 1),
        ("qubo", "qubo_er_deg2", 96, 2),
        ("qubo", "qubo_modular_sparse", 128, 1),
    ]


def grouping_stats(graph, p: int):
    cones = extract_lightcones(graph, p)
    if not cones:
        return 0, 0, 0, 0, 0, 0
    sizes = Counter(c.k for c in cones)
    topo = Counter(lightcone_topology_signature(c) for c in cones)
    return max(c.k for c in cones), len(cones), len(sizes), len(topo), max(sizes.values()), max(topo.values())


def run_case(objective: str, family: str, n: int, p: int, args) -> list[A6Row]:
    seed = 310000 + 37 * n + 131 * p
    graph = make_graph(objective, family, n, seed)
    gammas, betas = params_for_depth(p, seed=0)
    cmet = cone_metrics(graph, p)
    gmet = graph_metrics(graph)
    kmax, n_cones, n_size_groups, n_topology_groups, largest_size, largest_topo = grouping_stats(graph, p)
    case = f"{family}_n{n}_p{p}"
    print(f"A6 case={case} objective={objective} kmax={kmax} cones={n_cones}", flush=True)
    modes = [
        ("no_batching_per_cone", True, "size", np.complex64, np.float32, "one cone per batch"),
        ("size_batching", False, "size", np.complex64, np.float32, "default size-batched LC"),
        ("topology_grouping", False, "topology", np.complex64, np.float32, "groups by topology signature; can fragment"),
        ("mixed_precision_c64_f32", False, "size", np.complex64, np.float32, "complex64 state and float32 cost"),
        ("high_precision_c128_f64", False, "size", np.complex128, np.float64, "complex128 state and float64 cost"),
    ]
    rows: list[A6Row] = []
    naive_t = float("nan")
    size_t = float("nan")
    size_val = float("nan")
    for mode, naive, group_by, cdtype, fdtype, note in modes:
        if cmet["kmax"] > args.max_k:
            status, value, seconds, peak = f"skipped_kmax_{cmet['kmax']}_over_{args.max_k}", float("nan"), 0.0, 0.0
        else:
            try:
                stat = lightcone_expectation(
                    graph,
                    gammas,
                    betas,
                    p=p,
                    prefer_gpu=True,
                    max_k=args.max_k,
                    max_batch_states=args.max_batch_states,
                    naive=naive,
                    group_by=group_by,
                    complex_dtype=cdtype,
                    float_dtype=fdtype,
                )
                status, value, seconds, peak = stat.status, stat.value, stat.seconds, stat.peak_pool_bytes / 1024**2
            except Exception as exc:
                status, value, seconds, peak = f"failed:{type(exc).__name__}", float("nan"), 0.0, 0.0
                note = note + f"; {str(exc)[:120]}"
        if mode == "no_batching_per_cone" and status == "ok":
            naive_t = seconds
        if mode == "size_batching" and status == "ok":
            size_t, size_val = seconds, value
        launch_proxy = n_cones if naive else (n_topology_groups if group_by == "topology" else n_size_groups)
        fragmentation = (n_topology_groups / max(1, n_size_groups)) if group_by == "topology" else 1.0
        rows.append(
            A6Row(
                case=case,
                objective=objective,
                family=family,
                n=n,
                p=p,
                m=int(gmet["m"]),
                kmax=int(cmet["kmax"]),
                total_cone_states=int(cmet["total_cone_states"]),
                n_cones=n_cones,
                n_size_groups=n_size_groups,
                n_topology_groups=n_topology_groups,
                largest_size_group=largest_size,
                largest_topology_group=largest_topo,
                mode=mode,
                status=status,
                value=value,
                seconds=seconds,
                speedup_vs_naive=naive_t / seconds if status == "ok" and math.isfinite(naive_t) and seconds > 0 else float("nan"),
                speedup_vs_size=size_t / seconds if status == "ok" and math.isfinite(size_t) and seconds > 0 else float("nan"),
                abs_error_vs_size=abs(value - size_val) if status == "ok" and math.isfinite(size_val) else float("nan"),
                peak_mb=peak,
                kernel_launch_proxy=launch_proxy,
                fragmentation_ratio=fragmentation,
                notes=note,
            )
        )
    return rows


def write_md(rows: list[A6Row], path: Path) -> None:
    lines = [
        "# A6 Ablation Beyond Size Batching",
        "",
        "Kernel-launch proxy is the number of cone batches implied by the grouping scheme; it is not a hardware counter.",
        "",
        "| Case | Mode | Status | Seconds | speedup vs naive | speedup vs size | launch proxy | fragmentation | peak MB | note |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r.case} | {r.mode} | {r.status} | {r.seconds:.4g} | {r.speedup_vs_naive:.3g} | {r.speedup_vs_size:.3g} | "
            f"{r.kernel_launch_proxy} | {r.fragmentation_ratio:.3g} | {r.peak_mb:.4g} | {r.notes} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "benchmark_suite_20260704" / "A6_ablation")
    parser.add_argument("--max-k", type=int, default=24)
    parser.add_argument("--max-batch-states", type=int, default=1 << 19)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    specs = case_specs()[:3] if args.quick else case_specs()
    rows: list[A6Row] = []
    for spec in specs:
        rows.extend(run_case(*spec, args=args))
    csv_path = args.out_dir / "A6_ablation.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
    write_md(rows, args.out_dir / "A6_ablation.md")
    print(f"WROTE {csv_path}")


if __name__ == "__main__":
    main()
