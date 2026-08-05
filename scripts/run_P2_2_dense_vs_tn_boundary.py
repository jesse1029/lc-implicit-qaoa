from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from lcqaoa.graphs import WeightedGraph
from lcqaoa.qaoa import full_state_expectation
from benchmark_common import params_for_depth


PREFIX = Path.home() / "lc_implicit_qaoa_peers"
QTENSOR_PY = PREFIX / "venvs" / "qtensor-py38" / "bin" / "python"
if not QTENSOR_PY.exists():
    QTENSOR_PY = PREFIX / "venvs" / "qtensor-py310" / "bin" / "python"


@dataclass
class BoundaryRow:
    family: str
    k: int
    p: int
    seed: int
    m: int
    density: float
    treewidth_min_fill: float
    dense_status: str
    dense_seconds: float
    dense_peak_mb: float
    qtensor_status: str
    qtensor_seconds: float
    qtensor_wall_seconds: float
    qtensor_over_dense: float
    abs_value_diff: float
    recommended_engine: str
    notes: str


def make_graph(family: str, k: int, seed: int) -> WeightedGraph:
    import networkx as nx

    rng = np.random.default_rng(seed)
    if family == "path":
        g = nx.path_graph(k)
    elif family == "cycle":
        g = nx.cycle_graph(k)
    elif family == "tree":
        if hasattr(nx, "random_labeled_tree"):
            g = nx.random_labeled_tree(k, seed=seed)
        else:
            g = nx.random_tree(k, seed=seed)
    elif family == "grid":
        side = int(math.ceil(math.sqrt(k)))
        g0 = nx.grid_2d_graph(side, side)
        g = nx.convert_node_labels_to_integers(g0.subgraph(list(g0.nodes())[:k]).copy())
    elif family == "er_sparse":
        g = nx.erdos_renyi_graph(k, min(0.35, 2.0 / max(k - 1, 1)), seed=seed)
    elif family == "er_mid":
        g = nx.erdos_renyi_graph(k, 0.25, seed=seed)
    elif family == "er_dense":
        g = nx.erdos_renyi_graph(k, 0.55, seed=seed)
    elif family == "clique":
        g = nx.complete_graph(k)
    else:
        raise ValueError(family)
    if g.number_of_edges() == 0 and k >= 2:
        g.add_edge(0, 1)
    # QTensor's MaxCut energy path is graph-structural; keep unit weights so
    # value checks are not confounded by peer-specific weight conventions.
    edges = tuple((int(i), int(j), 1.0) for i, j in g.edges())
    return WeightedGraph(k, edges, objective="maxcut")


def treewidth(graph: WeightedGraph) -> float:
    try:
        import networkx as nx
        from networkx.algorithms.approximation import treewidth_min_fill_in

        g = nx.Graph()
        g.add_nodes_from(range(graph.n))
        g.add_edges_from((int(i), int(j)) for i, j, _ in graph.edges)
        width, _ = treewidth_min_fill_in(g)
        return float(width)
    except Exception:
        return float("nan")


def run_qtensor(graph: WeightedGraph, gammas, betas, work_dir: Path, timeout: int) -> dict:
    if not QTENSOR_PY.exists():
        return {"status": f"missing_python:{QTENSOR_PY}", "seconds": 0.0, "wall_seconds": 0.0, "value": float("nan"), "notes": ""}
    case = {
        "n": graph.n,
        "edges": [list(e) for e in graph.edges],
        "fields": [],
        "objective": graph.objective,
        "gammas": list(map(float, gammas)),
        "betas": list(map(float, betas)),
    }
    case_path = work_dir / f"case_{abs(hash(json.dumps(case, sort_keys=True)))}.json"
    out_path = work_dir / f"qtensor_{case_path.stem}.json"
    case_path.write_text(json.dumps(case, indent=2), encoding="utf-8")
    cmd = [
        str(QTENSOR_PY),
        str(ROOT / "scripts" / "external_peer_runner.py"),
        "--method",
        "qtensor_cpu",
        "--case",
        str(case_path),
        "--out",
        str(out_path),
        "--qtensor-transform",
        "paper_qiskit_inverse",
    ]
    import time

    t0 = time.perf_counter()
    try:
        cp = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
        wall = time.perf_counter() - t0
    except subprocess.TimeoutExpired as exc:
        return {"status": f"timeout_{timeout}s", "seconds": 0.0, "wall_seconds": float(timeout), "value": float("nan"), "notes": str(exc)[:500]}
    if out_path.exists():
        data = json.loads(out_path.read_text(encoding="utf-8"))
    else:
        return {"status": f"process_failed:{cp.returncode}", "seconds": 0.0, "wall_seconds": wall, "value": float("nan"), "notes": cp.stdout[-500:]}
    data["wall_seconds"] = wall
    return data


def run_case(family: str, k: int, p: int, seed_id: int, args) -> BoundaryRow:
    seed = 930000 + 1009 * seed_id + 37 * k + 131 * p
    graph = make_graph(family, k, seed)
    gammas, betas = params_for_depth(p, seed=seed_id)
    tw = treewidth(graph)
    density = 2.0 * graph.m / max(k * (k - 1), 1)
    print(f"P2-2 family={family} k={k} p={p} seed={seed_id} m={graph.m} tw={tw}", flush=True)
    try:
        dense = full_state_expectation(
            graph,
            gammas,
            betas,
            method="precompute",
            prefer_gpu=True,
            max_qubits=args.max_dense_k,
            complex_dtype=np.complex64,
            float_dtype=np.float32,
        )
        dstatus, dseconds, dpeak, dvalue = dense.status, dense.seconds, dense.peak_pool_bytes / 1024**2, dense.value
    except Exception as exc:
        dstatus, dseconds, dpeak, dvalue = f"failed:{type(exc).__name__}", 0.0, 0.0, float("nan")
    qt = run_qtensor(graph, gammas, betas, args.work_dir, args.timeout)
    qstatus = str(qt.get("status", ""))
    qsec = float(qt.get("seconds", 0.0) or 0.0)
    qwall = float(qt.get("wall_seconds", 0.0) or 0.0)
    qvalue = float(qt.get("value", float("nan")))
    if dstatus == "ok" and qstatus == "ok" and dseconds > 0 and qsec > 0:
        ratio = qsec / dseconds
        rec = "dense_statevector" if dseconds <= qsec else "qtensor"
    else:
        ratio = float("nan")
        rec = "dense_statevector" if dstatus == "ok" else ("qtensor" if qstatus == "ok" else "neither")
    diff = abs(dvalue - qvalue) if math.isfinite(dvalue) and math.isfinite(qvalue) else float("nan")
    return BoundaryRow(
        family=family,
        k=k,
        p=p,
        seed=seed_id,
        m=graph.m,
        density=density,
        treewidth_min_fill=tw,
        dense_status=dstatus,
        dense_seconds=dseconds,
        dense_peak_mb=dpeak,
        qtensor_status=qstatus,
        qtensor_seconds=qsec,
        qtensor_wall_seconds=qwall,
        qtensor_over_dense=ratio,
        abs_value_diff=diff,
        recommended_engine=rec,
        notes=str(qt.get("notes", ""))[:160],
    )


def write_md(rows: list[BoundaryRow], path: Path) -> None:
    lines = [
        "# P2-2 Dense Statevector vs Tensor-Network Boundary",
        "",
        "Rows compare an exact dense local-statevector subproblem against QTensor CPU when the peer environment is available.",
        "",
        "| Family | k | p | m | tw | Dense s | QTensor s | QTensor/Dense | Recommendation |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r.family} | {r.k} | {r.p} | {r.m} | {r.treewidth_min_fill:.3g} | {r.dense_seconds:.4g} | "
            f"{r.qtensor_seconds:.4g} | {r.qtensor_over_dense:.3g} | {r.recommended_engine} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "aaai27_required_experiments_20260710" / "P2_2_dense_vs_tn_boundary")
    parser.add_argument("--families", nargs="+", default=["path", "tree", "er_sparse", "er_mid", "er_dense", "grid", "clique"])
    parser.add_argument("--ks", nargs="+", type=int, default=[8, 10, 12, 14, 16, 18, 20, 22])
    parser.add_argument("--ps", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--max-dense-k", type=int, default=24)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if args.quick:
        args.families = args.families[:3]
        args.ks = args.ks[:3]
        args.ps = args.ps[:1]
        args.seeds = 1
        args.timeout = min(args.timeout, 60)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.work_dir = args.out_dir / "cases"
    args.work_dir.mkdir(exist_ok=True)
    rows: list[BoundaryRow] = []
    csv_path = args.out_dir / "P2_2_dense_vs_tn_boundary.csv"
    for family in args.families:
        for k in args.ks:
            if family == "grid" and int(math.ceil(math.sqrt(k))) ** 2 < k:
                continue
            if family == "clique" and k > 18:
                continue
            for p in args.ps:
                for seed_id in range(args.seeds):
                    rows.append(run_case(family, k, p, seed_id, args))
                    with csv_path.open("w", newline="", encoding="utf-8") as f:
                        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
                        writer.writeheader()
                        for row in rows:
                            writer.writerow(asdict(row))
                    write_md(rows, args.out_dir / "P2_2_dense_vs_tn_boundary.md")
    print(f"WROTE {csv_path}")


if __name__ == "__main__":
    main()
