from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))


QTENSOR_PY_CANDIDATES = (
    Path.home() / "lc_implicit_qaoa_peers" / "venvs" / "qtensor-py38" / "bin" / "python",
    Path.home() / "lc_implicit_qaoa_peers" / "venvs" / "qtensor-py310" / "bin" / "python",
)


@dataclass
class ResultRow:
    case: str
    family: str
    n: int
    p: int
    seed: int
    m: int
    treewidth_min_fill: int
    kmax: int
    total_cone_states: int
    profile_route: str
    gpu_execution_order: str
    lc_status: str
    lc_preprocess_seconds: float
    lc_first_seconds: float
    lc_steady_seconds: float
    lc_peak_pool_mb: float
    global_status: str
    global_preprocess_seconds: float
    global_first_seconds: float
    global_steady_seconds: float
    global_peak_pool_mb: float
    qtensor_status: str
    qtensor_process_startup_seconds: float
    qtensor_import_seconds: float
    qtensor_first_seconds: float
    qtensor_first_planning_seconds: float
    qtensor_first_contraction_seconds: float
    qtensor_steady_seconds: float
    qtensor_steady_planning_seconds: float
    qtensor_steady_contraction_seconds: float
    value_abs_diff_lc_qtensor: float
    oracle_q1: str
    oracle_q10: str
    oracle_q100: str
    route_correct_q1: bool
    route_correct_q10: bool
    route_correct_q100: bool
    route_regret_q1: float
    route_regret_q10: float
    route_regret_q100: float
    notes: str


def query_angles(p: int, seed: int, count: int) -> list[tuple[list[float], list[float]]]:
    rng = np.random.default_rng(780000 + 101 * seed + 17 * p)
    queries = []
    for query_id in range(count):
        gammas = [
            0.18 + 0.055 * layer + 0.006 * query_id + float(rng.uniform(-0.01, 0.01))
            for layer in range(p)
        ]
        betas = [
            0.35 - 0.035 * layer - 0.004 * query_id + float(rng.uniform(-0.008, 0.008))
            for layer in range(p)
        ]
        queries.append((gammas, betas))
    return queries


def make_graph(family: str, n: int, seed: int):
    import networkx as nx

    from lcqaoa.graphs import WeightedGraph

    if family == "3regular":
        graph = nx.random_regular_graph(3, n, seed=seed)
    elif family == "path":
        graph = nx.path_graph(n)
    elif family == "grid":
        side = int(round(math.sqrt(n)))
        if side * side != n:
            raise ValueError(f"grid n must be a square, got {n}")
        graph = nx.convert_node_labels_to_integers(nx.grid_2d_graph(side, side))
    elif family == "star":
        graph = nx.star_graph(n - 1)
    elif family == "balanced_tree":
        graph = nx.balanced_tree(2, int(round(math.log2(n + 1))) - 1)
        if graph.number_of_nodes() != n:
            raise ValueError(f"balanced_tree expected 2^(h+1)-1 nodes, got {n}")
    else:
        raise ValueError(family)
    edges = tuple((int(i), int(j), 1.0) for i, j in graph.edges())
    return WeightedGraph(n, edges, objective="maxcut")


def min_fill_treewidth(graph) -> int:
    import networkx as nx
    from networkx.algorithms.approximation import treewidth_min_fill_in

    nx_graph = nx.Graph()
    nx_graph.add_nodes_from(range(graph.n))
    nx_graph.add_edges_from((int(i), int(j)) for i, j, _ in graph.edges)
    width, _ = treewidth_min_fill_in(nx_graph)
    return int(width)


def prepare_lc(graph, p: int, max_k: int, max_batch_states: int):
    from lcqaoa.lightcone import _split_batches, extract_lightcones

    started = time.perf_counter()
    problems = extract_lightcones(graph, p)
    kmax = max((problem.k for problem in problems), default=0)
    total_cone_states = int(sum(1 << problem.k for problem in problems))
    if kmax > max_k:
        return None, time.perf_counter() - started, kmax, total_cone_states, f"rejected_kmax_{kmax}_over_{max_k}"
    groups: dict[int, list] = {}
    for problem in problems:
        groups.setdefault(problem.k, []).append(problem)
    batches = []
    for k in sorted(groups):
        batches.extend(_split_batches(groups[k], max_batch_states))
    return batches, time.perf_counter() - started, kmax, total_cone_states, "ok"


def evaluate_lc_queries(graph, p: int, queries, max_k: int, max_batch_states: int) -> dict:
    from lcqaoa.backend import get_backend
    from lcqaoa.lightcone import _evaluate_batch

    batches, prep, kmax, total_states, status = prepare_lc(graph, p, max_k, max_batch_states)
    if status != "ok":
        return {
            "status": status,
            "preprocess_seconds": prep,
            "query_seconds": [],
            "values": [],
            "peak_pool_mb": 0.0,
            "kmax": kmax,
            "total_cone_states": total_states,
        }
    backend = get_backend(True)
    backend.free_memory_pool()
    xp = backend.xp
    timings = []
    values = []
    for gammas, betas in queries:
        started = time.perf_counter()
        value = float(getattr(graph, "constant_offset", 0.0))
        for batch in batches:
            value += _evaluate_batch(
                batch,
                gammas,
                betas,
                graph.objective,
                xp,
                np.complex64,
                np.float32,
            )
        backend.sync()
        timings.append(time.perf_counter() - started)
        values.append(float(value))
    return {
        "status": "ok",
        "preprocess_seconds": prep,
        "query_seconds": timings,
        "values": values,
        "peak_pool_mb": backend.memory_pool_bytes() / 1024**2,
        "kmax": kmax,
        "total_cone_states": total_states,
    }


def evaluate_global_queries(graph, queries, max_n: int) -> dict:
    from lcqaoa.backend import get_backend
    from lcqaoa.qaoa import (
        apply_cost_precomputed,
        apply_mixer_inplace,
        cost_table,
        expectation_from_state,
    )

    if graph.n > max_n:
        return {
            "status": f"not_run_n_{graph.n}_over_{max_n}",
            "preprocess_seconds": 0.0,
            "query_seconds": [],
            "values": [],
            "peak_pool_mb": 0.0,
        }
    backend = get_backend(True)
    backend.free_memory_pool()
    xp = backend.xp
    started = time.perf_counter()
    cost = cost_table(
        graph.n,
        graph.edges,
        graph.fields,
        graph.objective,
        xp,
        np.float32,
    )
    backend.sync()
    prep = time.perf_counter() - started
    nstates = 1 << graph.n
    timings = []
    values = []
    try:
        for gammas, betas in queries:
            started = time.perf_counter()
            psi = xp.empty(nstates, dtype=np.complex64)
            psi.fill(1.0 / math.sqrt(nstates))
            for gamma, beta in zip(gammas, betas):
                apply_cost_precomputed(psi, cost, gamma, xp)
                apply_mixer_inplace(psi, graph.n, beta, xp)
            value = expectation_from_state(psi, cost, xp)
            backend.sync()
            timings.append(time.perf_counter() - started)
            values.append(float(value + getattr(graph, "constant_offset", 0.0)))
    except Exception as exc:
        return {
            "status": f"failed:{type(exc).__name__}",
            "preprocess_seconds": prep,
            "query_seconds": timings,
            "values": values,
            "peak_pool_mb": backend.memory_pool_bytes() / 1024**2,
        }
    return {
        "status": "ok",
        "preprocess_seconds": prep,
        "query_seconds": timings,
        "values": values,
        "peak_pool_mb": backend.memory_pool_bytes() / 1024**2,
    }


def qtensor_angles(gammas, betas):
    return (
        [-float(gamma) / (2.0 * math.pi) for gamma in gammas],
        [float(beta) / math.pi for beta in betas],
    )


def qtensor_worker(case_path: Path, out_path: Path) -> None:
    process_started = time.perf_counter()
    import_started = time.perf_counter()
    import networkx as nx
    import qtensor
    import qtree

    import_seconds = time.perf_counter() - import_started
    case = json.loads(case_path.read_text(encoding="utf-8"))
    graph = nx.Graph()
    graph.add_nodes_from(range(int(case["n"])))
    graph.add_edges_from((int(i), int(j)) for i, j, _ in case["edges"])
    composer = getattr(
        qtensor,
        "DefaultQAOAComposer",
        getattr(qtensor, "QtreeQAOAComposer", getattr(qtensor, "QAOAComposer")),
    )
    simulator = qtensor.QAOAQtreeSimulator(composer)

    def edge_value(edge, gammas, betas, peo=None):
        stage = {}
        started = time.perf_counter()
        circuit = simulator._edge_energy_circuit(graph, gammas, betas, edge)
        stage["circuit_seconds"] = time.perf_counter() - started
        started = time.perf_counter()
        simulator._new_circuit(circuit)
        simulator._create_buckets()
        simulator._set_free_qubits([])
        stage["bucket_build_seconds"] = time.perf_counter() - started
        if peo is None:
            started = time.perf_counter()
            simulator._optimize_buckets()
            stage["planning_seconds"] = time.perf_counter() - started
            peo = list(simulator.peo)
        else:
            simulator.peo = peo
            stage["planning_seconds"] = 0.0
        started = time.perf_counter()
        simulator._reorder_buckets()
        slice_dict = simulator._get_slice_dict()
        sliced_buckets = simulator.tn.slice(slice_dict)
        stage["reorder_slice_seconds"] = time.perf_counter() - started
        started = time.perf_counter()
        result = qtree.optimizer.bucket_elimination(
            sliced_buckets,
            simulator.bucket_backend.process_bucket,
            n_var_nosum=len(simulator.tn.free_vars),
        )
        result_data = simulator.bucket_backend.get_result_data(result).flatten()
        stage["contraction_seconds"] = time.perf_counter() - started
        return complex(result_data[0]), peo, stage

    cached_peos = None
    rows = []
    for query_id, query in enumerate(case["queries"]):
        gammas, betas = qtensor_angles(query["gammas"], query["betas"])
        started = time.perf_counter()
        total = 0.0 + 0.0j
        next_peos = []
        stage_totals = {
            "circuit_seconds": 0.0,
            "bucket_build_seconds": 0.0,
            "planning_seconds": 0.0,
            "reorder_slice_seconds": 0.0,
            "contraction_seconds": 0.0,
        }
        for edge_id, edge in enumerate(graph.edges()):
            peo = None if cached_peos is None else cached_peos[edge_id]
            value, planned, stage = edge_value(edge, gammas, betas, peo)
            total += value
            next_peos.append(planned)
            for key in stage_totals:
                stage_totals[key] += float(stage[key])
        cached_peos = next_peos
        value = float(simulator._post_process_energy(graph, total))
        rows.append(
            {
                "query_id": query_id,
                "value": value,
                "end_to_end_seconds": time.perf_counter() - started,
                **stage_totals,
            }
        )
    payload = {
        "status": "ok",
        "import_seconds": import_seconds,
        "worker_internal_seconds": time.perf_counter() - process_started,
        "queries": rows,
        "notes": "QTensor CPU; one PEO per edge planned on query 0 and reused for all later angle queries",
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_qtensor(case: dict, out_dir: Path, timeout: int) -> dict:
    qtensor_python = next((path for path in QTENSOR_PY_CANDIDATES if path.exists()), None)
    if qtensor_python is None:
        return {"status": "missing_qtensor_python", "queries": []}
    case_path = out_dir / f"{case['case']}_seed{case['seed']}.case.json"
    out_path = out_dir / f"{case['case']}_seed{case['seed']}.qtensor.json"
    case_path.write_text(json.dumps(case, indent=2), encoding="utf-8")
    command = [
        str(qtensor_python),
        str(Path(__file__).resolve()),
        "--qtensor-worker",
        "--case",
        str(case_path),
        "--worker-out",
        str(out_path),
    ]
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            env=env,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "status": f"timeout_{timeout}s",
            "queries": [],
            "process_wall_seconds": float(timeout),
            "notes": str(exc)[:500],
        }
    wall = time.perf_counter() - started
    if completed.returncode != 0 or not out_path.exists():
        return {
            "status": f"failed_rc_{completed.returncode}",
            "queries": [],
            "process_wall_seconds": wall,
            "notes": completed.stdout[-1000:],
        }
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    payload["process_wall_seconds"] = wall
    payload["process_startup_seconds"] = max(
        0.0, wall - float(payload.get("worker_internal_seconds", 0.0))
    )
    return payload


def first_and_steady(payload: dict, key: str = "query_seconds"):
    values = payload.get(key, [])
    if not values:
        return float("nan"), float("nan")
    first = float(values[0])
    steady_values = [float(value) for value in values[1:]]
    return first, float(median(steady_values)) if steady_values else first


def total_time(preprocess: float, first: float, steady: float, queries: int) -> float:
    if not all(math.isfinite(value) for value in (preprocess, first, steady)):
        return float("nan")
    return preprocess + first + max(0, queries - 1) * steady


def best_backend(times: dict[str, float]) -> str:
    valid = {name: value for name, value in times.items() if math.isfinite(value) and value >= 0.0}
    return min(valid, key=valid.get) if valid else "none"


def profile_route(n: int, kmax: int, total_states: int, treewidth_value: int) -> str:
    # Frozen before this follow-up: the global cutoff is the lower edge of the
    # measured n=22--24 crossover, and the LC limits are the manuscript's
    # existing p=2 execution band. Treewidth is consulted only after LC rejects.
    if n <= 22:
        return "global"
    if kmax <= 24 and total_states <= 20_000_000:
        return "lc"
    if treewidth_value <= 4:
        return "qtensor"
    return "reject"


def selected_time(route: str, times: dict[str, float]) -> float:
    return float(times.get(route, float("nan")))


def make_row(case_name: str, family: str, n: int, p: int, seed_id: int, args) -> ResultRow:
    graph_seed = 990000 + 1009 * seed_id + 37 * n + 131 * p
    graph = make_graph(family, n, graph_seed)
    treewidth_value = min_fill_treewidth(graph)
    queries = query_angles(p, seed_id, args.query_count)
    if args.gpu_order == "alternate":
        gpu_order = "lc_then_global" if seed_id % 2 == 0 else "global_then_lc"
    else:
        gpu_order = args.gpu_order
    if gpu_order == "global_then_lc":
        global_result = evaluate_global_queries(graph, queries, args.global_max_n)
        lc = evaluate_lc_queries(graph, p, queries, args.max_k, args.max_batch_states)
    else:
        lc = evaluate_lc_queries(graph, p, queries, args.max_k, args.max_batch_states)
        global_result = evaluate_global_queries(graph, queries, args.global_max_n)
    case = {
        "case": case_name,
        "family": family,
        "n": n,
        "p": p,
        "seed": seed_id,
        "edges": [list(edge) for edge in graph.edges],
        "queries": [
            {"gammas": list(map(float, gammas)), "betas": list(map(float, betas))}
            for gammas, betas in queries
        ],
    }
    qtensor = (
        {"status": "skipped_by_protocol", "queries": []}
        if args.skip_qtensor
        else run_qtensor(case, args.out_dir / "cases", args.timeout)
    )

    lc_first, lc_steady = first_and_steady(lc)
    global_first, global_steady = first_and_steady(global_result)
    qt_queries = qtensor.get("queries", [])
    qt_first, qt_steady = first_and_steady(
        {"query_seconds": [row["end_to_end_seconds"] for row in qt_queries]}
    )
    qt_plan_first, qt_plan_steady = first_and_steady(
        {"query_seconds": [row["planning_seconds"] for row in qt_queries]}
    )
    qt_contract_first, qt_contract_steady = first_and_steady(
        {"query_seconds": [row["contraction_seconds"] for row in qt_queries]}
    )
    qt_preprocess = float(qtensor.get("process_startup_seconds", float("nan"))) + float(
        qtensor.get("import_seconds", float("nan"))
    )
    route = profile_route(
        graph.n,
        int(lc["kmax"]),
        int(lc["total_cone_states"]),
        treewidth_value,
    )
    query_counts = (1, 10, 100)
    all_times: dict[int, dict[str, float]] = {}
    for count in query_counts:
        all_times[count] = {
            "lc": total_time(float(lc["preprocess_seconds"]), lc_first, lc_steady, count)
            if lc["status"] == "ok"
            else float("nan"),
            "global": total_time(
                float(global_result["preprocess_seconds"]),
                global_first,
                global_steady,
                count,
            )
            if global_result["status"] == "ok"
            else float("nan"),
            "qtensor": total_time(qt_preprocess, qt_first, qt_steady, count)
            if qtensor.get("status") == "ok"
            else float("nan"),
        }
    oracles = {count: best_backend(all_times[count]) for count in query_counts}
    correctness = {count: route == oracles[count] for count in query_counts}
    regrets = {}
    for count in query_counts:
        oracle_time = all_times[count].get(oracles[count], float("nan"))
        route_time = selected_time(route, all_times[count])
        regrets[count] = (
            route_time / oracle_time
            if math.isfinite(route_time) and math.isfinite(oracle_time) and oracle_time > 0
            else float("nan")
        )
    diff = float("nan")
    if lc.get("values") and qt_queries:
        diff = abs(float(lc["values"][0]) - float(qt_queries[0]["value"]))

    return ResultRow(
        case=case_name,
        family=family,
        n=n,
        p=p,
        seed=seed_id,
        m=graph.m,
        treewidth_min_fill=treewidth_value,
        kmax=int(lc["kmax"]),
        total_cone_states=int(lc["total_cone_states"]),
        profile_route=route,
        gpu_execution_order=gpu_order,
        lc_status=str(lc["status"]),
        lc_preprocess_seconds=float(lc["preprocess_seconds"]),
        lc_first_seconds=lc_first,
        lc_steady_seconds=lc_steady,
        lc_peak_pool_mb=float(lc["peak_pool_mb"]),
        global_status=str(global_result["status"]),
        global_preprocess_seconds=float(global_result["preprocess_seconds"]),
        global_first_seconds=global_first,
        global_steady_seconds=global_steady,
        global_peak_pool_mb=float(global_result["peak_pool_mb"]),
        qtensor_status=str(qtensor.get("status", "")),
        qtensor_process_startup_seconds=float(
            qtensor.get("process_startup_seconds", float("nan"))
        ),
        qtensor_import_seconds=float(qtensor.get("import_seconds", float("nan"))),
        qtensor_first_seconds=qt_first,
        qtensor_first_planning_seconds=qt_plan_first,
        qtensor_first_contraction_seconds=qt_contract_first,
        qtensor_steady_seconds=qt_steady,
        qtensor_steady_planning_seconds=qt_plan_steady,
        qtensor_steady_contraction_seconds=qt_contract_steady,
        value_abs_diff_lc_qtensor=diff,
        oracle_q1=oracles[1],
        oracle_q10=oracles[10],
        oracle_q100=oracles[100],
        route_correct_q1=correctness[1],
        route_correct_q10=correctness[10],
        route_correct_q100=correctness[100],
        route_regret_q1=regrets[1],
        route_regret_q10=regrets[10],
        route_regret_q100=regrets[100],
        notes=(
            "Objective-only targeted dispatch check; QTensor CPU plan is reused across "
            "distinct angle queries; global cost table and LC cones are also reused."
        ),
    )


def summarize(rows: list[ResultRow], out_dir: Path) -> None:
    summary_rows = []
    for count in (1, 10, 100):
        correctness = [getattr(row, f"route_correct_q{count}") for row in rows]
        regrets = [
            getattr(row, f"route_regret_q{count}")
            for row in rows
            if math.isfinite(getattr(row, f"route_regret_q{count}"))
        ]
        summary_rows.append(
            {
                "query_count": count,
                "cells": len(rows),
                "route_accuracy": sum(correctness) / len(correctness) if correctness else float("nan"),
                "median_route_regret": median(regrets) if regrets else float("nan"),
                "max_route_regret": max(regrets) if regrets else float("nan"),
            }
        )
    with (out_dir / "dispatch_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    lines = [
        "# Targeted QTensor Plan-Reuse Dispatch Check",
        "",
        "QTensor plans each edge contraction on the first query and reuses the cached ordering for later, distinct angle queries. LC reuses extracted cones and batches; the global route reuses its diagonal cost table.",
        "",
        "| Queries | Route accuracy | Median regret | Max regret |",
        "|---:|---:|---:|---:|",
    ]
    for item in summary_rows:
        lines.append(
            f"| {item['query_count']} | {item['route_accuracy']:.3f} | "
            f"{item['median_route_regret']:.3f} | {item['max_route_regret']:.3f} |"
        )
    lines.extend(
        [
            "",
            "| Case | n | p | tw | kmax | Route | Oracle q=1/10/100 | LC steady | QTensor steady | Global steady |",
            "|---|---:|---:|---:|---:|---|---|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row.case} s{row.seed} | {row.n} | {row.p} | {row.treewidth_min_fill} | "
            f"{row.kmax} | {row.profile_route} | {row.oracle_q1}/{row.oracle_q10}/{row.oracle_q100} | "
            f"{row.lc_steady_seconds:.4g} | {row.qtensor_steady_seconds:.4g} | "
            f"{row.global_steady_seconds:.4g} |"
        )
    (out_dir / "dispatch_plan_reuse_results.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def capture_environment(out_dir: Path) -> None:
    commands = {
        "hostname": ["hostname"],
        "uname": ["uname", "-a"],
        "nvidia_smi": [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,driver_version",
            "--format=csv,noheader",
        ],
    }
    payload = {"python": sys.version, "executable": sys.executable}
    for name, command in commands.items():
        try:
            completed = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=30,
            )
            payload[name] = completed.stdout.strip()
        except Exception as exc:
            payload[name] = f"{type(exc).__name__}: {exc}"
    (out_dir / "environment.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qtensor-worker", action="store_true")
    parser.add_argument("--case", type=Path)
    parser.add_argument("--worker-out", type=Path)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "results" / "followup_20260723" / "dispatch_plan_reuse",
    )
    parser.add_argument("--query-count", type=int, default=6)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--max-k", type=int, default=24)
    parser.add_argument("--max-batch-states", type=int, default=1 << 21)
    parser.add_argument("--global-max-n", type=int, default=24)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--skip-qtensor", action="store_true")
    parser.add_argument(
        "--gpu-order",
        choices=("lc_then_global", "global_then_lc", "alternate"),
        default="lc_then_global",
    )
    parser.add_argument(
        "--case-filter",
        action="append",
        default=[],
        help="Run only the named case; repeat the flag to select multiple cases.",
    )
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.qtensor_worker:
        qtensor_worker(args.case, args.worker_out)
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "cases").mkdir(exist_ok=True)
    capture_environment(args.out_dir)
    cases = [
        ("small_global", "3regular", 20, 2),
        ("near_crossover", "3regular", 24, 2),
        ("moderate_dense", "3regular", 64, 2),
        ("smallcone_lowwidth", "path", 64, 3),
        ("moderate_grid", "grid", 25, 2),
        ("near_tensor_boundary", "balanced_tree", 31, 3),
        ("largecone_lowwidth_star", "star", 48, 2),
        ("largecone_lowwidth_tree", "balanced_tree", 63, 3),
    ]
    if args.case_filter:
        selected = set(args.case_filter)
        cases = [case for case in cases if case[0] in selected]
        missing = selected.difference(case[0] for case in cases)
        if missing:
            raise ValueError(f"unknown case filters: {sorted(missing)}")
    if args.smoke:
        cases = cases[:3]
        args.seeds = 1
        args.query_count = 2
        args.timeout = min(args.timeout, 180)

    rows = []
    csv_path = args.out_dir / "dispatch_plan_reuse_raw.csv"
    for case_name, family, n, p in cases:
        for seed_id in range(args.seeds):
            print(
                f"RUN case={case_name} family={family} n={n} p={p} seed={seed_id}",
                flush=True,
            )
            row = make_row(case_name, family, n, p, seed_id, args)
            rows.append(row)
            with csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
                writer.writeheader()
                for item in rows:
                    writer.writerow(asdict(item))
            summarize(rows, args.out_dir)
            print(
                f"DONE route={row.profile_route} oracle100={row.oracle_q100} "
                f"lc={row.lc_status} qtensor={row.qtensor_status}",
                flush=True,
            )
    print(f"WROTE {csv_path}")


if __name__ == "__main__":
    main()
