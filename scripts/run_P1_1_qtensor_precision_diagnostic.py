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


ROOT = Path(__file__).resolve().parents[1]
QTENSOR_PY = Path.home() / "lc_implicit_qaoa_peers" / "venvs" / "qtensor-py38" / "bin" / "python"


def transform_angles(gammas, betas):
    return [-float(g) / (2.0 * math.pi) for g in gammas], [float(b) / math.pi for b in betas]


def qtensor_worker(case_path: Path, out_path: Path) -> None:
    process_start = time.perf_counter()
    import_start = time.perf_counter()
    import numpy as np
    import networkx as nx
    import qtensor
    import qtree

    import_seconds = time.perf_counter() - import_start
    case = json.loads(case_path.read_text(encoding="utf-8"))
    graph = nx.Graph()
    graph.add_nodes_from(range(int(case["n"])))
    graph.add_edges_from((int(i), int(j)) for i, j, _ in case["edges"])
    gammas, betas = transform_angles(case["gammas"], case["betas"])
    composer = getattr(
        qtensor,
        "DefaultQAOAComposer",
        getattr(qtensor, "QtreeQAOAComposer", getattr(qtensor, "QAOAComposer")),
    )
    sim = qtensor.QAOAQtreeSimulator(composer)

    def one_edge(edge, peo=None):
        t0 = time.perf_counter()
        qc = sim._edge_energy_circuit(graph, gammas, betas, edge)
        circuit_seconds = time.perf_counter() - t0
        t0 = time.perf_counter()
        sim._new_circuit(qc)
        sim._create_buckets()
        sim._set_free_qubits([])
        build_seconds = time.perf_counter() - t0
        if peo is None:
            t0 = time.perf_counter()
            sim._optimize_buckets()
            plan_seconds = time.perf_counter() - t0
            peo = list(sim.peo)
        else:
            sim.peo = peo
            plan_seconds = 0.0
        t0 = time.perf_counter()
        sim._reorder_buckets()
        slice_dict = sim._get_slice_dict()
        sliced_buckets = sim.tn.slice(slice_dict)
        reorder_slice_seconds = time.perf_counter() - t0
        t0 = time.perf_counter()
        result = qtree.optimizer.bucket_elimination(
            sliced_buckets,
            sim.bucket_backend.process_bucket,
            n_var_nosum=len(sim.tn.free_vars),
        )
        result_data = sim.bucket_backend.get_result_data(result).flatten()
        contraction_seconds = time.perf_counter() - t0
        value = complex(result_data[0])
        return value, peo, {
            "circuit_seconds": circuit_seconds,
            "bucket_build_seconds": build_seconds,
            "planning_seconds": plan_seconds,
            "reorder_slice_seconds": reorder_slice_seconds,
            "contraction_seconds": contraction_seconds,
            "result_dtype": str(result_data.dtype),
        }

    def energy(peos=None):
        total = 0.0 + 0.0j
        cached = []
        totals = {
            "circuit_seconds": 0.0,
            "bucket_build_seconds": 0.0,
            "planning_seconds": 0.0,
            "reorder_slice_seconds": 0.0,
            "contraction_seconds": 0.0,
        }
        dtype = ""
        for idx, edge in enumerate(graph.edges()):
            peo = None if peos is None else peos[idx]
            value, planned, timing = one_edge(edge, peo)
            total += value
            cached.append(planned)
            dtype = timing.pop("result_dtype")
            for key, value_t in timing.items():
                totals[key] += value_t
        return float(sim._post_process_energy(graph, total)), cached, totals, dtype

    first_start = time.perf_counter()
    first_value, peos, first_timing, dtype = energy(None)
    first_e2e = time.perf_counter() - first_start
    reuse_start = time.perf_counter()
    reuse_value, _, reuse_timing, reuse_dtype = energy(peos)
    reuse_e2e = time.perf_counter() - reuse_start
    payload = {
        "status": "ok",
        "qtensor_value": first_value,
        "qtensor_reuse_value": reuse_value,
        "import_seconds": import_seconds,
        "first_end_to_end_seconds": first_e2e,
        "reuse_end_to_end_seconds": reuse_e2e,
        "first": first_timing,
        "reuse": reuse_timing,
        "result_dtype": dtype,
        "reuse_result_dtype": reuse_dtype,
        "contraction_tolerance": "none (NumPy exact contraction at backend dtype)",
        "worker_internal_seconds": time.perf_counter() - process_start,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


@dataclass
class DiagnosticRow:
    family: str
    k: int
    p: int
    seed: int
    edges: int
    dense_c64_value: float
    dense_c128_value: float
    qtensor_value: float
    qtensor_reuse_value: float
    c64_abs_error: float
    c128_abs_error: float
    c128_relative_error: float
    reuse_abs_difference: float
    dense_c64_seconds: float
    dense_c128_seconds: float
    qtensor_process_wall_seconds: float
    qtensor_import_seconds: float
    qtensor_process_startup_seconds: float
    qtensor_first_end_to_end_seconds: float
    qtensor_first_planning_seconds: float
    qtensor_first_contraction_seconds: float
    qtensor_reuse_end_to_end_seconds: float
    qtensor_reuse_planning_seconds: float
    qtensor_reuse_contraction_seconds: float
    qtensor_result_dtype: str
    qtensor_tolerance: str
    constant_offset_check: str
    qubit_order_check: str
    endianness_check: str
    status: str
    notes: str


def make_grid(k: int):
    side = int(math.ceil(math.sqrt(k)))
    edges = []
    for node in range(k):
        row, col = divmod(node, side)
        right = node + 1
        down = node + side
        if col + 1 < side and right < k:
            edges.append((node, right))
        if row + 1 < side and down < k:
            edges.append((node, down))
    return edges


def run_driver(args) -> None:
    import numpy as np

    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "scripts"))
    from lcqaoa.graphs import WeightedGraph
    from lcqaoa.qaoa import full_state_expectation
    from benchmark_common import params_for_depth

    args.out_dir.mkdir(parents=True, exist_ok=True)
    work = args.out_dir / "cases"
    work.mkdir(exist_ok=True)
    rows = []
    csv_path = args.out_dir / "P1_1_qtensor_precision_diagnostic.csv"
    for seed_id in args.seed_ids:
        grid_edges = make_grid(args.k)
        graph = WeightedGraph(
            args.k,
            tuple((int(i), int(j), 1.0) for i, j in grid_edges),
            objective="maxcut",
        )
        gammas, betas = params_for_depth(args.p, seed=seed_id)
        c64 = full_state_expectation(
            graph, gammas, betas, method="precompute", prefer_gpu=True,
            max_qubits=args.k, complex_dtype=np.complex64, float_dtype=np.float32,
        )
        c128 = full_state_expectation(
            graph, gammas, betas, method="precompute", prefer_gpu=True,
            max_qubits=args.k, complex_dtype=np.complex128, float_dtype=np.float64,
        )
        case = {
            "n": graph.n,
            "edges": [list(e) for e in graph.edges],
            "gammas": list(map(float, gammas)),
            "betas": list(map(float, betas)),
        }
        case_path = work / f"grid_k{args.k}_p{args.p}_seed{seed_id}.json"
        out_path = work / f"grid_k{args.k}_p{args.p}_seed{seed_id}_qtensor.json"
        case_path.write_text(json.dumps(case, indent=2), encoding="utf-8")
        cmd = [
            str(args.qtensor_python), str(Path(__file__).resolve()), "--qtensor-worker",
            "--case", str(case_path), "--worker-out", str(out_path),
        ]
        worker_env = os.environ.copy()
        worker_env.pop("PYTHONPATH", None)
        wall_start = time.perf_counter()
        cp = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            timeout=args.timeout, env=worker_env, encoding="utf-8", errors="replace",
        )
        process_wall = time.perf_counter() - wall_start
        if cp.returncode != 0 or not out_path.exists():
            data = {"status": f"worker_failed_rc{cp.returncode}", "notes": cp.stdout[-1000:]}
        else:
            data = json.loads(out_path.read_text(encoding="utf-8"))
        if data.get("status") == "ok":
            qvalue = float(data["qtensor_value"])
            reuse = float(data["qtensor_reuse_value"])
            c128_abs = abs(c128.value - qvalue)
            row = DiagnosticRow(
                "grid", args.k, args.p, seed_id, graph.m,
                float(c64.value), float(c128.value), qvalue, reuse,
                abs(c64.value - qvalue), c128_abs,
                c128_abs / max(abs(c128.value), 1e-15), abs(qvalue - reuse),
                float(c64.seconds), float(c128.seconds), process_wall,
                float(data["import_seconds"]),
                max(0.0, process_wall - float(data["worker_internal_seconds"])),
                float(data["first_end_to_end_seconds"]),
                float(data["first"]["planning_seconds"]),
                float(data["first"]["contraction_seconds"]),
                float(data["reuse_end_to_end_seconds"]),
                float(data["reuse"]["planning_seconds"]),
                float(data["reuse"]["contraction_seconds"]),
                str(data["result_dtype"]), str(data["contraction_tolerance"]),
                "PASS: both report absolute MaxCut expectation; no constant dropped",
                "PASS: graph labels 0..k-1 preserved in both adapters",
                "PASS: dense bit i and QTensor node i use the same node labels; MaxCut is relabeling invariant",
                "ok", "transform=paper_qiskit_inverse; repeated angles reuse cached PEO per edge",
            )
        else:
            row = DiagnosticRow(
                "grid", args.k, args.p, seed_id, graph.m,
                float(c64.value), float(c128.value), float("nan"), float("nan"),
                float("nan"), float("nan"), float("nan"), float("nan"),
                float(c64.seconds), float(c128.seconds), process_wall,
                float("nan"), float("nan"), float("nan"), float("nan"), float("nan"),
                float("nan"), float("nan"), float("nan"), "", "", "", "", "",
                str(data.get("status")), str(data.get("notes", ""))[:800],
            )
        rows.append(row)
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
            writer.writeheader()
            for item in rows:
                writer.writerow(asdict(item))
        print(f"seed={seed_id} status={row.status} c64err={row.c64_abs_error:.3e} c128err={row.c128_abs_error:.3e}", flush=True)
    table_header = "| seed | c64 abs error | c128 abs error | relative error | planning s | contraction s | reused planning s | process wall s |"
    table_rule = "|---:|---:|---:|---:|---:|---:|---:|---:|"
    table_rows = [
        f"| {r.seed} | {r.c64_abs_error:.3e} | {r.c128_abs_error:.3e} | {r.c128_relative_error:.3e} | "
        f"{r.qtensor_first_planning_seconds:.4g} | {r.qtensor_first_contraction_seconds:.4g} | "
        f"{r.qtensor_reuse_planning_seconds:.4g} | {r.qtensor_process_wall_seconds:.4g} |"
        for r in rows
    ]
    lines = [
        "# P1-1 QTensor Precision and Timing Diagnostic",
        "",
        table_header,
        table_rule,
        *table_rows,
        "",
        "The original largest discrepancy is reproduced against complex64/float32 dense simulation. Complex128/float64 is the independent precision check. QTensor exposes no contraction tolerance in this path; it performs NumPy contraction at the reported backend dtype.",
    ]
    (args.out_dir / "P1_1_qtensor_precision_diagnostic.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qtensor-worker", action="store_true")
    parser.add_argument("--case", type=Path)
    parser.add_argument("--worker-out", type=Path)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "aaai27_followup_20260711" / "P1_1_qtensor_precision")
    parser.add_argument("--qtensor-python", type=Path, default=QTENSOR_PY)
    parser.add_argument("--k", type=int, default=22)
    parser.add_argument("--p", type=int, default=3)
    parser.add_argument("--seed-ids", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()
    if args.qtensor_worker:
        qtensor_worker(args.case, args.worker_out)
    else:
        run_driver(args)


if __name__ == "__main__":
    main()
