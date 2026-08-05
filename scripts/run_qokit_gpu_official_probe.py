from __future__ import annotations

import argparse
import csv
import os
import random
import subprocess
import time
from pathlib import Path


FAMILY_SEED_OFFSET = {
    "3regular": 101,
    "er_deg2": 202,
    "er_deg3": 303,
    "modular_sparse": 404,
}


def params_for(p: int) -> tuple[list[float], list[float]]:
    return [0.20 + 0.05 * i for i in range(p)], [0.32 - 0.035 * i for i in range(p)]


def graph_edges_for(family: str, n: int, seed: int) -> tuple[tuple[int, int], ...]:
    if family == "3regular":
        import networkx as nx

        g = nx.random_regular_graph(3, n, seed=seed)
        return tuple((int(i), int(j)) for i, j in g.edges())

    r = random.Random(seed)
    edges: list[tuple[int, int]] = []
    if family == "er_deg2":
        prob = min(0.45, 2.0 / max(2, n))
    elif family == "er_deg3":
        prob = min(0.45, 3.0 / max(2, n))
    elif family == "modular_sparse":
        modules = max(4, n // 16)
        block = max(1, n // modules)
        labels = [min(modules - 1, i // block) for i in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                prob = 0.22 if labels[i] == labels[j] else 0.0025
                if r.random() < prob:
                    edges.append((i, j))
        return tuple(edges or [(0, 1)])
    else:
        raise ValueError(family)

    for i in range(n):
        for j in range(i + 1, n):
            if r.random() < prob:
                edges.append((i, j))
    return tuple(edges or [(0, 1)])


def gpu_memory_mb() -> float:
    gpu_id = os.environ.get("QOKIT_PROBE_PHYSICAL_GPU")
    command = [
        "nvidia-smi",
        "--query-gpu=memory.used",
        "--format=csv,noheader,nounits",
    ]
    if gpu_id:
        command.insert(1, f"--id={gpu_id}")
    try:
        cp = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=10,
        )
    except Exception:
        return float("nan")
    if cp.returncode != 0:
        return float("nan")
    lines = [x.strip() for x in cp.stdout.splitlines() if x.strip()]
    return float(lines[0]) if lines else float("nan")


def run_case(family: str, n: int, p: int, repeats: int) -> dict[str, object]:
    import importlib.metadata as metadata
    import networkx as nx
    import numba.cuda as cuda
    from qokit.fur.nbcuda.qaoa_simulator import QAOAFURXSimulatorGPU
    from qokit.maxcut import get_maxcut_terms

    seed = 73000 + 97 * n + 13 * p + FAMILY_SEED_OFFSET[family]
    edges = graph_edges_for(family, n, seed)
    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    graph.add_edges_from(edges)
    gammas, betas = params_for(p)
    qokit_gammas = [2.0 * float(x) for x in gammas]
    qokit_betas = [float(x) for x in betas]

    row: dict[str, object] = {
        "family": family,
        "n": n,
        "p": p,
        "m": len(edges),
        "seed": seed,
        "status": "error",
        "value": float("nan"),
        "setup_seconds": float("nan"),
        "warmup_seconds": float("nan"),
        "eval_median_seconds": float("nan"),
        "eval_min_seconds": float("nan"),
        "eval_repeats": repeats,
        "gpu_mem_before_mb": gpu_memory_mb(),
        "gpu_mem_after_setup_mb": float("nan"),
        "gpu_mem_after_eval_mb": float("nan"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "qokit_version": metadata.version("qokit"),
        "numba_version": metadata.version("numba"),
        "numba_cuda_version": metadata.version("numba-cuda"),
        "notes": "official QOKit 0.1.4 pinned source-build GPU path; gammas doubled to match QOKit convention",
    }

    try:
        terms = get_maxcut_terms(graph)
        t0 = time.perf_counter()
        sim = QAOAFURXSimulatorGPU(n, terms=terms)
        cuda.synchronize()
        row["setup_seconds"] = time.perf_counter() - t0
        row["gpu_mem_after_setup_mb"] = gpu_memory_mb()

        t0 = time.perf_counter()
        result = sim.simulate_qaoa(qokit_gammas, qokit_betas)
        value = sim.get_expectation(result)
        cuda.synchronize()
        row["warmup_seconds"] = time.perf_counter() - t0
        row["value"] = float(value)

        timings: list[float] = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            result = sim.simulate_qaoa(qokit_gammas, qokit_betas)
            value = sim.get_expectation(result)
            cuda.synchronize()
            timings.append(time.perf_counter() - t0)
        timings.sort()
        row["value"] = float(value)
        row["eval_median_seconds"] = timings[len(timings) // 2]
        row["eval_min_seconds"] = timings[0]
        row["gpu_mem_after_eval_mb"] = gpu_memory_mb()
        row["status"] = "ok"
    except Exception as exc:
        row["status"] = f"{type(exc).__name__}: {exc}"
    finally:
        # The intended timing protocol launches this script once per case so
        # the process teardown releases the CUDA context cleanly. Avoid closing
        # the context here; CuPy/Numba module destructors can otherwise observe
        # invalid handles before Python exits.
        pass
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        help="Case as family,n,p. If omitted, a compact official-regime probe is used.",
    )
    args = parser.parse_args()
    cases = []
    if args.case:
        for item in args.case:
            family, n, p = item.split(",")
            cases.append((family, int(n), int(p)))
    else:
        cases = [
            ("3regular", 24, 2),
            ("3regular", 26, 2),
            ("3regular", 28, 2),
            ("er_deg2", 24, 2),
            ("er_deg3", 24, 2),
            ("modular_sparse", 24, 2),
        ]

    rows = [run_case(family, n, p, args.repeats) for family, n, p in cases]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
