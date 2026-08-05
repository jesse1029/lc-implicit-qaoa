from __future__ import annotations

import argparse
import csv
import os
import subprocess
from pathlib import Path

from lcqaoa.graphs import erdos_renyi_graph, modular_graph, random_regular_graph
from lcqaoa.lightcone import extract_lightcones, lightcone_expectation


FAMILY_SEED_OFFSET = {
    "3regular": 101,
    "er_deg2": 202,
    "er_deg3": 303,
    "modular_sparse": 404,
}


def params_for(p: int) -> tuple[list[float], list[float]]:
    return [0.20 + 0.05 * i for i in range(p)], [0.32 - 0.035 * i for i in range(p)]


def graph_for(family: str, n: int, seed: int):
    if family == "3regular":
        return random_regular_graph(n, 3, seed=seed)
    if family == "er_deg2":
        return erdos_renyi_graph(n, min(0.45, 2.0 / max(2, n)), seed=seed)
    if family == "er_deg3":
        return erdos_renyi_graph(n, min(0.45, 3.0 / max(2, n)), seed=seed)
    if family == "modular_sparse":
        modules = max(4, n // 16)
        return modular_graph(n, modules=modules, p_in=0.22, p_out=0.0025, seed=seed)
    raise ValueError(family)


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
    seed = 73000 + 97 * n + 13 * p + FAMILY_SEED_OFFSET[family]
    graph = graph_for(family, n, seed=seed)
    gammas, betas = params_for(p)
    cones = extract_lightcones(graph, p)
    kmax = max((cone.k for cone in cones), default=0)
    total_cone_states = sum(1 << cone.k for cone in cones)
    row: dict[str, object] = {
        "family": family,
        "n": n,
        "p": p,
        "m": graph.m,
        "seed": seed,
        "status": "error",
        "value": float("nan"),
        "warmup_seconds": float("nan"),
        "eval_median_seconds": float("nan"),
        "eval_min_seconds": float("nan"),
        "eval_repeats": repeats,
        "kmax": kmax,
        "total_cone_states": total_cone_states,
        "peak_pool_mb": float("nan"),
        "gpu_mem_before_mb": gpu_memory_mb(),
        "gpu_mem_after_eval_mb": float("nan"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "notes": "LC-Implicit-QAOA same-host GPU probe using the artifact implementation",
    }
    try:
        warmup = lightcone_expectation(graph, gammas, betas, p=p, prefer_gpu=True)
        row["warmup_seconds"] = warmup.seconds
        timings: list[float] = []
        value = warmup.value
        peak = warmup.peak_pool_bytes
        for _ in range(repeats):
            result = lightcone_expectation(graph, gammas, betas, p=p, prefer_gpu=True)
            timings.append(result.seconds)
            value = result.value
            peak = max(peak, result.peak_pool_bytes)
        timings.sort()
        row["status"] = "ok"
        row["value"] = value
        row["eval_median_seconds"] = timings[len(timings) // 2]
        row["eval_min_seconds"] = timings[0]
        row["peak_pool_mb"] = peak / (1024 * 1024)
        row["gpu_mem_after_eval_mb"] = gpu_memory_mb()
    except Exception as exc:
        row["status"] = f"{type(exc).__name__}: {exc}"
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--case", action="append", default=[])
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
            ("3regular", 30, 2),
            ("3regular", 32, 2),
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
