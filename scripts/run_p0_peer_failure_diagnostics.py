from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lcqaoa.graphs import erdos_renyi_graph, random_regular_graph
from lcqaoa.lightcone import extract_lightcones, lightcone_expectation


PREFIX = Path.home() / "lc_implicit_qaoa_peers"
CORE_PY = PREFIX / "venvs" / "lcqaoa-core" / "bin" / "python"
CUAOA_PY = PREFIX / "venvs" / "cuaoa-py312" / "bin" / "python"
FAMILY_SEED_OFFSET = {"3regular": 101, "er_deg3": 303}


def params_for(p: int) -> tuple[list[float], list[float]]:
    return [0.20 + 0.05 * i for i in range(p)], [0.32 - 0.035 * i for i in range(p)]


def graph_for(family: str, n: int, seed: int):
    if family == "3regular":
        return random_regular_graph(n, 3, seed=seed)
    if family == "er_deg3":
        return erdos_renyi_graph(n, min(0.45, 3.0 / max(2, n)), seed=seed)
    raise ValueError(family)


def official_seed(family: str, n: int, p: int) -> int:
    return 73000 + 97 * n + 13 * p + FAMILY_SEED_OFFSET[family]


def query_gpu_used_mb(gpu_index: int) -> float:
    cmd = [
        "nvidia-smi",
        f"--id={gpu_index}",
        "--query-gpu=memory.used",
        "--format=csv,noheader,nounits",
    ]
    cp = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, timeout=5)
    if cp.returncode != 0 or not cp.stdout.strip():
        return float("nan")
    return float(cp.stdout.strip().splitlines()[0].strip())


def memory_sampler(gpu_index: int, stop: threading.Event, samples: list[tuple[float, float]], interval: float) -> None:
    t0 = time.perf_counter()
    while not stop.is_set():
        try:
            used = query_gpu_used_mb(gpu_index)
        except Exception:
            used = float("nan")
        samples.append((time.perf_counter() - t0, used))
        stop.wait(interval)


def write_memory_csv(samples: list[tuple[float, float]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["seconds", "gpu_memory_used_mb"])
        writer.writerows(samples)


def run_adapter(
    method: str,
    py: Path,
    case_path: Path,
    out_json: Path,
    raw_prefix: Path,
    timeout: int,
    gpu_index: int,
) -> dict:
    cmd = [
        str(py),
        str(ROOT / "scripts" / "external_peer_runner.py"),
        "--method",
        method,
        "--case",
        str(case_path),
        "--out",
        str(out_json),
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    env["CUDA_LAUNCH_BLOCKING"] = "1"
    env["CUDA_MODULE_LOADING"] = "EAGER"
    samples: list[tuple[float, float]] = []
    stop = threading.Event()
    sampler = threading.Thread(target=memory_sampler, args=(gpu_index, stop, samples, 0.2), daemon=True)
    before_mb = query_gpu_used_mb(gpu_index)
    sampler.start()
    t0 = time.perf_counter()
    timed_out = False
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        proc.kill()
        stdout, stderr = proc.communicate()
    wall = time.perf_counter() - t0
    stop.set()
    sampler.join(timeout=2)
    after_mb = query_gpu_used_mb(gpu_index)
    raw_prefix.parent.mkdir(parents=True, exist_ok=True)
    (raw_prefix.with_suffix(".stdout.txt")).write_text(stdout or "", encoding="utf-8", errors="replace")
    (raw_prefix.with_suffix(".stderr.txt")).write_text(stderr or "", encoding="utf-8", errors="replace")
    write_memory_csv(samples, raw_prefix.with_suffix(".nvidia_smi.csv"))
    result = {}
    if out_json.exists():
        try:
            result = json.loads(out_json.read_text(encoding="utf-8"))
        except Exception as exc:
            result = {"status": f"unreadable_json:{type(exc).__name__}"}
    status = result.get("status", "")
    if timed_out:
        status = f"timeout_{timeout}s"
    elif proc.returncode != 0 and not status:
        status = f"process_returncode_{proc.returncode}"
    peak_mb = max((v for _, v in samples if v == v), default=float("nan"))
    return {
        "method": method,
        "python": str(py),
        "process_returncode": proc.returncode,
        "timed_out": timed_out,
        "status": status,
        "result_value": result.get("value", float("nan")),
        "result_seconds": result.get("seconds", 0.0),
        "wall_seconds": wall,
        "gpu_memory_before_mb": before_mb,
        "gpu_memory_peak_mb": peak_mb,
        "gpu_memory_after_mb": after_mb,
        "gpu_memory_peak_delta_mb": peak_mb - before_mb if peak_mb == peak_mb and before_mb == before_mb else float("nan"),
        "stdout_tail": (stdout or "")[-500:].replace("\n", " "),
        "stderr_tail": (stderr or "")[-500:].replace("\n", " "),
        "out_json": str(out_json),
        "raw_prefix": str(raw_prefix),
    }


def write_summary(rows: list[dict], out_dir: Path) -> None:
    csv_path = out_dir / "p0_peer_failure_diagnostics.csv"
    fieldnames = sorted({key for row in rows for key in row})
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    md_lines = [
        "# P0 Peer Failure Diagnostics",
        "",
        "All runs set `CUDA_VISIBLE_DEVICES=0`, `CUDA_LAUNCH_BLOCKING=1`, and sampled `nvidia-smi` memory every 0.2 seconds.",
        "The 3-regular family has no valid n=27 instance because 3n must be even, so the boundary pair is n=26/n=28; an ER degree-3 n=27 row is included only as a size diagnostic.",
        "",
        "| Case | Method | Status | RC | Wall s | Peak MB | Delta MB | Notes |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        notes = row.get("stderr_tail") or row.get("stdout_tail") or ""
        md_lines.append(
            "| {case} | {method} | {status} | {rc} | {wall:.3g} | {peak:.3g} | {delta:.3g} | {notes} |".format(
                case=row["case"],
                method=row["method"],
                status=row["status"],
                rc=row["process_returncode"],
                wall=float(row["wall_seconds"]),
                peak=float(row["gpu_memory_peak_mb"]),
                delta=float(row["gpu_memory_peak_delta_mb"]),
                notes=str(notes)[:180].replace("|", "/"),
            )
        )
    (out_dir / "p0_peer_failure_diagnostics.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"wrote {csv_path}")
    print(f"wrote {out_dir / 'p0_peer_failure_diagnostics.md'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "p0_peer_failure_diagnostics")
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--gpu-index", type=int, default=0)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cases = [
        ("3regular_n26_p2", "3regular", 26, 2),
        ("3regular_n28_p2", "3regular", 28, 2),
        ("er_deg3_n27_p2", "er_deg3", 27, 2),
        ("er_deg3_n28_p2", "er_deg3", 28, 2),
    ]
    methods = [
        ("cuaoa_gpu", CUAOA_PY),
        ("cudaq_observe", CORE_PY),
    ]
    rows: list[dict] = []
    for case_name, family, n, p in cases:
        seed = official_seed(family, n, p)
        graph = graph_for(family, n, seed)
        gammas, betas = params_for(p)
        cones = extract_lightcones(graph, p)
        lc = lightcone_expectation(graph, gammas, betas, p=p, prefer_gpu=True, max_k=24, max_batch_states=1 << 21)
        case = {
            "n": graph.n,
            "edges": [list(edge) for edge in graph.edges],
            "fields": [list(field) for field in graph.fields],
            "gammas": gammas,
            "betas": betas,
            "objective": graph.objective,
        }
        case_path = args.out_dir / f"{case_name}.json"
        case_path.write_text(json.dumps(case, indent=2), encoding="utf-8")
        for method, py in methods:
            out_json = args.out_dir / f"{case_name}_{method}.json"
            raw_prefix = args.out_dir / "raw" / f"{case_name}_{method}"
            if not py.exists():
                rows.append(
                    {
                        "case": case_name,
                        "family": family,
                        "n": n,
                        "p": p,
                        "m": graph.m,
                        "kmax": max(c.k for c in cones),
                        "total_cone_states": sum(1 << c.k for c in cones),
                        "lc_status": lc.status,
                        "lc_seconds": lc.seconds,
                        "lc_peak_mb": lc.peak_pool_bytes / 1024**2,
                        "method": method,
                        "status": f"missing_python:{py}",
                        "process_returncode": "",
                        "wall_seconds": 0.0,
                        "gpu_memory_peak_mb": float("nan"),
                        "gpu_memory_peak_delta_mb": float("nan"),
                    }
                )
                continue
            row = run_adapter(method, py, case_path, out_json, raw_prefix, args.timeout, args.gpu_index)
            row.update(
                {
                    "case": case_name,
                    "family": family,
                    "n": n,
                    "p": p,
                    "m": graph.m,
                    "kmax": max(c.k for c in cones),
                    "total_cone_states": sum(1 << c.k for c in cones),
                    "lc_status": lc.status,
                    "lc_seconds": lc.seconds,
                    "lc_peak_mb": lc.peak_pool_bytes / 1024**2,
                }
            )
            rows.append(row)
            print(f"{case_name} {method}: {row['status']} rc={row['process_returncode']} peak={row['gpu_memory_peak_mb']:.1f}MB")
    write_summary(rows, args.out_dir)


if __name__ == "__main__":
    main()
