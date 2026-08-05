from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_official_regime_matrix import FAMILY_SEED_OFFSET, graph_for, params_for


PREFIX = Path.home() / "lc_implicit_qaoa_peers"
QTENSOR_PY = PREFIX / "venvs" / "qtensor-py310" / "bin" / "python"
QTENSOR_GPU_SOURCE = PREFIX / "src" / "QTensor-cupybackend"


def write_case(path: Path, family: str, n: int, p: int) -> None:
    graph = graph_for(family, n, seed=73000 + 97 * n + 13 * p + FAMILY_SEED_OFFSET[family])
    gammas, betas = params_for(p)
    case = {
        "family": family,
        "n": graph.n,
        "p": p,
        "edges": [list(edge) for edge in graph.edges],
        "fields": [list(field) for field in graph.fields],
        "gammas": gammas,
        "betas": betas,
        "objective": graph.objective,
    }
    path.write_text(json.dumps(case, indent=2), encoding="utf-8")


def run_case(case_path: Path, out_path: Path, timeout: int) -> dict:
    env = os.environ.copy()
    env["QTENSOR_GPU_SOURCE"] = str(QTENSOR_GPU_SOURCE)
    cmd = [
        str(QTENSOR_PY),
        str(ROOT / "scripts" / "external_peer_runner.py"),
        "--method",
        "qtensor_gpu",
        "--case",
        str(case_path),
        "--out",
        str(out_path),
        "--qtensor-transform",
        "paper_qiskit_inverse",
    ]
    t0 = time.perf_counter()
    try:
        cp = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=timeout, env=env)
        wall = time.perf_counter() - t0
    except subprocess.TimeoutExpired as exc:
        return {
            "method": "qtensor_gpu_external",
            "status": f"timeout_{timeout}s",
            "value": float("nan"),
            "seconds": 0.0,
            "wall_seconds": float(timeout),
            "peak_pool_mb": float("nan"),
            "backend": "",
            "notes": (exc.stdout or "")[-1000:] if isinstance(exc.stdout, str) else "",
        }
    if out_path.exists():
        data = json.loads(out_path.read_text(encoding="utf-8"))
    else:
        data = {
            "method": "qtensor_gpu_external",
            "status": f"process_failed:{cp.returncode}",
            "value": float("nan"),
            "seconds": 0.0,
            "peak_pool_mb": float("nan"),
            "backend": "",
            "notes": cp.stdout[-1000:],
        }
    data["wall_seconds"] = wall
    if cp.returncode != 0 and data.get("status") == "ok":
        data["status"] = f"process_failed:{cp.returncode}"
        data["notes"] = cp.stdout[-1000:]
    return data


def write_markdown(rows: list[dict], path: Path) -> None:
    lines = [
        "# QTensor GPU Refresh",
        "",
        "Runs the official QTensor `origin/cupybackend` branch through the artifact adapter.",
        "",
        "| Family | n | p | status | seconds | wall seconds | peak pool MB | value |",
        "|---|---:|---:|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['family']} | {row['n']} | {row['p']} | {row['status']} | "
            f"{row['seconds']:.4g} | {row['wall_seconds']:.4g} | {row['peak_pool_mb']:.4g} | {row['value']:.8g} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "qtensor_gpu_refresh.csv")
    parser.add_argument("--markdown", type=Path, default=ROOT / "results" / "qtensor_gpu_refresh.md")
    parser.add_argument("--work-dir", type=Path, default=ROOT / "results" / "qtensor_gpu_refresh_cases")
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.work_dir.mkdir(parents=True, exist_ok=True)

    cases = [
        ("3regular", 24, 2),
        ("3regular", 26, 2),
        ("3regular", 28, 2),
        ("3regular", 30, 2),
        ("3regular", 32, 2),
        ("3regular", 64, 2),
        ("3regular", 128, 2),
        ("er_deg2", 24, 2),
        ("er_deg2", 32, 2),
        ("er_deg2", 64, 2),
        ("er_deg2", 128, 2),
        ("er_deg3", 24, 2),
        ("er_deg3", 32, 2),
        ("er_deg3", 64, 2),
        ("modular_sparse", 24, 2),
        ("modular_sparse", 64, 2),
        ("modular_sparse", 128, 1),
        ("modular_sparse", 128, 2),
    ]
    rows: list[dict] = []
    for family, n, p in cases:
        print(f"RUN {family} n={n} p={p}", flush=True)
        case_path = args.work_dir / f"qtensor_gpu_{family}_n{n}_p{p}.json"
        result_path = args.work_dir / f"qtensor_gpu_{family}_n{n}_p{p}.result.json"
        write_case(case_path, family, n, p)
        result = run_case(case_path, result_path, timeout=args.timeout)
        result.update({"family": family, "n": n, "p": p})
        rows.append(result)

    fieldnames = [
        "family",
        "n",
        "p",
        "method",
        "status",
        "seconds",
        "wall_seconds",
        "peak_pool_mb",
        "value",
        "backend",
        "notes",
    ]
    with args.out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    write_markdown(rows, args.markdown)
    print(f"WROTE {args.out}")
    print(f"WROTE {args.markdown}")


if __name__ == "__main__":
    main()
