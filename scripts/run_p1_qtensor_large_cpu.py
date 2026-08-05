from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from lcqaoa.graphs import random_regular_graph
from lcqaoa.lightcone import extract_lightcones
from run_benchmarks import params_for


PREFIX = Path.home() / "lc_implicit_qaoa_peers"
QTENSOR_PY = PREFIX / "venvs" / "qtensor-py310" / "bin" / "python"


def extended_seed(n: int, p: int) -> int:
    return 260000 + n * 17 + p * 101 + 1


def load_lc_extended_rows() -> dict[tuple[int, int], dict]:
    path = ROOT / "results" / "extended_reach.csv"
    rows: dict[tuple[int, int], dict] = {}
    if not path.exists():
        return rows
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("family") != "3regular" or row.get("task") != "objective":
                continue
            rows[(int(row["n"]), int(row["p"]))] = row
    return rows


def run_qtensor(case_path: Path, out_json: Path, timeout: int) -> dict:
    if not QTENSOR_PY.exists():
        return {
            "method": "qtensor_cpu_external",
            "status": f"missing_python:{QTENSOR_PY}",
            "value": float("nan"),
            "seconds": 0.0,
            "wall_seconds": 0.0,
            "notes": "",
        }
    cmd = [
        str(QTENSOR_PY),
        str(ROOT / "scripts" / "external_peer_runner.py"),
        "--method",
        "qtensor_cpu",
        "--case",
        str(case_path),
        "--out",
        str(out_json),
    ]
    t0 = time.perf_counter()
    try:
        cp = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=timeout)
        wall = time.perf_counter() - t0
    except subprocess.TimeoutExpired as exc:
        return {
            "method": "qtensor_cpu_external",
            "status": f"timeout_{timeout}s",
            "value": float("nan"),
            "seconds": 0.0,
            "wall_seconds": float(timeout),
            "notes": (exc.stdout or "")[-800:] if isinstance(exc.stdout, str) else "",
        }
    if out_json.exists():
        data = json.loads(out_json.read_text(encoding="utf-8"))
    else:
        data = {
            "method": "qtensor_cpu_external",
            "status": f"process_failed:{cp.returncode}",
            "value": float("nan"),
            "seconds": 0.0,
            "notes": cp.stdout[-800:],
        }
    data["process_returncode"] = cp.returncode
    data["wall_seconds"] = wall
    if cp.returncode != 0 and data.get("status") == "ok":
        data["status"] = f"process_failed:{cp.returncode}"
        data["notes"] = cp.stdout[-800:]
    return data


def write_outputs(rows: list[dict], out_dir: Path) -> None:
    csv_path = out_dir / "p1_qtensor_large_cpu.csv"
    fieldnames = sorted({key for row in rows for key in row})
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    md_lines = [
        "# P1 QTensor CPU Large-n Probe",
        "",
        "3-regular p=2 cases use the same seed rule as `run_extended_reach.py`, so LC rows align with the existing bounded-cone stress table.",
        "",
        "| n | kmax | total cone states | LC obj s | QTensor CPU status | QTensor CPU s | QTensor/LC time | Notes |",
        "|---:|---:|---:|---:|---|---:|---:|---|",
    ]
    for row in rows:
        md_lines.append(
            "| {n} | {kmax} | {total_cone_states} | {lc_seconds:.4g} | {qt_status} | {qt_seconds:.4g} | {ratio:.4g} | {notes} |".format(
                n=row["n"],
                kmax=row["kmax"],
                total_cone_states=row["total_cone_states"],
                lc_seconds=float(row.get("lc_seconds", 0.0)),
                qt_status=row.get("qtensor_status", ""),
                qt_seconds=float(row.get("qtensor_seconds", 0.0)),
                ratio=float(row.get("qtensor_over_lc_time", float("nan"))),
                notes=str(row.get("notes", ""))[:160].replace("|", "/"),
            )
        )
    (out_dir / "p1_qtensor_large_cpu.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"wrote {csv_path}")
    print(f"wrote {out_dir / 'p1_qtensor_large_cpu.md'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "p1_qtensor_large_cpu")
    parser.add_argument("--timeout", type=int, default=1200)
    parser.add_argument("--ns", nargs="+", type=int, default=[512, 2048, 16384])
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    p = 2
    gammas, betas = params_for(p)
    lc_rows = load_lc_extended_rows()
    rows: list[dict] = []
    for n in args.ns:
        graph = random_regular_graph(n, 3, seed=extended_seed(n, p))
        cones = extract_lightcones(graph, p)
        kmax = max(c.k for c in cones)
        total_cone_states = sum(1 << c.k for c in cones)
        case = {
            "n": graph.n,
            "edges": [list(edge) for edge in graph.edges],
            "fields": [list(field) for field in graph.fields],
            "gammas": gammas,
            "betas": betas,
            "objective": graph.objective,
        }
        case_path = args.out_dir / f"qtensor_3regular_n{n}_p{p}.json"
        case_path.write_text(json.dumps(case, indent=2), encoding="utf-8")
        data = run_qtensor(case_path, args.out_dir / f"qtensor_3regular_n{n}_p{p}.out.json", args.timeout)
        lc = lc_rows.get((n, p), {})
        lc_seconds = float(lc.get("seconds", "nan")) if lc else float("nan")
        qt_seconds = float(data.get("seconds", 0.0))
        rows.append(
            {
                "family": "3regular",
                "n": n,
                "p": p,
                "m": graph.m,
                "kmax": kmax,
                "total_cone_states": total_cone_states,
                "lc_status": lc.get("status", "missing_extended_lc_row"),
                "lc_seconds": lc_seconds,
                "lc_peak_mb": float(lc.get("peak_pool_mb", "nan")) if lc else float("nan"),
                "qtensor_status": data.get("status", ""),
                "qtensor_seconds": qt_seconds,
                "qtensor_wall_seconds": float(data.get("wall_seconds", 0.0)),
                "qtensor_value": data.get("value", float("nan")),
                "qtensor_over_lc_time": qt_seconds / lc_seconds if lc_seconds == lc_seconds and lc_seconds > 0 else float("nan"),
                "backend": data.get("backend", ""),
                "process_returncode": data.get("process_returncode", ""),
                "notes": data.get("notes", ""),
            }
        )
        print(f"n={n}: {data.get('status')} qtensor={qt_seconds:.4g}s lc={lc_seconds:.4g}s")
    write_outputs(rows, args.out_dir)


if __name__ == "__main__":
    main()
