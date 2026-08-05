from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lcqaoa.graphs import erdos_renyi_graph, modular_graph, random_regular_graph
from lcqaoa.lightcone import extract_lightcones, lightcone_expectation
from lcqaoa.qaoa import full_state_expectation


PREFIX = Path.home() / "lc_implicit_qaoa_peers"
CORE_PY = PREFIX / "venvs" / "lcqaoa-core" / "bin" / "python"
QTENSOR_PY = PREFIX / "venvs" / "qtensor-py310" / "bin" / "python"
CUAOA_PY = PREFIX / "venvs" / "cuaoa-py312" / "bin" / "python"
FAMILY_SEED_OFFSET = {
    "3regular": 101,
    "er_deg2": 202,
    "er_deg3": 303,
    "modular_sparse": 404,
}


@dataclass
class RegimeRow:
    family: str
    n: int
    p: int
    m: int
    avg_degree: float
    max_degree: int
    kmax: int
    total_cone_states: int
    lc_status: str
    lc_seconds: float
    lc_peak_mb: float
    full_status: str
    full_seconds: float
    full_peak_mb: float
    qokit_cpu_status: str
    qokit_cpu_seconds: float
    qokit_gpu_status: str
    cuaoa_status: str
    cuaoa_seconds: float
    cudaq_status: str
    cudaq_seconds: float
    qtensor_cpu_status: str
    qtensor_cpu_seconds: float
    qtensor_gpu_status: str
    value_lc: float
    abs_error_vs_full: float
    notes: str


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


def params_for(p: int) -> tuple[list[float], list[float]]:
    return [0.20 + 0.05 * i for i in range(p)], [0.32 - 0.035 * i for i in range(p)]


def degree_stats(graph) -> tuple[float, int]:
    deg = [0 for _ in range(graph.n)]
    for i, j, _ in graph.edges:
        deg[int(i)] += 1
        deg[int(j)] += 1
    return (sum(deg) / graph.n if graph.n else 0.0, max(deg) if deg else 0)


def qokit_gpu_source_status() -> str:
    if not CORE_PY.exists():
        return "missing_core_python"
    code = (
        "from pathlib import Path; import qokit; "
        "p=Path(qokit.__file__).resolve().parent/'fur'/'nbcuda'/'furx.cu'; "
        "print('available' if p.exists() else 'missing_furx_cu')"
    )
    try:
        cp = subprocess.run([str(CORE_PY), "-c", code], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=30)
    except Exception as exc:
        return f"probe_failed:{type(exc).__name__}"
    return cp.stdout.strip().splitlines()[-1] if cp.returncode == 0 and cp.stdout.strip() else f"probe_failed:{cp.returncode}"


def qtensor_gpu_status() -> str:
    # The installed adapter used in this artifact exposes QTensor/QTree through
    # NumPy CPU. A GPU tensor-contraction path is therefore not claimed.
    if not QTENSOR_PY.exists():
        return "missing_qtensor_python"
    code = "import importlib.util; print('cupy_present' if importlib.util.find_spec('cupy') else 'no_cupy_in_qtensor_env')"
    try:
        cp = subprocess.run([str(QTENSOR_PY), "-c", code], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=30)
    except Exception as exc:
        return f"probe_failed:{type(exc).__name__}"
    flag = cp.stdout.strip().splitlines()[-1] if cp.stdout.strip() else f"probe_failed:{cp.returncode}"
    return "no_official_gpu_adapter:" + flag


def run_external(method: str, case_path: Path, out: Path, py: Path, timeout: int, extra: list[str] | None = None) -> dict:
    if not py.exists():
        return {"method": method, "status": f"missing_python:{py}", "seconds": 0.0, "value": float("nan"), "notes": ""}
    cmd = [
        str(py),
        str(ROOT / "scripts" / "external_peer_runner.py"),
        "--method",
        method,
        "--case",
        str(case_path),
        "--out",
        str(out),
    ]
    if extra:
        cmd.extend(extra)
    try:
        t0 = time.perf_counter()
        cp = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=timeout)
        wall = time.perf_counter() - t0
    except subprocess.TimeoutExpired as exc:
        return {
            "method": method,
            "status": f"timeout_{timeout}s",
            "seconds": 0.0,
            "wall_seconds": float(timeout),
            "value": float("nan"),
            "notes": (exc.stdout or "")[-500:] if isinstance(exc.stdout, str) else "",
        }
    if out.exists():
        data = json.loads(out.read_text(encoding="utf-8"))
    else:
        data = {
            "method": method,
            "status": f"process_failed:{cp.returncode}",
            "seconds": 0.0,
            "value": float("nan"),
            "notes": cp.stdout[-500:],
        }
    data["wall_seconds"] = wall
    if cp.returncode != 0 and data.get("status") == "ok":
        data["status"] = f"process_failed:{cp.returncode}"
        data["notes"] = cp.stdout[-500:]
    return data


def run_case(family: str, n: int, p: int, args, qokit_gpu: str, qtensor_gpu: str) -> RegimeRow:
    graph = graph_for(family, n, seed=73000 + 97 * n + 13 * p + FAMILY_SEED_OFFSET[family])
    gammas, betas = params_for(p)
    avg_degree, max_degree = degree_stats(graph)
    cones = extract_lightcones(graph, p)
    kmax = max(c.k for c in cones) if cones else 0
    total_cone_states = sum(1 << c.k for c in cones)

    lc = lightcone_expectation(
        graph,
        gammas,
        betas,
        p=p,
        prefer_gpu=True,
        max_k=args.max_k,
        max_batch_states=args.max_batch_states,
    )

    try:
        full = full_state_expectation(
            graph,
            gammas,
            betas,
            method="precompute",
            prefer_gpu=True,
            max_qubits=args.full_cap,
        )
    except Exception as exc:
        class _FailedFull:
            value = float("nan")
            seconds = 0.0
            peak_pool_bytes = 0
            status = ""

        full = _FailedFull()
        full.status = f"failed:{type(exc).__name__}"
    ref_value = full.value if full.status == "ok" else float("nan")

    case = {
        "n": graph.n,
        "edges": [list(edge) for edge in graph.edges],
        "fields": [list(field) for field in graph.fields],
        "gammas": gammas,
        "betas": betas,
        "objective": graph.objective,
    }
    case_path = args.work_dir / f"official_matrix_{family}_n{n}_p{p}.json"
    case_path.write_text(json.dumps(case, indent=2), encoding="utf-8")

    qokit_cpu = {"status": "not_run_cpu_cost", "seconds": 0.0, "value": float("nan")}
    if n <= args.qokit_cpu_max_n:
        qokit_cpu = run_external(
            "qokit_cpu",
            case_path,
            args.work_dir / f"qokit_cpu_{family}_n{n}_p{p}.json",
            CORE_PY,
            args.qokit_timeout,
        )

    cuaoa = {"status": "not_run_large_by_protocol", "seconds": 0.0, "value": float("nan")}
    if graph.objective == "maxcut" and n <= args.cuaoa_max_n:
        cuaoa = run_external(
            "cuaoa_gpu",
            case_path,
            args.work_dir / f"cuaoa_{family}_n{n}_p{p}.json",
            CUAOA_PY,
            args.peer_timeout,
        )

    cudaq = {"status": "not_run_large", "seconds": 0.0, "value": float("nan")}
    if n <= args.cudaq_max_n:
        cudaq = run_external(
            "cudaq_observe",
            case_path,
            args.work_dir / f"cudaq_{family}_n{n}_p{p}.json",
            CORE_PY,
            args.peer_timeout,
        )

    qtensor = {"status": "not_run_large_or_non_3regular", "seconds": 0.0, "value": float("nan")}
    if graph.objective == "maxcut" and n <= args.qtensor_max_n:
        qtensor = run_external(
            "qtensor_cpu",
            case_path,
            args.work_dir / f"qtensor_{family}_n{n}_p{p}.json",
            QTENSOR_PY,
            args.peer_timeout,
            ["--qtensor-transform", "paper_qiskit_inverse"],
        )

    abs_error = abs(lc.value - ref_value) if lc.status == "ok" and math.isfinite(ref_value) else float("nan")
    return RegimeRow(
        family=family,
        n=n,
        p=p,
        m=graph.m,
        avg_degree=avg_degree,
        max_degree=max_degree,
        kmax=kmax,
        total_cone_states=total_cone_states,
        lc_status=lc.status,
        lc_seconds=lc.seconds,
        lc_peak_mb=lc.peak_pool_bytes / 1024**2,
        full_status=full.status,
        full_seconds=full.seconds,
        full_peak_mb=full.peak_pool_bytes / 1024**2,
        qokit_cpu_status=str(qokit_cpu.get("status", "")),
        qokit_cpu_seconds=float(qokit_cpu.get("seconds", 0.0) or 0.0),
        qokit_gpu_status=qokit_gpu,
        cuaoa_status=str(cuaoa.get("status", "")),
        cuaoa_seconds=float(cuaoa.get("seconds", 0.0) or 0.0),
        cudaq_status=str(cudaq.get("status", "")),
        cudaq_seconds=float(cudaq.get("seconds", 0.0) or 0.0),
        qtensor_cpu_status=str(qtensor.get("status", "")),
        qtensor_cpu_seconds=float(qtensor.get("seconds", 0.0) or 0.0),
        qtensor_gpu_status=qtensor_gpu,
        value_lc=lc.value,
        abs_error_vs_full=abs_error,
        notes="official executable status matrix; proxy rows excluded",
    )


def write_markdown(rows: list[RegimeRow], path: Path) -> None:
    lines = [
        "# Official Regime Matrix",
        "",
        "Proxy rows are excluded. GPU-unavailable statuses mean the installed artifact did not expose a runnable official GPU adapter.",
        "",
        "| Family | n | p | kmax | sum 2^k | LC | Full-state | CUAOA | CUDA-Q | QOKit CPU/GPU | QTensor CPU/GPU |",
        "|---|---:|---:|---:|---:|---|---|---|---|---|---|",
    ]
    for r in rows:
        lc = f"{r.lc_status}, {r.lc_seconds:.3g}s, {r.lc_peak_mb:.3g}MB"
        full = f"{r.full_status}, {r.full_seconds:.3g}s, {r.full_peak_mb:.3g}MB"
        cuaoa = f"{r.cuaoa_status}, {r.cuaoa_seconds:.3g}s"
        cudaq = f"{r.cudaq_status}, {r.cudaq_seconds:.3g}s"
        qokit = f"{r.qokit_cpu_status}, {r.qokit_cpu_seconds:.3g}s / {r.qokit_gpu_status}"
        qtensor = f"{r.qtensor_cpu_status}, {r.qtensor_cpu_seconds:.3g}s / {r.qtensor_gpu_status}"
        lines.append(
            f"| {r.family} | {r.n} | {r.p} | {r.kmax} | {r.total_cone_states} | {lc} | {full} | {cuaoa} | {cudaq} | {qokit} | {qtensor} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "official_regime_matrix.csv")
    parser.add_argument("--markdown", type=Path, default=ROOT / "results" / "official_regime_matrix.md")
    parser.add_argument("--work-dir", type=Path, default=ROOT / "results" / "official_regime_matrix_cases")
    parser.add_argument("--max-k", type=int, default=24)
    parser.add_argument("--max-batch-states", type=int, default=1 << 21)
    parser.add_argument("--full-cap", type=int, default=28)
    parser.add_argument("--peer-timeout", type=int, default=90)
    parser.add_argument("--qokit-timeout", type=int, default=220)
    parser.add_argument("--qokit-cpu-max-n", type=int, default=24)
    parser.add_argument("--cuaoa-max-n", type=int, default=30)
    parser.add_argument("--cudaq-max-n", type=int, default=30)
    parser.add_argument("--qtensor-max-n", type=int, default=128)
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
    qokit_gpu = qokit_gpu_source_status()
    qtensor_gpu = qtensor_gpu_status()
    rows: list[RegimeRow] = []
    for family, n, p in cases:
        print(f"RUN {family} n={n} p={p}", flush=True)
        rows.append(run_case(family, n, p, args, qokit_gpu, qtensor_gpu))

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
