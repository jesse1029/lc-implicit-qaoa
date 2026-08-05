from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lcqaoa.graphs import erdos_renyi_graph, modular_graph, random_regular_graph
from lcqaoa.lightcone import extract_lightcones, lightcone_expectation
from lcqaoa.proxies import bmqsim_proxy_quantized_expectation, queen_proxy_fused_expectation
from lcqaoa.qaoa import full_state_expectation


PREFIX = Path.home() / "lc_implicit_qaoa_peers"
CORE_PY = PREFIX / "venvs" / "lcqaoa-core" / "bin" / "python"
QTENSOR_PY = PREFIX / "venvs" / "qtensor-py310" / "bin" / "python"
CUAOA_PY = PREFIX / "venvs" / "cuaoa-py312" / "bin" / "python"


@dataclass
class PeerRow:
    family: str
    n: int
    m: int
    p: int
    kmax: int
    method: str
    status: str
    value: float
    seconds: float
    backend: str
    abs_error_vs_full: float
    notes: str


def graph_for(family: str, n: int, seed: int):
    if family == "3regular":
        return random_regular_graph(n, 3, seed=seed)
    if family == "er_sparse":
        return erdos_renyi_graph(n, min(0.18, 3.0 / max(2, n)), seed=seed)
    if family == "modular":
        return modular_graph(n, modules=3, p_in=0.35, p_out=0.02, seed=seed)
    raise ValueError(family)


def params_for(p: int) -> tuple[list[float], list[float]]:
    return [0.20 + 0.05 * i for i in range(p)], [0.32 - 0.035 * i for i in range(p)]


def run_external(py: Path, method: str, case_path: Path, out: Path, extra: list[str] | None = None) -> dict:
    if not py.exists():
        return {
            "method": method,
            "status": f"missing_python:{py}",
            "value": float("nan"),
            "seconds": 0.0,
            "backend": "",
            "notes": "",
        }
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
    cp = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=600)
    if out.exists():
        data = json.loads(out.read_text(encoding="utf-8"))
    else:
        data = {
            "method": method,
            "status": f"process_failed:{cp.returncode}",
            "value": float("nan"),
            "seconds": 0.0,
            "backend": "",
            "notes": cp.stdout[-1000:],
        }
    data["process_returncode"] = cp.returncode
    if cp.returncode != 0 and data.get("status") == "ok":
        data["status"] = f"process_failed:{cp.returncode}"
        data["notes"] = cp.stdout[-1000:]
    return data


def to_row(family: str, graph, p: int, kmax: int, data: dict, ref_value: float, notes: str | None = None) -> PeerRow:
    value = float(data.get("value", float("nan")))
    err = abs(value - ref_value) if math.isfinite(value) and math.isfinite(ref_value) else float("nan")
    return PeerRow(
        family=family,
        n=graph.n,
        m=graph.m,
        p=p,
        kmax=kmax,
        method=str(data.get("method", "")),
        status=str(data.get("status", "")),
        value=value,
        seconds=float(data.get("seconds", 0.0)),
        backend=str(data.get("backend", ""))[:160].replace("\n", " "),
        abs_error_vs_full=err,
        notes=notes if notes is not None else str(data.get("notes", ""))[:240].replace("\n", " "),
    )


def run_case(family: str, n: int, p: int, work_dir: Path) -> list[PeerRow]:
    graph = graph_for(family, n, seed=50000 + n * 19 + p)
    gammas, betas = params_for(p)
    cones = extract_lightcones(graph, p)
    kmax = max(c.k for c in cones)
    rows: list[PeerRow] = []

    ref = full_state_expectation(graph, gammas, betas, method="precompute", prefer_gpu=True, max_qubits=22)
    rows.append(
        to_row(
            family,
            graph,
            p,
            kmax,
            {
                "method": "full_precompute_gpu",
                "status": ref.status,
                "value": ref.value,
                "seconds": ref.seconds,
                "backend": ref.backend,
                "notes": "QOKit/CUAOA-style full-state diagonal materialization",
            },
            ref.value if ref.status == "ok" else float("nan"),
        )
    )
    ref_value = ref.value if ref.status == "ok" else float("nan")

    queen = queen_proxy_fused_expectation(graph, gammas, betas, prefer_gpu=True, fusion_width=4, max_qubits=22)
    rows.append(
        to_row(
            family,
            graph,
            p,
            kmax,
            {
                "method": "queen_proxy_fused_gpu",
                "status": queen.status,
                "value": queen.value,
                "seconds": queen.seconds,
                "backend": queen.backend,
                "notes": "paper-derived QueenV2-style exact full-state fused-mixer proxy; not official QueenV2",
            },
            ref_value,
        )
    )

    bmq = bmqsim_proxy_quantized_expectation(
        graph,
        gammas,
        betas,
        prefer_gpu=True,
        quant_bits=8,
        block_states=1 << 16,
        max_qubits=22,
    )
    rows.append(
        to_row(
            family,
            graph,
            p,
            kmax,
            {
                "method": "bmqsim_proxy_quantized_gpu",
                "status": bmq.status,
                "value": bmq.value,
                "seconds": bmq.seconds,
                "backend": bmq.backend,
                "notes": "paper-derived BMQSim-style 8-bit block-quantized state checkpoint proxy; not official BMQSim",
            },
            ref_value,
        )
    )

    implicit = full_state_expectation(graph, gammas, betas, method="implicit", prefer_gpu=True, max_qubits=22)
    rows.append(
        to_row(
            family,
            graph,
            p,
            kmax,
            {
                "method": "full_implicit_gpu",
                "status": implicit.status,
                "value": implicit.value,
                "seconds": implicit.seconds,
                "backend": implicit.backend,
                "notes": "full-state route without persistent global cost vector",
            },
            ref_value,
        )
    )

    lc = lightcone_expectation(graph, gammas, betas, p=p, prefer_gpu=True, max_k=24, max_batch_states=1 << 21)
    rows.append(
        to_row(
            family,
            graph,
            p,
            kmax,
            {
                "method": "lc_batched_gpu",
                "status": lc.status,
                "value": lc.value,
                "seconds": lc.seconds,
                "backend": lc.backend,
                "notes": "LC-Implicit-QAOA exact batched light-cone evaluator",
            },
            ref_value,
        )
    )

    case = {
        "n": graph.n,
        "edges": [list(edge) for edge in graph.edges],
        "gammas": gammas,
        "betas": betas,
        "objective": graph.objective,
    }
    case_path = work_dir / f"case_{family}_n{n}_p{p}.json"
    case_path.write_text(json.dumps(case, indent=2), encoding="utf-8")

    externals = [
        (CORE_PY, "qokit_cpu", []),
        (CUAOA_PY, "cuaoa_gpu", []),
        (CORE_PY, "cudaq_observe", []),
        (CORE_PY, "qblaze_cpu", []),
        (CORE_PY, "juliqaoa_cpu", []),
        (CORE_PY, "mps_juliqaoa", []),
        (QTENSOR_PY, "qtensor_cpu", ["--qtensor-transform", "paper_qiskit_inverse"]),
    ]
    for py, method, extra in externals:
        out = work_dir / f"{method}_{family}_n{n}_p{p}.json"
        data = run_external(py, method, case_path, out, extra)
        rows.append(to_row(family, graph, p, kmax, data, ref_value))

    return rows


def write_markdown(rows: list[PeerRow], path: Path) -> None:
    lines = [
        "# External Peer Benchmark",
        "",
        "| Family | n | m | p | kmax | Method | Status | Value | Error | Seconds | Backend | Notes |",
        "|---|---:|---:|---:|---:|---|---|---:|---:|---:|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r.family} | {r.n} | {r.m} | {r.p} | {r.kmax} | {r.method} | {r.status} | "
            f"{r.value:.7g} | {r.abs_error_vs_full:.3g} | {r.seconds:.4g} | {r.backend} | {r.notes} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "external_peer_benchmark.csv")
    parser.add_argument("--markdown", type=Path, default=ROOT / "results" / "external_peer_benchmark.md")
    parser.add_argument("--work-dir", type=Path, default=ROOT / "results" / "external_peer_cases")
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.work_dir.mkdir(parents=True, exist_ok=True)

    cases = [
        ("3regular", 10, 1),
        ("3regular", 10, 2),
        ("3regular", 14, 1),
        ("3regular", 14, 2),
        ("er_sparse", 14, 1),
        ("er_sparse", 14, 2),
        ("modular", 16, 1),
        ("modular", 16, 2),
    ]

    rows: list[PeerRow] = []
    for family, n, p in cases:
        print(f"RUN external family={family} n={n} p={p}", flush=True)
        rows.extend(run_case(family, n, p, args.work_dir))

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
