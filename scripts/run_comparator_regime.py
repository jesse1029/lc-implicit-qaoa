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

from lcqaoa.graphs import (
    WeightedGraph,
    erdos_renyi_graph,
    modular_graph,
    random_regular_graph,
    scale_free_graph,
    weighted_modular_qubo_graph,
    weighted_qubo_graph,
)
from lcqaoa.lightcone import lightcone_expectation, lightcone_gradient_adjoint
from lcqaoa.qaoa import full_state_expectation
from benchmark_common import cone_metrics, graph_metrics, normalize_status, params_for_depth

PREFIX = Path.home() / "lc_implicit_qaoa_peers"
CORE_PY = PREFIX / "venvs" / "lcqaoa-core" / "bin" / "python"
QTENSOR_PY = PREFIX / "venvs" / "qtensor-py310" / "bin" / "python"
if not QTENSOR_PY.exists():
    QTENSOR_PY = PREFIX / "venvs" / "qtensor-py38" / "bin" / "python"
CUAOA_PY = PREFIX / "venvs" / "cuaoa-py312" / "bin" / "python"


@dataclass
class A1Row:
    family: str
    objective: str
    n: int
    p: int
    seed: int
    mean_degree: float
    max_degree: int
    degeneracy: float
    clustering: float
    m: int
    fields: int
    kmax: int
    k_median: float
    k_p95: float
    total_cone_states: int
    max_batch_state_elements: int
    lc_obj_status: str
    lc_obj_seconds: float
    lc_grad_status: str
    lc_grad_seconds: float
    lc_peak_mb: float
    full_precompute_status: str
    full_precompute_seconds: float
    full_precompute_peak_mb: float
    full_implicit_status: str
    full_implicit_seconds: float
    full_implicit_peak_mb: float
    cuaoa_status: str
    cuaoa_seconds: float
    qokit_status: str
    qokit_seconds: float
    cudaq_status: str
    cudaq_seconds: float
    qtensor_cpu_status: str
    qtensor_cpu_seconds: float
    qtensor_gpu_status: str
    qtensor_gpu_seconds: float
    juliqaoa_status: str
    juliqaoa_seconds: float
    mps_juliqaoa_status: str
    mps_juliqaoa_seconds: float
    qblaze_status: str
    qblaze_seconds: float
    exact_error: float
    failure_reason: str


def make_graph(family: str, n: int, seed: int) -> WeightedGraph:
    if family == "3regular":
        return random_regular_graph(n, 3, seed=seed)
    if family == "er_deg2":
        return erdos_renyi_graph(n, min(0.45, 2.0 / max(2, n)), seed=seed)
    if family == "er_deg3":
        return erdos_renyi_graph(n, min(0.45, 3.0 / max(2, n)), seed=seed)
    if family == "er_dense":
        return erdos_renyi_graph(n, 0.35, seed=seed)
    if family == "scale_free":
        return scale_free_graph(n, attachment=2, seed=seed)
    if family == "modular_dense":
        return modular_graph(n, modules=max(4, n // 24), p_in=0.34, p_out=0.015, seed=seed)
    if family == "qubo_modular_sparse":
        return weighted_modular_qubo_graph(n, modules=max(4, n // 16), p_in=0.14, p_out=0.0015, seed=seed)
    if family == "weighted_sparse_qubo":
        return weighted_qubo_graph(n, min(0.40, 2.0 / max(2, n)), seed=seed, field_scale=0.7)
    if family == "qubo_modular_dense":
        return weighted_modular_qubo_graph(n, modules=max(4, n // 24), p_in=0.34, p_out=0.012, seed=seed)
    raise ValueError(f"unknown family: {family}")


def case_grid() -> list[tuple[str, list[int], list[int]]]:
    return [
        ("3regular", [24, 32, 48, 64, 96, 128, 256, 512], [1, 2, 3]),
        ("er_deg2", [24, 32, 48, 64, 96, 128, 256], [1, 2, 3]),
        ("er_deg3", [24, 32, 48, 64, 96, 128], [1, 2]),
        ("qubo_modular_sparse", [24, 64, 128, 256, 512], [1, 2]),
        ("weighted_sparse_qubo", [24, 48, 64, 96, 128, 256], [1, 2]),
        ("er_dense", [24, 32, 48, 64, 96, 128], [1, 2, 3]),
        ("scale_free", [24, 32, 48, 64, 96, 128], [1, 2, 3]),
        ("qubo_modular_dense", [24, 32, 48, 64, 96, 128], [1, 2, 3]),
    ]


def write_case_json(path: Path, graph: WeightedGraph, gammas: list[float], betas: list[float]) -> None:
    path.write_text(
        json.dumps(
            {
                "n": graph.n,
                "edges": [list(e) for e in graph.edges],
                "fields": [list(f) for f in graph.fields],
                "objective": graph.objective,
                "gammas": gammas,
                "betas": betas,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def run_external(method: str, case_path: Path, out_path: Path, py: Path, timeout: int, extra: list[str] | None = None) -> tuple[str, float, float, str]:
    if not py.exists():
        return "NOT_RUN_EXPLAINED", 0.0, float("nan"), f"missing python {py}"
    cmd = [str(py), str(ROOT / "scripts" / "external_peer_runner.py"), "--method", method, "--case", str(case_path), "--out", str(out_path)]
    if extra:
        cmd.extend(extra)
    try:
        cp = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return "TIMEOUT", float(timeout), float("nan"), f"timeout_{timeout}s"
    if out_path.exists():
        data = json.loads(out_path.read_text(encoding="utf-8"))
    else:
        return f"FAILED_RC_{cp.returncode}", 0.0, float("nan"), cp.stdout[-500:]
    status = normalize_status(data.get("status", ""))
    return status, float(data.get("seconds", 0.0) or 0.0), float(data.get("value", float("nan"))), str(data.get("notes", ""))[:500]


def peer_not_run(method: str, graph: WeightedGraph, n: int, args) -> tuple[str, float, float, str]:
    if args.peer_mode == "none":
        return "NOT_RUN_EXPLAINED", 0.0, float("nan"), "peer_mode_none; LC/status matrix only"
    if graph.objective != "maxcut" and method in {"cuaoa", "qokit", "cudaq", "qtensor_cpu", "qtensor_gpu", "juliqaoa", "mps_juliqaoa", "qblaze"}:
        return "UNSUPPORTED_OBJECTIVE", 0.0, float("nan"), f"{method} adapter is MaxCut-only in this artifact"
    if n > args.peer_exact_max_n and method in {"cuaoa", "qokit", "cudaq", "juliqaoa", "mps_juliqaoa", "qblaze"}:
        return "NOT_RUN_EXPLAINED", 0.0, float("nan"), f"global-state peer guarded above n={args.peer_exact_max_n}"
    if n > args.qtensor_max_n and method.startswith("qtensor"):
        return "NOT_RUN_EXPLAINED", 0.0, float("nan"), f"QTensor guarded above n={args.qtensor_max_n}"
    return "", 0.0, float("nan"), ""


def eval_full(graph: WeightedGraph, gammas: list[float], betas: list[float], method: str, cap: int) -> tuple[str, float, float, float]:
    if graph.n > cap:
        return "NOT_RUN_EXPLAINED", 0.0, 0.0, float("nan")
    try:
        stats = full_state_expectation(graph, gammas, betas, method=method, prefer_gpu=True, max_qubits=None)
        return normalize_status(stats.status), stats.seconds, stats.peak_pool_bytes / 1024**2, stats.value
    except Exception as exc:
        return normalize_status(f"failed:{type(exc).__name__}"), 0.0, 0.0, float("nan")


def run_case(family: str, n: int, p: int, seed_id: int, args) -> A1Row:
    seed = 240000 + 997 * seed_id + 37 * n + 131 * p
    graph = make_graph(family, n, seed)
    gammas, betas = params_for_depth(p, seed=seed_id)
    gmet = graph_metrics(graph)
    cmet = cone_metrics(graph, p)
    failure: list[str] = []

    lc_obj_status = "NOT_RUN_EXPLAINED"
    lc_obj_seconds = 0.0
    lc_grad_status = "NOT_RUN_EXPLAINED"
    lc_grad_seconds = 0.0
    lc_peak = 0.0
    lc_value = float("nan")
    if cmet["kmax"] > args.max_k:
        failure.append(f"LC kmax {cmet['kmax']} exceeds max_k {args.max_k}")
        lc_obj_status = f"NOT_RUN_EXPLAINED"
        lc_grad_status = f"NOT_RUN_EXPLAINED"
    elif cmet["total_cone_states"] > args.max_total_cone_states:
        failure.append(f"LC total cone states {cmet['total_cone_states']} exceeds cap {args.max_total_cone_states}")
    else:
        try:
            lc = lightcone_expectation(graph, gammas, betas, p=p, prefer_gpu=True, max_k=args.max_k, max_batch_states=args.max_batch_states)
            lc_obj_status = normalize_status(lc.status)
            lc_obj_seconds = lc.seconds
            lc_peak = max(lc_peak, lc.peak_pool_bytes / 1024**2)
            lc_value = lc.value
        except Exception as exc:
            lc_obj_status = normalize_status(f"failed:{type(exc).__name__}")
            failure.append(str(exc)[:180])
        try:
            grad = lightcone_gradient_adjoint(graph, gammas, betas, p=p, prefer_gpu=True, max_k=args.max_k, max_batch_states=args.max_batch_states)
            lc_grad_status = normalize_status(grad.status)
            lc_grad_seconds = grad.seconds
            lc_peak = max(lc_peak, grad.peak_pool_bytes / 1024**2)
        except Exception as exc:
            lc_grad_status = normalize_status(f"failed:{type(exc).__name__}")
            failure.append(str(exc)[:180])

    full_pre_s, full_pre_t, full_pre_mb, full_pre_val = eval_full(graph, gammas, betas, "precompute", args.full_cap)
    full_imp_s, full_imp_t, full_imp_mb, _ = eval_full(graph, gammas, betas, "implicit", args.full_cap)
    exact_error = abs(lc_value - full_pre_val) if math.isfinite(lc_value) and math.isfinite(full_pre_val) else float("nan")

    work_dir = args.out_dir / "cases"
    work_dir.mkdir(parents=True, exist_ok=True)
    case_path = work_dir / f"{family}_n{n}_p{p}_seed{seed_id}.json"
    write_case_json(case_path, graph, gammas, betas)

    def maybe(method: str, py: Path, external_name: str, extra: list[str] | None = None, max_n_override: int | None = None) -> tuple[str, float, float, str]:
        if args.peer_mode != "none" and seed_id >= args.peer_seeds:
            return "NOT_RUN_EXPLAINED", 0.0, float("nan"), f"peer run restricted to first {args.peer_seeds} seed(s)"
        nr_s, nr_t, nr_v, nr_note = peer_not_run(method, graph, n, args)
        if nr_s:
            return nr_s, nr_t, nr_v, nr_note
        if max_n_override is not None and n > max_n_override:
            return "NOT_RUN_EXPLAINED", 0.0, float("nan"), f"{method} guarded above n={max_n_override}"
        return run_external(external_name, case_path, work_dir / f"{method}_{family}_n{n}_p{p}_seed{seed_id}.json", py, args.peer_timeout, extra)

    cuaoa_s, cuaoa_t, _, cuaoa_note = maybe("cuaoa", CUAOA_PY, "cuaoa_gpu", max_n_override=args.cuaoa_max_n)
    qokit_s, qokit_t, _, qokit_note = maybe("qokit", CORE_PY, "qokit_cpu", max_n_override=args.qokit_cpu_max_n)
    cudaq_s, cudaq_t, _, cudaq_note = maybe("cudaq", CORE_PY, "cudaq_observe", max_n_override=args.cudaq_max_n)
    qtensor_s, qtensor_t, _, qtensor_note = maybe("qtensor_cpu", QTENSOR_PY, "qtensor_cpu", ["--qtensor-transform", "paper_qiskit_inverse"], max_n_override=args.qtensor_max_n)
    qtensor_gpu_s, qtensor_gpu_t, _, qtensor_gpu_note = maybe("qtensor_gpu", QTENSOR_PY, "qtensor_gpu", ["--qtensor-transform", "paper_qiskit_inverse"], max_n_override=args.qtensor_gpu_max_n)
    juli_s, juli_t, _, juli_note = maybe("juliqaoa", CORE_PY, "juliqaoa_cpu", max_n_override=args.juliqaoa_max_n)
    mps_s, mps_t, _, mps_note = maybe("mps_juliqaoa", CORE_PY, "mps_juliqaoa", max_n_override=args.mps_max_n)
    qblaze_s, qblaze_t, _, qblaze_note = maybe("qblaze", CORE_PY, "qblaze_cpu", max_n_override=args.qblaze_max_n)

    for note in [cuaoa_note, qokit_note, cudaq_note, qtensor_note, qtensor_gpu_note, juli_note, mps_note, qblaze_note]:
        if note and ("failed" in note.lower() or "missing" in note.lower()):
            failure.append(note[:160])

    return A1Row(
        family=family,
        objective=graph.objective,
        n=n,
        p=p,
        seed=seed_id,
        mean_degree=gmet["mean_degree"],
        max_degree=int(gmet["max_degree"]),
        degeneracy=gmet["degeneracy"],
        clustering=gmet["clustering"],
        m=int(gmet["m"]),
        fields=len(graph.fields),
        kmax=int(cmet["kmax"]),
        k_median=float(cmet["k_median"]),
        k_p95=float(cmet["k_p95"]),
        total_cone_states=int(cmet["total_cone_states"]),
        max_batch_state_elements=int(cmet["max_batch_state_elements"]),
        lc_obj_status=lc_obj_status,
        lc_obj_seconds=lc_obj_seconds,
        lc_grad_status=lc_grad_status,
        lc_grad_seconds=lc_grad_seconds,
        lc_peak_mb=lc_peak,
        full_precompute_status=full_pre_s,
        full_precompute_seconds=full_pre_t,
        full_precompute_peak_mb=full_pre_mb,
        full_implicit_status=full_imp_s,
        full_implicit_seconds=full_imp_t,
        full_implicit_peak_mb=full_imp_mb,
        cuaoa_status=cuaoa_s,
        cuaoa_seconds=cuaoa_t,
        qokit_status=qokit_s,
        qokit_seconds=qokit_t,
        cudaq_status=cudaq_s,
        cudaq_seconds=cudaq_t,
        qtensor_cpu_status=qtensor_s,
        qtensor_cpu_seconds=qtensor_t,
        qtensor_gpu_status=qtensor_gpu_s,
        qtensor_gpu_seconds=qtensor_gpu_t,
        juliqaoa_status=juli_s,
        juliqaoa_seconds=juli_t,
        mps_juliqaoa_status=mps_s,
        mps_juliqaoa_seconds=mps_t,
        qblaze_status=qblaze_s,
        qblaze_seconds=qblaze_t,
        exact_error=exact_error,
        failure_reason="; ".join(dict.fromkeys(failure)),
    )


def write_summary(rows: list[A1Row], out: Path) -> None:
    groups: dict[tuple[str, int], list[A1Row]] = {}
    for r in rows:
        groups.setdefault((r.family, r.p), []).append(r)
    lines = [
        "# Comparator Regime Measurements",
        "",
        "This table is seed-level in CSV. The markdown summarizes LC feasibility and peer status without proxy rows.",
        "",
        "| Family | p | LC obj success | LC grad success | max n LC obj | median kmax(success) | full precompute OOM/not-run | CUAOA success | QOKit success | CUDA-Q success | QTensor CPU success |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for (family, p), subset in sorted(groups.items()):
        lc_ok = [r for r in subset if r.lc_obj_status == "SUCCESS"]
        grad_ok = [r for r in subset if r.lc_grad_status == "SUCCESS"]
        full_bad = [r for r in subset if r.full_precompute_status != "SUCCESS"]
        q = lambda attr: sum(1 for r in subset if getattr(r, attr) == "SUCCESS")
        kmed = float(np.median([r.kmax for r in lc_ok])) if lc_ok else float("nan")
        max_n = max([r.n for r in lc_ok] or [0])
        lines.append(
            f"| {family} | {p} | {len(lc_ok)}/{len(subset)} | {len(grad_ok)}/{len(subset)} | {max_n} | {kmed:.3g} | "
            f"{len(full_bad)}/{len(subset)} | {q('cuaoa_status')} | {q('qokit_status')} | {q('cudaq_status')} | {q('qtensor_cpu_status')} |"
        )
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "comparator_regime")
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--max-k", type=int, default=24)
    parser.add_argument("--max-total-cone-states", type=int, default=1 << 31)
    parser.add_argument("--max-batch-states", type=int, default=1 << 21)
    parser.add_argument("--full-cap", type=int, default=27)
    parser.add_argument("--peer-timeout", type=int, default=120)
    parser.add_argument("--peer-mode", choices=["none", "small", "all"], default="small")
    parser.add_argument("--peer-seeds", type=int, default=1)
    parser.add_argument("--peer-exact-max-n", type=int, default=24)
    parser.add_argument("--cuaoa-max-n", type=int, default=24)
    parser.add_argument("--qokit-cpu-max-n", type=int, default=24)
    parser.add_argument("--cudaq-max-n", type=int, default=28)
    parser.add_argument("--qtensor-max-n", type=int, default=128)
    parser.add_argument("--qtensor-gpu-max-n", type=int, default=32)
    parser.add_argument("--juliqaoa-max-n", type=int, default=20)
    parser.add_argument("--mps-max-n", type=int, default=24)
    parser.add_argument("--qblaze-max-n", type=int, default=24)
    parser.add_argument("--families", nargs="*", default=None)
    parser.add_argument("--ns", nargs="*", type=int, default=None)
    parser.add_argument("--ps", nargs="*", type=int, default=None)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[A1Row] = []
    grid = case_grid()
    if args.families:
        grid = [g for g in grid if g[0] in set(args.families)]
    if args.ns:
        wanted_ns = set(args.ns)
        grid = [(fam, [n for n in ns if n in wanted_ns], ps) for fam, ns, ps in grid]
        grid = [g for g in grid if g[1]]
    if args.ps:
        wanted_ps = set(args.ps)
        grid = [(fam, ns, [p for p in ps if p in wanted_ps]) for fam, ns, ps in grid]
        grid = [g for g in grid if g[2]]
    if args.quick:
        grid = [(fam, ns[:2], ps[:1]) for fam, ns, ps in grid[:3]]
        args.seeds = min(args.seeds, 2)
    csv_path = args.out_dir / "comparator_regime.csv"
    for family, ns, ps in grid:
        for n in ns:
            for p in ps:
                for seed_id in range(args.seeds):
                    print(f"comparator family={family} n={n} p={p} seed={seed_id}", flush=True)
                    row = run_case(family, n, p, seed_id, args)
                    rows.append(row)
                    with csv_path.open("w", newline="", encoding="utf-8") as f:
                        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
                        writer.writeheader()
                        for r in rows:
                            writer.writerow(asdict(r))
    write_summary(rows, args.out_dir / "comparator_regime.md")
    print(f"WROTE {csv_path}")


if __name__ == "__main__":
    main()
