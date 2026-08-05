from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from lcqaoa.graphs import random_regular_graph, weighted_modular_qubo_graph, weighted_qubo_graph
from lcqaoa.lightcone import lightcone_expectation, lightcone_gradient_adjoint
from benchmark_common import cone_metrics, params_for_depth


@dataclass
class PrecisionRow:
    case: str
    family: str
    n: int
    p: int
    seed: int
    precision: str
    complex_dtype: str
    float_dtype: str
    kmax: int
    total_cone_states: int
    objective_status: str
    gradient_status: str
    objective_value: float
    objective_seconds: float
    gradient_seconds: float
    peak_mb: float
    abs_obj_error_vs_c128f64: float
    rel_obj_error_vs_c128f64: float
    rel_grad_l2_error_vs_c128f64: float
    max_grad_abs_error_vs_c128f64: float
    grad_cosine_vs_c128f64: float
    adam_steps: int
    adam_final_objective: float
    adam_wall_seconds: float
    adam_status: str


def make_graph(family: str, n: int, seed: int):
    if family == "3regular":
        return random_regular_graph(n, 3, seed=seed)
    if family == "weighted_qubo_er2":
        return weighted_qubo_graph(n, min(0.40, 2.0 / max(2, n)), seed=seed, field_scale=0.7)
    if family == "qubo_modular_sparse":
        return weighted_modular_qubo_graph(n, modules=max(4, n // 16), p_in=0.12, p_out=0.0015, seed=seed)
    raise ValueError(family)


def case_specs() -> list[tuple[str, int, int]]:
    return [
        ("3regular", 24, 2),
        ("3regular", 128, 2),
        ("3regular", 512, 2),
        ("3regular", 24, 3),
        ("weighted_qubo_er2", 96, 2),
        ("qubo_modular_sparse", 128, 1),
    ]


def adam_short_run(graph, gammas, betas, p: int, complex_dtype, float_dtype, steps: int, lr: float, args) -> tuple[float, float, str]:
    import time

    theta = np.asarray(list(gammas) + list(betas), dtype=np.float64)
    m = np.zeros_like(theta)
    v = np.zeros_like(theta)
    beta1, beta2 = 0.9, 0.999
    t0 = time.perf_counter()
    status = "ok"
    value = float("nan")
    for step in range(1, steps + 1):
        gams = theta[:p].tolist()
        bets = theta[p:].tolist()
        grad = lightcone_gradient_adjoint(
            graph,
            gams,
            bets,
            p=p,
            prefer_gpu=True,
            max_k=args.max_k,
            max_batch_states=args.max_batch_states,
            complex_dtype=complex_dtype,
            float_dtype=float_dtype,
        )
        if grad.gradient is None or grad.status != "ok":
            status = grad.status
            break
        value = grad.value
        g = -np.asarray(grad.gradient, dtype=np.float64)
        m = beta1 * m + (1.0 - beta1) * g
        v = beta2 * v + (1.0 - beta2) * (g * g)
        theta -= lr * (m / (1.0 - beta1**step)) / (np.sqrt(v / (1.0 - beta2**step)) + 1e-8)
        theta = ((theta + math.pi) % (2.0 * math.pi)) - math.pi
    return value, time.perf_counter() - t0, status


def run_case(family: str, n: int, p: int, seed_id: int, args) -> list[PrecisionRow]:
    seed = 910000 + 1009 * seed_id + 37 * n + 131 * p
    graph = make_graph(family, n, seed)
    gammas, betas = params_for_depth(p, seed=seed_id)
    cmet = cone_metrics(graph, p)
    case = f"{family}_n{n}_p{p}"
    configs = [
        ("complex64_float32", np.complex64, np.float32),
        ("complex64_float64_mixed", np.complex64, np.float64),
        ("complex128_float64", np.complex128, np.float64),
    ]
    ref_value = float("nan")
    ref_grad = None
    tmp: list[dict] = []
    for name, cdtype, fdtype in configs:
        print(f"P1-3 case={case} seed={seed_id} precision={name}", flush=True)
        try:
            obj = lightcone_expectation(
                graph,
                gammas,
                betas,
                p=p,
                prefer_gpu=True,
                max_k=args.max_k,
                max_batch_states=args.max_batch_states,
                complex_dtype=cdtype,
                float_dtype=fdtype,
            )
            grad = lightcone_gradient_adjoint(
                graph,
                gammas,
                betas,
                p=p,
                prefer_gpu=True,
                max_k=args.max_k,
                max_batch_states=args.max_batch_states,
                complex_dtype=cdtype,
                float_dtype=fdtype,
            )
        except Exception as exc:
            obj = type("Obj", (), dict(status=f"failed:{type(exc).__name__}", value=float("nan"), seconds=0.0, peak_pool_bytes=0))()
            grad = type("Grad", (), dict(status=f"failed:{type(exc).__name__}", value=float("nan"), seconds=0.0, peak_pool_bytes=0, gradient=None))()
        if name == "complex128_float64" and grad.gradient is not None and grad.status == "ok":
            ref_value = obj.value
            ref_grad = np.asarray(grad.gradient, dtype=np.float64)
        tmp.append(dict(name=name, cdtype=cdtype, fdtype=fdtype, obj=obj, grad=grad))

    rows: list[PrecisionRow] = []
    for item in tmp:
        obj = item["obj"]
        grad = item["grad"]
        g = np.asarray(grad.gradient, dtype=np.float64) if grad.gradient is not None else None
        if ref_grad is not None and g is not None:
            diff = g - ref_grad
            rel_g = float(np.linalg.norm(diff) / max(np.linalg.norm(ref_grad), 1e-30))
            max_abs = float(np.max(np.abs(diff)))
            cosine = float(np.dot(g, ref_grad) / max(np.linalg.norm(g) * np.linalg.norm(ref_grad), 1e-30))
        else:
            rel_g, max_abs, cosine = float("nan"), float("nan"), float("nan")
        abs_obj = abs(obj.value - ref_value) if math.isfinite(obj.value) and math.isfinite(ref_value) else float("nan")
        rel_obj = abs_obj / max(abs(ref_value), 1e-30) if math.isfinite(abs_obj) else float("nan")
        final_obj, adam_s, adam_status = adam_short_run(
            graph,
            gammas,
            betas,
            p,
            item["cdtype"],
            item["fdtype"],
            args.adam_steps,
            args.adam_lr,
            args,
        )
        rows.append(
            PrecisionRow(
                case=case,
                family=family,
                n=n,
                p=p,
                seed=seed_id,
                precision=item["name"],
                complex_dtype=np.dtype(item["cdtype"]).name,
                float_dtype=np.dtype(item["fdtype"]).name,
                kmax=int(cmet["kmax"]),
                total_cone_states=int(cmet["total_cone_states"]),
                objective_status=obj.status,
                gradient_status=grad.status,
                objective_value=obj.value,
                objective_seconds=obj.seconds,
                gradient_seconds=grad.seconds,
                peak_mb=max(obj.peak_pool_bytes, grad.peak_pool_bytes) / 1024**2,
                abs_obj_error_vs_c128f64=abs_obj,
                rel_obj_error_vs_c128f64=rel_obj,
                rel_grad_l2_error_vs_c128f64=rel_g,
                max_grad_abs_error_vs_c128f64=max_abs,
                grad_cosine_vs_c128f64=cosine,
                adam_steps=args.adam_steps,
                adam_final_objective=final_obj,
                adam_wall_seconds=adam_s,
                adam_status=adam_status,
            )
        )
    return rows


def write_md(rows: list[PrecisionRow], path: Path) -> None:
    lines = [
        "# P1-3 Precision Ablation",
        "",
        "Errors are against the complex128/float64 LC run for the same graph and angles.",
        "",
        "| Case | Precision | Obj s | Grad s | Peak MB | rel grad err | cosine | Adam final | Adam s |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r.case} seed{r.seed} | {r.precision} | {r.objective_seconds:.4g} | {r.gradient_seconds:.4g} | "
            f"{r.peak_mb:.4g} | {r.rel_grad_l2_error_vs_c128f64:.3g} | {r.grad_cosine_vs_c128f64:.6g} | "
            f"{r.adam_final_objective:.6g} | {r.adam_wall_seconds:.4g} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "aaai27_required_experiments_20260710" / "P1_3_precision_ablation")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--adam-steps", type=int, default=30)
    parser.add_argument("--adam-lr", type=float, default=0.04)
    parser.add_argument("--max-k", type=int, default=24)
    parser.add_argument("--max-batch-states", type=int, default=1 << 19)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if args.quick:
        args.seeds = min(args.seeds, 1)
        args.adam_steps = min(args.adam_steps, 3)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[PrecisionRow] = []
    specs = case_specs()[:2] if args.quick else case_specs()
    csv_path = args.out_dir / "P1_3_precision_ablation.csv"
    for family, n, p in specs:
        for seed_id in range(args.seeds):
            rows.extend(run_case(family, n, p, seed_id, args))
            with csv_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
                writer.writeheader()
                for row in rows:
                    writer.writerow(asdict(row))
            write_md(rows, args.out_dir / "P1_3_precision_ablation.md")
    print(f"WROTE {csv_path}")


if __name__ == "__main__":
    main()
