from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from lcqaoa.backend import get_backend
from lcqaoa.graphs import random_regular_graph
from lcqaoa.lightcone import LightConeProblem, _x_sum_batched, extract_lightcones
from lcqaoa.qaoa import apply_mixer_batched_inplace, cost_table, term_table
from benchmark_common import params_for_depth


@dataclass
class MicrobatchRow:
    n: int
    p: int
    seed: int
    repeat: int
    mode: str
    cap_terms_at_kmax: int
    max_batch_states: int
    kmax: int
    bucket_k: int
    bucket_terms: int
    bucket_Bk: int
    bucket_batches: int
    total_terms: int
    total_batches: int
    seconds: float
    terms_per_second: float
    value: float
    status: str
    peak_allocated_mb: float
    peak_reserved_mb: float
    predicted_active_mb: float
    predicted_forward_state_mb: float
    predicted_temp_mb: float
    predicted_checkpoint_mb: float
    measured_predicted_ratio: float
    backend: str
    notes: str


def mempool_stats(xp, gpu: bool) -> tuple[int, int]:
    if not gpu:
        return 0, 0
    pool = xp.get_default_memory_pool()
    return int(pool.used_bytes()), int(pool.total_bytes())


def update_peak(xp, gpu: bool, peak_used: int, peak_total: int) -> tuple[int, int]:
    used, total = mempool_stats(xp, gpu)
    return max(peak_used, used), max(peak_total, total)


def split_batches(items: list[LightConeProblem], max_batch_states: int) -> list[list[LightConeProblem]]:
    batches: list[list[LightConeProblem]] = []
    current: list[LightConeProblem] = []
    current_states = 0
    for item in items:
        states = 1 << item.k
        if current and current_states + states > max_batch_states:
            batches.append(current)
            current = []
            current_states = 0
        current.append(item)
        current_states += states
    if current:
        batches.append(current)
    return batches


def predicted_bytes(mode: str, active_states: int, p: int, complex_dtype, float_dtype) -> tuple[int, int, int, int]:
    c = np.dtype(complex_dtype).itemsize
    f = np.dtype(float_dtype).itemsize
    forward = active_states * (c + 2 * f)  # psi, local cost, local term table
    if mode == "objective":
        temp = active_states * (c + 2 * f)  # phase/mixer/probability temporaries
        checkpoint = 0
    else:
        checkpoint = active_states * (2 * p * c)  # after-cost and after-layer states
        temp = active_states * (3 * c + f)  # adjoint, x-sum, mixer copy, reductions
    return forward + temp + checkpoint, forward, temp, checkpoint


def evaluate_batch_instrumented(
    batch: list[LightConeProblem],
    gammas: list[float],
    betas: list[float],
    objective: str,
    mode: str,
    xp,
    gpu: bool,
    complex_dtype,
    float_dtype,
) -> tuple[float, np.ndarray | None, int, int]:
    p = len(gammas)
    k = batch[0].k
    nstates = 1 << k
    bsz = len(batch)
    peak_used, peak_total = update_peak(xp, gpu, 0, 0)

    psi = xp.empty((bsz, nstates), dtype=complex_dtype)
    psi.fill(1.0 / math.sqrt(nstates))
    cost = xp.empty((bsz, nstates), dtype=float_dtype)
    terms = xp.empty((bsz, nstates), dtype=float_dtype)
    peak_used, peak_total = update_peak(xp, gpu, peak_used, peak_total)

    for row, problem in enumerate(batch):
        cost[row, :] = cost_table(k, problem.edges, problem.fields, objective, xp, float_dtype)
        terms[row, :] = term_table(
            k,
            problem.term_kind,
            problem.local_term_nodes,
            problem.weight,
            objective,
            xp,
            float_dtype,
        )
        peak_used, peak_total = update_peak(xp, gpu, peak_used, peak_total)

    after_costs = []
    after_layers = []
    for gamma, beta in zip(gammas, betas):
        psi *= xp.exp((-1j * float(gamma)) * cost)
        if mode == "adjoint":
            after_costs.append(psi.copy())
        peak_used, peak_total = update_peak(xp, gpu, peak_used, peak_total)
        apply_mixer_batched_inplace(psi, k, beta, xp)
        if mode == "adjoint":
            after_layers.append(psi.copy())
        peak_used, peak_total = update_peak(xp, gpu, peak_used, peak_total)

    values = xp.sum((xp.abs(psi) ** 2) * terms, axis=1)
    total = xp.sum(values)
    peak_used, peak_total = update_peak(xp, gpu, peak_used, peak_total)

    grad = None
    if mode == "adjoint":
        grad_gamma = xp.zeros(p, dtype=xp.float64)
        grad_beta = xp.zeros(p, dtype=xp.float64)
        adjoint = terms * psi
        peak_used, peak_total = update_peak(xp, gpu, peak_used, peak_total)
        for layer in range(p - 1, -1, -1):
            x_sum = _x_sum_batched(after_layers[layer], k, xp)
            d_beta_state = -1j * x_sum
            grad_beta[layer] = 2.0 * xp.real(xp.sum(xp.conj(adjoint) * d_beta_state))
            apply_mixer_batched_inplace(adjoint, k, -float(betas[layer]), xp)
            d_gamma_state = -1j * cost * after_costs[layer]
            grad_gamma[layer] = 2.0 * xp.real(xp.sum(xp.conj(adjoint) * d_gamma_state))
            adjoint *= xp.exp((1j * float(gammas[layer])) * cost)
            peak_used, peak_total = update_peak(xp, gpu, peak_used, peak_total)
        g = xp.concatenate([grad_gamma, grad_beta])
        grad = g.get() if hasattr(g, "get") else np.asarray(g)

    if gpu:
        xp.cuda.Stream.null.synchronize()
    value = float(total.get() if hasattr(total, "get") else total)
    return value, grad, peak_used, peak_total


def bucket_rows(
    n: int,
    p: int,
    seed: int,
    repeat: int,
    mode: str,
    cap: int,
    max_batch_states: int,
    problems: list[LightConeProblem],
    seconds: float,
    value: float,
    status: str,
    peak_used: int,
    peak_total: int,
    predicted: tuple[int, int, int, int],
    backend: str,
    notes: str,
) -> list[MicrobatchRow]:
    by_k: dict[int, list[LightConeProblem]] = {}
    for problem in problems:
        by_k.setdefault(problem.k, []).append(problem)
    total_batches = 0
    bucket_meta: dict[int, tuple[int, int]] = {}
    for k, group in by_k.items():
        bk = max(1, max_batch_states // (1 << k))
        batches = math.ceil(len(group) / bk)
        total_batches += batches
        bucket_meta[k] = (bk, batches)
    kmax = max(by_k) if by_k else 0
    total_terms = len(problems)
    pred_total, pred_forward, pred_temp, pred_checkpoint = predicted
    ratio = (peak_used / pred_total) if pred_total > 0 else float("nan")
    rows = []
    for k in sorted(by_k):
        bk, batches = bucket_meta[k]
        rows.append(
            MicrobatchRow(
                n=n,
                p=p,
                seed=seed,
                repeat=repeat,
                mode=mode,
                cap_terms_at_kmax=cap,
                max_batch_states=max_batch_states,
                kmax=kmax,
                bucket_k=k,
                bucket_terms=len(by_k[k]),
                bucket_Bk=bk,
                bucket_batches=batches,
                total_terms=total_terms,
                total_batches=total_batches,
                seconds=seconds,
                terms_per_second=(total_terms / seconds) if seconds > 0 and status == "ok" else float("nan"),
                value=value,
                status=status,
                peak_allocated_mb=peak_used / 1024**2,
                peak_reserved_mb=peak_total / 1024**2,
                predicted_active_mb=pred_total / 1024**2,
                predicted_forward_state_mb=pred_forward / 1024**2,
                predicted_temp_mb=pred_temp / 1024**2,
                predicted_checkpoint_mb=pred_checkpoint / 1024**2,
                measured_predicted_ratio=ratio,
                backend=backend,
                notes=notes,
            )
        )
    return rows


def run_one(
    n: int,
    p: int,
    seed_id: int,
    repeat: int,
    cap: int,
    mode: str,
    args,
    *,
    graph=None,
    problems: list[LightConeProblem] | None = None,
    gammas: list[float] | None = None,
    betas: list[float] | None = None,
) -> list[MicrobatchRow]:
    if graph is None:
        graph = random_regular_graph(n, 3, seed=310000 + 101 * seed_id + n)
    if gammas is None or betas is None:
        gammas, betas = params_for_depth(p, seed=seed_id)
    if problems is None:
        problems = extract_lightcones(graph, p)
    kmax = max(problem.k for problem in problems)
    max_batch_states = cap * (1 << args.kmax_reference)

    backend = get_backend(prefer_gpu=not args.cpu)
    xp = backend.xp
    gpu = backend.gpu
    backend.free_memory_pool()

    if kmax > args.max_k:
        return bucket_rows(
            n,
            p,
            seed_id,
            repeat,
            mode,
            cap,
            max_batch_states,
            problems,
            0.0,
            float("nan"),
            f"skipped_kmax_{kmax}_over_{args.max_k}",
            0,
            0,
            (0, 0, 0, 0),
            backend.name,
            "guardrail",
        )

    groups: dict[int, list[LightConeProblem]] = {}
    for problem in problems:
        groups.setdefault(problem.k, []).append(problem)

    total_value = 0.0
    total_grad = np.zeros(2 * p, dtype=np.float64)
    peak_used, peak_total = 0, 0
    pred_peak = (0, 0, 0, 0)
    status = "ok"
    t0 = time.perf_counter()
    try:
        for k in sorted(groups):
            for batch in split_batches(groups[k], max_batch_states):
                active_states = len(batch) * (1 << k)
                pred = predicted_bytes(mode, active_states, p, args.complex_dtype, args.float_dtype)
                if pred[0] > pred_peak[0]:
                    pred_peak = pred
                value, grad, used, total = evaluate_batch_instrumented(
                    batch,
                    gammas,
                    betas,
                    graph.objective,
                    mode,
                    xp,
                    gpu,
                    args.complex_dtype,
                    args.float_dtype,
                )
                total_value += value
                if grad is not None:
                    total_grad += grad
                peak_used = max(peak_used, used)
                peak_total = max(peak_total, total)
    except Exception as exc:
        status = f"failed:{type(exc).__name__}"
        notes = str(exc)[:240]
    else:
        notes = f"grad_norm={float(np.linalg.norm(total_grad)):.6g}" if mode == "adjoint" else ""
    seconds = time.perf_counter() - t0
    return bucket_rows(
        n,
        p,
        seed_id,
        repeat,
        mode,
        cap,
        max_batch_states,
        problems,
        seconds,
        total_value,
        status,
        peak_used,
        peak_total,
        pred_peak,
        backend.name,
        notes,
    )


def write_md(rows: list[MicrobatchRow], path: Path) -> None:
    lines = [
        "# P0-1 Microbatch Memory-Model Verification",
        "",
        "Each row in the CSV records one size bucket; the summary below reports the largest observed bucket-level peak for each run.",
        "",
        "| n | mode | B@k=14 | seeds | repeats | median sec | median reserved MB | median predicted MB | median ratio | status |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    keys = sorted({(r.n, r.mode, r.cap_terms_at_kmax) for r in rows})
    for key in keys:
        sub_all = [r for r in rows if (r.n, r.mode, r.cap_terms_at_kmax) == key]
        run_keys = sorted({(r.seed, r.repeat) for r in sub_all})
        per_run = []
        statuses = []
        for rk in run_keys:
            sub = [r for r in sub_all if (r.seed, r.repeat) == rk]
            statuses.extend(r.status for r in sub)
            per_run.append(
                (
                    max(r.seconds for r in sub),
                    max(r.peak_reserved_mb for r in sub),
                    max(r.predicted_active_mb for r in sub),
                    max(r.measured_predicted_ratio for r in sub),
                )
            )
        ok = sum(1 for s in statuses if s == "ok")
        total = len(statuses)
        med = lambda idx: float(np.nanmedian([x[idx] for x in per_run])) if per_run else float("nan")
        lines.append(
            f"| {key[0]} | {key[1]} | {key[2]} | {len(set(r.seed for r in sub_all))} | {len(set(r.repeat for r in sub_all))} | "
            f"{med(0):.4g} | {med(1):.4g} | {med(2):.4g} | {med(3):.4g} | {ok}/{total} bucket rows ok |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_dtype(name: str):
    return {
        "complex64": np.complex64,
        "complex128": np.complex128,
        "float32": np.float32,
        "float64": np.float64,
    }[name]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "aaai27_required_experiments_20260710" / "P0_1_microbatch_memory")
    parser.add_argument("--ns", nargs="*", type=int, default=[512, 2048, 8192, 16384])
    parser.add_argument("--p", type=int, default=2)
    parser.add_argument("--caps", nargs="*", type=int, default=[16, 32, 64, 128, 256, 512, 1024])
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-k", type=int, default=24)
    parser.add_argument("--kmax-reference", type=int, default=14)
    parser.add_argument("--complex-dtype", type=parse_dtype, default=np.complex64)
    parser.add_argument("--float-dtype", type=parse_dtype, default=np.float32)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if args.quick:
        args.ns = [512]
        args.caps = [16, 128]
        args.seeds = 1
        args.repeats = 1
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[MicrobatchRow] = []
    csv_path = args.out_dir / "P0_1_microbatch_memory.csv"
    for n in args.ns:
        seed_count = max(args.seeds, 2 if n == 16384 and not args.quick else args.seeds)
        for seed in range(seed_count):
            print(f"P0-1 prepare graph/lightcones n={n} p={args.p} seed={seed}", flush=True)
            graph = random_regular_graph(n, 3, seed=310000 + 101 * seed + n)
            problems = extract_lightcones(graph, args.p)
            gammas, betas = params_for_depth(args.p, seed=seed)
            for cap in args.caps:
                for repeat in range(args.repeats):
                    for mode in ["objective", "adjoint"]:
                        print(f"P0-1 n={n} p={args.p} seed={seed} repeat={repeat} cap={cap} mode={mode}", flush=True)
                        rows.extend(
                            run_one(
                                n,
                                args.p,
                                seed,
                                repeat,
                                cap,
                                mode,
                                args,
                                graph=graph,
                                problems=problems,
                                gammas=gammas,
                                betas=betas,
                            )
                        )
                        with csv_path.open("w", newline="", encoding="utf-8") as f:
                            writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
                            writer.writeheader()
                            for row in rows:
                                writer.writerow(asdict(row))
                        write_md(rows, args.out_dir / "P0_1_microbatch_memory.md")
    print(f"WROTE {csv_path}")


if __name__ == "__main__":
    main()
