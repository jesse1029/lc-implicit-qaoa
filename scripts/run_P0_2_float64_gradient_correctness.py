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

from lcqaoa.graphs import erdos_renyi_graph, modular_graph, random_regular_graph, weighted_modular_qubo_graph, weighted_qubo_graph
from lcqaoa.lightcone import lightcone_gradient_adjoint
from lcqaoa.qaoa import apply_mixer_inplace, cost_table, expectation_from_state
from benchmark_common import params_for_depth


@dataclass
class P02Row:
    family: str
    n: int
    p: int
    graph_seed: int
    init_id: int
    m: int
    fields: int
    reference: str
    status: str
    value_lc: float
    value_ref: float
    value_abs_error: float
    max_abs_error: float
    relative_l2_error: float
    cosine_similarity: float
    worst_parameter: str
    taylor_slope: float
    taylor_max_abs_remainder: float
    seconds_lc: float
    seconds_ref: float
    seconds_fd: float
    notes: str


def make_graph(family: str, n: int, seed: int):
    if family == "3regular":
        return random_regular_graph(n, 3, seed=seed)
    if family == "er2":
        return erdos_renyi_graph(n, min(0.45, 2.0 / max(2, n)), seed=seed)
    if family == "modular_sparse":
        return modular_graph(n, modules=max(2, min(4, n // 4)), p_in=0.28, p_out=0.015, seed=seed)
    if family == "weighted_qubo_fields":
        return weighted_qubo_graph(n, min(0.45, 2.0 / max(2, n)), field_prob=1.0, seed=seed, weight_scale=1.0, field_scale=0.7)
    if family == "weighted_modular_qubo":
        return weighted_modular_qubo_graph(n, modules=max(2, min(4, n // 4)), p_in=0.18, p_out=0.01, seed=seed)
    raise ValueError(family)


def x_sum_state(psi: np.ndarray, k: int) -> np.ndarray:
    out = np.zeros_like(psi)
    for q in range(k):
        step = 1 << q
        block = step << 1
        src = psi.reshape(-1, block)
        dst = out.reshape(-1, block)
        dst[:, :step] += src[:, step:block]
        dst[:, step:block] += src[:, :step]
    return out


def global_value_and_adjoint(graph, gammas: list[float], betas: list[float]) -> tuple[float, np.ndarray]:
    p = len(gammas)
    nstates = 1 << graph.n
    psi = np.empty(nstates, dtype=np.complex128)
    psi.fill(1.0 / math.sqrt(nstates))
    cost = cost_table(graph.n, graph.edges, graph.fields, graph.objective, np, np.float64)
    after_costs = []
    after_layers = []
    for gamma, beta in zip(gammas, betas):
        psi *= np.exp((-1j * float(gamma)) * cost)
        after_costs.append(psi.copy())
        apply_mixer_inplace(psi, graph.n, beta, np)
        after_layers.append(psi.copy())
    value = expectation_from_state(psi, cost, np)
    grad_gamma = np.zeros(p, dtype=np.float64)
    grad_beta = np.zeros(p, dtype=np.float64)
    adjoint = cost * psi
    for layer in range(p - 1, -1, -1):
        xsum = x_sum_state(after_layers[layer], graph.n)
        d_beta = -1j * xsum
        grad_beta[layer] = 2.0 * np.real(np.vdot(adjoint, d_beta))
        apply_mixer_inplace(adjoint, graph.n, -float(betas[layer]), np)
        d_gamma = -1j * cost * after_costs[layer]
        grad_gamma[layer] = 2.0 * np.real(np.vdot(adjoint, d_gamma))
        adjoint *= np.exp((1j * float(gammas[layer])) * cost)
    return float(value), np.concatenate([grad_gamma, grad_beta])


def global_value(graph, theta: np.ndarray, p: int) -> float:
    gammas = theta[:p].astype(float).tolist()
    betas = theta[p:].astype(float).tolist()
    nstates = 1 << graph.n
    psi = np.empty(nstates, dtype=np.complex128)
    psi.fill(1.0 / math.sqrt(nstates))
    cost = cost_table(graph.n, graph.edges, graph.fields, graph.objective, np, np.float64)
    for gamma, beta in zip(gammas, betas):
        psi *= np.exp((-1j * float(gamma)) * cost)
        apply_mixer_inplace(psi, graph.n, beta, np)
    return expectation_from_state(psi, cost, np)


def central_fd(graph, theta: np.ndarray, p: int, eps: float) -> tuple[np.ndarray, float]:
    grad = np.zeros_like(theta)
    t0 = time.perf_counter()
    for i in range(theta.size):
        plus = theta.copy()
        minus = theta.copy()
        plus[i] += eps
        minus[i] -= eps
        grad[i] = (global_value(graph, plus, p) - global_value(graph, minus, p)) / (2.0 * eps)
    return grad, time.perf_counter() - t0


def gradient_metrics(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float, int]:
    diff = a - b
    rel = float(np.linalg.norm(diff) / max(np.linalg.norm(b), 1e-30))
    max_abs = float(np.max(np.abs(diff)))
    cos = float(np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-30))
    worst = int(np.argmax(np.abs(diff)))
    return max_abs, rel, cos, worst


def taylor_test(graph, theta: np.ndarray, p: int, grad: np.ndarray, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    direction = rng.normal(size=theta.size)
    direction /= max(np.linalg.norm(direction), 1e-30)
    f0 = global_value(graph, theta, p)
    hs = np.asarray([1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4], dtype=np.float64)
    rem = []
    for h in hs:
        r = abs(global_value(graph, theta + h * direction, p) - f0 - h * float(np.dot(grad, direction)))
        rem.append(max(r, 1e-300))
    logh = np.log(hs)
    logr = np.log(np.asarray(rem))
    slope = float(np.polyfit(logh, logr, 1)[0])
    return slope, float(max(rem))


def run_one(family: str, n: int, p: int, graph_seed: int, init_id: int, args) -> list[P02Row]:
    seed = 320000 + graph_seed * 997 + n * 37 + p * 131
    graph = make_graph(family, n, seed)
    gammas, betas = params_for_depth(p, seed=graph_seed, init_id=init_id)
    theta = np.asarray(gammas + betas, dtype=np.float64)
    rows: list[P02Row] = []
    t0 = time.perf_counter()
    lc = lightcone_gradient_adjoint(
        graph,
        gammas,
        betas,
        p=p,
        prefer_gpu=False,
        max_k=args.max_k,
        max_batch_states=args.max_batch_states,
        complex_dtype=np.complex128,
        float_dtype=np.float64,
    )
    lc_seconds = time.perf_counter() - t0
    if lc.status != "ok" or lc.gradient is None:
        rows.append(
            P02Row(
                family,
                n,
                p,
                graph_seed,
                init_id,
                graph.m,
                len(graph.fields),
                "global_adjoint_float64",
                lc.status,
                float(lc.value),
                float("nan"),
                float("nan"),
                float("nan"),
                float("nan"),
                float("nan"),
                "",
                float("nan"),
                float("nan"),
                lc_seconds,
                0.0,
                0.0,
                "LC did not complete",
            )
        )
        return rows

    t0 = time.perf_counter()
    value_ref, grad_ref = global_value_and_adjoint(graph, gammas, betas)
    ref_seconds = time.perf_counter() - t0
    max_abs, rel, cos, worst = gradient_metrics(np.asarray(lc.gradient), grad_ref)
    slope, max_rem = taylor_test(graph, theta, p, grad_ref, seed + 17 * init_id)
    labels = [f"gamma_{i}" for i in range(p)] + [f"beta_{i}" for i in range(p)]
    rows.append(
        P02Row(
            family,
            n,
            p,
            graph_seed,
            init_id,
            graph.m,
            len(graph.fields),
            "global_adjoint_float64",
            "ok",
            float(lc.value),
            value_ref,
            abs(float(lc.value) - value_ref),
            max_abs,
            rel,
            cos,
            labels[worst],
            slope,
            max_rem,
            lc_seconds,
            ref_seconds,
            0.0,
            "independent dense global adjoint; complex128/float64",
        )
    )

    if args.include_fd:
        for eps in args.fd_eps:
            fd, fd_seconds = central_fd(graph, theta, p, eps)
            max_abs, rel, cos, worst = gradient_metrics(np.asarray(lc.gradient), fd)
            rows.append(
                P02Row(
                    family,
                    n,
                    p,
                    graph_seed,
                    init_id,
                    graph.m,
                    len(graph.fields),
                    f"central_fd_eps_{eps:g}",
                    "ok",
                    float(lc.value),
                    value_ref,
                    abs(float(lc.value) - value_ref),
                    max_abs,
                    rel,
                    cos,
                    labels[worst],
                    slope,
                    max_rem,
                    lc_seconds,
                    ref_seconds,
                    fd_seconds,
                    "FD is secondary; central FD over 2p parameters uses 4p objective calls",
                )
            )
    return rows


def write_md(rows: list[P02Row], path: Path) -> None:
    lines = [
        "# P0-2 Independent Float64 Gradient Correctness",
        "",
        "Primary reference is an independent dense global-state adjoint implemented in complex128/float64. Central finite differences are secondary diagnostics.",
        "",
        "| family | n | p | reference | rows | max rel L2 | max abs | min cosine | median Taylor slope |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|",
    ]
    keys = sorted({(r.family, r.n, r.p, r.reference) for r in rows})
    for key in keys:
        sub = [r for r in rows if (r.family, r.n, r.p, r.reference) == key and r.status == "ok"]
        if not sub:
            continue
        lines.append(
            f"| {key[0]} | {key[1]} | {key[2]} | {key[3]} | {len(sub)} | "
            f"{np.nanmax([r.relative_l2_error for r in sub]):.3e} | {np.nanmax([r.max_abs_error for r in sub]):.3e} | "
            f"{np.nanmin([r.cosine_similarity for r in sub]):.12g} | {np.nanmedian([r.taylor_slope for r in sub]):.3g} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "aaai27_required_experiments_20260710" / "P0_2_float64_gradient_correctness")
    parser.add_argument("--families", nargs="*", default=["3regular", "er2", "modular_sparse", "weighted_qubo_fields"])
    parser.add_argument("--ns", nargs="*", type=int, default=[8, 12, 16])
    parser.add_argument("--ps", nargs="*", type=int, default=[1, 2, 3])
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--inits", type=int, default=5)
    parser.add_argument("--max-k", type=int, default=28)
    parser.add_argument("--max-batch-states", type=int, default=1 << 20)
    parser.add_argument("--include-fd", action="store_true")
    parser.add_argument("--fd-eps", nargs="*", type=float, default=[1e-3])
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if args.quick:
        args.families = ["3regular"]
        args.ns = [8]
        args.ps = [1, 2]
        args.seeds = 1
        args.inits = 1
        args.include_fd = True
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[P02Row] = []
    csv_path = args.out_dir / "P0_2_float64_gradient_correctness.csv"
    for family in args.families:
        for n in args.ns:
            if family == "3regular" and (n * 3) % 2 != 0:
                continue
            for p in args.ps:
                for seed in range(args.seeds):
                    for init_id in range(args.inits):
                        print(f"P0-2 family={family} n={n} p={p} seed={seed} init={init_id}", flush=True)
                        rows.extend(run_one(family, n, p, seed, init_id, args))
                        with csv_path.open("w", newline="", encoding="utf-8") as f:
                            writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
                            writer.writeheader()
                            for row in rows:
                                writer.writerow(asdict(row))
                        write_md(rows, args.out_dir / "P0_2_float64_gradient_correctness.md")
    print(f"WROTE {csv_path}")


if __name__ == "__main__":
    main()
