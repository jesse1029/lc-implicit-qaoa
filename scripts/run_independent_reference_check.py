"""Independent complex128 dense reference for the value-and-gradient correctness grid.

Independence contract enforced by construction: this file imports the graph
generators and the angle schedule, which are the shared *inputs* of the
comparison, and the light-cone entry point, which is the system under test.
It imports no cost table, no mixer, no state-evolution routine, no expectation
reduction and no gradient routine from the evaluator package. The diagonal cost,
the state preparation, the mixer, the expectation and the reverse pass below are
written here from the definition of the objective, with different index
arithmetic from the evaluator, so that a shared bit-order, sign, XOR/AND or
field-indexing mistake cannot cancel on both sides.

The directional Taylor test is applied to the light-cone value and the
light-cone gradient, not to the dense reference, so that it certifies the pair
the paper actually claims.
"""
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

# shared inputs only
from lcqaoa.graphs import (  # noqa: E402
    erdos_renyi_graph,
    modular_graph,
    random_regular_graph,
    weighted_modular_qubo_graph,
    weighted_qubo_graph,
)
from benchmark_common import params_for_depth  # noqa: E402

# system under test
from lcqaoa.lightcone import lightcone_gradient_adjoint  # noqa: E402

FAMILIES = ("3regular", "er2", "modular_sparse", "weighted_qubo_fields")


@dataclass
class Row:
    family: str
    n: int
    p: int
    graph_seed: int
    init_id: int
    status: str
    lc_value: float
    ref_value: float
    abs_value_error: float
    rel_gradient_l2_error: float
    max_abs_gradient_error: float
    cosine_similarity: float
    lc_taylor_slope: float
    lc_taylor_max_remainder: float
    lc_seconds: float
    ref_seconds: float


def make_graph(family: str, n: int, seed: int):
    if family == "3regular":
        return random_regular_graph(n, 3, seed=seed)
    if family == "er2":
        return erdos_renyi_graph(n, min(0.45, 2.0 / max(2, n)), seed=seed)
    if family == "modular_sparse":
        return modular_graph(n, modules=max(2, min(4, n // 4)), p_in=0.28, p_out=0.015, seed=seed)
    if family == "weighted_qubo_fields":
        return weighted_qubo_graph(n, min(0.45, 2.0 / max(2, n)), field_prob=1.0, seed=seed,
                                   weight_scale=1.0, field_scale=0.7)
    if family == "weighted_modular_qubo":
        return weighted_modular_qubo_graph(n, modules=max(2, min(4, n // 4)), p_in=0.18, p_out=0.01, seed=seed)
    raise ValueError(family)


# --------------------------------------------------------------- own primitives
def bit_of(states: np.ndarray, q: int) -> np.ndarray:
    """Bit q of each basis index, as float64. Own formulation: divide-and-mod
    rather than shift-and-mask, so a shift/mask convention error cannot be shared."""
    return ((states // (2 ** q)) % 2).astype(np.float64)


def diagonal_cost(graph) -> np.ndarray:
    """Diagonal of C in the computational basis, built from the objective definition.

    maxcut : sum over edges of w * [x_i != x_j]
    qubo   : sum over edges of w * x_i * x_j  +  sum over fields of h * x_i
    """
    n = graph.n
    states = np.arange(1 << n, dtype=np.int64)
    out = np.zeros(1 << n, dtype=np.float64)
    if graph.objective == "maxcut":
        for i, j, w in graph.edges:
            out += float(w) * np.abs(bit_of(states, int(i)) - bit_of(states, int(j)))
    elif graph.objective == "qubo":
        for i, j, w in graph.edges:
            out += float(w) * bit_of(states, int(i)) * bit_of(states, int(j))
        for i, w in graph.fields:
            out += float(w) * bit_of(states, int(i))
    else:
        raise ValueError(f"unknown objective: {graph.objective}")
    return out


def apply_mixer(psi: np.ndarray, n: int, beta: float) -> np.ndarray:
    """exp(-i beta sum_q X_q) applied out of place.

    Own index arithmetic: reshape to (high, 2, low) and mix along the middle axis,
    which is a different decomposition from a (-1, block) slice-and-copy scheme.
    """
    c = math.cos(float(beta))
    s = math.sin(float(beta))
    out = psi
    for q in range(n):
        low = 1 << q
        high = psi.size // (low * 2)
        v = out.reshape(high, 2, low)
        a = v[:, 0, :]
        b = v[:, 1, :]
        out = np.stack((c * a - 1j * s * b, c * b - 1j * s * a), axis=1).reshape(-1)
    return out


def x_sum(psi: np.ndarray, n: int) -> np.ndarray:
    """(sum_q X_q) |psi>, built with the same reshape convention."""
    acc = np.zeros_like(psi)
    for q in range(n):
        low = 1 << q
        high = psi.size // (low * 2)
        v = psi.reshape(high, 2, low)
        acc += np.stack((v[:, 1, :], v[:, 0, :]), axis=1).reshape(-1)
    return acc


def dense_value_and_gradient(graph, gammas, betas):
    """Exact value and all shared-parameter derivatives, reverse mode, complex128."""
    n = graph.n
    p = len(gammas)
    cost = diagonal_cost(graph)
    dim = 1 << n
    psi = np.full(dim, 1.0 / math.sqrt(dim), dtype=np.complex128)

    after_cost = []   # state leaving the cost phase of each layer
    after_mix = []    # state leaving the mixer of each layer
    for layer in range(p):
        psi = np.exp(-1j * float(gammas[layer]) * cost) * psi
        after_cost.append(psi)
        psi = apply_mixer(psi, n, float(betas[layer]))
        after_mix.append(psi)

    value = float(np.real(np.vdot(psi, cost * psi)))

    # F = <psi|C|psi> is real, so dF/dalpha = 2 Re<C psi, dpsi/dalpha>.
    # For U = exp(-i alpha H), dU psi_in / dalpha = -i H U psi_in = -i H psi_out,
    # so each derivative pairs the adjoint at a gate's OUTPUT with H applied to that
    # same output state.
    lam = cost * psi
    g_gamma = np.zeros(p, dtype=np.float64)
    g_beta = np.zeros(p, dtype=np.float64)
    for layer in range(p - 1, -1, -1):
        g_beta[layer] = 2.0 * float(np.real(np.vdot(lam, -1j * x_sum(after_mix[layer], n))))
        lam = apply_mixer(lam, n, -float(betas[layer]))
        g_gamma[layer] = 2.0 * float(np.real(np.vdot(lam, -1j * cost * after_cost[layer])))
        lam = np.exp(1j * float(gammas[layer]) * cost) * lam
    return value, np.concatenate([g_gamma, g_beta])


# --------------------------------------------------------------- LC helpers
def lc_call(graph, gammas, betas, args, want_gradient: bool):
    return lightcone_gradient_adjoint(
        graph, list(gammas), list(betas), p=len(gammas), prefer_gpu=False,
        max_k=args.max_k, max_batch_states=args.max_batch_states,
        complex_dtype=np.complex128, float_dtype=np.float64,
    )


def lc_taylor(graph, theta, p, lc_grad, args, seed: int):
    """Directional Taylor test on the LC value against the LC gradient."""
    rng = np.random.default_rng(seed)
    v = rng.normal(size=theta.size)
    v /= max(np.linalg.norm(v), 1e-30)
    f0 = float(lc_call(graph, theta[:p], theta[p:], args, False).value)
    hs = np.asarray([1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4], dtype=np.float64)
    rem = []
    for h in hs:
        t = theta + h * v
        fh = float(lc_call(graph, t[:p], t[p:], args, False).value)
        rem.append(max(abs(fh - f0 - h * float(np.dot(lc_grad, v))), 1e-300))
    slope = float(np.polyfit(np.log(hs), np.log(np.asarray(rem)), 1)[0])
    return slope, float(max(rem))


def run_one(family, n, p, graph_seed, init_id, args) -> Row:
    seed = 320000 + graph_seed * 997 + n * 37 + p * 131
    graph = make_graph(family, n, seed)
    gammas, betas = params_for_depth(p, seed=graph_seed, init_id=init_id)
    theta = np.asarray(list(gammas) + list(betas), dtype=np.float64)

    t0 = time.perf_counter()
    lc = lc_call(graph, gammas, betas, args, True)
    lc_seconds = time.perf_counter() - t0
    if lc.status != "ok" or lc.gradient is None:
        return Row(family, n, p, graph_seed, init_id, lc.status, float(lc.value), float("nan"),
                   *[float("nan")] * 6, lc_seconds, 0.0)

    t0 = time.perf_counter()
    ref_value, ref_grad = dense_value_and_gradient(graph, gammas, betas)
    ref_seconds = time.perf_counter() - t0

    lc_grad = np.asarray(lc.gradient, dtype=np.float64)
    diff = lc_grad - ref_grad
    rel = float(np.linalg.norm(diff) / max(np.linalg.norm(ref_grad), 1e-30))
    max_abs = float(np.max(np.abs(diff)))
    cos = float(np.dot(lc_grad, ref_grad) /
                max(np.linalg.norm(lc_grad) * np.linalg.norm(ref_grad), 1e-30))

    if args.taylor:
        slope, max_rem = lc_taylor(graph, theta, p, lc_grad, args, seed + 17 * init_id)
    else:
        slope, max_rem = float("nan"), float("nan")

    return Row(family, n, p, graph_seed, init_id, "ok", float(lc.value), float(ref_value),
               abs(float(lc.value) - float(ref_value)), rel, max_abs, cos, slope, max_rem,
               lc_seconds, ref_seconds)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--sizes", type=int, nargs="+", default=[8, 12, 16])
    ap.add_argument("--depths", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--graphs", type=int, default=10)
    ap.add_argument("--inits", type=int, default=5)
    ap.add_argument("--max-k", type=int, default=24)
    ap.add_argument("--max-batch-states", type=int, default=1 << 22)
    ap.add_argument("--taylor", action="store_true")
    ap.add_argument("--families", nargs="+", default=list(FAMILIES))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[Row] = []
    total = len(args.families) * len(args.sizes) * len(args.depths) * args.graphs * args.inits
    done = 0
    for family in args.families:
        for n in args.sizes:
            for p in args.depths:
                for g in range(args.graphs):
                    for i in range(args.inits):
                        rows.append(run_one(family, n, p, g, i, args))
                        done += 1
                        if done % 100 == 0:
                            print(f"  {done}/{total}", flush=True)

    out = args.out_dir / "independent_reference_check.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))

    ok = [r for r in rows if r.status == "ok"]
    print()
    print(f"comparisons: {len(ok)}/{len(rows)}")
    print(f"worst relative gradient L2 error : {max(r.rel_gradient_l2_error for r in ok):.6g}")
    print(f"worst absolute component error   : {max(r.max_abs_gradient_error for r in ok):.6g}")
    print(f"worst absolute value error       : {max(r.abs_value_error for r in ok):.6g}")
    print(f"minimum cosine similarity        : {min(r.cosine_similarity for r in ok):.17g}")
    if args.taylor:
        sl = [r.lc_taylor_slope for r in ok if r.lc_taylor_slope == r.lc_taylor_slope]
        print(f"LC Taylor slope median/min/max   : {np.median(sl):.4g} / {min(sl):.4g} / {max(sl):.4g}")
    print()
    print("per family:")
    for family in args.families:
        sub = [r for r in ok if r.family == family]
        if not sub:
            continue
        print("  %-22s rows=%-5d max rel=%.4g  max abs=%.4g  min cos=%.17g"
              % (family, len(sub), max(r.rel_gradient_l2_error for r in sub),
                 max(r.max_abs_gradient_error for r in sub),
                 min(r.cosine_similarity for r in sub)))
    print("WROTE", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
