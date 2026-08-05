from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence
import math
import time

import numpy as np

from .backend import Backend, get_backend
from .graphs import Edge, Field, WeightedGraph


@dataclass
class EvalStats:
    value: float
    seconds: float
    backend: str
    peak_pool_bytes: int = 0
    state_qubits: int = 0
    status: str = "ok"


def _arange_states(xp, start: int, stop: int):
    return xp.arange(start, stop, dtype=xp.uint64)


def cost_table(
    k: int,
    edges: Sequence[Edge],
    fields: Sequence[Field],
    objective: str,
    xp,
    float_dtype=np.float32,
    offset: int = 0,
    size: int | None = None,
):
    if size is None:
        size = 1 << k
    idx = _arange_states(xp, offset, offset + size)
    out = xp.zeros(size, dtype=float_dtype)
    if objective == "maxcut":
        for i, j, w in edges:
            bits = ((idx >> int(i)) ^ (idx >> int(j))) & xp.uint64(1)
            out += float(w) * bits.astype(float_dtype, copy=False)
    elif objective == "qubo":
        for i, j, w in edges:
            bi = (idx >> int(i)) & xp.uint64(1)
            bj = (idx >> int(j)) & xp.uint64(1)
            out += float(w) * (bi & bj).astype(float_dtype, copy=False)
        for i, w in fields:
            bi = (idx >> int(i)) & xp.uint64(1)
            out += float(w) * bi.astype(float_dtype, copy=False)
    else:
        raise ValueError(f"unknown objective: {objective}")
    return out


def term_table(
    k: int,
    term_kind: str,
    term_nodes: tuple[int, ...],
    weight: float,
    objective: str,
    xp,
    float_dtype=np.float32,
):
    idx = _arange_states(xp, 0, 1 << k)
    out = xp.zeros(1 << k, dtype=float_dtype)
    if term_kind == "edge":
        i, j = term_nodes
        if objective == "maxcut":
            bits = ((idx >> int(i)) ^ (idx >> int(j))) & xp.uint64(1)
            out += float(weight) * bits.astype(float_dtype, copy=False)
        elif objective == "qubo":
            bi = (idx >> int(i)) & xp.uint64(1)
            bj = (idx >> int(j)) & xp.uint64(1)
            out += float(weight) * (bi & bj).astype(float_dtype, copy=False)
        else:
            raise ValueError(f"unknown objective: {objective}")
    elif term_kind == "field":
        (i,) = term_nodes
        bi = (idx >> int(i)) & xp.uint64(1)
        out += float(weight) * bi.astype(float_dtype, copy=False)
    else:
        raise ValueError(f"unknown term kind: {term_kind}")
    return out


def apply_cost_precomputed(psi, cost, gamma: float, xp) -> None:
    psi *= xp.exp((-1j * float(gamma)) * cost)


def apply_cost_implicit(
    psi,
    k: int,
    edges: Sequence[Edge],
    fields: Sequence[Field],
    objective: str,
    gamma: float,
    xp,
    float_dtype=np.float32,
    chunk_states: int = 1 << 22,
) -> None:
    nstates = 1 << k
    for start in range(0, nstates, chunk_states):
        stop = min(nstates, start + chunk_states)
        cost = cost_table(k, edges, fields, objective, xp, float_dtype, offset=start, size=stop - start)
        psi[start:stop] *= xp.exp((-1j * float(gamma)) * cost)


def apply_mixer_inplace(psi, k: int, beta: float, xp) -> None:
    c = math.cos(float(beta))
    s = math.sin(float(beta))
    for q in range(k):
        step = 1 << q
        block = step << 1
        view = psi.reshape(-1, block)
        a = view[:, :step].copy()
        b = view[:, step:block].copy()
        view[:, :step] = c * a - 1j * s * b
        view[:, step:block] = c * b - 1j * s * a


def apply_mixer_batched_inplace(psi, k: int, beta: float, xp) -> None:
    c = math.cos(float(beta))
    s = math.sin(float(beta))
    batch = psi.shape[0]
    for q in range(k):
        step = 1 << q
        block = step << 1
        view = psi.reshape(batch, -1, block)
        a = view[:, :, :step].copy()
        b = view[:, :, step:block].copy()
        view[:, :, :step] = c * a - 1j * s * b
        view[:, :, step:block] = c * b - 1j * s * a


def expectation_from_state(psi, cost, xp) -> float:
    probs = xp.abs(psi) ** 2
    value = xp.sum(probs * cost)
    return float(value.get() if hasattr(value, "get") else value)


def expectation_from_state_implicit(
    psi,
    k: int,
    edges: Sequence[Edge],
    fields: Sequence[Field],
    objective: str,
    xp,
    float_dtype=np.float32,
    chunk_states: int = 1 << 22,
) -> float:
    nstates = 1 << k
    total = 0.0
    for start in range(0, nstates, chunk_states):
        stop = min(nstates, start + chunk_states)
        cost = cost_table(k, edges, fields, objective, xp, float_dtype, offset=start, size=stop - start)
        probs = xp.abs(psi[start:stop]) ** 2
        value = xp.sum(probs * cost)
        total += float(value.get() if hasattr(value, "get") else value)
    return total


def full_state_expectation(
    graph: WeightedGraph,
    gammas: Sequence[float],
    betas: Sequence[float],
    *,
    method: str = "precompute",
    prefer_gpu: bool = True,
    complex_dtype=np.complex64,
    float_dtype=np.float32,
    max_qubits: int | None = None,
    chunk_states: int = 1 << 22,
) -> EvalStats:
    if len(gammas) != len(betas):
        raise ValueError("gammas and betas must have the same length")
    if max_qubits is not None and graph.n > max_qubits:
        return EvalStats(
            value=float("nan"),
            seconds=0.0,
            backend="skipped",
            state_qubits=graph.n,
            status=f"skipped_over_{max_qubits}_qubits",
        )

    backend = get_backend(prefer_gpu)
    xp = backend.xp
    backend.free_memory_pool()
    start_time = time.perf_counter()
    nstates = 1 << graph.n
    psi = xp.empty(nstates, dtype=complex_dtype)
    psi.fill(1.0 / math.sqrt(nstates))
    cost = None
    if method == "precompute":
        cost = cost_table(graph.n, graph.edges, graph.fields, graph.objective, xp, float_dtype)
    elif method != "implicit":
        raise ValueError("method must be 'precompute' or 'implicit'")

    for gamma, beta in zip(gammas, betas):
        if method == "precompute":
            apply_cost_precomputed(psi, cost, gamma, xp)
        else:
            apply_cost_implicit(
                psi,
                graph.n,
                graph.edges,
                graph.fields,
                graph.objective,
                gamma,
                xp,
                float_dtype,
                chunk_states,
            )
        apply_mixer_inplace(psi, graph.n, beta, xp)

    if method == "precompute":
        value = expectation_from_state(psi, cost, xp)
    else:
        value = expectation_from_state_implicit(
            psi,
            graph.n,
            graph.edges,
            graph.fields,
            graph.objective,
            xp,
            float_dtype,
            chunk_states,
        )
    backend.sync()
    seconds = time.perf_counter() - start_time
    return EvalStats(
        value=value,
        seconds=seconds,
        backend=backend.name,
        peak_pool_bytes=backend.memory_pool_bytes(),
        state_qubits=graph.n,
        status="ok",
    )


ObjectiveFn = Callable[[Sequence[float], Sequence[float]], EvalStats]


def finite_difference_gradient(
    fn: ObjectiveFn,
    gammas: Sequence[float],
    betas: Sequence[float],
    eps: float = 1e-3,
) -> tuple[np.ndarray, float]:
    gammas_arr = np.asarray(gammas, dtype=np.float64)
    betas_arr = np.asarray(betas, dtype=np.float64)
    params = np.concatenate([gammas_arr, betas_arr])
    grad = np.zeros_like(params)
    t0 = time.perf_counter()
    for i in range(params.size):
        plus = params.copy()
        minus = params.copy()
        plus[i] += eps
        minus[i] -= eps
        gp, bp = plus[: gammas_arr.size], plus[gammas_arr.size :]
        gm, bm = minus[: gammas_arr.size], minus[gammas_arr.size :]
        vp = fn(gp, bp).value
        vm = fn(gm, bm).value
        grad[i] = (vp - vm) / (2.0 * eps)
    return grad, time.perf_counter() - t0

