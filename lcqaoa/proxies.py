from __future__ import annotations

from typing import Sequence
import math
import time

import numpy as np

from .backend import get_backend
from .graphs import WeightedGraph
from .qaoa import (
    EvalStats,
    apply_cost_implicit,
    apply_cost_precomputed,
    cost_table,
    expectation_from_state,
    expectation_from_state_implicit,
)


def _as_float(x) -> float:
    return float(x.get() if hasattr(x, "get") else x)


def _mixer_unitary(width: int, beta: float, xp, complex_dtype):
    c = math.cos(float(beta))
    s = math.sin(float(beta))
    one = np.asarray([[c, -1j * s], [-1j * s, c]], dtype=complex_dtype)
    out = one
    for _ in range(1, width):
        out = np.kron(one, out)
    return xp.asarray(out, dtype=complex_dtype)


def apply_mixer_fused_inplace(
    psi,
    k: int,
    beta: float,
    xp,
    *,
    fusion_width: int = 4,
    complex_dtype=np.complex64,
) -> None:
    """Apply X mixers in small fused blocks.

    This is a QueenV2-style proxy rather than an official QueenV2 kernel. It
    keeps the state-vector route but reduces gate granularity by applying a
    tensor-product block for adjacent qubits, matching the gate-fusion idea.
    """
    if fusion_width <= 0:
        raise ValueError("fusion_width must be positive")
    q = 0
    while q < k:
        width = min(fusion_width, k - q)
        step = 1 << q
        subspace = 1 << width
        block = step * subspace
        unitary = _mixer_unitary(width, beta, xp, complex_dtype)
        view = psi.reshape(-1, block).reshape(-1, subspace, step)
        mixed = xp.empty_like(view)
        for row in range(subspace):
            acc = unitary[row, 0] * view[:, 0, :]
            for col in range(1, subspace):
                acc = acc + unitary[row, col] * view[:, col, :]
            mixed[:, row, :] = acc
        view[...] = mixed
        q += width


def queen_proxy_fused_expectation(
    graph: WeightedGraph,
    gammas: Sequence[float],
    betas: Sequence[float],
    *,
    prefer_gpu: bool = True,
    fusion_width: int = 4,
    complex_dtype=np.complex64,
    float_dtype=np.float32,
    max_qubits: int | None = None,
) -> EvalStats:
    """Exact full-state proxy for Queen/QueenV2-like gate fusion."""
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
    t0 = time.perf_counter()
    nstates = 1 << graph.n
    psi = xp.empty(nstates, dtype=complex_dtype)
    psi.fill(1.0 / math.sqrt(nstates))
    cost = cost_table(graph.n, graph.edges, graph.fields, graph.objective, xp, float_dtype)

    for gamma, beta in zip(gammas, betas):
        apply_cost_precomputed(psi, cost, gamma, xp)
        apply_mixer_fused_inplace(
            psi,
            graph.n,
            beta,
            xp,
            fusion_width=fusion_width,
            complex_dtype=complex_dtype,
        )

    value = expectation_from_state(psi, cost, xp)
    backend.sync()
    return EvalStats(
        value=value,
        seconds=time.perf_counter() - t0,
        backend=f"{backend.name}_fused_w{fusion_width}",
        peak_pool_bytes=backend.memory_pool_bytes(),
        state_qubits=graph.n,
        status="ok",
    )


def _quantized_payload_bytes(nstates: int, bits: int, block_states: int) -> int:
    component_bytes = max(1, (int(bits) + 7) // 8)
    blocks = (nstates + block_states - 1) // block_states
    return nstates * 2 * component_bytes + blocks * 4


def quantize_state_blockwise_inplace(
    psi,
    xp,
    *,
    bits: int = 8,
    block_states: int = 1 << 16,
    complex_dtype=np.complex64,
) -> None:
    """Round real/imag parts to block-scaled signed integers, then decompress.

    The resident Python object remains a state vector; the returned memory
    metric in the caller is the compressed payload estimate. This is only a
    BMQSim-style proxy for objective-level sensitivity experiments.
    """
    if bits < 2:
        raise ValueError("bits must be at least 2")
    qmax = float((1 << (int(bits) - 1)) - 1)
    nstates = int(psi.shape[0])
    for start in range(0, nstates, block_states):
        stop = min(nstates, start + block_states)
        chunk = psi[start:stop]
        max_abs = xp.max(xp.maximum(xp.abs(chunk.real), xp.abs(chunk.imag)))
        scale_value = _as_float(max_abs) / qmax
        if scale_value == 0.0:
            continue
        scale = max_abs / qmax
        real = xp.clip(xp.round(chunk.real / scale), -qmax, qmax) * scale
        imag = xp.clip(xp.round(chunk.imag / scale), -qmax, qmax) * scale
        chunk[...] = (real + 1j * imag).astype(complex_dtype, copy=False)
    norm = xp.sqrt(xp.sum(xp.abs(psi) ** 2))
    norm_value = _as_float(norm)
    if norm_value != 0.0:
        psi /= norm


def bmqsim_proxy_quantized_expectation(
    graph: WeightedGraph,
    gammas: Sequence[float],
    betas: Sequence[float],
    *,
    prefer_gpu: bool = True,
    quant_bits: int = 8,
    block_states: int = 1 << 16,
    complex_dtype=np.complex64,
    float_dtype=np.float32,
    max_qubits: int | None = None,
    chunk_states: int = 1 << 22,
) -> EvalStats:
    """Approximate BMQSim-style lossy-compressed full-state proxy.

    BMQSim itself uses a more sophisticated GPU compression and memory manager.
    This proxy quantizes state-vector checkpoints to test whether a lossy
    state-vector route is a credible competitor for QAOA objective evaluation.
    """
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
    t0 = time.perf_counter()
    nstates = 1 << graph.n
    psi = xp.empty(nstates, dtype=complex_dtype)
    psi.fill(1.0 / math.sqrt(nstates))

    from .qaoa import apply_mixer_inplace

    quantize_state_blockwise_inplace(
        psi,
        xp,
        bits=quant_bits,
        block_states=block_states,
        complex_dtype=complex_dtype,
    )
    for gamma, beta in zip(gammas, betas):
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
        quantize_state_blockwise_inplace(
            psi,
            xp,
            bits=quant_bits,
            block_states=block_states,
            complex_dtype=complex_dtype,
        )

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
    compressed_bytes = _quantized_payload_bytes(nstates, quant_bits, block_states)
    return EvalStats(
        value=value,
        seconds=time.perf_counter() - t0,
        backend=f"{backend.name}_block_quant{quant_bits}",
        peak_pool_bytes=compressed_bytes,
        state_qubits=graph.n,
        status="ok_proxy",
    )
