#!/usr/bin/env python3
"""Small deterministic CPU check for an installed LC-Implicit-QAOA package."""

from __future__ import annotations

import numpy as np

from lcqaoa import (
    full_state_expectation,
    lightcone_expectation,
    lightcone_gradient_adjoint,
)
from lcqaoa.graphs import random_regular_graph


def main() -> None:
    graph = random_regular_graph(n=8, degree=3, seed=7)
    gammas = np.asarray([0.21, 0.37], dtype=float)
    betas = np.asarray([0.18, 0.29], dtype=float)

    full = full_state_expectation(
        graph,
        gammas,
        betas,
        method="precompute",
        prefer_gpu=False,
        complex_dtype=np.complex128,
        float_dtype=np.float64,
        max_qubits=12,
    )
    local = lightcone_expectation(
        graph,
        gammas,
        betas,
        p=2,
        prefer_gpu=False,
        max_k=12,
        complex_dtype=np.complex128,
        float_dtype=np.float64,
    )
    gradient = lightcone_gradient_adjoint(
        graph,
        gammas,
        betas,
        p=2,
        prefer_gpu=False,
        max_k=12,
        complex_dtype=np.complex128,
        float_dtype=np.float64,
    )

    if full.status != "ok" or local.status != "ok" or gradient.status != "ok":
        raise RuntimeError(
            f"smoke status failure: full={full.status}, local={local.status}, "
            f"gradient={gradient.status}"
        )
    error = abs(float(full.value) - float(local.value))
    if error > 1e-11:
        raise RuntimeError(f"LC/full mismatch: {error:.3e}")
    if gradient.gradient is None or not np.all(np.isfinite(gradient.gradient)):
        raise RuntimeError("adjoint returned no finite gradient")

    print(
        "SMOKE_OK "
        f"objective={local.value:.12g} "
        f"abs_error={error:.3e} "
        f"gradient_norm={np.linalg.norm(gradient.gradient):.6g}"
    )


if __name__ == "__main__":
    main()
