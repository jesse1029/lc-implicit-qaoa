from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lcqaoa.graphs import random_regular_graph
from lcqaoa.proxies import bmqsim_proxy_quantized_expectation, queen_proxy_fused_expectation
from lcqaoa.qaoa import full_state_expectation


def main() -> None:
    graph = random_regular_graph(10, 3, seed=1)
    gammas = [0.20, 0.25]
    betas = [0.32, 0.285]
    ref = full_state_expectation(graph, gammas, betas, prefer_gpu=True, max_qubits=16)
    queen = queen_proxy_fused_expectation(graph, gammas, betas, prefer_gpu=True, max_qubits=16)
    bmq = bmqsim_proxy_quantized_expectation(graph, gammas, betas, prefer_gpu=True, max_qubits=16)

    queen_err = abs(queen.value - ref.value)
    bmq_err = abs(bmq.value - ref.value)
    print(f"ref status={ref.status} value={ref.value:.9g}")
    print(f"queen status={queen.status} value={queen.value:.9g} error={queen_err:.3g} backend={queen.backend}")
    print(
        f"bmqsim status={bmq.status} value={bmq.value:.9g} "
        f"error={bmq_err:.3g} backend={bmq.backend} estimated_bytes={bmq.peak_pool_bytes}"
    )
    if ref.status != "ok" or queen.status != "ok" or not bmq.status.startswith("ok"):
        raise SystemExit("PROXY_SMOKE_FAILED")
    if queen_err > 2e-5:
        raise SystemExit(f"PROXY_SMOKE_FAILED queen_err={queen_err}")
    print("PROXY_SMOKE_OK")


if __name__ == "__main__":
    main()
