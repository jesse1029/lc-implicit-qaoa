from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lcqaoa.graphs import WeightedGraph, random_regular_graph
from lcqaoa.qaoa import full_state_expectation


def adjacency(graph: WeightedGraph, sign: float = 1.0) -> np.ndarray:
    mat = np.zeros((graph.n, graph.n), dtype=np.float64)
    for i, j, w in graph.edges:
        mat[i, j] += sign * float(w)
        mat[j, i] += sign * float(w)
    return mat


def run_cuaoa_value(graph: WeightedGraph, gammas, betas, *, sign: float, gamma_scale: float, beta_scale: float):
    import pycuaoa

    params = pycuaoa.Parameters(
        np.asarray([beta_scale * b for b in betas], dtype=np.float64),
        np.asarray([gamma_scale * g for g in gammas], dtype=np.float64),
    )
    sim = pycuaoa.CUAOA(adjacency(graph, sign=sign), depth=len(gammas), parameters=params)
    # On the RTX 3070 / CUDA 13.3 host, pycuaoa 0.1.0 segfaults with
    # create_handle(..., exact=True). The default handle path is stable and is
    # the one used in upstream examples.
    handle = pycuaoa.create_handle(graph.n)
    t0 = time.perf_counter()
    try:
        value = float(sim.expectation_value(handle))
    finally:
        try:
            handle.destroy()
        except Exception:
            pass
    return value, time.perf_counter() - t0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "cuaoa_calibration.json")
    args = parser.parse_args()
    graph = random_regular_graph(10, 3, seed=50010)
    gammas = [0.20, 0.25]
    betas = [0.32, 0.285]
    ref = full_state_expectation(graph, gammas, betas, prefer_gpu=True, max_qubits=16)

    rows = []
    for sign in [1.0, -1.0]:
        for gamma_scale in [1.0, -1.0, 2.0, -2.0, 0.5, -0.5]:
            for beta_scale in [1.0, -1.0, 2.0, -2.0, 0.5, -0.5]:
                try:
                    raw, seconds = run_cuaoa_value(
                        graph,
                        gammas,
                        betas,
                        sign=sign,
                        gamma_scale=gamma_scale,
                        beta_scale=beta_scale,
                    )
                    candidates = {
                        "raw": raw,
                        "neg_raw": -raw,
                        "m_plus_raw": graph.m + raw,
                        "m_minus_raw": graph.m - raw,
                    }
                    best_name, best_value = min(
                        candidates.items(),
                        key=lambda item: abs(float(item[1]) - ref.value),
                    )
                    err = abs(float(best_value) - ref.value)
                    status = "ok"
                except Exception as exc:
                    raw = float("nan")
                    seconds = 0.0
                    best_name = ""
                    best_value = float("nan")
                    err = float("nan")
                    status = f"failed:{type(exc).__name__}:{str(exc)[:160]}"
                rows.append(
                    {
                        "status": status,
                        "sign": sign,
                        "gamma_scale": gamma_scale,
                        "beta_scale": beta_scale,
                        "raw": raw,
                        "transform": best_name,
                        "value": float(best_value),
                        "error": err,
                        "seconds": seconds,
                    }
                )
    ok_rows = [r for r in rows if r["status"] == "ok" and math.isfinite(r["error"])]
    best = min(ok_rows, key=lambda r: r["error"]) if ok_rows else None
    result = {
        "reference": ref.value,
        "graph_m": graph.m,
        "best": best,
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"reference": ref.value, "best": best}, indent=2))


if __name__ == "__main__":
    main()
