from __future__ import annotations

import csv
import math
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from run_benchmarks import Row, graph_for, params_for
from lcqaoa.lightcone import extract_lightcones, lightcone_expectation
from lcqaoa.qaoa import full_state_expectation


def run_sparse_scale() -> list[Row]:
    rows: list[Row] = []
    families = ["3regular", "er_sparse", "modular", "scale_free"]
    ns = [32, 48, 64, 96, 128]
    ps = [1, 2]
    for family in families:
        for n in ns:
            for p in ps:
                graph = graph_for(family, n, seed=20000 + n * 17 + p)
                gammas, betas = params_for(p)
                cones = extract_lightcones(graph, p)
                kmax = max(c.k for c in cones) if cones else 0
                total_cone_states = sum(1 << c.k for c in cones)
                print(f"SCALE family={family} n={n} m={graph.m} p={p} kmax={kmax} total={total_cone_states}", flush=True)

                full = full_state_expectation(
                    graph,
                    gammas,
                    betas,
                    method="precompute",
                    prefer_gpu=True,
                    max_qubits=24,
                )
                rows.append(
                    Row(
                        family,
                        graph.n,
                        graph.m,
                        p,
                        "full_precompute_gpu",
                        full.status,
                        full.value,
                        full.seconds,
                        full.backend,
                        full.state_qubits,
                        full.peak_pool_bytes / 1024**2,
                        0.0 if full.status == "ok" else float("nan"),
                        float("nan"),
                        "full-state reference capped at n<=24 on the 8GB RTX 3070 run",
                    )
                )

                if total_cone_states > (1 << 28):
                    rows.append(
                        Row(
                            family,
                            graph.n,
                            graph.m,
                            p,
                            "lc_batched_gpu",
                            f"skipped_total_cone_states_{total_cone_states}",
                            float("nan"),
                            0.0,
                            "cupy",
                            kmax,
                            0.0,
                            float("nan"),
                            float("nan"),
                            "sparse-scaling guardrail",
                        )
                    )
                    continue

                lc = lightcone_expectation(
                    graph,
                    gammas,
                    betas,
                    p=p,
                    prefer_gpu=True,
                    max_k=24,
                    max_batch_states=1 << 21,
                )
                rows.append(
                    Row(
                        family,
                        graph.n,
                        graph.m,
                        p,
                        "lc_batched_gpu",
                        lc.status,
                        lc.value,
                        lc.seconds,
                        lc.backend,
                        kmax,
                        lc.peak_pool_bytes / 1024**2,
                        abs(lc.value - full.value) if math.isfinite(full.value) and lc.status == "ok" else float("nan"),
                        float("nan"),
                        f"total_cone_states={total_cone_states}",
                    )
                )
    return rows


def write_markdown(rows: list[Row], path: Path) -> None:
    lines = [
        "# Sparse Scaling Benchmark",
        "",
        "| Family | n | m | p | Method | Status | Value | Time s | k/qubits | Peak MB | Notes |",
        "|---|---:|---:|---:|---|---|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r.family} | {r.n} | {r.m} | {r.p} | {r.method} | {r.status} | "
            f"{r.value:.6g} | {r.seconds:.4g} | {r.kmax_or_qubits} | {r.peak_pool_mb:.1f} | {r.notes} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out = Path("results/benchmark_sparse_scale.csv")
    md = Path("results/benchmark_sparse_scale.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = run_sparse_scale()
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
    write_markdown(rows, md)
    print(f"WROTE {out}")
    print(f"WROTE {md}")


if __name__ == "__main__":
    main()

