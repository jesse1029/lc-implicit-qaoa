from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from lcqaoa.lightcone import lightcone_expectation
from lcqaoa.qaoa import full_state_expectation
from run_benchmarks import params_for
from run_sota_sparse_scale import FAMILY_SEED_OFFSETS, cone_stats, degree_stats, graph_for_scale


@dataclass
class MultiSeedRow:
    family: str
    n: int
    m: int
    p: int
    seed_id: int
    graph_seed: int
    avg_degree: float
    max_degree: int
    kmax: int
    total_cone_states: int
    method: str
    status: str
    value: float
    seconds: float
    peak_pool_mb: float
    abs_error_vs_full: float
    notes: str


def specs() -> list[tuple[str, int, int]]:
    return [
        ("3regular", 24, 2),
        ("3regular", 128, 2),
        ("3regular", 512, 2),
        ("er_deg2", 24, 2),
        ("er_deg2", 128, 2),
        ("er_deg3", 24, 2),
        ("modular_sparse", 24, 2),
        ("modular_sparse", 128, 1),
        ("modular_sparse", 512, 1),
        ("scale_free_a1", 24, 1),
        ("scale_free_a1", 64, 1),
    ]


def run_spec(
    family: str,
    n: int,
    p: int,
    seed_id: int,
    *,
    full_cap: int,
    max_k: int,
    max_total_cone_states: int,
    max_batch_states: int,
) -> list[MultiSeedRow]:
    graph_seed = 100000 + seed_id * 1009 + n * 37 + p * 131 + FAMILY_SEED_OFFSETS.get(family, 0)
    graph = graph_for_scale(family, n, seed=graph_seed)
    gammas, betas = params_for(p)
    avg_degree, max_degree = degree_stats(graph)
    kmax, total_cone_states = cone_stats(graph, p)
    print(
        f"MULTISEED family={family} n={n} p={p} seed={seed_id} "
        f"kmax={kmax} total={total_cone_states}",
        flush=True,
    )
    rows: list[MultiSeedRow] = []
    ref_value = float("nan")
    if n <= full_cap:
        try:
            full = full_state_expectation(graph, gammas, betas, method="precompute", prefer_gpu=True, max_qubits=full_cap)
            ref_value = full.value if full.status == "ok" else float("nan")
            rows.append(
                MultiSeedRow(
                    family,
                    n,
                    graph.m,
                    p,
                    seed_id,
                    graph_seed,
                    avg_degree,
                    max_degree,
                    kmax,
                    total_cone_states,
                    "full_precompute_gpu",
                    full.status,
                    full.value,
                    full.seconds,
                    full.peak_pool_bytes / 1024**2,
                    0.0 if full.status == "ok" else float("nan"),
                    "full-state reference for seed robustness",
                )
            )
        except Exception as exc:
            rows.append(
                MultiSeedRow(
                    family,
                    n,
                    graph.m,
                    p,
                    seed_id,
                    graph_seed,
                    avg_degree,
                    max_degree,
                    kmax,
                    total_cone_states,
                    "full_precompute_gpu",
                    f"failed:{type(exc).__name__}",
                    float("nan"),
                    0.0,
                    0.0,
                    float("nan"),
                    str(exc)[:180],
                )
            )
    else:
        rows.append(
            MultiSeedRow(
                family,
                n,
                graph.m,
                p,
                seed_id,
                graph_seed,
                avg_degree,
                max_degree,
                kmax,
                total_cone_states,
                "full_precompute_gpu",
                f"skipped_over_{full_cap}_qubits",
                float("nan"),
                0.0,
                0.0,
                float("nan"),
                "full-state cap",
            )
        )

    if kmax > max_k:
        status = f"skipped_kmax_{kmax}_over_{max_k}"
        value = float("nan")
        seconds = 0.0
        peak = 0.0
    elif total_cone_states > max_total_cone_states:
        status = f"skipped_total_cone_states_{total_cone_states}"
        value = float("nan")
        seconds = 0.0
        peak = 0.0
    else:
        try:
            lc = lightcone_expectation(
                graph,
                gammas,
                betas,
                p=p,
                prefer_gpu=True,
                max_k=max_k,
                max_batch_states=max_batch_states,
            )
            status = lc.status
            value = lc.value
            seconds = lc.seconds
            peak = lc.peak_pool_bytes / 1024**2
        except Exception as exc:
            status = f"failed:{type(exc).__name__}"
            value = float("nan")
            seconds = 0.0
            peak = 0.0

    err = abs(value - ref_value) if math.isfinite(value) and math.isfinite(ref_value) and status == "ok" else float("nan")
    rows.append(
        MultiSeedRow(
            family,
            n,
            graph.m,
            p,
            seed_id,
            graph_seed,
            avg_degree,
            max_degree,
            kmax,
            total_cone_states,
            "lc_batched_gpu",
            status,
            value,
            seconds,
            peak,
            err,
            "LC seed robustness row",
        )
    )
    return rows


def fmean(values: list[float]) -> float:
    finite = [v for v in values if math.isfinite(v)]
    return statistics.mean(finite) if finite else float("nan")


def fstd(values: list[float]) -> float:
    finite = [v for v in values if math.isfinite(v)]
    return statistics.stdev(finite) if len(finite) >= 2 else 0.0 if finite else float("nan")


def fp(values: list[float], q: float) -> float:
    finite = sorted(v for v in values if math.isfinite(v))
    if not finite:
        return float("nan")
    idx = min(len(finite) - 1, max(0, round((len(finite) - 1) * q)))
    return finite[idx]


def write_markdown(rows: list[MultiSeedRow], path: Path) -> None:
    lines = [
        "# Multi-Seed Robustness Benchmark",
        "",
        "| Family | n | p | Method | OK | Mean s | Std s | P90 s | Mean kmax | Max kmax | Mean peak MB | Max full-check error |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for family, n, p, method in sorted({(r.family, r.n, r.p, r.method) for r in rows}):
        subset = [r for r in rows if (r.family, r.n, r.p, r.method) == (family, n, p, method)]
        ok = [r for r in subset if r.status == "ok"]
        errors = [r.abs_error_vs_full for r in ok if math.isfinite(r.abs_error_vs_full)]
        lines.append(
            f"| {family} | {n} | {p} | {method} | {len(ok)}/{len(subset)} | "
            f"{fmean([r.seconds for r in ok]):.4g} | {fstd([r.seconds for r in ok]):.3g} | "
            f"{fp([r.seconds for r in ok], 0.9):.4g} | {fmean([r.kmax for r in ok]):.3g} | "
            f"{max([r.kmax for r in ok] or [0])} | {fmean([r.peak_pool_mb for r in ok]):.4g} | "
            f"{max(errors or [float('nan')]):.3g} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "multiseed_stats.csv")
    parser.add_argument("--markdown", type=Path, default=ROOT / "results" / "multiseed_stats.md")
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--full-cap", type=int, default=24)
    parser.add_argument("--max-k", type=int, default=24)
    parser.add_argument("--max-total-cone-states", type=int, default=1 << 30)
    parser.add_argument("--max-batch-states", type=int, default=1 << 21)
    args = parser.parse_args()

    rows: list[MultiSeedRow] = []
    for family, n, p in specs():
        for seed_id in range(args.seeds):
            rows.extend(
                run_spec(
                    family,
                    n,
                    p,
                    seed_id,
                    full_cap=args.full_cap,
                    max_k=args.max_k,
                    max_total_cone_states=args.max_total_cone_states,
                    max_batch_states=args.max_batch_states,
                )
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
    write_markdown(rows, args.markdown)
    print(f"WROTE {args.out}")
    print(f"WROTE {args.markdown}")


if __name__ == "__main__":
    main()
