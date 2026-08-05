from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from lcqaoa.lightcone import extract_lightcones, lightcone_expectation
from run_benchmarks import params_for
from run_extended_reach import graph_for


@dataclass
class P3Row:
    family: str
    n: int
    p: int
    m: int
    kmax: int
    total_cone_states: int
    status: str
    seconds: float
    peak_pool_mb: float
    value: float
    notes: str


def cone_stats(graph, p: int) -> tuple[int, int]:
    cones = extract_lightcones(graph, p)
    return max(c.k for c in cones), sum(1 << c.k for c in cones)


def run_case(n: int, *, max_k: int, max_total: int, max_batch_states: int) -> P3Row:
    p = 3
    seed = 260000 + n * 17 + p * 101 + 1
    graph = graph_for("3regular", n, seed)
    kmax, total = cone_stats(graph, p)
    if kmax > max_k:
        return P3Row(
            "3regular",
            n,
            p,
            graph.m,
            kmax,
            total,
            f"skipped_kmax_{kmax}_over_{max_k}",
            0.0,
            0.0,
            float("nan"),
            "p=3 random 3-regular cones exceed the exact single-GPU local-state cap",
        )
    if total > max_total:
        return P3Row(
            "3regular",
            n,
            p,
            graph.m,
            kmax,
            total,
            f"skipped_total_cone_states_{total}",
            0.0,
            0.0,
            float("nan"),
            f"total-cone work exceeds cap {max_total}",
        )
    gammas, betas = params_for(p)
    stats = lightcone_expectation(
        graph,
        gammas,
        betas,
        p=p,
        prefer_gpu=True,
        max_k=max_k,
        max_batch_states=max_batch_states,
    )
    return P3Row(
        "3regular",
        n,
        p,
        graph.m,
        kmax,
        total,
        stats.status,
        float(stats.seconds),
        float(stats.peak_pool_bytes) / 1024**2,
        float(stats.value),
        "exact p=3 objective row when cone cap permits",
    )


def write_outputs(rows: list[P3Row], out_dir: Path) -> None:
    csv_path = out_dir / "p2_p3_cone_sweep.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
    md_lines = [
        "# P2 p=3 Cone Sweep",
        "",
        "This sweep tests whether 3-regular p=3 remains inside the exact local-state cap on the 8GB RTX 3070. It is retained as regime-boundary evidence, not as a positive scalability claim.",
        "",
        "| n | p | m | kmax | total cone states | Status | Seconds | Peak MB | Notes |",
        "|---:|---:|---:|---:|---:|---|---:|---:|---|",
    ]
    for row in rows:
        md_lines.append(
            f"| {row.n} | {row.p} | {row.m} | {row.kmax} | {row.total_cone_states} | {row.status} | "
            f"{row.seconds:.4g} | {row.peak_pool_mb:.4g} | {row.notes} |"
        )
    (out_dir / "p2_p3_cone_sweep.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"wrote {csv_path}")
    print(f"wrote {out_dir / 'p2_p3_cone_sweep.md'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "p2_p3_cone_sweep")
    parser.add_argument("--max-k", type=int, default=24)
    parser.add_argument("--max-total-cone-states", type=int, default=100_000_000)
    parser.add_argument("--max-batch-states", type=int, default=1 << 21)
    parser.add_argument("--ns", nargs="+", type=int, default=[24, 28, 32, 40, 48, 64, 96, 128, 256, 512])
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        run_case(n, max_k=args.max_k, max_total=args.max_total_cone_states, max_batch_states=args.max_batch_states)
        for n in args.ns
    ]
    write_outputs(rows, args.out_dir)


if __name__ == "__main__":
    main()
