from __future__ import annotations

import argparse
import csv
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
class StressRepeatRow:
    family: str
    n: int
    p: int
    seed_offset: int
    repeat: int
    m: int
    kmax: int
    total_cone_states: int
    status: str
    seconds: float
    peak_pool_mb: float
    value: float


def seed_for(n: int, p: int, seed_offset: int) -> int:
    return 260000 + n * 17 + p * 101 + seed_offset


def cone_stats(graph, p: int) -> tuple[int, int]:
    cones = extract_lightcones(graph, p)
    return max(c.k for c in cones), sum(1 << c.k for c in cones)


def write_markdown(rows: list[StressRepeatRow], path: Path) -> None:
    by_seed: dict[int, list[StressRepeatRow]] = {}
    for row in rows:
        by_seed.setdefault(row.seed_offset, []).append(row)
    lines = [
        "# P2 n=16384 Stress Repeats",
        "",
        "Objective-only repeated timings for the bounded-cone 3-regular p=2 stress row.",
        "",
        "| seed offset | repeats | kmax | status set | median s | min s | max s | median peak MB |",
        "|---:|---:|---:|---|---:|---:|---:|---:|",
    ]
    for seed, group in sorted(by_seed.items()):
        seconds = sorted(float(r.seconds) for r in group if r.status == "ok")
        peaks = sorted(float(r.peak_pool_mb) for r in group if r.status == "ok")
        statuses = ",".join(sorted({r.status for r in group}))
        if seconds:
            med = seconds[len(seconds) // 2]
            min_s = min(seconds)
            max_s = max(seconds)
            med_peak = peaks[len(peaks) // 2]
        else:
            med = min_s = max_s = med_peak = float("nan")
        lines.append(
            f"| {seed} | {len(group)} | {group[0].kmax} | {statuses} | {med:.4g} | {min_s:.4g} | {max_s:.4g} | {med_peak:.4g} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "p2_stress_repeats")
    parser.add_argument("--n", type=int, default=16384)
    parser.add_argument("--p", type=int, default=2)
    parser.add_argument("--seed-offsets", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--max-batch-states", type=int, default=1 << 21)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "p2_stress_repeats.csv"
    rows: list[StressRepeatRow] = []
    gammas, betas = params_for(args.p)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(StressRepeatRow.__dataclass_fields__.keys()))
        writer.writeheader()
        for seed_offset in args.seed_offsets:
            graph = graph_for("3regular", args.n, seed_for(args.n, args.p, seed_offset))
            kmax, total = cone_stats(graph, args.p)
            for repeat in range(args.repeats):
                stats = lightcone_expectation(
                    graph,
                    gammas,
                    betas,
                    p=args.p,
                    prefer_gpu=True,
                    max_k=24,
                    max_batch_states=args.max_batch_states,
                )
                row = StressRepeatRow(
                    "3regular",
                    graph.n,
                    args.p,
                    seed_offset,
                    repeat,
                    graph.m,
                    kmax,
                    total,
                    stats.status,
                    float(stats.seconds),
                    float(stats.peak_pool_bytes) / 1024**2,
                    float(stats.value),
                )
                rows.append(row)
                writer.writerow(asdict(row))
                f.flush()
                print(f"seed_offset={seed_offset} repeat={repeat} status={row.status} seconds={row.seconds:.4g}", flush=True)
    write_markdown(rows, args.out_dir / "p2_stress_repeats.md")
    print(f"wrote {csv_path}", flush=True)


if __name__ == "__main__":
    main()
