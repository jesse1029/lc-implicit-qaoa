from __future__ import annotations

import argparse
import csv
import gc
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lcqaoa.graphs import random_regular_graph
from lcqaoa.qaoa import full_state_expectation


@dataclass
class BoundaryRow:
    family: str
    n: int
    p: int
    method: str
    status: str
    seconds: float
    peak_mb: float
    value: float
    theoretical_state_cost_mb: float
    notes: str


def params_for(p: int) -> tuple[list[float], list[float]]:
    return [0.20 + 0.05 * i for i in range(p)], [0.32 - 0.035 * i for i in range(p)]


def theoretical_mb(n: int) -> float:
    # complex64 state plus float32 cost table, before temporary mixer/cost buffers.
    return ((8 + 4) * (1 << n)) / 1024**2


def run_one(n: int, p: int) -> BoundaryRow:
    graph = random_regular_graph(n, 3, seed=88000 + 17 * n + p)
    gammas, betas = params_for(p)
    try:
        result = full_state_expectation(graph, gammas, betas, method="precompute", prefer_gpu=True, max_qubits=None)
        return BoundaryRow(
            family="3regular",
            n=n,
            p=p,
            method="full_precompute_gpu",
            status=result.status,
            seconds=result.seconds,
            peak_mb=result.peak_pool_bytes / 1024**2,
            value=result.value,
            theoretical_state_cost_mb=theoretical_mb(n),
            notes="complex64 state plus float32 cost table; measured peak includes temporary buffers and CuPy pool behavior",
        )
    except Exception as exc:
        try:
            from lcqaoa.backend import get_backend

            get_backend(True).free_memory_pool()
        except Exception:
            pass
        gc.collect()
        return BoundaryRow(
            family="3regular",
            n=n,
            p=p,
            method="full_precompute_gpu",
            status=f"failed:{type(exc).__name__}",
            seconds=0.0,
            peak_mb=0.0,
            value=float("nan"),
            theoretical_state_cost_mb=theoretical_mb(n),
            notes=str(exc).splitlines()[0][:240],
        )


def write_markdown(rows: list[BoundaryRow], path: Path) -> None:
    lines = [
        "# Full-State Boundary Probe",
        "",
        "| n | p | Status | Seconds | Peak MB | Theoretical state+cost MB | Notes |",
        "|---:|---:|---|---:|---:|---:|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r.n} | {r.p} | {r.status} | {r.seconds:.3g} | {r.peak_mb:.3g} | {r.theoretical_state_cost_mb:.3g} | {r.notes} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "fullstate_boundary_probe.csv")
    parser.add_argument("--markdown", type=Path, default=ROOT / "results" / "fullstate_boundary_probe.md")
    parser.add_argument("--p", type=int, default=2)
    parser.add_argument(
        "--n-values",
        default="24,26,28,30",
        help="Comma-separated n values. For 3-regular graphs, n must be even.",
    )
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows: list[BoundaryRow] = []
    n_values = [int(item.strip()) for item in args.n_values.split(",") if item.strip()]
    for n in n_values:
        if (n * 3) % 2:
            rows.append(
                BoundaryRow(
                    family="3regular",
                    n=n,
                    p=args.p,
                    method="full_precompute_gpu",
                    status="invalid_3regular_odd_n",
                    seconds=0.0,
                    peak_mb=0.0,
                    value=float("nan"),
                    theoretical_state_cost_mb=theoretical_mb(n),
                    notes="3-regular simple graphs require n*degree to be even",
                )
            )
            continue
        print(f"RUN full-state boundary n={n} p={args.p}", flush=True)
        rows.append(run_one(n, args.p))
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
