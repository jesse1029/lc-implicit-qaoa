from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lcqaoa.graphs import Edge, Field, WeightedGraph
from lcqaoa.lightcone import extract_lightcones, lightcone_expectation, lightcone_gradient_adjoint
from lcqaoa.qaoa import full_state_expectation


@dataclass
class CaseStudyRow:
    dataset: str
    variant: str
    n: int
    m: int
    p: int
    avg_degree: float
    max_degree: int
    kmax: int
    method: str
    status: str
    objective_value: float
    objective_seconds: float
    gradient_seconds: float
    gradient_norm: float
    abs_error_vs_full: float
    peak_pool_mb: float
    notes: str


def standardize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    mean = np.nanmean(x, axis=0)
    std = np.nanstd(x, axis=0)
    std[std < 1e-12] = 1.0
    return np.nan_to_num((x - mean) / std)


def feature_selection_qubo(
    x: np.ndarray,
    y: np.ndarray,
    *,
    top_features: int | None = None,
    top_degree: int = 3,
    redundancy_scale: float = 0.55,
) -> WeightedGraph:
    x = standardize(x)
    y = np.asarray(y, dtype=np.float64)
    y = (y - y.mean()) / (y.std() if y.std() > 1e-12 else 1.0)
    relevance = np.abs(x.T @ y) / max(1, x.shape[0] - 1)
    relevance = np.nan_to_num(relevance)
    if relevance.max() > 0:
        relevance = relevance / relevance.max()
    order = np.argsort(-relevance)
    if top_features is not None:
        order = order[:top_features]
    x = x[:, order]
    relevance = relevance[order]
    n = x.shape[1]
    corr = np.abs(np.corrcoef(x, rowvar=False))
    corr = np.nan_to_num(corr)
    np.fill_diagonal(corr, 0.0)
    edges_set: dict[tuple[int, int], float] = {}
    for i in range(n):
        choices = np.argsort(-corr[i])[: min(top_degree, max(0, n - 1))]
        for j in choices:
            if i == j or corr[i, j] <= 0:
                continue
            a, b = sorted((int(i), int(j)))
            weight = -float(redundancy_scale * corr[i, j])
            if (a, b) not in edges_set or abs(weight) > abs(edges_set[(a, b)]):
                edges_set[(a, b)] = weight
    edges: tuple[Edge, ...] = tuple((i, j, w) for (i, j), w in sorted(edges_set.items()))
    fields: tuple[Field, ...] = tuple((int(i), float(relevance[i])) for i in range(n))
    return WeightedGraph(n=n, edges=edges, fields=fields, objective="qubo")


def load_datasets() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    try:
        from sklearn.datasets import load_breast_cancer, load_digits
    except Exception as exc:
        raise SystemExit(f"scikit-learn is required for real QUBO case studies: {exc}")
    breast = load_breast_cancer()
    digits = load_digits()
    return {
        "breast_cancer": (breast.data, breast.target),
        "digits": (digits.data, digits.target),
    }


def degree_stats(graph: WeightedGraph) -> tuple[float, int]:
    deg = [0] * graph.n
    for i, j, _ in graph.edges:
        deg[i] += 1
        deg[j] += 1
    return (sum(deg) / graph.n if graph.n else 0.0, max(deg) if deg else 0)


def kmax_for(graph: WeightedGraph, p: int) -> int:
    cones = extract_lightcones(graph, p)
    return max((c.k for c in cones), default=0)


def params_for_case(p: int) -> tuple[list[float], list[float]]:
    gammas = [0.22 + 0.07 * i for i in range(p)]
    betas = [0.31 - 0.04 * i for i in range(p)]
    return gammas, betas


def run_variant(
    dataset: str,
    variant: str,
    graph: WeightedGraph,
    p: int,
    *,
    max_k: int,
    max_batch_states: int,
    full_cap: int,
) -> list[CaseStudyRow]:
    gammas, betas = params_for_case(p)
    avg_degree, max_degree = degree_stats(graph)
    kmax = kmax_for(graph, p)
    print(
        "REAL_QUBO "
        f"dataset={dataset} variant={variant} n={graph.n} m={graph.m} p={p} kmax={kmax}",
        flush=True,
    )
    rows: list[CaseStudyRow] = []
    full_value = float("nan")
    if graph.n <= full_cap:
        full = full_state_expectation(graph, gammas, betas, method="precompute", prefer_gpu=True, max_qubits=full_cap)
        full_value = full.value
        rows.append(
            CaseStudyRow(
                dataset=dataset,
                variant=variant,
                n=graph.n,
                m=graph.m,
                p=p,
                avg_degree=avg_degree,
                max_degree=max_degree,
                kmax=kmax,
                method="full_precompute_gpu",
                status=full.status,
                objective_value=full.value,
                objective_seconds=full.seconds,
                gradient_seconds=0.0,
                gradient_norm=float("nan"),
                abs_error_vs_full=0.0,
                peak_pool_mb=full.peak_pool_bytes / 1024**2,
                notes="exact full-state reference for small feature-selection QUBO",
            )
        )
    else:
        rows.append(
            CaseStudyRow(
                dataset=dataset,
                variant=variant,
                n=graph.n,
                m=graph.m,
                p=p,
                avg_degree=avg_degree,
                max_degree=max_degree,
                kmax=kmax,
                method="full_precompute_gpu",
                status=f"skipped_over_{full_cap}_qubits",
                objective_value=float("nan"),
                objective_seconds=0.0,
                gradient_seconds=0.0,
                gradient_norm=float("nan"),
                abs_error_vs_full=float("nan"),
                peak_pool_mb=0.0,
                notes="global full-state reference intentionally capped",
            )
        )

    lc = lightcone_expectation(
        graph,
        gammas,
        betas,
        p=p,
        prefer_gpu=True,
        max_k=max_k,
        max_batch_states=max_batch_states,
    )
    grad = lightcone_gradient_adjoint(
        graph,
        gammas,
        betas,
        p=p,
        prefer_gpu=True,
        max_k=max_k,
        max_batch_states=max_batch_states,
    )
    rows.append(
        CaseStudyRow(
            dataset=dataset,
            variant=variant,
            n=graph.n,
            m=graph.m,
            p=p,
            avg_degree=avg_degree,
            max_degree=max_degree,
            kmax=kmax,
            method="lc_batched_gpu_adjoint",
            status=lc.status if lc.status != "ok" else grad.status,
            objective_value=lc.value,
            objective_seconds=lc.seconds,
            gradient_seconds=grad.seconds,
            gradient_norm=float(np.linalg.norm(grad.gradient)) if grad.gradient is not None else float("nan"),
            abs_error_vs_full=abs(lc.value - full_value) if math.isfinite(full_value) else float("nan"),
            peak_pool_mb=max(lc.peak_pool_bytes, grad.peak_pool_bytes) / 1024**2,
            notes="real-data sparse feature-selection QUBO; LC objective plus exact adjoint gradient",
        )
    )
    return rows


def write_markdown(rows: list[CaseStudyRow], path: Path) -> None:
    lines = [
        "# Real-Data QUBO Case Study",
        "",
        "Feature-selection QUBOs are built from sklearn's breast-cancer and digits datasets.",
        "Node fields encode feature relevance; sparse negative pairwise terms penalize redundant correlated features.",
        "",
        "| Dataset | Variant | n | m | p | kmax | Method | Status | Obj | Obj s | Grad s | Err vs full | Peak MB |",
        "|---|---|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r.dataset} | {r.variant} | {r.n} | {r.m} | {r.p} | {r.kmax} | {r.method} | {r.status} | "
            f"{r.objective_value:.6g} | {r.objective_seconds:.4g} | {r.gradient_seconds:.4g} | "
            f"{r.abs_error_vs_full:.3g} | {r.peak_pool_mb:.3g} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "real_qubo_case_study.csv")
    parser.add_argument("--markdown", type=Path, default=ROOT / "results" / "real_qubo_case_study.md")
    parser.add_argument("--max-k", type=int, default=24)
    parser.add_argument("--max-batch-states", type=int, default=1 << 21)
    parser.add_argument("--full-cap", type=int, default=24)
    args = parser.parse_args()

    datasets = load_datasets()
    variants = [
        ("breast_cancer", "top20_deg3", 20, 3, [1, 2]),
        ("breast_cancer", "all30_deg3", 30, 3, [1, 2]),
        ("digits", "all64_deg2", 64, 2, [1, 2]),
    ]
    rows: list[CaseStudyRow] = []
    for dataset, variant, top_features, top_degree, ps in variants:
        x, y = datasets[dataset]
        graph = feature_selection_qubo(x, y, top_features=top_features, top_degree=top_degree)
        for p in ps:
            rows.extend(
                run_variant(
                    dataset,
                    variant,
                    graph,
                    p,
                    max_k=args.max_k,
                    max_batch_states=args.max_batch_states,
                    full_cap=args.full_cap,
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
