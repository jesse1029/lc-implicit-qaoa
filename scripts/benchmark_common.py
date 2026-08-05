from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def params_for_depth(p: int, *, seed: int = 0, init_id: int = 0) -> tuple[list[float], list[float]]:
    """Deterministic but non-degenerate QAOA angles for benchmark rows."""
    rng = random.Random(910000 + 1009 * p + 9176 * seed + init_id)
    gammas = [0.16 + 0.07 * layer + rng.uniform(-0.015, 0.015) for layer in range(p)]
    betas = [0.34 - 0.035 * layer + rng.uniform(-0.012, 0.012) for layer in range(p)]
    return gammas, betas


def pack_params(gammas: Iterable[float], betas: Iterable[float]) -> np.ndarray:
    return np.asarray(list(gammas) + list(betas), dtype=np.float64)


def unpack_params(x: np.ndarray, p: int) -> tuple[list[float], list[float]]:
    return x[:p].astype(float).tolist(), x[p:].astype(float).tolist()


def wrap_angles(x: np.ndarray) -> np.ndarray:
    return ((x + math.pi) % (2.0 * math.pi)) - math.pi


def now_seconds() -> float:
    return time.perf_counter()


def normalize_status(status: str) -> str:
    low = str(status)
    if low == "ok":
        return "SUCCESS"
    if "outofmemory" in low.lower() or "oom" in low.lower():
        return "OOM_GPU"
    if "timeout" in low.lower():
        return "TIMEOUT"
    if "unsupported_objective" in low.lower():
        return "UNSUPPORTED_OBJECTIVE"
    if "unsupported_gradient" in low.lower():
        return "UNSUPPORTED_GRADIENT"
    if "cpu_only" in low.lower():
        return "CPU_ONLY"
    if "not_run" in low.lower() or "skipped" in low.lower():
        return "NOT_RUN_EXPLAINED"
    return low


def safe_float(value) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


@dataclass(frozen=True)
class TimerResult:
    seconds: float
    status: str
    error: str = ""


def graph_metrics(graph) -> dict[str, float]:
    deg = [0 for _ in range(graph.n)]
    for i, j, _ in graph.edges:
        deg[int(i)] += 1
        deg[int(j)] += 1
    metrics = {
        "m": len(graph.edges),
        "mean_degree": float(sum(deg) / graph.n) if graph.n else 0.0,
        "max_degree": int(max(deg) if deg else 0),
        "degeneracy": float("nan"),
        "clustering": float("nan"),
    }
    try:
        import networkx as nx

        g = nx.Graph()
        g.add_nodes_from(range(graph.n))
        g.add_edges_from((int(i), int(j)) for i, j, _ in graph.edges)
        core = nx.core_number(g) if graph.n else {}
        metrics["degeneracy"] = float(max(core.values()) if core else 0)
        metrics["clustering"] = float(nx.average_clustering(g)) if graph.n else 0.0
    except Exception:
        pass
    return metrics


def cone_metrics(graph, p: int) -> dict[str, float]:
    from lcqaoa.lightcone import extract_lightcones

    cones = extract_lightcones(graph, p)
    if not cones:
        return {
            "term_count": 0,
            "kmax": 0,
            "k_median": 0.0,
            "k_p95": 0.0,
            "total_cone_states": 0,
            "max_batch_state_elements": 0,
        }
    ks = np.asarray([c.k for c in cones], dtype=np.int64)
    counts: dict[int, int] = {}
    for k in ks:
        counts[int(k)] = counts.get(int(k), 0) + 1
    max_batch = max(count * (1 << k) for k, count in counts.items())
    return {
        "term_count": int(len(cones)),
        "kmax": int(ks.max()),
        "k_median": float(np.median(ks)),
        "k_p95": float(np.percentile(ks, 95)),
        "total_cone_states": int(sum(1 << int(k) for k in ks)),
        "max_batch_state_elements": int(max_batch),
    }


def write_markdown_table(path: Path, title: str, rows: list[dict], columns: list[str], limit: int | None = None) -> None:
    lines = [f"# {title}", ""]
    show_rows = rows[:limit] if limit is not None else rows
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("|" + "|".join("---" for _ in columns) + "|")
    for row in show_rows:
        lines.append("| " + " | ".join(str(row.get(c, "")) for c in columns) + " |")
    if limit is not None and len(rows) > limit:
        lines.append("")
        lines.append(f"Showing {limit} of {len(rows)} rows; see CSV for full data.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
