from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lcqaoa.graphs import Edge, Field, WeightedGraph
from lcqaoa.lightcone import extract_lightcones, lightcone_expectation, lightcone_gradient_adjoint
from lcqaoa.qaoa import full_state_expectation


@dataclass
class BiomedicalRuntimeRow:
    dataset: str
    variant: str
    n: int
    m: int
    p: int
    kmax: int
    method: str
    status: str
    objective_value: float
    objective_seconds: float
    gradient_seconds: float
    abs_error_vs_full: float
    peak_pool_mb: float
    notes: str


@dataclass
class BiomedicalSelectorRow:
    dataset: str
    variant: str
    n: int
    m: int
    budget: int
    selector: str
    status: str
    selected_count: int
    qubo_score: float
    cv_metric: str
    cv_mean: float
    cv_std: float
    seconds: float
    selected_features: str
    notes: str


def standardize_numeric(x):
    import pandas as pd
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler

    if isinstance(x, pd.DataFrame):
        x = x.select_dtypes(include=[np.number])
        cols = [str(c) for c in x.columns]
        arr = x.to_numpy(dtype=np.float64)
    else:
        arr = np.asarray(x, dtype=np.float64)
        cols = [f"f{i}" for i in range(arr.shape[1])]
    arr = SimpleImputer(strategy="median").fit_transform(arr)
    keep = np.nanstd(arr, axis=0) > 1e-12
    arr = arr[:, keep]
    cols = [c for c, ok in zip(cols, keep) if ok]
    arr = StandardScaler().fit_transform(arr)
    return arr, cols


def load_datasets(include_openml: bool):
    from sklearn.datasets import fetch_openml, load_breast_cancer
    from sklearn.preprocessing import LabelEncoder

    datasets = []
    breast = load_breast_cancer()
    x, cols = standardize_numeric(breast.data)
    y = np.asarray(breast.target, dtype=int)
    datasets.append(("breast_cancer_wisconsin", x, y, list(breast.feature_names), "binary_cancer"))

    if include_openml:
        try:
            mice = fetch_openml(data_id=40966, as_frame=True, parser="auto")
            x, cols = standardize_numeric(mice.data)
            y = LabelEncoder().fit_transform(np.asarray(mice.target).astype(str))
            datasets.append(("mice_protein_openml40966", x, y, cols, "multiclass_protein_expression"))
        except Exception as exc:
            print(f"OPENML_SKIP mice_protein_openml40966 {exc!r}", flush=True)
    return datasets


def relevance_scores(x: np.ndarray, y: np.ndarray, seed: int) -> np.ndarray:
    from sklearn.feature_selection import mutual_info_classif

    scores = mutual_info_classif(x, y, random_state=seed)
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    if scores.max() > 0:
        scores = scores / scores.max()
    return scores


def build_feature_qubo(
    x: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    *,
    top_features: int,
    top_degree: int,
    seed: int,
    redundancy_scale: float = 0.55,
):
    relevance_all = relevance_scores(x, y, seed)
    order = np.argsort(-relevance_all)[: min(top_features, x.shape[1])]
    x_sel = x[:, order]
    names = [feature_names[i] for i in order]
    relevance = relevance_all[order]
    corr = np.abs(np.corrcoef(x_sel, rowvar=False))
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(corr, 0.0)
    edges_map: dict[tuple[int, int], float] = {}
    for i in range(x_sel.shape[1]):
        for j in np.argsort(-corr[i])[: min(top_degree, max(0, x_sel.shape[1] - 1))]:
            if i == j or corr[i, j] <= 0:
                continue
            a, b = sorted((int(i), int(j)))
            w = -float(redundancy_scale * corr[i, j])
            if (a, b) not in edges_map or abs(w) > abs(edges_map[(a, b)]):
                edges_map[(a, b)] = w
    edges: tuple[Edge, ...] = tuple((i, j, w) for (i, j), w in sorted(edges_map.items()))
    fields: tuple[Field, ...] = tuple((int(i), float(relevance[i])) for i in range(x_sel.shape[1]))
    graph = WeightedGraph(n=x_sel.shape[1], edges=edges, fields=fields, objective="qubo")
    return graph, x_sel, y, names, relevance, corr


def qubo_score(graph: WeightedGraph, selected: list[int]) -> float:
    s = set(selected)
    total = 0.0
    for i, w in graph.fields:
        if i in s:
            total += w
    for i, j, w in graph.edges:
        if i in s and j in s:
            total += w
    return float(total)


def topk_from_scores(scores: np.ndarray, budget: int) -> list[int]:
    return [int(i) for i in np.argsort(-np.abs(scores))[:budget]]


def mrmr_greedy(relevance: np.ndarray, corr: np.ndarray, budget: int) -> list[int]:
    chosen: list[int] = []
    remaining = set(range(len(relevance)))
    while len(chosen) < budget and remaining:
        best = None
        best_score = -float("inf")
        for i in remaining:
            redundancy = float(np.mean([corr[i, j] for j in chosen])) if chosen else 0.0
            score = float(relevance[i] - redundancy)
            if score > best_score:
                best_score = score
                best = i
        chosen.append(int(best))
        remaining.remove(int(best))
    return chosen


def qubo_greedy(graph: WeightedGraph, budget: int) -> list[int]:
    chosen: list[int] = []
    remaining = set(range(graph.n))
    while len(chosen) < budget and remaining:
        best = None
        best_score = -float("inf")
        current = qubo_score(graph, chosen)
        for i in remaining:
            score = qubo_score(graph, chosen + [i]) - current
            if score > best_score:
                best = i
                best_score = score
        chosen.append(int(best))
        remaining.remove(int(best))
    return chosen


def qubo_simulated_annealing(graph: WeightedGraph, budget: int, seed: int, steps: int = 4000, restarts: int = 16) -> list[int]:
    rng = np.random.default_rng(seed)
    best_sel = qubo_greedy(graph, budget)
    best_score = qubo_score(graph, best_sel)
    all_nodes = np.arange(graph.n)
    for _ in range(restarts):
        sel = list(rng.choice(all_nodes, size=budget, replace=False))
        score = qubo_score(graph, sel)
        for step in range(steps):
            temp = max(0.01, 1.0 - step / steps)
            out_pos = int(rng.integers(0, budget))
            candidates = list(set(range(graph.n)) - set(sel))
            if not candidates:
                break
            new_node = int(rng.choice(candidates))
            proposal = sel.copy()
            proposal[out_pos] = new_node
            new_score = qubo_score(graph, proposal)
            delta = new_score - score
            if delta >= 0 or rng.random() < math.exp(delta / temp):
                sel = proposal
                score = new_score
                if score > best_score:
                    best_sel = sel.copy()
                    best_score = score
    return [int(i) for i in best_sel]


def selector_outputs(x: np.ndarray, y: np.ndarray, relevance: np.ndarray, corr: np.ndarray, graph: WeightedGraph, budget: int, seed: int):
    outputs: list[tuple[str, str, list[int], float, str]] = []
    t0 = time.perf_counter()
    outputs.append(("mutual_info_topk", "ok", topk_from_scores(relevance, budget), time.perf_counter() - t0, "filter baseline"))

    t0 = time.perf_counter()
    outputs.append(("mrmr_greedy", "ok", mrmr_greedy(relevance, corr, budget), time.perf_counter() - t0, "minimum-redundancy maximum-relevance greedy baseline"))

    t0 = time.perf_counter()
    outputs.append(("qubo_greedy", "ok", qubo_greedy(graph, budget), time.perf_counter() - t0, "greedy maximization of the same sparse QUBO"))

    t0 = time.perf_counter()
    outputs.append(("qubo_simulated_annealing", "ok", qubo_simulated_annealing(graph, budget, seed), time.perf_counter() - t0, "classical stochastic QUBO heuristic"))

    try:
        from sklearn.ensemble import RandomForestClassifier

        t0 = time.perf_counter()
        rf = RandomForestClassifier(n_estimators=300, random_state=seed, class_weight="balanced_subsample", n_jobs=1)
        rf.fit(x, y)
        outputs.append(("random_forest_importance", "ok", topk_from_scores(rf.feature_importances_, budget), time.perf_counter() - t0, "tree-importance wrapper baseline"))
    except Exception as exc:
        outputs.append(("random_forest_importance", f"failed_{type(exc).__name__}", [], 0.0, repr(exc)))

    for name, penalty, l1_ratio in [("lasso_logistic_l1", "l1", None), ("elastic_net_logistic", "elasticnet", 0.5)]:
        try:
            from sklearn.linear_model import LogisticRegression

            t0 = time.perf_counter()
            solver = "saga" if len(np.unique(y)) > 2 or penalty == "elasticnet" else "liblinear"
            kwargs = dict(
                penalty=penalty,
                solver=solver,
                max_iter=5000,
                random_state=seed,
                class_weight="balanced",
                C=0.5,
            )
            if l1_ratio is not None:
                kwargs["l1_ratio"] = l1_ratio
            model = LogisticRegression(**kwargs)
            model.fit(x, y)
            coef = np.sum(np.abs(model.coef_), axis=0)
            outputs.append((name, "ok", topk_from_scores(coef, budget), time.perf_counter() - t0, "sparse linear-model feature selector"))
        except Exception as exc:
            outputs.append((name, f"failed_{type(exc).__name__}", [], 0.0, repr(exc)))

    try:
        from sklearn.feature_selection import RFE
        from sklearn.svm import LinearSVC

        t0 = time.perf_counter()
        estimator = LinearSVC(C=0.2, class_weight="balanced", dual="auto", max_iter=5000, random_state=seed)
        rfe = RFE(estimator, n_features_to_select=budget, step=0.2)
        rfe.fit(x, y)
        outputs.append(("svm_rfe", "ok", [int(i) for i in np.flatnonzero(rfe.support_)], time.perf_counter() - t0, "linear-SVM recursive feature elimination"))
    except Exception as exc:
        outputs.append(("svm_rfe", f"failed_{type(exc).__name__}", [], 0.0, repr(exc)))

    return outputs


def cv_metric(x: np.ndarray, y: np.ndarray, selected: list[int], seed: int) -> tuple[str, float, float]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if not selected:
        return "balanced_accuracy", float("nan"), float("nan")
    counts = np.bincount(y)
    folds = max(2, min(5, int(counts[counts > 0].min())))
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    scoring = "roc_auc" if len(np.unique(y)) == 2 else "balanced_accuracy"
    solver = "liblinear" if len(np.unique(y)) == 2 else "lbfgs"
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=5000, solver=solver, class_weight="balanced", random_state=seed),
    )
    scores = cross_val_score(model, x[:, selected], y, cv=cv, scoring=scoring, n_jobs=1)
    return scoring, float(np.mean(scores)), float(np.std(scores))


def kmax_for(graph: WeightedGraph, p: int) -> int:
    return max((c.k for c in extract_lightcones(graph, p)), default=0)


def run_runtime(dataset: str, variant: str, graph: WeightedGraph, p: int, *, max_k: int, max_batch_states: int, full_cap: int):
    gammas = [0.22 + 0.07 * i for i in range(p)]
    betas = [0.31 - 0.04 * i for i in range(p)]
    kmax = kmax_for(graph, p)
    rows: list[BiomedicalRuntimeRow] = []
    full_value = float("nan")
    if graph.n <= full_cap:
        full = full_state_expectation(graph, gammas, betas, method="precompute", prefer_gpu=True, max_qubits=full_cap)
        full_value = full.value
        rows.append(
            BiomedicalRuntimeRow(dataset, variant, graph.n, graph.m, p, kmax, "full_precompute_gpu", full.status, full.value, full.seconds, 0.0, 0.0, full.peak_pool_bytes / 1024**2, "global full-state reference")
        )
    else:
        rows.append(
            BiomedicalRuntimeRow(dataset, variant, graph.n, graph.m, p, kmax, "full_precompute_gpu", f"skipped_over_{full_cap}_qubits", float("nan"), 0.0, 0.0, float("nan"), 0.0, "global full-state reference intentionally capped")
        )
    lc = lightcone_expectation(graph, gammas, betas, p=p, prefer_gpu=True, max_k=max_k, max_batch_states=max_batch_states)
    grad = lightcone_gradient_adjoint(graph, gammas, betas, p=p, prefer_gpu=True, max_k=max_k, max_batch_states=max_batch_states)
    rows.append(
        BiomedicalRuntimeRow(
            dataset,
            variant,
            graph.n,
            graph.m,
            p,
            kmax,
            "lc_batched_gpu_adjoint",
            lc.status if lc.status != "ok" else grad.status,
            lc.value,
            lc.seconds,
            grad.seconds,
            abs(lc.value - full_value) if math.isfinite(full_value) else float("nan"),
            max(lc.peak_pool_bytes, grad.peak_pool_bytes) / 1024**2,
            "LC objective plus exact reverse-mode adjoint gradient",
        )
    )
    return rows


def write_markdown(runtime_rows: list[BiomedicalRuntimeRow], selector_rows: list[BiomedicalSelectorRow], path: Path) -> None:
    lines = [
        "# Biomedical Feature-Selection QUBO Benchmark",
        "",
        "This benchmark is an application validation section, not the main systems claim.",
        "Feature-selection QUBOs use relevance fields and sparse redundancy penalties.",
        "",
        "## LC Runtime",
        "",
        "| Dataset | Variant | n | m | p | kmax | Method | Status | Obj s | Grad s | Err vs full | Peak MB |",
        "|---|---|---:|---:|---:|---:|---|---|---:|---:|---:|---:|",
    ]
    for r in runtime_rows:
        lines.append(
            f"| {r.dataset} | {r.variant} | {r.n} | {r.m} | {r.p} | {r.kmax} | {r.method} | {r.status} | "
            f"{r.objective_seconds:.4g} | {r.gradient_seconds:.4g} | {r.abs_error_vs_full:.3g} | {r.peak_pool_mb:.3g} |"
        )
    lines.extend(
        [
            "",
            "## Feature-Selection Baselines",
            "",
            "| Dataset | Variant | Budget | Selector | Status | QUBO score | CV metric | CV mean | CV std | Seconds |",
            "|---|---|---:|---|---|---:|---|---:|---:|---:|",
        ]
    )
    for r in selector_rows:
        lines.append(
            f"| {r.dataset} | {r.variant} | {r.budget} | {r.selector} | {r.status} | "
            f"{r.qubo_score:.4g} | {r.cv_metric} | {r.cv_mean:.4g} | {r.cv_std:.3g} | {r.seconds:.4g} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-runtime", type=Path, default=ROOT / "results" / "biomedical_runtime.csv")
    parser.add_argument("--out-selectors", type=Path, default=ROOT / "results" / "biomedical_selectors.csv")
    parser.add_argument("--markdown", type=Path, default=ROOT / "results" / "biomedical_feature_selection.md")
    parser.add_argument("--include-openml", action="store_true")
    parser.add_argument("--seed", type=int, default=20260701)
    parser.add_argument("--max-k", type=int, default=24)
    parser.add_argument("--max-batch-states", type=int, default=1 << 21)
    parser.add_argument("--full-cap", type=int, default=24)
    args = parser.parse_args()

    runtime_rows: list[BiomedicalRuntimeRow] = []
    selector_rows: list[BiomedicalSelectorRow] = []
    for dataset, x, y, names, task in load_datasets(args.include_openml):
        variants = [(20, 3, 8), (min(64, x.shape[1]), 3 if x.shape[1] <= 32 else 2, min(12, max(4, x.shape[1] // 5)))]
        seen = set()
        for top_features, top_degree, budget in variants:
            top_features = min(top_features, x.shape[1])
            key = (top_features, top_degree, budget)
            if key in seen:
                continue
            seen.add(key)
            variant = f"top{top_features}_deg{top_degree}_budget{budget}"
            graph, x_sel, y_sel, local_names, relevance, corr = build_feature_qubo(
                x,
                y,
                names,
                top_features=top_features,
                top_degree=top_degree,
                seed=args.seed,
            )
            print(f"BIOMED dataset={dataset} variant={variant} n={graph.n} m={graph.m}", flush=True)
            for p in (1, 2):
                runtime_rows.extend(
                    run_runtime(
                        dataset,
                        variant,
                        graph,
                        p,
                        max_k=args.max_k,
                        max_batch_states=args.max_batch_states,
                        full_cap=args.full_cap,
                    )
                )
            for selector, status, selected, seconds, notes in selector_outputs(x_sel, y_sel, relevance, corr, graph, budget, args.seed):
                if status == "ok":
                    metric, mean, std = cv_metric(x_sel, y_sel, selected, args.seed)
                    score = qubo_score(graph, selected)
                    selected_names = ",".join(local_names[i] for i in selected)
                else:
                    metric, mean, std, score, selected_names = "NA", float("nan"), float("nan"), float("nan"), ""
                selector_rows.append(
                    BiomedicalSelectorRow(
                        dataset,
                        variant,
                        graph.n,
                        graph.m,
                        budget,
                        selector,
                        status,
                        len(selected),
                        score,
                        metric,
                        mean,
                        std,
                        seconds,
                        selected_names,
                        notes,
                    )
                )

    args.out_runtime.parent.mkdir(parents=True, exist_ok=True)
    with args.out_runtime.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(runtime_rows[0]).keys()))
        writer.writeheader()
        for row in runtime_rows:
            writer.writerow(asdict(row))
    with args.out_selectors.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(selector_rows[0]).keys()))
        writer.writeheader()
        for row in selector_rows:
            writer.writerow(asdict(row))
    write_markdown(runtime_rows, selector_rows, args.markdown)
    print(f"WROTE {args.out_runtime}")
    print(f"WROTE {args.out_selectors}")
    print(f"WROTE {args.markdown}")


if __name__ == "__main__":
    main()
