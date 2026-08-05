from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from lcqaoa.graphs import WeightedGraph
from run_biomedical_feature_selection import (
    build_feature_qubo,
    mrmr_greedy,
    qubo_greedy,
    qubo_score,
    qubo_simulated_annealing,
    relevance_scores,
    standardize_numeric,
    topk_from_scores,
)


@dataclass
class DownstreamRow:
    dataset: str
    outer_fold: int
    selector: str
    classifier: str
    status: str
    n_train: int
    n_test: int
    n_features_pool: int
    budget: int
    selected_count: int
    qubo_score: float
    balanced_accuracy: float
    auroc: float
    selector_seconds: float
    classifier_seconds: float
    selected_features: str
    notes: str


@dataclass
class StabilityRow:
    dataset: str
    selector: str
    folds: int
    mean_jaccard: float
    median_jaccard: float
    min_jaccard: float


def load_datasets(include_openml: bool, max_features: int):
    from sklearn.datasets import fetch_openml, load_breast_cancer, load_digits
    from sklearn.preprocessing import LabelEncoder

    out = []
    breast = load_breast_cancer()
    x, names = standardize_numeric(breast.data)
    out.append(("breast_cancer_wisconsin", x, np.asarray(breast.target, dtype=int), list(map(str, breast.feature_names))))

    digits = load_digits()
    x, names = standardize_numeric(digits.data)
    out.append(("digits_64", x, np.asarray(digits.target, dtype=int), [f"pix{i}" for i in range(x.shape[1])]))

    if include_openml:
        try:
            mice = fetch_openml(data_id=40966, as_frame=True, parser="auto")
            x, names = standardize_numeric(mice.data)
            y = LabelEncoder().fit_transform(np.asarray(mice.target).astype(str))
            out.append(("mice_protein_openml40966", x, y, names))
        except Exception as exc:
            print(f"P1-4 OPENML_SKIP mice_protein_openml40966 {exc!r}", flush=True)
    trimmed = []
    for name, x, y, names in out:
        if x.shape[1] > max_features:
            scores = relevance_scores(x, y, 20260710)
            keep = np.argsort(-scores)[:max_features]
            trimmed.append((name, x[:, keep], y, [names[i] for i in keep]))
        else:
            trimmed.append((name, x, y, names))
    return trimmed


def train_qubo_on_subset(x_train, y_train, names, top_features: int, top_degree: int):
    graph, x_sel, y_sel, local_names, relevance, corr = build_feature_qubo(
        x_train,
        y_train,
        names,
        top_features=min(top_features, x_train.shape[1]),
        top_degree=top_degree,
        seed=20260710,
    )
    # build_feature_qubo ranks the training-fold columns by relevance and returns the
    # reordered view x_sel; graph node i therefore denotes original column order[i].
    # Callers must index the data with this order, not with the original column order,
    # or the indices returned by the relevance- and QUBO-space selectors address the
    # wrong features.
    index_of = {name: i for i, name in enumerate(names)}
    order = np.asarray([index_of[n] for n in local_names], dtype=int)
    return graph, local_names, relevance, corr, order


def select_features(selector: str, x_train, y_train, graph: WeightedGraph, relevance, corr, budget: int, seed: int):
    t0 = time.perf_counter()
    notes = ""
    status = "ok"
    selected: list[int] = []
    try:
        if selector == "mrmr":
            selected = mrmr_greedy(relevance, corr, budget)
        elif selector == "mutual_info_topk":
            selected = topk_from_scores(relevance, budget)
        elif selector == "qubo_greedy":
            selected = qubo_greedy(graph, budget)
        elif selector == "qubo_sa":
            selected = qubo_simulated_annealing(graph, budget, seed, steps=2500, restarts=8)
        elif selector == "l1_logistic":
            from sklearn.linear_model import LogisticRegression

            clf = LogisticRegression(
                penalty="l1",
                solver="saga",
                C=0.5,
                max_iter=4000,
                class_weight="balanced",
                random_state=seed,
                n_jobs=1,
            )
            clf.fit(x_train, y_train)
            scores = np.sum(np.abs(clf.coef_), axis=0)
            selected = topk_from_scores(scores, budget)
        elif selector == "rf_importance":
            from sklearn.ensemble import RandomForestClassifier

            rf = RandomForestClassifier(n_estimators=250, random_state=seed, class_weight="balanced_subsample", n_jobs=1)
            rf.fit(x_train, y_train)
            selected = topk_from_scores(rf.feature_importances_, budget)
        elif selector == "svm_rfe":
            from sklearn.feature_selection import RFE
            from sklearn.svm import LinearSVC

            estimator = LinearSVC(C=0.2, class_weight="balanced", dual="auto", max_iter=5000, random_state=seed)
            rfe = RFE(estimator, n_features_to_select=budget, step=0.25)
            rfe.fit(x_train, y_train)
            selected = [int(i) for i in np.flatnonzero(rfe.support_)]
        else:
            raise ValueError(selector)
    except Exception as exc:
        status = f"failed:{type(exc).__name__}"
        notes = str(exc)[:180]
        selected = []
    return selected, time.perf_counter() - t0, status, notes


def tune_classifier(name: str, x_train, y_train, seed: int):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import LinearSVC

    classes, counts = np.unique(y_train, return_counts=True)
    folds = max(2, min(3, int(counts.min())))
    scoring = "roc_auc" if len(classes) == 2 else "balanced_accuracy"
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    if name == "logistic":
        candidates = [
            make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=4000, solver="lbfgs", class_weight="balanced", random_state=seed))
            for c in [0.1, 1.0, 10.0]
        ]
    elif name == "linear_svm":
        candidates = [
            make_pipeline(StandardScaler(), LinearSVC(C=c, class_weight="balanced", dual="auto", max_iter=5000, random_state=seed))
            for c in [0.05, 0.2, 1.0]
        ]
    elif name == "random_forest":
        candidates = [
            RandomForestClassifier(n_estimators=200, max_depth=depth, random_state=seed, class_weight="balanced_subsample", n_jobs=1)
            for depth in [None, 8]
        ]
    else:
        raise ValueError(name)
    best_score = -float("inf")
    best = candidates[0]
    for model in candidates:
        try:
            score = float(np.mean(cross_val_score(model, x_train, y_train, cv=cv, scoring=scoring, n_jobs=1)))
        except Exception:
            score = -float("inf")
        if score > best_score:
            best_score = score
            best = model
    return best


def evaluate_classifier(model, x_train, y_train, x_test, y_test):
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score

    t0 = time.perf_counter()
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    bal = float(balanced_accuracy_score(y_test, pred))
    auroc = float("nan")
    if len(np.unique(y_train)) == 2:
        try:
            if hasattr(model, "predict_proba"):
                score = model.predict_proba(x_test)[:, 1]
            elif hasattr(model, "decision_function"):
                score = model.decision_function(x_test)
            else:
                score = pred
            auroc = float(roc_auc_score(y_test, score))
        except Exception:
            pass
    return bal, auroc, time.perf_counter() - t0


def jaccard_summary(rows: list[DownstreamRow]) -> list[StabilityRow]:
    out: list[StabilityRow] = []
    for dataset in sorted({r.dataset for r in rows}):
        for selector in sorted({r.selector for r in rows if r.dataset == dataset}):
            # A selector's subset depends on the outer fold only, not on the downstream
            # classifier, so the three classifier rows of one fold carry the same subset.
            # Pool one set per outer fold; pooling all rows would inject same-fold pairs
            # with Jaccard 1.0 and bias the median upward.
            by_fold: dict[int, set[int]] = {}
            for r in rows:
                if r.dataset == dataset and r.selector == selector and r.status == "ok" and r.selected_features:
                    by_fold.setdefault(int(r.outer_fold),
                                       set(int(x) for x in r.selected_features.split(",") if x != ""))
            sets = [by_fold[k] for k in sorted(by_fold)]
            vals = []
            for a, b in combinations(sets, 2):
                vals.append(len(a & b) / max(len(a | b), 1))
            out.append(
                StabilityRow(
                    dataset=dataset,
                    selector=selector,
                    folds=len(sets),
                    mean_jaccard=float(np.mean(vals)) if vals else float("nan"),
                    median_jaccard=float(np.median(vals)) if vals else float("nan"),
                    min_jaccard=float(np.min(vals)) if vals else float("nan"),
                )
            )
    return out


def run_dataset(name: str, x, y, names, args) -> list[DownstreamRow]:
    from sklearn.model_selection import StratifiedKFold

    budget = min(args.budget, max(2, x.shape[1] // 5))
    top_features = min(args.top_features, x.shape[1])
    outer = StratifiedKFold(n_splits=args.outer_folds, shuffle=True, random_state=args.seed)
    rows: list[DownstreamRow] = []
    selectors = ["mutual_info_topk", "mrmr", "qubo_greedy", "qubo_sa", "l1_logistic", "rf_importance", "svm_rfe"]
    classifiers = ["logistic", "linear_svm", "random_forest"]
    for fold, (tr, te) in enumerate(outer.split(x, y)):
        x_tr_raw, x_te_raw = x[tr], x[te]
        y_tr, y_te = y[tr], y[te]
        graph, local_names, relevance, corr, order = train_qubo_on_subset(x_tr_raw, y_tr, names, top_features, args.top_degree)
        x_tr = x_tr_raw[:, order]
        x_te = x_te_raw[:, order]
        for selector in selectors:
            selected, sel_s, status, notes = select_features(selector, x_tr, y_tr, graph, relevance, corr, budget, args.seed + fold)
            selected = [int(i) for i in selected[:budget]]
            for clf_name in classifiers:
                print(f"P1-4 dataset={name} fold={fold} selector={selector} clf={clf_name}", flush=True)
                if status != "ok" or not selected:
                    bal, auroc, clf_s = float("nan"), float("nan"), 0.0
                    row_status = status
                else:
                    try:
                        model = tune_classifier(clf_name, x_tr[:, selected], y_tr, args.seed + fold)
                        bal, auroc, clf_s = evaluate_classifier(model, x_tr[:, selected], y_tr, x_te[:, selected], y_te)
                        row_status = "ok"
                    except Exception as exc:
                        bal, auroc, clf_s = float("nan"), float("nan"), 0.0
                        row_status = f"failed:{type(exc).__name__}"
                        notes = (notes + "; " + str(exc)[:160]).strip("; ")
                rows.append(
                    DownstreamRow(
                        dataset=name,
                        outer_fold=fold,
                        selector=selector,
                        classifier=clf_name,
                        status=row_status,
                        n_train=len(tr),
                        n_test=len(te),
                        n_features_pool=graph.n,
                        budget=budget,
                        selected_count=len(selected),
                        qubo_score=qubo_score(graph, selected) if selected else float("nan"),
                        balanced_accuracy=bal,
                        auroc=auroc,
                        selector_seconds=sel_s,
                        classifier_seconds=clf_s,
                        # record ORIGINAL column ids: the per-fold relevance ordering makes
                        # graph-space indices incomparable across folds, so fold-to-fold
                        # selector stability must be measured in the original feature space.
                        selected_features=",".join(str(int(order[i])) for i in selected),
                        notes=notes,
                    )
                )
    return rows


def write_md(rows: list[DownstreamRow], stability: list[StabilityRow], path: Path) -> None:
    lines = [
        "# P1-4 AI Downstream Nested-CV Evidence",
        "",
        "Selector construction is repeated inside each outer training fold. Classifier hyperparameters are selected by inner CV on the training fold.",
        "",
        "| Dataset | Selector | Classifier | OK rows | bal acc median | AUROC median | selector s median | classifier s median |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for dataset in sorted({r.dataset for r in rows}):
        for selector in sorted({r.selector for r in rows if r.dataset == dataset}):
            for clf in sorted({r.classifier for r in rows if r.dataset == dataset and r.selector == selector}):
                sub = [r for r in rows if r.dataset == dataset and r.selector == selector and r.classifier == clf and r.status == "ok"]
                if not sub:
                    lines.append(f"| {dataset} | {selector} | {clf} | 0 | nan | nan | nan | nan |")
                    continue
                lines.append(
                    f"| {dataset} | {selector} | {clf} | {len(sub)} | "
                    f"{float(np.median([r.balanced_accuracy for r in sub])):.4g} | "
                    f"{float(np.nanmedian([r.auroc for r in sub])):.4g} | "
                    f"{float(np.median([r.selector_seconds for r in sub])):.4g} | "
                    f"{float(np.median([r.classifier_seconds for r in sub])):.4g} |"
                )
    lines.extend(["", "## Stability", "", "| Dataset | Selector | folds | mean Jaccard | median Jaccard |", "|---|---|---:|---:|---:|"])
    for s in stability:
        lines.append(f"| {s.dataset} | {s.selector} | {s.folds} | {s.mean_jaccard:.4g} | {s.median_jaccard:.4g} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "aaai27_required_experiments_20260710" / "P1_4_ai_downstream")
    parser.add_argument("--include-openml", action="store_true")
    parser.add_argument("--max-features", type=int, default=64)
    parser.add_argument("--top-features", type=int, default=64)
    parser.add_argument("--top-degree", type=int, default=3)
    parser.add_argument("--budget", type=int, default=12)
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if args.quick:
        args.outer_folds = 2
        args.max_features = min(args.max_features, 24)
        args.top_features = min(args.top_features, 24)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[DownstreamRow] = []
    csv_path = args.out_dir / "P1_4_ai_downstream.csv"
    for dataset in load_datasets(args.include_openml, args.max_features):
        name, x, y, names = dataset
        if args.quick and name != "breast_cancer_wisconsin":
            continue
        rows.extend(run_dataset(name, x, y, names, args))
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
            writer.writeheader()
            for row in rows:
                writer.writerow(asdict(row))
        stability = jaccard_summary(rows)
        with (args.out_dir / "P1_4_ai_downstream_stability.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(asdict(stability[0]).keys()))
            writer.writeheader()
            for row in stability:
                writer.writerow(asdict(row))
        write_md(rows, stability, args.out_dir / "P1_4_ai_downstream.md")
    print(f"WROTE {csv_path}")


if __name__ == "__main__":
    main()
