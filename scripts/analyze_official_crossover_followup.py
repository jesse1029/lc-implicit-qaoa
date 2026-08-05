from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


Q_VALUES = (1, 5, 10, 25, 50, 100, 250)
FEATURES = ("log_total_cone_states", "log_kmax")


def peer_pairs(frame: pd.DataFrame, peer_token: str, dataset: str) -> pd.DataFrame:
    frame = frame.copy()
    key = ["family", "n", "p", "seed"]
    metric = [
        "preprocess_seconds",
        "setup_seconds",
        "cold_seconds",
        "steady_median_seconds",
    ]
    extra = [name for name in ("kmax", "total_cone_states") if name in frame]
    lc = frame[
        frame["backend"].astype(str).str.startswith("LC") & (frame["status"] == "ok")
    ][key + metric + extra].copy()
    peer = frame[
        frame["backend"].astype(str).str.contains(peer_token, regex=False)
        & (frame["status"] == "ok")
    ][key + metric].copy()
    lc = lc.rename(columns={name: f"lc_{name}" for name in metric})
    peer = peer.rename(columns={name: f"peer_{name}" for name in metric})
    pairs = lc.merge(peer, on=key, how="inner", validate="one_to_one")
    pairs.insert(0, "dataset", dataset)
    pairs["peer_over_lc"] = (
        pairs["peer_steady_median_seconds"] / pairs["lc_steady_median_seconds"]
    )
    return pairs


def bootstrap_median(values: np.ndarray, rng: np.random.Generator, reps: int) -> tuple[float, float]:
    draws = values[rng.integers(0, len(values), size=(reps, len(values)))]
    medians = np.median(draws, axis=1)
    return tuple(float(x) for x in np.percentile(medians, [2.5, 97.5]))


def paired_summary(pairs: pd.DataFrame, reps: int) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(20260712)
    for keys, cell in pairs.groupby(["dataset", "family", "n", "p"], sort=True):
        values = cell["peer_over_lc"].to_numpy(dtype=float)
        low, high = bootstrap_median(values, rng, reps)
        faster = int(np.count_nonzero(values > 1.0))
        slower = int(np.count_nonzero(values < 1.0))
        non_ties = faster + slower
        sign_p = float(binomtest(faster, non_ties, 0.5).pvalue) if non_ties else 1.0
        try:
            wilcoxon_p = float(wilcoxon(np.log(values), zero_method="wilcox").pvalue)
        except ValueError:
            wilcoxon_p = 1.0
        decision = "lc_faster" if low > 1.0 else "peer_faster" if high < 1.0 else "tie_or_unstable"
        rows.append(
            {
                "dataset": keys[0],
                "family": keys[1],
                "n": int(keys[2]),
                "p": int(keys[3]),
                "paired_seeds": len(values),
                "median_peer_over_lc": float(np.median(values)),
                "q1_peer_over_lc": float(np.percentile(values, 25)),
                "q3_peer_over_lc": float(np.percentile(values, 75)),
                "bootstrap_ci_low": low,
                "bootstrap_ci_high": high,
                "lc_faster_seed_fraction": faster / len(values),
                "paired_sign_test_p": sign_p,
                "wilcoxon_log_ratio_p": wilcoxon_p,
                "decision": decision,
            }
        )
    return pd.DataFrame(rows)


def total_time(frame: pd.DataFrame, prefix: str, q: int) -> pd.Series:
    startup = (
        frame[f"{prefix}_preprocess_seconds"]
        + frame[f"{prefix}_setup_seconds"]
        + frame[f"{prefix}_cold_seconds"]
    )
    return startup + max(q - 1, 0) * frame[f"{prefix}_steady_median_seconds"]


def break_even_q(lc_start: float, lc_steady: float, peer_start: float, peer_steady: float) -> float:
    if lc_steady >= peer_steady:
        return math.inf
    if lc_start <= peer_start:
        return 1.0
    threshold = 1.0 + (lc_start - peer_start) / (peer_steady - lc_steady)
    return float(max(1, math.ceil(threshold - 1e-12)))


def peer_overtake_q(lc_start: float, lc_steady: float, peer_start: float, peer_steady: float) -> float:
    if peer_steady >= lc_steady:
        return math.inf
    if peer_start <= lc_start:
        return 1.0
    threshold = 1.0 + (peer_start - lc_start) / (lc_steady - peer_steady)
    return float(max(1, math.ceil(threshold - 1e-12)))


def break_even_tables(pairs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    seed_rows = []
    for _, row in pairs.iterrows():
        lc_start = float(
            row.lc_preprocess_seconds + row.lc_setup_seconds + row.lc_cold_seconds
        )
        peer_start = float(
            row.peer_preprocess_seconds + row.peer_setup_seconds + row.peer_cold_seconds
        )
        base = {
            "dataset": row.dataset,
            "family": row.family,
            "n": int(row.n),
            "p": int(row.p),
            "seed": int(row.seed),
            "q_break_even": break_even_q(
                lc_start,
                float(row.lc_steady_median_seconds),
                peer_start,
                float(row.peer_steady_median_seconds),
            ),
            "q_peer_overtakes_lc": peer_overtake_q(
                lc_start,
                float(row.lc_steady_median_seconds),
                peer_start,
                float(row.peer_steady_median_seconds),
            ),
        }
        for q in Q_VALUES:
            base[f"lc_T{q}"] = float(total_time(pd.DataFrame([row]), "lc", q).iloc[0])
            base[f"peer_T{q}"] = float(total_time(pd.DataFrame([row]), "peer", q).iloc[0])
            base[f"winner_T{q}"] = "LC" if base[f"lc_T{q}"] < base[f"peer_T{q}"] else "peer"
        base["optimizer_100_break_even"] = base["lc_T100"] < base["peer_T100"]
        seed_rows.append(base)
    seed_table = pd.DataFrame(seed_rows)

    summary_rows = []
    for keys, cell in pairs.groupby(["dataset", "family", "n", "p"], sort=True):
        row = {
            "dataset": keys[0],
            "family": keys[1],
            "n": int(keys[2]),
            "p": int(keys[3]),
            "paired_seeds": len(cell),
        }
        lc_start = float(
            np.median(
                cell.lc_preprocess_seconds + cell.lc_setup_seconds + cell.lc_cold_seconds
            )
        )
        peer_start = float(
            np.median(
                cell.peer_preprocess_seconds + cell.peer_setup_seconds + cell.peer_cold_seconds
            )
        )
        lc_steady = float(np.median(cell.lc_steady_median_seconds))
        peer_steady = float(np.median(cell.peer_steady_median_seconds))
        row["q_break_even_from_component_medians"] = break_even_q(
            lc_start, lc_steady, peer_start, peer_steady
        )
        row["q_peer_overtakes_lc_from_component_medians"] = peer_overtake_q(
            lc_start, lc_steady, peer_start, peer_steady
        )
        finite_seed_breaks = seed_table[
            (seed_table.dataset == keys[0])
            & (seed_table.family == keys[1])
            & (seed_table.n == keys[2])
            & (seed_table.p == keys[3])
            & np.isfinite(seed_table.q_break_even)
        ].q_break_even
        row["median_seed_q_break_even"] = (
            float(np.median(finite_seed_breaks)) if len(finite_seed_breaks) else math.inf
        )
        for q in Q_VALUES:
            lc_values = total_time(cell, "lc", q)
            peer_values = total_time(cell, "peer", q)
            row[f"median_lc_T{q}"] = float(np.median(lc_values))
            row[f"median_peer_T{q}"] = float(np.median(peer_values))
            row[f"winner_T{q}"] = "LC" if row[f"median_lc_T{q}"] < row[f"median_peer_T{q}"] else "peer"
        row["optimizer_100_break_even"] = row["winner_T100"] == "LC"
        summary_rows.append(row)
    return seed_table, pd.DataFrame(summary_rows)


def frozen_dispatcher(
    train_path: Path, external: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    train = pd.read_csv(train_path)
    train = train[
        (train["precision"] == "c64_f32")
        & (train["p"] >= 2)
        & (train["stage"] == "value+gradient")
        & (train["status"] == "ok")
    ].copy()
    key = ["family", "n", "p", "seed", "kmax", "total_cone_states"]
    lc = train[train.method.str.startswith("LC")][key + ["steady_median_seconds"]].rename(
        columns={"steady_median_seconds": "lc_seconds"}
    )
    peer = train[train.method.str.startswith("global-state")][
        key + ["steady_median_seconds"]
    ].rename(columns={"steady_median_seconds": "peer_seconds"})
    train_pairs = lc.merge(peer, on=key, validate="one_to_one")
    train_pairs["log_total_cone_states"] = np.log(train_pairs.total_cone_states)
    train_pairs["log_kmax"] = np.log(train_pairs.kmax)
    train_y = (train_pairs.peer_seconds > train_pairs.lc_seconds).astype(int)
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            class_weight="balanced", max_iter=2000, random_state=20260712
        ),
    )
    model.fit(train_pairs[list(FEATURES)], train_y)

    tested = external.copy()
    tested["log_total_cone_states"] = np.log(tested.total_cone_states)
    tested["log_kmax"] = np.log(tested.kmax)
    tested["actual_choose_lc"] = (tested.peer_over_lc > 1.0).astype(int)
    tested["probability_choose_lc"] = model.predict_proba(tested[list(FEATURES)])[:, 1]
    tested["predicted_choose_lc"] = (tested.probability_choose_lc >= 0.5).astype(int)
    tested["predefined_target_choose_lc"] = (
        (tested.kmax <= 14) & (tested.total_cone_states <= 2.0e7)
    ).astype(int)
    tested["near_crossover"] = tested.peer_over_lc.between(0.75, 1.33)

    metrics = []
    for dataset, cell in tested.groupby("dataset", sort=True):
        actual = cell.actual_choose_lc.to_numpy()
        for rule, column in (
            ("frozen_logistic_threshold_0.5", "predicted_choose_lc"),
            ("predefined_target_k14_S2e7", "predefined_target_choose_lc"),
        ):
            predicted = cell[column].to_numpy()
            matrix = confusion_matrix(actual, predicted, labels=[0, 1])
            correct = predicted == actual
            metrics.append(
                {
                    "dataset": dataset,
                    "rule": rule,
                    "external_cells": len(cell),
                    "accuracy": accuracy_score(actual, predicted),
                    "balanced_accuracy": balanced_accuracy_score(actual, predicted),
                    "precision_choose_lc": precision_score(actual, predicted, zero_division=0),
                    "recall_choose_lc": recall_score(actual, predicted, zero_division=0),
                    "tn": int(matrix[0, 0]),
                    "fp": int(matrix[0, 1]),
                    "fn": int(matrix[1, 0]),
                    "tp": int(matrix[1, 1]),
                    "near_crossover_errors": int((cell.near_crossover & ~correct).sum()),
                }
            )
    metadata = {
        "training_source": str(train_path),
        "training_cells": len(train_pairs),
        "features": list(FEATURES),
        "fixed_probability_threshold": 0.5,
        "predefined_target_rule": "kmax<=14 and total_cone_states<=2e7",
        "threshold_retuned_on_official_cells": False,
        "training_positive_fraction": float(train_y.mean()),
        "interpretation": "external sanity check, not a learned runtime predictor",
    }
    return tested, pd.DataFrame(metrics), metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuaoa-3070", type=Path, required=True)
    parser.add_argument("--cuaoa-3090", type=Path, required=True)
    parser.add_argument("--lightning-3090", type=Path, required=True)
    parser.add_argument("--cone-lookup", type=Path)
    parser.add_argument("--dispatcher-train", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-reps", type=int, default=50000)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    cuaoa_3070 = peer_pairs(pd.read_csv(args.cuaoa_3070), "CUAOA", "cuaoa_rtx3070_c128")
    cuaoa_3090 = peer_pairs(pd.read_csv(args.cuaoa_3090), "CUAOA", "cuaoa_rtx3090_c128")
    lightning = peer_pairs(
        pd.read_csv(args.lightning_3090), "Lightning", "lightning_rtx3090_c64"
    )
    lookup_sources = [cuaoa_3070, cuaoa_3090]
    if args.cone_lookup:
        lookup_sources.append(
            peer_pairs(pd.read_csv(args.cone_lookup), "CUAOA", "cone_lookup_only")
        )
    cone_lookup = pd.concat(lookup_sources, ignore_index=True)[
        ["family", "n", "p", "seed", "kmax", "total_cone_states"]
    ].drop_duplicates(["family", "n", "p", "seed"])
    lightning = lightning.merge(
        cone_lookup, on=["family", "n", "p", "seed"], how="left", validate="one_to_one"
    )
    if lightning[["kmax", "total_cone_states"]].isna().any().any():
        raise RuntimeError("missing cone metrics for Lightning external cells")
    pairs = pd.concat([cuaoa_3070, cuaoa_3090, lightning], ignore_index=True)
    pairs.to_csv(args.out_dir / "paired_seed_ratios.csv", index=False)

    paired = paired_summary(pairs, args.bootstrap_reps)
    paired.to_csv(args.out_dir / "paired_crossover_summary.csv", index=False)
    seed_breaks, break_summary = break_even_tables(pairs)
    seed_breaks.to_csv(args.out_dir / "break_even_by_seed.csv", index=False)
    break_summary.to_csv(args.out_dir / "break_even_summary.csv", index=False)

    dispatch_rows, dispatch_metrics, dispatch_meta = frozen_dispatcher(
        args.dispatcher_train, pairs
    )
    dispatch_rows.to_csv(args.out_dir / "external_dispatcher_cells.csv", index=False)
    dispatch_metrics.to_csv(args.out_dir / "external_dispatcher_metrics.csv", index=False)
    (args.out_dir / "external_dispatcher_metadata.json").write_text(
        json.dumps(dispatch_meta, indent=2), encoding="utf-8"
    )

    requested = paired[
        (
            (paired.dataset == "lightning_rtx3090_c64")
            & (paired.family == "weighted_qubo_er2")
            & (paired.n == 22)
        )
        | (
            (paired.dataset == "cuaoa_rtx3070_c128")
            & (paired.family == "weighted_qubo_er2")
            & (paired.n == 22)
        )
        | (
            (paired.dataset == "cuaoa_rtx3090_c128")
            & (paired.family == "weighted_qubo_er2")
            & (paired.n == 24)
        )
        | (
            (paired.dataset == "cuaoa_rtx3090_c128")
            & (paired.family == "3regular")
            & (paired.n == 24)
        )
    ]
    report = [
        "# Official-Gradient Crossover Follow-up",
        "",
        "## Requested near-crossover cells",
        "",
        requested.to_markdown(index=False),
        "",
        "## Repeated-query break-even",
        "",
        "The declared calculation is `T(q)=T_preprocess+T_setup+T_cold+(q-1)T_steady`; process-launch overhead is retained separately in the raw benchmark but is not added because it is outside the supplied formula.",
        "",
        break_summary.to_markdown(index=False),
        "",
        "## Frozen dispatcher external sanity check",
        "",
        "The classifier was fitted only on the earlier matched CuPy global-state-adjoint cells using `log(sum_t 2^k_t)` and `log(kmax)`. The 0.5 decision threshold was not adjusted after seeing CUAOA or Lightning timings.",
        "",
        dispatch_metrics.to_markdown(index=False),
        "",
        "These scores test backend ranking, not LC feasibility. Weak external accuracy therefore narrows the dispatch claim: the cone profile remains a feasibility and workload descriptor but is not sufficient by itself to choose the fastest implementation across backend families.",
    ]
    (args.out_dir / "OFFICIAL_CROSSOVER_FOLLOWUP.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    summary = {
        "status": "PASS",
        "paired_seed_rows": len(pairs),
        "paired_cells": len(paired),
        "break_even_cells": len(break_summary),
        "external_dispatcher_cells": len(dispatch_rows),
        "bootstrap_reps": args.bootstrap_reps,
        "all_requested_cells_present": len(requested) == 4,
        "dispatcher_threshold_retuned": False,
    }
    (args.out_dir / "FOLLOWUP_ANALYSIS_VALIDATION.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    if not summary["all_requested_cells_present"]:
        raise RuntimeError("one or more requested near-crossover cells are missing")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
