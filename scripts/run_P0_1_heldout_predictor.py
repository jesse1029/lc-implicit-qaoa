from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import balanced_accuracy_score, precision_score, r2_score, recall_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
FEATURES = ["log_total_cone_states", "log_term_count", "log_kmax"]
SUCCESS = {"success", "ok"}


def parse_dataset(spec: str) -> tuple[str, Path]:
    label, sep, raw = spec.partition("=")
    if not sep:
        raise ValueError(f"dataset must be LABEL=PATH: {spec}")
    return label, Path(raw)


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    numeric = [
        "n", "p", "seed", "m", "fields", "kmax", "total_cone_states",
        "lc_obj_seconds", "lc_grad_seconds", "full_precompute_seconds",
        "full_implicit_seconds",
    ]
    for col in numeric:
        if col in out:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    out["term_count"] = out.get("m", 0).fillna(0) + out.get("fields", 0).fillna(0)
    out["log_total_cone_states"] = np.log(out["total_cone_states"].astype(float).clip(lower=1))
    out["log_term_count"] = np.log(out["term_count"].astype(float).clip(lower=1))
    out["log_kmax"] = np.log(out["kmax"].astype(float).clip(lower=1))
    out["row_id"] = np.arange(len(out), dtype=int)
    return out[out["p"] >= 2].reset_index(drop=True)


def blocked_n_split(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    train = np.zeros(len(df), dtype=bool)
    test = np.zeros(len(df), dtype=bool)
    cutoffs: dict[str, int] = {}
    for family, sub in df.groupby("family"):
        levels = sorted(int(x) for x in sub["n"].dropna().unique())
        if len(levels) < 2:
            continue
        n_train = max(1, int(math.ceil(2 * len(levels) / 3)))
        n_train = min(n_train, len(levels) - 1)
        cutoff = levels[n_train - 1]
        cutoffs[str(family)] = cutoff
        idx = sub.index.to_numpy()
        train[idx] = sub["n"].to_numpy() <= cutoff
        test[idx] = sub["n"].to_numpy() > cutoff
    return np.flatnonzero(train), np.flatnonzero(test), cutoffs


def runtime_splits(df: pd.DataFrame):
    families = sorted(df["family"].astype(str).unique())
    for family in families:
        te = np.flatnonzero(df["family"].astype(str).to_numpy() == family)
        tr = np.flatnonzero(df["family"].astype(str).to_numpy() != family)
        if len(tr) and len(te):
            yield "leave_one_family_out", family, tr, te
    tr, te, cutoffs = blocked_n_split(df)
    if len(tr) and len(te):
        yield "blocked_by_n", json.dumps(cutoffs, sort_keys=True), tr, te
    groups = df["seed"].astype(str).to_numpy()
    unique = np.unique(groups)
    if len(unique) >= 2:
        splitter = GroupKFold(n_splits=min(5, len(unique)))
        for fold, (tr, te) in enumerate(splitter.split(df, groups=groups)):
            yield "grouped_by_seed", f"fold_{fold}", tr, te


def bootstrap_regression(y: np.ndarray, pred: np.ndarray, rng, reps: int = 2000):
    def calc(a, b):
        mape = float(np.mean(np.abs(b - a) / np.maximum(np.abs(a), 1e-12)))
        r2 = float(r2_score(a, b)) if len(a) >= 2 and np.ptp(a) > 0 else float("nan")
        rho = float(spearmanr(a, b).statistic) if len(a) >= 2 else float("nan")
        return np.asarray([mape, r2, rho], dtype=float)

    point = calc(y, pred)
    boots = []
    for _ in range(reps):
        idx = rng.integers(0, len(y), size=len(y))
        boots.append(calc(y[idx], pred[idx]))
    arr = np.asarray(boots)
    return point, np.nanpercentile(arr, 2.5, axis=0), np.nanpercentile(arr, 97.5, axis=0)


def bootstrap_classification(y: np.ndarray, pred: np.ndarray, rng, reps: int = 2000):
    def calc(a, b):
        return np.asarray(
            [
                precision_score(a, b, zero_division=0),
                recall_score(a, b, zero_division=0),
                balanced_accuracy_score(a, b),
            ],
            dtype=float,
        )

    point = calc(y, pred)
    boots = []
    for _ in range(reps):
        idx = rng.integers(0, len(y), size=len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        boots.append(calc(y[idx], pred[idx]))
    arr = np.asarray(boots)
    if len(arr) == 0:
        nan = np.full(3, np.nan)
        return point, nan, nan
    return point, np.nanpercentile(arr, 2.5, axis=0), np.nanpercentile(arr, 97.5, axis=0)


def run_runtime(df: pd.DataFrame, dataset: str, target: str, rng):
    status_col = "lc_obj_status" if target == "objective" else "lc_grad_status"
    time_col = "lc_obj_seconds" if target == "objective" else "lc_grad_seconds"
    data = df[
        df[status_col].astype(str).str.lower().isin(SUCCESS)
        & np.isfinite(df[time_col])
        & (df[time_col] > 0)
    ].copy().reset_index(drop=True)
    pred_rows = []
    for protocol, fold, tr, te in runtime_splits(data):
        model = LinearRegression().fit(data.loc[tr, FEATURES], np.log(data.loc[tr, time_col]))
        pred = np.exp(model.predict(data.loc[te, FEATURES]))
        for pos, value in zip(te, pred):
            row = data.iloc[pos]
            pred_rows.append(
                {
                    "dataset": dataset,
                    "target": target,
                    "protocol": protocol,
                    "fold": fold,
                    "family": row.family,
                    "n": int(row.n),
                    "p": int(row.p),
                    "seed": int(row.seed),
                    "actual_seconds": float(row[time_col]),
                    "predicted_seconds": float(value),
                    "log_residual": float(np.log(row[time_col]) - np.log(value)),
                }
            )
    predictions = pd.DataFrame(pred_rows)
    metrics = []
    for protocol, sub in predictions.groupby("protocol"):
        point, low, high = bootstrap_regression(
            sub["actual_seconds"].to_numpy(), sub["predicted_seconds"].to_numpy(), rng
        )
        metrics.append(
            {
                "dataset": dataset,
                "target": target,
                "protocol": protocol,
                "rows": len(sub),
                "mape": point[0], "mape_ci_low": low[0], "mape_ci_high": high[0],
                "r2": point[1], "r2_ci_low": low[1], "r2_ci_high": high[1],
                "spearman": point[2], "spearman_ci_low": low[2], "spearman_ci_high": high[2],
            }
        )
    family_residual = (
        predictions[predictions["protocol"] == "leave_one_family_out"]
        .groupby(["dataset", "target", "family"])
        .agg(
            rows=("actual_seconds", "size"),
            mape=("log_residual", lambda x: float(np.mean(np.abs(np.exp(-x) - 1.0)))),
            median_log_residual=("log_residual", "median"),
            mean_log_residual=("log_residual", "mean"),
        )
        .reset_index()
    )
    return predictions, pd.DataFrame(metrics), family_residual


def explicit_failure(status: pd.Series) -> pd.Series:
    text = status.astype(str).str.lower()
    return ~text.isin(SUCCESS) & ~text.str.contains("not_run|not recorded|unsupported|nan")


def add_labels(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["guardrail_objective"] = out["lc_obj_status"].astype(str).str.lower().isin(SUCCESS).astype(int)
    out["guardrail_gradient"] = out["lc_grad_status"].astype(str).str.lower().isin(SUCCESS).astype(int)
    pre_ok = out["full_precompute_status"].astype(str).str.lower().isin(SUCCESS)
    imp_ok = out["full_implicit_status"].astype(str).str.lower().isin(SUCCESS)
    pre_fail = explicit_failure(out["full_precompute_status"])
    imp_fail = explicit_failure(out["full_implicit_status"])
    baseline_observed = pre_ok | imp_ok | pre_fail | imp_fail
    times = pd.DataFrame(
        {
            "pre": out["full_precompute_seconds"].where(pre_ok & (out["full_precompute_seconds"] > 0)),
            "imp": out["full_implicit_seconds"].where(imp_ok & (out["full_implicit_seconds"] > 0)),
        }
    )
    best = times.min(axis=1, skipna=True)
    lc_ok = out["guardrail_objective"].astype(bool)
    out["global_win_valid"] = baseline_observed
    out["beats_global_objective"] = (
        lc_ok & ((best.notna() & (out["lc_obj_seconds"] < best)) | (best.isna() & (pre_fail | imp_fail)))
    ).astype(int)
    return out


def run_classifier(df: pd.DataFrame, dataset: str, label: str, rng):
    data = df.copy()
    if label == "beats_global_objective":
        data = data[data["global_win_valid"]].copy()
    data = data.reset_index(drop=True)
    pred_rows = []
    for protocol, fold, tr, te in runtime_splits(data):
        ytr = data.loc[tr, label].to_numpy(dtype=int)
        if len(np.unique(ytr)) < 2:
            continue
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
        model.fit(data.loc[tr, FEATURES], ytr)
        prob = model.predict_proba(data.loc[te, FEATURES])[:, 1]
        for pos, value in zip(te, prob):
            row = data.iloc[pos]
            pred_rows.append(
                {
                    "dataset": dataset, "label": label, "protocol": protocol, "fold": fold,
                    "family": row.family, "n": int(row.n), "p": int(row.p), "seed": int(row.seed),
                    "actual": int(row[label]), "probability": float(value), "predicted": int(value >= 0.5),
                }
            )
    predictions = pd.DataFrame(pred_rows)
    metrics = []
    for protocol, sub in predictions.groupby("protocol"):
        point, low, high = bootstrap_classification(
            sub["actual"].to_numpy(dtype=int), sub["predicted"].to_numpy(dtype=int), rng
        )
        metrics.append(
            {
                "dataset": dataset, "label": label, "protocol": protocol, "rows": len(sub),
                "precision": point[0], "precision_ci_low": low[0], "precision_ci_high": high[0],
                "recall": point[1], "recall_ci_low": low[1], "recall_ci_high": high[1],
                "balanced_accuracy": point[2],
                "balanced_accuracy_ci_low": low[2], "balanced_accuracy_ci_high": high[2],
            }
        )
    return predictions, pd.DataFrame(metrics)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", action="append", required=True, help="LABEL=CSV")
    parser.add_argument(
        "--out-dir", type=Path,
        default=ROOT / "results" / "aaai27_followup_20260711" / "P0_1_heldout_predictor",
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260711)
    runtime_predictions, runtime_metrics, family_residuals = [], [], []
    class_predictions, class_metrics = [], []
    for spec in args.dataset:
        label, path = parse_dataset(spec)
        data = add_labels(normalize(pd.read_csv(path)))
        for target in ["objective", "gradient"]:
            pred, metrics, residual = run_runtime(data, label, target, rng)
            runtime_predictions.append(pred)
            runtime_metrics.append(metrics)
            family_residuals.append(residual)
        for target_label in ["guardrail_objective", "guardrail_gradient", "beats_global_objective"]:
            pred, metrics = run_classifier(data, label, target_label, rng)
            class_predictions.append(pred)
            class_metrics.append(metrics)

    runtime_predictions_df = pd.concat(runtime_predictions, ignore_index=True)
    runtime_metrics_df = pd.concat(runtime_metrics, ignore_index=True)
    residual_df = pd.concat(family_residuals, ignore_index=True)
    class_predictions_df = pd.concat(class_predictions, ignore_index=True)
    class_metrics_df = pd.concat(class_metrics, ignore_index=True)
    runtime_predictions_df.to_csv(args.out_dir / "P0_1_runtime_predictions.csv", index=False)
    runtime_metrics_df.to_csv(args.out_dir / "P0_1_runtime_metrics.csv", index=False)
    residual_df.to_csv(args.out_dir / "P0_1_family_residuals.csv", index=False)
    class_predictions_df.to_csv(args.out_dir / "P0_1_dispatch_predictions.csv", index=False)
    class_metrics_df.to_csv(args.out_dir / "P0_1_dispatch_metrics.csv", index=False)

    decisive = runtime_metrics_df[runtime_metrics_df["protocol"].isin(["leave_one_family_out", "blocked_by_n"])]
    conclusions = []
    for (dataset, target), sub in decisive.groupby(["dataset", "target"]):
        worst = float(sub["mape"].max())
        conclusions.append(
            {
                "dataset": dataset,
                "target": target,
                "worst_heldout_mape": worst,
                "recommended_term": "predictive model" if worst <= 0.30 else "descriptive cost model / heuristic dispatch model",
            }
        )
    conclusion_df = pd.DataFrame(conclusions)
    conclusion_df.to_csv(args.out_dir / "P0_1_model_scope.csv", index=False)
    lines = [
        "# P0-1 Held-Out Runtime and Dispatch Validation",
        "",
        "Models use only log(total cone states), log(term count), and log(kmax); family identity is not a feature. Runtime models are fitted separately for objective and gradient rows with p>=2. Hardware datasets are never mixed.",
        "",
        "## Claim decision",
        "",
        conclusion_df.to_markdown(index=False),
        "",
        "## Held-out runtime metrics",
        "",
        runtime_metrics_df.to_markdown(index=False),
        "",
        "## Dispatch metrics",
        "",
        class_metrics_df.to_markdown(index=False),
        "",
        "`beats_global_objective` uses only rows where a matched global value baseline completed or recorded an explicit failure. The source matrix has no official global-gradient timing, so no global-gradient win label is fabricated.",
    ]
    (args.out_dir / "P0_1_heldout_validation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(conclusion_df.to_string(index=False))


if __name__ == "__main__":
    main()
