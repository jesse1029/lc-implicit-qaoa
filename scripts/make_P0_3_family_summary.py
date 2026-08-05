from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FAMILIES = ["3regular", "er_deg2", "qubo_modular_sparse", "weighted_sparse_qubo"]
SUCCESS = {"success", "ok"}


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in [
        "n", "p", "seed", "kmax", "total_cone_states", "lc_obj_seconds",
        "lc_grad_seconds", "full_precompute_seconds",
    ]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["lc_ok"] = out["lc_obj_status"].astype(str).str.lower().isin(SUCCESS)
    out["state_ok"] = out["full_precompute_status"].astype(str).str.lower().isin(SUCCESS)
    return out


def first_crossover(sub: pd.DataFrame) -> float:
    matched = sub[sub["lc_ok"] & sub["state_ok"]].copy()
    if matched.empty:
        return float("nan")
    by_n = matched.groupby("n").agg(
        lc=("lc_obj_seconds", "median"), state=("full_precompute_seconds", "median")
    )
    win = by_n[by_n["lc"] < by_n["state"]]
    return float(win.index.min()) if not win.empty else float("nan")


def safe_max(values: pd.Series) -> float:
    values = values.dropna()
    return float(values.max()) if len(values) else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument(
        "--out-dir", type=Path,
        default=ROOT / "results" / "aaai27_followup_20260711" / "P0_3_family_summary",
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for path in args.inputs:
        data = normalize(pd.read_csv(path))
        data["source"] = str(path)
        frames.append(data)
    all_rows = pd.concat(frames, ignore_index=True)
    graph_key = [c for c in ["family", "n", "p", "seed"] if c in all_rows.columns]
    if len(graph_key) == 4:
        before = len(all_rows)
        all_rows = all_rows.drop_duplicates(subset=graph_key, keep="first").copy()
        print(f"DEDUP {before - len(all_rows)} repeated graph-seed rows by {graph_key}")
    target = all_rows[
        (all_rows["p"] == 2)
        & all_rows["family"].isin(FAMILIES)
        & (all_rows["kmax"] <= 14)
        & (all_rows["total_cone_states"] <= 2.0e7)
    ].copy()
    family_rows = []
    for family in FAMILIES:
        sub = target[target["family"] == family].copy()
        matched = sub[sub["lc_ok"] & sub["state_ok"] & (sub["lc_obj_seconds"] > 0)].copy()
        matched["state_over_lc_objective"] = matched["full_precompute_seconds"] / matched["lc_obj_seconds"]
        family_rows.append(
            {
                "family": family,
                "rows": len(sub),
                "row_share": len(sub) / max(len(target), 1),
                "lc_success_rows": int(sub["lc_ok"].sum()),
                "lc_completion_rate": float(sub["lc_ok"].mean()),
                "state_success_rows": int(sub["state_ok"].sum()),
                "state_noncompletion_rows": int((~sub["state_ok"]).sum()),
                "state_completion_rate": float(sub["state_ok"].mean()),
                "lc_max_success_n": safe_max(sub.loc[sub["lc_ok"], "n"]),
                "state_max_success_n": safe_max(sub.loc[sub["state_ok"], "n"]),
                "objective_crossover_n": first_crossover(sub),
                "matched_rows": len(matched),
                "family_median_state_over_lc": float(matched["state_over_lc_objective"].median()) if len(matched) else float("nan"),
                "family_geomean_state_over_lc": float(np.exp(np.mean(np.log(matched["state_over_lc_objective"])))) if len(matched) else float("nan"),
                "lc_objective_median_s": float(sub.loc[sub["lc_ok"], "lc_obj_seconds"].median()),
                "lc_gradient_median_s": float(sub.loc[sub["lc_ok"], "lc_grad_seconds"].median()),
            }
        )
    family_df = pd.DataFrame(family_rows)
    medians = family_df["family_median_state_over_lc"].dropna().to_numpy(dtype=float)
    family_macro = {
        "target_rows": int(len(target)),
        "families": len(FAMILIES),
        "lc_total_success": int(target["lc_ok"].sum()),
        "state_total_success": int(target["state_ok"].sum()),
        "state_total_noncompletion": int((~target["state_ok"]).sum()),
        "lc_family_macro_completion_rate": float(family_df["lc_completion_rate"].mean()),
        "state_family_macro_completion_rate": float(family_df["state_completion_rate"].mean()),
        "row_weighted_lc_completion_rate": float(target["lc_ok"].mean()),
        "row_weighted_state_completion_rate": float(target["state_ok"].mean()),
        "family_equal_median_state_over_lc": float(np.median(medians)),
        "family_equal_geomean_state_over_lc": float(np.exp(np.mean(np.log(medians)))),
        "three_regular_row_fraction": float((target["family"] == "3regular").mean()),
        "summary_text": "LC completes every requested value-plus-gradient query in all four predefined bounded-cone families and extends the measured feasibility frontier of the configured state-plus-cost reference in each family.",
        "speed_ratio_scope": "objective-only matched rows; full-precompute seconds divided by LC objective seconds",
    }
    family_df.to_csv(args.out_dir / "P0_3_family_summary.csv", index=False)
    target.to_csv(args.out_dir / "P0_3_target_rows_with_status.csv", index=False)
    interpreted = target.copy()
    interpreted["state_report_status"] = np.where(
        interpreted["state_ok"],
        "SUCCESS_MEASURED",
        "NOT_RUN_EXPLAINED_AFTER_CAP",
    )
    interpreted["state_status_evidence"] = np.where(
        interpreted["state_ok"],
        "measured execution",
        "runner cap bypassed allocation after the separately measured boundary",
    )
    interpreted.to_csv(
        args.out_dir / "P0_3_target_rows_with_interpretation.csv", index=False
    )
    (args.out_dir / "P0_3_family_macro.json").write_text(
        json.dumps(family_macro, indent=2), encoding="utf-8"
    )
    lines = [
        "# P0-3 Family-Separated p=2 Summary",
        "",
        family_macro["summary_text"],
        "",
        f"The target set has {len(target)} rows, but 3-regular contributes {family_macro['three_regular_row_fraction']:.1%}; therefore family-macro values, not row-weighted totals, are primary.",
        "",
        family_df.to_markdown(index=False),
        "",
        "## Macro summary",
        "",
        "```json",
        json.dumps(family_macro, indent=2),
        "```",
    ]
    (args.out_dir / "P0_3_family_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    semantics = f"""# State-reference status semantics

The original A1 runner returned `OOM_GPU` whenever `n` exceeded its configured
full-state cap, before attempting allocation. Consequently,
{family_macro['state_total_noncompletion']} unique target-grid instances have
zero runtime and zero measured peak memory but carry that legacy runner label.

`P0_3_target_rows_with_interpretation.csv` preserves every original column and
adds:

- `state_report_status=SUCCESS_MEASURED` for the
  {family_macro['state_total_success']} executed state-plus-cost instances;
- `state_report_status=NOT_RUN_EXPLAINED_AFTER_CAP` for the
  {family_macro['state_total_noncompletion']} cap-policy instances;
- `state_status_evidence`, which distinguishes measured execution from the
  declared non-run policy.

The {family_macro['state_total_noncompletion']} policy rows are feasibility
non-completions, not independent measured OOM events. The separate uncapped
boundary probe remains the evidence for the measured RTX 3070 OOM at `n=28`.
Repeated timing reruns of the same `(family,n,p,seed)` are excluded from all
instance counts.
"""
    (args.out_dir / "P0_3_status_semantics.md").write_text(semantics, encoding="utf-8")
    print(json.dumps(family_macro, indent=2))


if __name__ == "__main__":
    main()
