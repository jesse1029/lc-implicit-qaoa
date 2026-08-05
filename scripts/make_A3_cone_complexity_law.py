from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from lcqaoa.lightcone import extract_lightcones
from run_extended_reach import graph_for as graph_for_extended
from run_sota_sparse_scale import graph_for_scale


OUT_DIR = ROOT / "results" / "benchmark_suite_20260704" / "A3_cone_complexity_law"
FIG_DIR = OUT_DIR / "figures"


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def normalize_rows() -> pd.DataFrame:
    frames = []
    sources = [
        (ROOT / "results" / "sota_sparse_scale.csv", "sota_sparse_scale"),
        (ROOT / "results" / "qubo_benchmark.csv", "qubo_benchmark"),
        (ROOT / "results" / "extended_reach.csv", "extended_reach"),
        (ROOT / "results" / "official_regime_matrix.csv", "official_regime_matrix"),
        (ROOT / "results" / "p2_p3_cone_sweep" / "p2_p3_cone_sweep.csv", "p2_p3_cone_sweep"),
        (ROOT / "results" / "p1_qtensor_large_cpu" / "p1_qtensor_large_cpu.csv", "p1_qtensor_large_cpu"),
    ]
    for path, source in sources:
        df = read_csv(path)
        if df.empty:
            continue
        df = df.copy()
        df["source"] = source
        if source == "official_regime_matrix":
            out = pd.DataFrame(
                {
                    "family": df["family"],
                    "n": df["n"],
                    "m": df["m"],
                    "p": df["p"],
                    "avg_degree": df["avg_degree"],
                    "max_degree": df["max_degree"],
                    "kmax": df["kmax"],
                    "total_cone_states": df["total_cone_states"],
                    "status": df["lc_status"],
                    "seconds": df["lc_seconds"],
                    "peak_pool_mb": df["lc_peak_mb"],
                    "method": "lc_batched_gpu",
                    "task": "objective",
                    "source": source,
                }
            )
        elif source == "p1_qtensor_large_cpu":
            out = pd.DataFrame(
                {
                    "family": df["family"],
                    "n": df["n"],
                    "m": df["m"],
                    "p": df["p"],
                    "avg_degree": np.nan,
                    "max_degree": np.nan,
                    "kmax": df["kmax"],
                    "total_cone_states": df["total_cone_states"],
                    "status": df["qtensor_status"],
                    "seconds": df["lc_seconds"],
                    "peak_pool_mb": df["lc_peak_mb"],
                    "method": "lc_batched_gpu",
                    "task": "objective",
                    "source": source,
                }
            )
        else:
            method = df["method"] if "method" in df.columns else "lc_batched_gpu"
            task = df["task"] if "task" in df.columns else "objective"
            out = pd.DataFrame(
                {
                    "family": df["family"],
                    "n": df["n"],
                    "m": df["m"],
                    "p": df["p"],
                    "avg_degree": df["avg_degree"] if "avg_degree" in df.columns else np.nan,
                    "max_degree": df["max_degree"] if "max_degree" in df.columns else np.nan,
                    "kmax": df["kmax"],
                    "total_cone_states": df["total_cone_states"],
                    "status": df["status"],
                    "seconds": df["seconds"],
                    "peak_pool_mb": df["peak_pool_mb"],
                    "method": method,
                    "task": task,
                    "source": source,
                }
            )
        frames.append(out)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    combined = combined[combined["method"].astype(str).str.contains("lc_batched|lc", regex=True, na=False)]
    for col in ["n", "m", "p", "avg_degree", "max_degree", "kmax", "total_cone_states", "seconds", "peak_pool_mb"]:
        combined[col] = pd.to_numeric(combined[col], errors="coerce")
    combined["success"] = combined["status"].astype(str).eq("ok")
    combined["max_state_elements"] = np.power(2.0, combined["kmax"].astype(float))
    return combined


def fit_log_law(df: pd.DataFrame, x_col: str, y_col: str) -> dict[str, float]:
    ok = df[(df["success"]) & (df[x_col] > 0) & (df[y_col] > 0)].copy()
    if len(ok) < 3:
        return {"count": len(ok), "slope": float("nan"), "intercept": float("nan"), "r2": float("nan")}
    x = np.log10(ok[x_col].astype(float).to_numpy())
    y = np.log10(ok[y_col].astype(float).to_numpy())
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"count": len(ok), "slope": float(slope), "intercept": float(intercept), "r2": float(r2)}


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def plot_regime_law(df: pd.DataFrame) -> None:
    set_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    obj = df[(df["task"].astype(str) == "objective") | (df["task"].isna())].copy()
    ok = obj[obj["success"] & (obj["seconds"] > 0) & (obj["total_cone_states"] > 0)]
    fail = obj[~obj["success"] & (obj["total_cone_states"] > 0)]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2))
    ax = axes[0, 0]
    for family, group in ok.groupby("family"):
        ax.scatter(group["total_cone_states"], group["seconds"], s=13, alpha=0.72, label=str(family))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"total cone work $\sum_t 2^{k_t}$")
    ax.set_ylabel("LC objective time (s)")
    ax.set_title("Time law")
    ax.grid(True, which="both", linewidth=0.25, alpha=0.35)
    ax.legend(frameon=False, ncol=2, fontsize=5.8)

    ax = axes[0, 1]
    mem_ok = obj[obj["success"] & (obj["peak_pool_mb"] > 0) & (obj["max_state_elements"] > 0)]
    for family, group in mem_ok.groupby("family"):
        ax.scatter(group["max_state_elements"], group["peak_pool_mb"], s=13, alpha=0.72, label=str(family))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"largest cone state size $2^{k_{\max}}$")
    ax.set_ylabel("peak pool memory (MB)")
    ax.set_title("Memory law")
    ax.grid(True, which="both", linewidth=0.25, alpha=0.35)

    ax = axes[1, 0]
    heat = obj.copy()
    heat["degree_bin"] = pd.cut(heat["avg_degree"], bins=[0, 2.5, 3.5, 5.0, 10.0, 100.0], labels=["<=2.5", "2.5-3.5", "3.5-5", "5-10", ">10"])
    pivot = heat.pivot_table(index="p", columns="degree_bin", values="success", aggfunc="mean", observed=False)
    im = ax.imshow(pivot.fillna(np.nan).to_numpy(), aspect="auto", vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(list(pivot.columns), rotation=35, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([str(x) for x in pivot.index])
    ax.set_xlabel("average-degree bin")
    ax.set_ylabel("QAOA depth p")
    ax.set_title("Success fraction")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[1, 1]
    hist_rows = cone_histogram_rows()
    for label, sizes in hist_rows:
        bins = np.arange(0, max(sizes) + 2) - 0.5
        ax.hist(sizes, bins=bins, histtype="step", linewidth=1.25, label=label)
    ax.set_xlabel(r"cone size $k_t$")
    ax.set_ylabel("local terms")
    ax.set_title("Cone-size histograms")
    ax.legend(frameon=False, fontsize=6)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "A3_cone_complexity_law.pdf")
    fig.savefig(FIG_DIR / "A3_cone_complexity_law.png", dpi=300)
    plt.close(fig)


def cone_histogram_rows() -> list[tuple[str, list[int]]]:
    rows = []
    specs = [
        ("3-regular", "3regular", 128, 2, 7771, "scale"),
        ("ER degree-2", "er_deg2", 128, 2, 7772, "scale"),
        ("ER degree-3", "er_deg3", 128, 2, 7773, "scale"),
        ("scale-free", "scale_free_a1", 128, 2, 7774, "scale"),
        ("modular sparse", "modular_sparse", 128, 2, 7775, "extended"),
    ]
    for label, family, n, p, seed, source in specs:
        graph = graph_for_extended(family, n, seed) if source == "extended" else graph_for_scale(family, n, seed)
        rows.append((label, [c.k for c in extract_lightcones(graph, p)]))
    return rows


def write_summary(df: pd.DataFrame, out_dir: Path) -> None:
    obj = df[(df["task"].astype(str) == "objective") | (df["task"].isna())].copy()
    time_fit = fit_log_law(obj, "total_cone_states", "seconds")
    mem_fit = fit_log_law(obj, "max_state_elements", "peak_pool_mb")
    status_counts = obj.groupby(["family", "p", "success"]).size().reset_index(name="count")
    status_counts.to_csv(out_dir / "A3_success_counts.csv", index=False)
    lines = [
        "# A3 Cone-Complexity Law",
        "",
        "This analysis pools LC rows from the existing benchmark CSVs and tests whether runtime and memory are predictable from cone statistics before simulation.",
        "",
        "## Log-log fits",
        "",
        f"- Time vs total cone work: count={time_fit['count']}, slope={time_fit['slope']:.3g}, R2={time_fit['r2']:.3g}.",
        f"- Memory vs largest cone state size: count={mem_fit['count']}, slope={mem_fit['slope']:.3g}, R2={mem_fit['r2']:.3g}.",
        "",
        "## Interpretation",
        "",
        "- Runtime is governed by total local-state work, not by n alone.",
        "- Memory is governed by the largest active cone/batch, not by global state size.",
        "- Failure rows are retained and should be explained through kmax and total cone work.",
        "",
        "See `figures/A3_cone_complexity_law.pdf` for the four-panel regime-law figure.",
    ]
    (out_dir / "A3_cone_complexity_law.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with (out_dir / "A3_fit_summary.json").open("w", encoding="utf-8") as f:
        import json

        json.dump({"time_fit": time_fit, "memory_fit": mem_fit}, f, indent=2)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = normalize_rows()
    df.to_csv(OUT_DIR / "A3_cone_complexity_rows.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    plot_regime_law(df)
    write_summary(df, OUT_DIR)
    print(f"WROTE {OUT_DIR / 'A3_cone_complexity_rows.csv'}")
    print(f"WROTE {OUT_DIR / 'figures' / 'A3_cone_complexity_law.pdf'}")


if __name__ == "__main__":
    main()
