from __future__ import annotations

import argparse
import math
import shutil
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
DEFAULT_OUT = RESULTS / "figures"


CB = {
    "blue": "#0077BB",
    "cyan": "#33BBEE",
    "teal": "#009988",
    "orange": "#EE7733",
    "red": "#CC3311",
    "magenta": "#EE3377",
    "grey": "#BBBBBB",
    "dark": "#222222",
    "light": "#F7F7F7",
}


def configure() -> None:
    matplotlib.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.5,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def read_csv(name: str) -> pd.DataFrame:
    path = RESULTS / name
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def save(fig: plt.Figure, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.pdf")
    fig.savefig(out_dir / f"{name}.png", dpi=300)


def copy_outputs(out_dir: Path, copy_to: Path | None, names: list[str]) -> None:
    if copy_to is None:
        return
    copy_to.mkdir(parents=True, exist_ok=True)
    for name in names:
        for suffix in (".pdf", ".png"):
            src = out_dir / f"{name}{suffix}"
            if src.exists():
                shutil.copy2(src, copy_to / src.name)


def card(ax, xy, wh, edge, face, lw=1.2):
    rect = Rectangle(xy, wh[0], wh[1], linewidth=lw, edgecolor=edge, facecolor=face)
    ax.add_patch(rect)
    return rect


def make_teaser(out_dir: Path) -> None:
    fig = plt.figure(figsize=(3.35, 2.55))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.5,
        0.965,
        "QAOA training: global state vs. local cones",
        ha="center",
        va="top",
        fontsize=8.8,
        fontweight="bold",
        color=CB["dark"],
    )

    card(ax, (0.04, 0.49), (0.42, 0.36), CB["red"], "#FFF4F1")
    ax.text(0.06, 0.795, "Full-state route", fontsize=7.9, fontweight="bold", color=CB["red"])
    ax.text(
        0.06,
        0.72,
        r"$\langle C\rangle=\sum_z |\psi_\theta(z)|^2 C(z)$",
        fontsize=7.9,
        color=CB["dark"],
    )
    ax.text(0.06, 0.645, r"$\psi_\theta, C[\cdot]\ \propto\ 2^n$", fontsize=7.9, color=CB["dark"])
    ax.text(0.06, 0.58, "state + cost table", fontsize=6.8, color=CB["dark"])
    ax.add_patch(Rectangle((0.06, 0.52), 0.36, 0.044, facecolor="#FFE0D6", edgecolor="none"))
    ax.text(0.24, 0.542, "3070: n=26 OK, n=28 OOM", fontsize=6.25, ha="center", va="center")

    card(ax, (0.54, 0.49), (0.42, 0.36), CB["blue"], "#F0F7FF")
    ax.text(0.56, 0.795, "LC-Implicit-QAOA", fontsize=7.9, fontweight="bold", color=CB["blue"])
    ax.text(
        0.56,
        0.72,
        r"$\langle C\rangle=\sum_t \langle C_t\rangle_{G[L_p(t)]}$",
        fontsize=7.9,
        color=CB["dark"],
    )
    ax.text(0.56, 0.645, r"workspace $\propto b_k2^k,\ k\ll n$", fontsize=7.9, color=CB["dark"])
    ax.text(0.56, 0.58, "batched local states", fontsize=6.8, color=CB["dark"])
    ax.add_patch(Rectangle((0.56, 0.52), 0.36, 0.044, facecolor="#DDF0FF", edgecolor="none"))
    ax.text(0.74, 0.542, "n=16,384, kmax=14", fontsize=6.55, ha="center", va="center")

    ax.annotate(
        "",
        xy=(0.535, 0.67),
        xytext=(0.465, 0.67),
        arrowprops=dict(arrowstyle="->", lw=1.2, color=CB["dark"]),
    )

    card(ax, (0.04, 0.28), (0.92, 0.15), "#555555", "#FAFAFA", lw=0.8)
    ax.text(0.06, 0.392, "Main RTX 3070 evidence", fontsize=7.2, fontweight="bold", color=CB["dark"])
    ax.text(0.06, 0.345, "bounded-cone n=24:", fontsize=6.7, color=CB["dark"])
    ax.text(0.39, 0.345, "25.6x speed", fontsize=7.5, color=CB["teal"], fontweight="bold")
    ax.text(0.61, 0.345, "and", fontsize=6.7, color=CB["dark"])
    ax.text(0.68, 0.345, "2.4e4x memory", fontsize=7.5, color=CB["teal"], fontweight="bold")
    ax.text(0.06, 0.305, "stress objective/gradient:", fontsize=6.7, color=CB["dark"])
    ax.text(0.42, 0.305, "121s / 126s", fontsize=7.2, color=CB["blue"], fontweight="bold")
    ax.text(0.63, 0.305, "at", fontsize=6.7, color=CB["dark"])
    ax.text(0.69, 0.305, "84.8 / 254 MB", fontsize=7.2, color=CB["blue"], fontweight="bold")

    card(ax, (0.04, 0.115), (0.92, 0.11), CB["orange"], "#FFF8EE", lw=0.9)
    ax.text(0.06, 0.188, "Regime condition", fontsize=6.9, fontweight="bold", color=CB["orange"])
    ax.text(
        0.06,
        0.148,
        r"small $k_{\max}$ and controlled $\sum_t 2^{k_t}$",
        fontsize=6.55,
        color=CB["dark"],
    )
    ax.text(
        0.06,
        0.126,
        "hubs, dense modules, or larger p make cones global.",
        fontsize=6.3,
        color=CB["dark"],
    )

    save(fig, out_dir, "fig0_teaser_summary")
    plt.close(fig)


def make_dashboard(out_dir: Path) -> None:
    boundary = read_csv("fullstate_boundary_probe.csv")
    qokit = read_csv("samehost_3090_lc_vs_qokit_gpu.csv")
    official = read_csv("official_regime_matrix.csv")
    grad = read_csv("adjoint_gradient_benchmark.csv")

    fig, axes = plt.subplots(2, 2, figsize=(6.9, 4.45))
    fig.subplots_adjust(wspace=0.35, hspace=0.48)

    # A. 3070 full-state boundary and LC stress row.
    ax = axes[0, 0]
    if not boundary.empty:
        ok = boundary[boundary["status"].astype(str).str.startswith("ok")]
        fail = boundary[~boundary["status"].astype(str).str.startswith("ok")]
        ax.plot(ok["n"], ok["peak_mb"], marker="o", color=CB["red"], label="full-state measured")
        if not fail.empty:
            oom = fail.iloc[0]
            ax.scatter([oom["n"]], [8192], marker="x", s=55, color=CB["red"], label="OOM boundary")
            ax.text(float(oom["n"]) + 0.15, 8192, "OOM", fontsize=7, va="center", color=CB["red"])
    ax.scatter([16384], [84.828125], marker="s", s=38, color=CB["teal"], label="LC stress row")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.axhline(8192, color="#777777", lw=0.8, ls="--")
    ax.text(24, 9000, "8 GB", fontsize=7, color="#555555")
    ax.set_title("A. Boundary on RTX 3070")
    ax.set_xlabel("problem size n")
    ax.set_ylabel("peak GPU memory (MB)")
    ax.legend(frameon=False, loc="lower right", fontsize=6.8)

    # B. Auxiliary same-host 3090 QOKit GPU ratio.
    ax = axes[0, 1]
    if not qokit.empty:
        rows = qokit[qokit["qokit_gpu_status"].astype(str).eq("ok")].copy()
        rows["label"] = rows["family"].replace(
            {
                "3regular": "3-reg.",
                "er_deg2": "ER2",
                "er_deg3": "ER3",
                "modular_sparse": "modular",
            }
        ) + " n=" + rows["n"].astype(str)
        rows = rows.sort_values("qokit_eval_over_lc")
        y = np.arange(len(rows))
        ratios = rows["qokit_eval_over_lc"].astype(float)
        colors = [CB["teal"] if r > 1 else CB["orange"] for r in ratios]
        ax.barh(y, ratios, color=colors, edgecolor="none")
        ax.axvline(1.0, color="#333333", lw=0.9)
        ax.set_xscale("log")
        ax.set_yticks(y)
        ax.set_yticklabels(rows["label"])
        ax.set_xlabel("QOKit GPU time / LC time")
        ax.set_title("B. Official QOKit GPU on RTX 3090")
        ax.text(1.04, len(rows) - 0.35, "LC faster", fontsize=6.8, color=CB["teal"])
        ax.text(0.11, -0.25, "QOKit faster", fontsize=6.8, color=CB["orange"])

    # C. Cone size explains the regime.
    ax = axes[1, 0]
    if not official.empty:
        plot_rows = official[
            official["family"].isin(["3regular", "er_deg2", "er_deg3", "modular_sparse"])
        ].copy()
        plot_rows["label"] = plot_rows["family"].replace(
            {
                "3regular": "3-reg.",
                "er_deg2": "ER2",
                "er_deg3": "ER3",
                "modular_sparse": "modular",
            }
        ) + " n=" + plot_rows["n"].astype(str) + " p=" + plot_rows["p"].astype(str)
        ok = plot_rows[plot_rows["lc_status"].astype(str).eq("ok")]
        skip = plot_rows[~plot_rows["lc_status"].astype(str).eq("ok")]
        ax.scatter(ok["kmax"], ok["lc_peak_mb"].replace(0, np.nan), s=34, color=CB["blue"], label="LC evaluated")
        if not skip.empty:
            ax.scatter(skip["kmax"], [1200] * len(skip), marker="x", s=38, color=CB["red"], label="guardrail skip")
        for _, row in plot_rows.iterrows():
            if row["family"] in ("er_deg3", "modular_sparse") and int(row["n"]) in (24, 128):
                yval = float(row["lc_peak_mb"]) if float(row["lc_peak_mb"]) > 0 else 1200
                ax.text(float(row["kmax"]) + 0.35, yval * 1.08, str(row["label"]), fontsize=5.9)
        ax.axvline(24, color="#777777", ls="--", lw=0.8)
        ax.set_yscale("log")
        ax.set_xlabel("maximum light-cone size kmax")
        ax.set_ylabel("LC peak memory (MB)")
        ax.set_title("C. Regime is governed by cone size")
        ax.legend(frameon=False, fontsize=6.8, loc="upper left")

    # D. Adjoint gradient replaces repeated objective calls.
    ax = axes[1, 1]
    if not grad.empty:
        fd = grad[grad["method"].astype(str).eq("lc_batched_gpu_fd")].copy()
        ad = grad[grad["method"].astype(str).eq("lc_batched_gpu_adjoint")].copy()
        merged = fd.merge(ad, on="case", suffixes=("_fd", "_adj"))
        merged = merged[merged["status_fd"].eq("ok") & merged["status_adj"].eq("ok")]
        merged["speedup"] = merged["seconds_fd"].astype(float) / merged["seconds_adj"].astype(float)
        order = ["3regular_n24_p2", "3regular_n128_p2", "3regular_n512_p2", "modular_sparse_n128_p1"]
        merged["ord"] = merged["case"].apply(lambda c: order.index(c) if c in order else 99)
        merged = merged.sort_values("ord").head(4)
        labels = [
            c.replace("3regular_", "3-reg. ").replace("modular_sparse_", "mod. ").replace("_", " ")
            for c in merged["case"]
        ]
        y = np.arange(len(merged))
        ax.barh(y, merged["speedup"], color=CB["cyan"], edgecolor="none")
        ax.axvline(1, color="#333333", lw=0.9)
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.set_xlabel("LC finite-diff time / adjoint time")
        ax.set_title("D. Gradient pass in the same local regime")
        for yi, (_, row) in enumerate(merged.iterrows()):
            ax.text(float(row["speedup"]) * 1.03, yi, f"{row['speedup']:.1f}x", va="center", fontsize=6.8)
        ax.set_xlim(0, max(merged["speedup"].max() * 1.35, 2.0))

    for ax in axes.flat:
        ax.grid(True, axis="y", color="#E8E8E8", lw=0.5)
        ax.set_axisbelow(True)

    save(fig, out_dir, "fig9_experiment_summary_dashboard")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate teaser and experiment-summary dashboard figures.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--copy-to", type=Path, default=None)
    args = parser.parse_args()

    configure()
    make_teaser(args.out_dir)
    make_dashboard(args.out_dir)
    copy_outputs(
        args.out_dir,
        args.copy_to,
        ["fig0_teaser_summary", "fig9_experiment_summary_dashboard"],
    )


if __name__ == "__main__":
    main()
