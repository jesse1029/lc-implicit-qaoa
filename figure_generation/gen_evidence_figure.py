"""Generate the four-panel submission figure from logged experiment CSVs."""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("LCQAOA_DATA_ROOT", PROJECT))
VERIFIED = DATA_ROOT / "results" / "verified_20260710"
CROSSOVER = (
    DATA_ROOT
    / "results"
    / "followup_20260712"
    / "official_crossover_analysis"
    / "paired_crossover_summary.csv"
)
CHECKPOINT = (
    DATA_ROOT
    / "results"
    / "E1_true_budgeted_checkpoint_20260717"
    / "E1_policy_summary.csv"
)
OUT = PROJECT / "figures"

GREEN = "#006955"
BLUE = "#245C8A"
ORANGE = "#B45214"
RED = "#A02D19"
PURPLE = "#6F4C9B"
GRAY = "#5F6669"


def configure() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "STIXGeneral"],
            "mathtext.fontset": "stix",
            "font.size": 9.0,
            "axes.titlesize": 9.5,
            "axes.labelsize": 9.0,
            "xtick.labelsize": 9.0,
            "ytick.labelsize": 9.0,
            "legend.fontsize": 9.0,
            "axes.linewidth": 0.65,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
        }
    )


def _numeric(frame: pd.DataFrame, columns: list[str]) -> None:
    for column in columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")


def make_figure() -> None:
    corr = pd.read_csv(VERIFIED / "P0_2_float64_gradient_correctness.csv")
    micro = pd.read_csv(VERIFIED / "P0_1_microbatch_memory_all.csv")
    cross = pd.read_csv(CROSSOVER)
    checkpoint = pd.read_csv(CHECKPOINT)
    _numeric(
        cross,
        [
            "n",
            "median_peer_over_lc",
            "bootstrap_ci_low",
            "bootstrap_ci_high",
        ],
    )
    _numeric(
        checkpoint,
        [
            "successful_repetitions",
            "seconds_median",
            "allocated_bytes_median",
        ],
    )

    fig, axes_grid = plt.subplots(2, 2, figsize=(7.02, 3.72))
    axes = axes_grid.ravel()
    fig.subplots_adjust(left=0.085, right=0.992, bottom=0.13, top=0.94, wspace=0.30, hspace=0.62)

    # A. Algebraic evaluator checked against an independent dense implementation.
    ax = axes[0]
    g = corr[(corr.status == "ok") & (corr.reference == "global_adjoint_float64")]
    families = ["3regular", "er2", "modular_sparse", "weighted_qubo_fields"]
    labels = ["3-reg.", "ER2", "modular", "weighted"]
    values = [g[g.family == family].relative_l2_error.astype(float).max() for family in families]
    ax.bar(np.arange(4), values, color=[GREEN, BLUE, ORANGE, PURPLE], width=0.66)
    ax.axhline(1e-8, color=RED, ls="--", lw=0.75, label=r"$10^{-8}$ criterion")
    ax.set_yscale("log")
    ax.set_ylim(1e-15, 3e-8)
    ax.set_xticks(np.arange(4))
    ax.set_xticklabels(labels, rotation=24, ha="right")
    ax.set_ylabel("worst relative error")
    ax.set_title("A. Float64 Gradient Agreement")
    ax.annotate(r"$10^{-8}$ criterion", xy=(0.04, 1.25e-8), fontsize=9.0, color=RED)

    # B. Same-device paired confidence intervals; ties use open markers.
    ax = axes[1]
    routes = [
        ("cuaoa_rtx3070_c128", "CUAOA 3070", BLUE, "o"),
        ("cuaoa_rtx3090_c128", "CUAOA 3090", ORANGE, "s"),
        ("lightning_rtx3090_c64", "Lightning 3090", GREEN, "^"),
    ]
    for dataset, label, color, marker in routes:
        q = cross[(cross.dataset == dataset) & (cross.family == "3regular")].sort_values("n")
        x = q.n.to_numpy(float)
        y = q.median_peer_over_lc.to_numpy(float)
        low = q.bootstrap_ci_low.to_numpy(float)
        high = q.bootstrap_ci_high.to_numpy(float)
        ax.plot(x, y, color=color, lw=0.9, label=label)
        for xi, yi, lo, hi, decision in zip(x, y, low, high, q.decision):
            tied = decision == "tie_or_unstable"
            ax.errorbar(
                xi,
                yi,
                yerr=np.array([[yi - lo], [hi - yi]]),
                fmt=marker,
                ms=3.4,
                mfc="white" if tied else color,
                mec=color,
                mew=0.7,
                ecolor=color,
                elinewidth=0.55,
                capsize=1.5,
            )
    ax.axhline(1.0, color=RED, ls="--", lw=0.75)
    ax.set_yscale("log")
    ax.set_xticks([18, 20, 22, 24, 26])
    ax.set_xlabel("qubits $n$ ($p=2$)")
    ax.set_ylabel("peer / LC time")
    ax.set_title("B. Matched Global-to-LC Crossover")
    ax.legend(frameon=False, loc="upper left", handlelength=1.5, ncol=1)

    # C. Active allocation follows B*2^k while runtime quickly saturates.
    ax = axes[2]
    q = micro[(micro.status == "ok") & (pd.to_numeric(micro.n) == 16384)].copy()
    _numeric(q, ["cap_terms_at_kmax", "peak_allocated_mb", "seconds"])
    for mode, color, marker in [("objective", BLUE, "o"), ("adjoint", GREEN, "s")]:
        points = []
        for cap in sorted(q.cap_terms_at_kmax.dropna().unique().astype(int)):
            z = q[(q["mode"] == mode) & (q.cap_terms_at_kmax == cap)]
            points.append((cap, float(z.peak_allocated_mb.median()), float(z.seconds.median())))
        ax.plot(
            [point[1] for point in points],
            [point[2] for point in points],
            "-" + marker,
            color=color,
            ms=3,
            lw=0.9,
            label=mode,
        )
        for cap, memory, seconds in points:
            if cap in (16, 64, 1024):
                ax.annotate(f"B={cap}", (memory, seconds), xytext=(4, 3), textcoords="offset points", fontsize=9.0)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("peak allocated MiB")
    ax.set_ylabel("wall time (s)")
    ax.set_title(r"C. Microbatch Pareto ($n=16{,}384$)")
    ax.legend(frameon=False, loc="upper right", handlelength=1.5)

    # D. Normalize every policy to cache-all on the same graph instance.
    ax = axes[3]
    ok = checkpoint[checkpoint.successful_repetitions > 0].copy()
    keys = ["case", "graph_seed_id"]
    baseline = ok[ok.config == "cache_all"][
        keys + ["seconds_median", "allocated_bytes_median"]
    ].rename(
        columns={
            "seconds_median": "cache_seconds",
            "allocated_bytes_median": "cache_bytes",
        }
    )
    ok = ok.merge(baseline, on=keys, how="inner")
    ok["time_ratio"] = ok.seconds_median / ok.cache_seconds
    ok["memory_ratio"] = ok.allocated_bytes_median / ok.cache_bytes
    styles = {
        "cache_all": ("cache", GREEN, "o"),
        "fixed_interval": ("fixed", BLUE, "s"),
        "recompute_all": ("recomp.", GRAY, "D"),
        "budgeted_0.25": ("25%", ORANGE, "v"),
        "budgeted_0.50": ("50%", ORANGE, "<"),
        "budgeted_1.00": ("100%", ORANGE, ">"),
    }
    annotation_offsets = {
        "cache_all": (4, -10),
        "fixed_interval": (-34, -5),
        "recompute_all": (4, 8),
        "budgeted_0.25": (4, 5),
        "budgeted_0.50": (-18, -11),
        "budgeted_1.00": (5, -24),
    }
    budget_points = []
    for config, (label, color, marker) in styles.items():
        z = ok[ok.config == config]
        if z.empty:
            continue
        x = float(z.memory_ratio.median())
        y = float(z.time_ratio.median())
        ax.scatter(x, y, s=17, marker=marker, facecolor=color, edgecolor="white", linewidth=0.45, zorder=3)
        ax.annotate(
            label,
            (x, y),
            xytext=annotation_offsets[config],
            textcoords="offset points",
            fontsize=9.0,
            color=color,
        )
        if config.startswith("budgeted_"):
            budget_points.append((x, y))
    if len(budget_points) > 1:
        budget_points.sort()
        ax.plot([p[0] for p in budget_points], [p[1] for p in budget_points], color=ORANGE, lw=0.65, ls=":")
    ax.axvline(1.0, color="#9AA0A3", ls="--", lw=0.55)
    ax.axhline(1.0, color="#9AA0A3", ls="--", lw=0.55)
    ax.set_xscale("log", base=2)
    ax.set_xlim(0.04, 1.55)
    ax.set_ylim(0.82, max(1.5, float(ok.time_ratio.max()) * 1.07))
    ax.set_xlabel("allocated / cache-all")
    ax.set_ylabel("time / cache-all")
    ax.text(
        0.03,
        0.96,
        "215 feasible\n55 rejected pre-allocation",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        color=RED,
    )
    ax.set_title("D. Byte-Budget Planning")

    for axis in axes:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.tick_params(length=2)

    OUT.mkdir(parents=True, exist_ok=True)
    for suffix, kwargs in [("pdf", {}), ("svg", {}), ("png", {"dpi": 320})]:
        fig.savefig(OUT / f"fig2_evidence_summary_v6.{suffix}", **kwargs)
    plt.close(fig)


if __name__ == "__main__":
    configure()
    make_figure()
    print("wrote fig2_evidence_summary_v6")
