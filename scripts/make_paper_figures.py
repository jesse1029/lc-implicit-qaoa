from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fnum(x: str) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def save(fig, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURES / f"{name}.png", dpi=220)
    fig.savefig(FIGURES / f"{name}.pdf")


def figure_sparse_scaling(plt, scale: list[dict[str, str]], extended: list[dict[str, str]]) -> None:
    rows = [
        r
        for r in scale
        if r.get("family") == "3regular" and r.get("p") == "2" and r.get("method") == "lc_batched_gpu"
    ]
    rows = sorted(rows, key=lambda r: int(r["n"]))
    ok = [r for r in rows if r["status"] == "ok"]
    if not ok:
        return
    ns = [int(r["n"]) for r in ok]
    times = [fnum(r["seconds"]) for r in ok]
    mem = [fnum(r["peak_pool_mb"]) for r in ok]
    max_canonical_n = max(ns)
    ext_ok = [
        r
        for r in extended
        if r.get("family") == "3regular"
        and r.get("p") == "2"
        and r.get("task") == "objective"
        and r.get("status") == "ok"
        and int(r["n"]) > max_canonical_n
    ]
    ext_ok = sorted(ext_ok, key=lambda r: int(r["n"]))
    ext_ns = [int(r["n"]) for r in ext_ok]
    ext_times = [fnum(r["seconds"]) for r in ext_ok]
    ext_mem = [fnum(r["peak_pool_mb"]) for r in ext_ok]
    fig, ax1 = plt.subplots(figsize=(6.0, 3.2))
    ax1.plot(ns, times, marker="o", color="#1b6ca8", label="LC time")
    if ext_ns:
        ax1.plot(ext_ns, ext_times, marker="^", linestyle="--", color="#1b6ca8", label="LC time, stress")
    ax1.set_xlabel("n qubits")
    ax1.set_ylabel("Objective time (s)", color="#1b6ca8")
    ax1.tick_params(axis="y", labelcolor="#1b6ca8")
    ax1.set_xscale("log", base=2)
    ax1.set_yscale("log")
    ax2 = ax1.twinx()
    ax2.plot(ns, mem, marker="s", color="#c44240", label="LC peak MB")
    if ext_ns:
        ax2.plot(ext_ns, ext_mem, marker="v", linestyle="--", color="#c44240", label="LC peak MB, stress")
    ax2.set_ylabel("Peak pool memory (MB)", color="#c44240")
    ax2.tick_params(axis="y", labelcolor="#c44240")
    ax2.set_yscale("log")
    ax1.axvline(24, color="#444444", linestyle="--", linewidth=1.0, label="_nolegend_")
    ax1.text(25, min(times) * 1.3, "full-state cap n=24", fontsize=8, color="#444444")
    ax1.set_title("3-regular MaxCut p=2 exact LC scaling")
    lines = ax1.get_lines() + ax2.get_lines()
    lines = [line for line in lines if not line.get_label().startswith("_")]
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, frameon=False, fontsize=7, loc="upper left")
    save(fig, "fig1_sparse_scaling_3regular_p2")
    plt.close(fig)


def figure_regime_map(plt, scale: list[dict[str, str]], extended: list[dict[str, str]]) -> None:
    agg: dict[tuple[str, int], int] = {}
    for row in scale:
        if row.get("method") != "lc_batched_gpu" or row.get("status") != "ok":
            continue
        key = (row["family"], int(row["p"]))
        agg[key] = max(agg.get(key, 0), int(row["n"]))
    for row in extended:
        if row.get("task") != "objective" or row.get("status") != "ok":
            continue
        key = (row["family"], int(row["p"]))
        agg[key] = max(agg.get(key, 0), int(row["n"]))
    families = sorted({k[0] for k in agg})
    ps = [1, 2, 3]
    if not families:
        return
    width = 0.25
    x = list(range(len(families)))
    colors = {1: "#2a9d8f", 2: "#e9c46a", 3: "#e76f51"}
    fig, ax = plt.subplots(figsize=(8.2, 3.8))
    for idx, p in enumerate(ps):
        vals = [agg.get((fam, p), 0) for fam in families]
        ax.bar([xx + (idx - 1) * width for xx in x], vals, width=width, label=f"p={p}", color=colors[p])
    ax.set_xticks(x)
    ax.set_xticklabels(families, rotation=35, ha="right")
    ax.set_yscale("log")
    ax.set_ylim(1, max(agg.values()) * 1.6)
    ax.set_ylabel("Max n with exact LC row (log)")
    ax.set_title("Exact LC applicability regime")
    ax.legend(frameon=False, ncols=3)
    save(fig, "fig2_lc_regime_map")
    plt.close(fig)


def figure_peer_medians(plt, external: list[dict[str, str]]) -> None:
    by_method: dict[str, list[float]] = defaultdict(list)
    for row in external:
        if row.get("status") in {"ok", "ok_proxy"}:
            by_method[row["method"]].append(fnum(row["seconds"]))
    methods = [
        "full_precompute_gpu",
        "lc_batched_gpu",
        "cuaoa_gpu_external",
        "qtensor_cpu_external",
        "qokit_cpu_external",
        "cudaq_observe_external",
        "juliqaoa_cpu_external",
        "mps_juliqaoa_external",
    ]
    vals = []
    labels = []
    for method in methods:
        finite = sorted(v for v in by_method.get(method, []) if math.isfinite(v))
        if finite:
            vals.append(finite[len(finite) // 2])
            labels.append(method.replace("_external", "").replace("_gpu", "").replace("_cpu", ""))
    if not vals:
        return
    fig, ax = plt.subplots(figsize=(7.8, 3.4))
    ax.bar(range(len(vals)), vals, color="#4f6d7a")
    ax.set_yscale("log")
    ax.set_ylabel("Median time (s, log)")
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_title("Small-n official peer comparison")
    save(fig, "fig3_peer_median_times")
    plt.close(fig)


def figure_qubo(plt, qubo: list[dict[str, str]]) -> None:
    rows = [
        r
        for r in qubo
        if r.get("method") == "lc_batched_gpu" and r.get("status") == "ok" and r.get("p") == "2"
    ]
    if not rows:
        return
    agg: dict[str, tuple[int, float]] = {}
    for row in rows:
        family = row["family"]
        n = int(row["n"])
        if family not in agg or n > agg[family][0]:
            agg[family] = (n, fnum(row["seconds"]))
    families = sorted(agg)
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    ax.bar(families, [agg[f][0] for f in families], color="#6a994e")
    ax.set_xticks(range(len(families)))
    ax.set_xticklabels(families, rotation=25, ha="right")
    ax.set_ylabel("Max n with exact weighted-QUBO LC row")
    ax.set_title("Weighted QUBO with linear fields")
    save(fig, "fig4_weighted_qubo_reach")
    plt.close(fig)


def figure_training(plt, opt: list[dict[str, str]]) -> None:
    rows = [r for r in opt if r.get("mode") == "trajectory" and r.get("status") == "ok"]
    if not rows:
        return
    labels = [r["case"].replace("maxcut_", "").replace("qubo_", "") + "\n" + r["method"].replace("_gpu", "") for r in rows]
    vals = [fnum(r["seconds_per_eval"]) for r in rows]
    fig, ax = plt.subplots(figsize=(8.4, 3.6))
    colors = ["#1b6ca8" if r["method"] == "lc_batched_gpu" else "#777777" for r in rows]
    ax.bar(range(len(rows)), vals, color=colors)
    ax.set_yscale("log")
    ax.set_ylabel("Seconds per objective eval (log)")
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=7)
    ax.set_title("Training-loop repeated objective cost")
    save(fig, "fig5_training_loop_cost")
    plt.close(fig)


def figure_adjoint_gradient(plt, adjoint: list[dict[str, str]]) -> None:
    rows = [
        r
        for r in adjoint
        if r.get("method") in {"lc_batched_gpu_fd", "lc_batched_gpu_adjoint"} and r.get("status") == "ok"
    ]
    if not rows:
        return
    wanted = ["3regular_n24_p2", "3regular_n128_p2", "3regular_n512_p2", "qubo_er_deg2_n24_p2"]
    filtered = [r for r in rows if r.get("case") in wanted]
    if not filtered:
        filtered = rows[:8]
    labels = [r["case"].replace("_", " ") + "\n" + ("adjoint" if r["method"].endswith("adjoint") else "FD") for r in filtered]
    vals = [fnum(r["seconds"]) for r in filtered]
    colors = ["#247ba0" if r["method"].endswith("adjoint") else "#888888" for r in filtered]
    fig, ax = plt.subplots(figsize=(8.0, 3.6))
    ax.bar(range(len(filtered)), vals, color=colors)
    ax.set_yscale("log")
    ax.set_ylabel("Gradient time (s, log)")
    ax.set_xticks(range(len(filtered)))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=7)
    ax.set_title("LC adjoint gradient vs finite difference")
    save(fig, "fig6_adjoint_gradient_cost")
    plt.close(fig)


def figure_real_qubo(plt, real_qubo: list[dict[str, str]]) -> None:
    rows = [r for r in real_qubo if r.get("method") == "lc_batched_gpu_adjoint" and r.get("status") == "ok"]
    if not rows:
        return
    labels = [f"{r['dataset']}\n{r['variant']} p={r['p']}" for r in rows]
    obj = [fnum(r["objective_seconds"]) for r in rows]
    grad = [fnum(r["gradient_seconds"]) for r in rows]
    x = list(range(len(rows)))
    fig, ax = plt.subplots(figsize=(8.2, 3.5))
    ax.bar([i - 0.18 for i in x], obj, width=0.36, label="objective", color="#2a9d8f")
    ax.bar([i + 0.18 for i in x], grad, width=0.36, label="adjoint gradient", color="#e76f51")
    ax.set_yscale("log")
    ax.set_ylabel("Seconds (log)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
    ax.set_title("Real-data sparse feature-selection QUBO")
    ax.legend(frameon=False)
    save(fig, "fig7_real_qubo_case_study")
    plt.close(fig)


def figure_biomedical(plt, selectors: list[dict[str, str]]) -> None:
    rows = [
        r
        for r in selectors
        if r.get("status") == "ok"
        and r.get("dataset") == "breast_cancer_wisconsin"
        and r.get("variant", "").startswith("top20")
    ]
    if not rows:
        rows = [r for r in selectors if r.get("status") == "ok"][:10]
    if not rows:
        return
    rows = sorted(rows, key=lambda r: fnum(r.get("cv_mean", "nan")), reverse=True)
    labels = [r["selector"].replace("_", "\n") for r in rows]
    cv = [fnum(r["cv_mean"]) for r in rows]
    qscore = [fnum(r["qubo_score"]) for r in rows]
    fig, ax1 = plt.subplots(figsize=(8.4, 3.6))
    ax1.bar(range(len(rows)), cv, color="#3a86ff", label="CV metric")
    ax1.set_ylabel(rows[0].get("cv_metric", "CV metric"))
    ax1.set_ylim(max(0.0, min(cv) - 0.05), min(1.02, max(cv) + 0.03))
    ax1.set_xticks(range(len(rows)))
    ax1.set_xticklabels(labels, rotation=35, ha="right", fontsize=7)
    ax2 = ax1.twinx()
    ax2.plot(range(len(rows)), qscore, color="#fb5607", marker="o", linewidth=1.5, label="QUBO score")
    ax2.set_ylabel("Sparse QUBO score")
    ax1.set_title("Biomedical feature-selection QUBO baselines")
    save(fig, "fig8_biomedical_feature_selection")
    plt.close(fig)


def main() -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise SystemExit(f"matplotlib is required for paper figures: {exc}")

    scale = read_csv(RESULTS / "sota_sparse_scale.csv")
    external = read_csv(RESULTS / "external_peer_benchmark_with_proxies.csv")
    qubo = read_csv(RESULTS / "qubo_benchmark.csv")
    opt = read_csv(RESULTS / "optimization_benchmark.csv")
    adjoint = read_csv(RESULTS / "adjoint_gradient_benchmark.csv")
    real_qubo = read_csv(RESULTS / "real_qubo_case_study.csv")
    biomedical_selectors = read_csv(RESULTS / "biomedical_selectors.csv")
    extended = read_csv(RESULTS / "extended_reach.csv")
    figure_sparse_scaling(plt, scale, extended)
    figure_regime_map(plt, scale, extended)
    figure_peer_medians(plt, external)
    figure_qubo(plt, qubo)
    figure_training(plt, opt)
    figure_adjoint_gradient(plt, adjoint)
    figure_real_qubo(plt, real_qubo)
    figure_biomedical(plt, biomedical_selectors)
    generated = sorted(p.name for p in FIGURES.glob("fig*.png"))
    (RESULTS / "figure_manifest.txt").write_text("\n".join(generated) + "\n", encoding="utf-8")
    print(f"WROTE {RESULTS / 'figure_manifest.txt'}")
    for name in generated:
        print(name)


if __name__ == "__main__":
    main()
