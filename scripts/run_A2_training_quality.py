from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from lcqaoa.graphs import random_regular_graph, erdos_renyi_graph, scale_free_graph, weighted_modular_qubo_graph, weighted_qubo_graph
from lcqaoa.lightcone import lightcone_expectation, lightcone_gradient_adjoint
from lcqaoa.qaoa import full_state_expectation
from benchmark_common import cone_metrics, graph_metrics, pack_params, params_for_depth, unpack_params, wrap_angles


@dataclass
class A2SummaryRow:
    case: str
    objective: str
    family: str
    n: int
    p: int
    seed: int
    init_id: int
    optimizer: str
    status: str
    budget: int
    objective_calls: int
    gradient_calls: int
    seconds_total: float
    initial_value: float
    final_value: float
    best_value: float
    normalized_improvement: float
    calls_to_95pct_best: int
    time_to_95pct_best: float
    kmax: int
    total_cone_states: int
    peak_mb: float
    full_state_initial: float
    full_state_final: float
    max_abs_error_vs_full: float
    notes: str


@dataclass
class A2StepRow:
    case: str
    seed: int
    init_id: int
    optimizer: str
    step: int
    objective_calls: int
    gradient_calls: int
    seconds_elapsed: float
    value: float
    best_value: float
    grad_norm: float
    peak_mb: float
    status: str


def make_graph(family: str, n: int, seed: int):
    if family == "3regular":
        return random_regular_graph(n, 3, seed=seed)
    if family == "weighted_sparse_qubo":
        return weighted_qubo_graph(n, min(0.40, 2.0 / max(2, n)), seed=seed, field_scale=0.7)
    if family == "qubo_modular_sparse":
        return weighted_modular_qubo_graph(n, modules=max(4, n // 16), p_in=0.14, p_out=0.0015, seed=seed)
    if family == "er_deg3":
        return erdos_renyi_graph(n, min(0.45, 3.0 / max(2, n)), seed=seed)
    if family == "scale_free":
        return scale_free_graph(n, attachment=2, seed=seed)
    raise ValueError(family)


def case_specs() -> list[tuple[str, list[int], list[int]]]:
    return [
        ("3regular", [24, 64, 128, 256, 512], [1, 2, 3]),
        ("weighted_sparse_qubo", [24, 64, 128, 256], [1, 2]),
        ("qubo_modular_sparse", [64, 128, 256, 512], [1, 2]),
        ("er_deg3", [24, 32], [1, 2]),
        ("scale_free", [24, 32], [1, 2]),
    ]


def budget_for(p: int) -> int:
    return {1: 100, 2: 200, 3: 300}.get(p, 100 * p)


class EvalCounter:
    def __init__(self, graph, p: int, args, mode: str = "lc") -> None:
        self.graph = graph
        self.p = p
        self.args = args
        self.mode = mode
        self.objective_calls = 0
        self.gradient_calls = 0
        self.peak_mb = 0.0

    def value(self, x: np.ndarray) -> tuple[float, str]:
        gammas, betas = unpack_params(x, self.p)
        if self.mode == "full":
            stats = full_state_expectation(self.graph, gammas, betas, method="precompute", prefer_gpu=True, max_qubits=self.args.full_cap)
        else:
            stats = lightcone_expectation(
                self.graph,
                gammas,
                betas,
                p=self.p,
                prefer_gpu=True,
                max_k=self.args.max_k,
                max_batch_states=self.args.max_batch_states,
            )
        self.objective_calls += 1
        self.peak_mb = max(self.peak_mb, stats.peak_pool_bytes / 1024**2)
        return float(stats.value), stats.status

    def value_grad(self, x: np.ndarray) -> tuple[float, np.ndarray, str]:
        gammas, betas = unpack_params(x, self.p)
        stats = lightcone_gradient_adjoint(
            self.graph,
            gammas,
            betas,
            p=self.p,
            prefer_gpu=True,
            max_k=self.args.max_k,
            max_batch_states=self.args.max_batch_states,
        )
        self.objective_calls += 1
        self.gradient_calls += 1
        self.peak_mb = max(self.peak_mb, stats.peak_pool_bytes / 1024**2)
        if stats.gradient is None:
            return float("nan"), np.zeros(2 * self.p), stats.status
        return float(stats.value), np.asarray(stats.gradient, dtype=np.float64), stats.status

    def fd_grad(self, x: np.ndarray, eps: float) -> tuple[float, np.ndarray, str]:
        base, status = self.value(x)
        if status != "ok":
            return base, np.zeros_like(x), status
        grad = np.zeros_like(x)
        for i in range(x.size):
            xp = x.copy()
            xm = x.copy()
            xp[i] += eps
            xm[i] -= eps
            vp, sp = self.value(xp)
            vm, sm = self.value(xm)
            if sp != "ok" or sm != "ok":
                return base, grad, f"fd_failed:{sp}/{sm}"
            grad[i] = (vp - vm) / (2.0 * eps)
        self.gradient_calls += 1
        return base, grad, "ok"


def adam_step(x: np.ndarray, grad: np.ndarray, step: int, state: dict, lr: float, amsgrad: bool = False) -> np.ndarray:
    b1, b2, eps = 0.9, 0.999, 1e-8
    state.setdefault("m", np.zeros_like(x))
    state.setdefault("v", np.zeros_like(x))
    state["m"] = b1 * state["m"] + (1.0 - b1) * grad
    state["v"] = b2 * state["v"] + (1.0 - b2) * (grad * grad)
    mhat = state["m"] / (1.0 - b1**step)
    vhat = state["v"] / (1.0 - b2**step)
    if amsgrad:
        state.setdefault("vhatmax", np.zeros_like(x))
        state["vhatmax"] = np.maximum(state["vhatmax"], vhat)
        vhat = state["vhatmax"]
    return wrap_angles(x + lr * mhat / (np.sqrt(vhat) + eps))


def summarize_steps(common: dict, optimizer: str, status: str, budget: int, counter: EvalCounter, steps: list[A2StepRow], initial: float, final: float, full_initial: float, full_final: float, max_full_err: float, notes: str) -> A2SummaryRow:
    best = max((s.best_value for s in steps), default=float("nan"))
    denom = max(abs(initial), 1.0)
    target = initial + 0.95 * (best - initial) if math.isfinite(best) and math.isfinite(initial) else float("nan")
    calls95 = 0
    time95 = float("nan")
    for s in steps:
        if math.isfinite(target) and s.best_value >= target:
            calls95 = s.objective_calls
            time95 = s.seconds_elapsed
            break
    return A2SummaryRow(
        **common,
        optimizer=optimizer,
        status=status,
        budget=budget,
        objective_calls=counter.objective_calls,
        gradient_calls=counter.gradient_calls,
        seconds_total=steps[-1].seconds_elapsed if steps else 0.0,
        initial_value=initial,
        final_value=final,
        best_value=best,
        normalized_improvement=(best - initial) / denom if math.isfinite(best) and math.isfinite(initial) else float("nan"),
        calls_to_95pct_best=calls95,
        time_to_95pct_best=time95,
        peak_mb=counter.peak_mb,
        full_state_initial=full_initial,
        full_state_final=full_final,
        max_abs_error_vs_full=max_full_err,
        notes=notes,
    )


def run_gradient_optimizer(common: dict, graph, p: int, x0: np.ndarray, optimizer: str, budget: int, args) -> tuple[list[A2StepRow], A2SummaryRow]:
    counter = EvalCounter(graph, p, args)
    x = x0.copy()
    state: dict = {}
    steps: list[A2StepRow] = []
    best = float("-inf")
    initial = float("nan")
    status = "ok"
    max_full_err = 0.0
    t0 = time.perf_counter()
    for step in range(budget):
        if optimizer == "lc_adjoint_adam":
            val, grad, status = counter.value_grad(x)
            x_next = adam_step(x, grad, step + 1, state, args.lr, amsgrad=False)
        elif optimizer == "lc_adjoint_amsgrad":
            val, grad, status = counter.value_grad(x)
            x_next = adam_step(x, grad, step + 1, state, args.lr, amsgrad=True)
        elif optimizer == "lc_fd_adam":
            val, grad, status = counter.fd_grad(x, args.fd_eps)
            x_next = adam_step(x, grad, step + 1, state, args.lr, amsgrad=False)
        else:
            raise ValueError(optimizer)
        if step == 0:
            initial = val
        best = max(best, val) if math.isfinite(val) else best
        full_val = float("nan")
        if graph.n <= args.full_cap and step in {0, budget - 1}:
            fs = EvalCounter(graph, p, args, mode="full")
            full_val, _ = fs.value(x)
            if math.isfinite(full_val) and math.isfinite(val):
                max_full_err = max(max_full_err, abs(full_val - val))
        steps.append(A2StepRow(common["case"], common["seed"], common["init_id"], optimizer, step, counter.objective_calls, counter.gradient_calls, time.perf_counter() - t0, val, best, float(np.linalg.norm(grad)), counter.peak_mb, status))
        if status != "ok":
            break
        x = x_next
        if optimizer == "lc_fd_adam" and counter.objective_calls >= budget:
            break
    final = steps[-1].value if steps else float("nan")
    full_initial = float("nan")
    full_final = float("nan")
    if graph.n <= args.full_cap and steps:
        fs = EvalCounter(graph, p, args, mode="full")
        full_initial, _ = fs.value(x0)
        full_final, _ = fs.value(x)
    summary = summarize_steps(common, optimizer, status, budget, counter, steps, initial, final, full_initial, full_final, max_full_err if max_full_err > 0 else float("nan"), f"{optimizer}; lr={args.lr}")
    return steps, summary


def run_random_search(common: dict, graph, p: int, x0: np.ndarray, budget: int, args) -> tuple[list[A2StepRow], A2SummaryRow]:
    rng = np.random.default_rng(180000 + common["seed"] * 31 + common["init_id"])
    counter = EvalCounter(graph, p, args)
    steps: list[A2StepRow] = []
    best = float("-inf")
    initial, status = counter.value(x0)
    t0 = time.perf_counter()
    for step in range(budget):
        x = x0 if step == 0 else rng.uniform(-math.pi, math.pi, size=2 * p)
        val, status = counter.value(x)
        best = max(best, val) if math.isfinite(val) else best
        steps.append(A2StepRow(common["case"], common["seed"], common["init_id"], "random_search", step, counter.objective_calls, counter.gradient_calls, time.perf_counter() - t0, val, best, float("nan"), counter.peak_mb, status))
        if status != "ok":
            break
    summary = summarize_steps(common, "random_search", status, budget, counter, steps, initial, steps[-1].value if steps else float("nan"), float("nan"), float("nan"), float("nan"), "uniform random QAOA parameter search")
    return steps, summary


def run_spsa(common: dict, graph, p: int, x0: np.ndarray, budget: int, args) -> tuple[list[A2StepRow], A2SummaryRow]:
    rng = np.random.default_rng(190000 + common["seed"] * 31 + common["init_id"])
    counter = EvalCounter(graph, p, args)
    x = x0.copy()
    steps: list[A2StepRow] = []
    best = float("-inf")
    t0 = time.perf_counter()
    initial, status = counter.value(x0)
    max_steps = max(1, budget // 2)
    for step in range(max_steps):
        delta = rng.choice([-1.0, 1.0], size=x.size)
        ck = args.spsa_c / ((step + 1) ** 0.101)
        ak = args.spsa_a / ((step + 1 + 10.0) ** 0.602)
        vp, sp = counter.value(x + ck * delta)
        vm, sm = counter.value(x - ck * delta)
        status = "ok" if sp == "ok" and sm == "ok" else f"spsa_failed:{sp}/{sm}"
        grad = ((vp - vm) / (2.0 * ck)) * delta
        val = 0.5 * (vp + vm)
        x = wrap_angles(x + ak * grad)
        best = max(best, val) if math.isfinite(val) else best
        steps.append(A2StepRow(common["case"], common["seed"], common["init_id"], "spsa", step, counter.objective_calls, counter.gradient_calls, time.perf_counter() - t0, val, best, float(np.linalg.norm(grad)), counter.peak_mb, status))
        if status != "ok" or counter.objective_calls >= budget:
            break
    summary = summarize_steps(common, "spsa", status, budget, counter, steps, initial, steps[-1].value if steps else float("nan"), float("nan"), float("nan"), float("nan"), f"SPSA a={args.spsa_a}, c={args.spsa_c}")
    return steps, summary


def run_scipy_optimizer(common: dict, graph, p: int, x0: np.ndarray, optimizer: str, budget: int, args) -> tuple[list[A2StepRow], A2SummaryRow]:
    counter = EvalCounter(graph, p, args)
    steps: list[A2StepRow] = []
    best = float("-inf")
    status = "ok"
    t0 = time.perf_counter()

    def record(x: np.ndarray, value: float, grad_norm: float) -> None:
        nonlocal best
        best = max(best, value) if math.isfinite(value) else best
        steps.append(
            A2StepRow(
                common["case"],
                common["seed"],
                common["init_id"],
                optimizer,
                len(steps),
                counter.objective_calls,
                counter.gradient_calls,
                time.perf_counter() - t0,
                value,
                best,
                grad_norm,
                counter.peak_mb,
                status,
            )
        )

    if optimizer == "lbfgsb":
        last_grad = np.zeros_like(x0)

        def fun(x: np.ndarray) -> float:
            nonlocal status, last_grad
            value, grad, status = counter.value_grad(wrap_angles(x))
            last_grad = grad
            record(x, value, float(np.linalg.norm(grad)))
            return -value

        def jac(x: np.ndarray) -> np.ndarray:
            return -last_grad

        res = minimize(
            fun,
            x0,
            method="L-BFGS-B",
            jac=jac,
            bounds=[(-math.pi, math.pi)] * x0.size,
            options={"maxiter": budget, "maxfun": budget, "ftol": 1e-9, "gtol": 1e-7},
        )
        status = "ok" if res.success or "TOTAL NO. of f AND g EVALUATIONS" in str(res.message) else f"scipy:{res.message}"
    elif optimizer == "nelder_mead":
        def fun(x: np.ndarray) -> float:
            nonlocal status
            value, status = counter.value(wrap_angles(x))
            record(x, value, float("nan"))
            return -value

        res = minimize(
            fun,
            x0,
            method="Nelder-Mead",
            options={"maxiter": budget, "maxfev": budget, "xatol": 1e-5, "fatol": 1e-5},
        )
        status = "ok" if res.success or "Maximum number" in str(res.message) else f"scipy:{res.message}"
    else:
        raise ValueError(optimizer)

    initial = steps[0].value if steps else float("nan")
    final = steps[-1].value if steps else float("nan")
    full_initial = float("nan")
    full_final = float("nan")
    max_full_err = float("nan")
    if graph.n <= args.full_cap and steps:
        fs = EvalCounter(graph, p, args, mode="full")
        full_initial, _ = fs.value(x0)
        full_final, _ = fs.value(wrap_angles(getattr(res, "x", x0)))
        if math.isfinite(full_initial) and math.isfinite(initial):
            max_full_err = abs(full_initial - initial)
        if math.isfinite(full_final) and math.isfinite(final):
            max_full_err = max(max_full_err if math.isfinite(max_full_err) else 0.0, abs(full_final - final))
    summary = summarize_steps(common, optimizer, status, budget, counter, steps, initial, final, full_initial, full_final, max_full_err, f"scipy {optimizer}; message={getattr(res, 'message', '')}")
    return steps, summary


def run_case(family: str, n: int, p: int, seed_id: int, init_id: int, args) -> tuple[list[A2StepRow], list[A2SummaryRow]]:
    seed = 260000 + seed_id * 977 + n * 37 + p * 131
    graph = make_graph(family, n, seed)
    cmet = cone_metrics(graph, p)
    gmet = graph_metrics(graph)
    case = f"{family}_n{n}_p{p}"
    common = {
        "case": case,
        "objective": graph.objective,
        "family": family,
        "n": n,
        "p": p,
        "seed": seed_id,
        "init_id": init_id,
        "kmax": int(cmet["kmax"]),
        "total_cone_states": int(cmet["total_cone_states"]),
    }
    if cmet["kmax"] > args.max_k:
        summary = A2SummaryRow(**common, optimizer="all", status=f"NOT_RUN_EXPLAINED:kmax_{cmet['kmax']}_over_{args.max_k}", budget=budget_for(p), objective_calls=0, gradient_calls=0, seconds_total=0.0, initial_value=float("nan"), final_value=float("nan"), best_value=float("nan"), normalized_improvement=float("nan"), calls_to_95pct_best=0, time_to_95pct_best=float("nan"), peak_mb=0.0, full_state_initial=float("nan"), full_state_final=float("nan"), max_abs_error_vs_full=float("nan"), notes=f"m={gmet['m']}, mean_degree={gmet['mean_degree']:.3g}")
        return [], [summary]
    gammas, betas = params_for_depth(p, seed=seed_id, init_id=init_id)
    x0 = pack_params(gammas, betas)
    budget = budget_for(p)
    steps: list[A2StepRow] = []
    summaries: list[A2SummaryRow] = []
    optimizers = args.optimizers or ["lc_adjoint_adam", "lc_adjoint_amsgrad", "random_search", "spsa"]
    if args.include_fd and (n <= args.fd_max_n or init_id == 0 and seed_id == 0):
        if "lc_fd_adam" not in optimizers:
            optimizers.append("lc_fd_adam")
    for opt in optimizers:
        print(f"A2 {case} seed={seed_id} init={init_id} opt={opt} budget={budget}", flush=True)
        if opt in {"lc_adjoint_adam", "lc_adjoint_amsgrad", "lc_fd_adam"}:
            s, r = run_gradient_optimizer(common, graph, p, x0, opt, budget, args)
        elif opt == "random_search":
            s, r = run_random_search(common, graph, p, x0, budget, args)
        elif opt in {"lbfgsb", "nelder_mead"}:
            s, r = run_scipy_optimizer(common, graph, p, x0, opt, budget, args)
        else:
            s, r = run_spsa(common, graph, p, x0, budget, args)
        steps.extend(s)
        summaries.append(r)
    return steps, summaries


def write_md(rows: list[A2SummaryRow], path: Path) -> None:
    lines = [
        "# A2 End-to-End QAOA Training Quality",
        "",
        "Rows are full optimizer runs using LC objective/adjoint where supported. Full-state agreement is recorded for n <= full_cap.",
        "",
        "| Case | Optimizer | Runs | success | mean best | mean norm impr | mean seconds | mean peak MB |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    keys = sorted({(r.case, r.optimizer) for r in rows})
    for key in keys:
        sub = [r for r in rows if (r.case, r.optimizer) == key]
        ok = [r for r in sub if r.status == "ok"]
        mean = lambda vals: float(np.nanmean(vals)) if vals else float("nan")
        lines.append(
            f"| {key[0]} | {key[1]} | {len(sub)} | {len(ok)} | {mean([r.best_value for r in ok]):.6g} | "
            f"{mean([r.normalized_improvement for r in ok]):.4g} | {mean([r.seconds_total for r in ok]):.4g} | {mean([r.peak_mb for r in ok]):.4g} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "benchmark_suite_20260704" / "A2_training_quality")
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--inits", type=int, default=5)
    parser.add_argument("--init-start", type=int, default=0)
    parser.add_argument("--max-k", type=int, default=24)
    parser.add_argument("--max-batch-states", type=int, default=1 << 21)
    parser.add_argument("--full-cap", type=int, default=24)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--fd-eps", type=float, default=1e-3)
    parser.add_argument("--include-fd", action="store_true")
    parser.add_argument("--fd-max-n", type=int, default=24)
    parser.add_argument("--spsa-a", type=float, default=0.04)
    parser.add_argument("--spsa-c", type=float, default=0.04)
    parser.add_argument("--families", nargs="*", default=None)
    parser.add_argument("--ns", nargs="*", type=int, default=None)
    parser.add_argument("--ps", nargs="*", type=int, default=None)
    parser.add_argument("--optimizers", nargs="*", default=None)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    grid = case_specs()
    if args.families:
        grid = [g for g in grid if g[0] in set(args.families)]
    if args.ns:
        wanted_ns = set(args.ns)
        grid = [(fam, [n for n in ns if n in wanted_ns], ps) for fam, ns, ps in grid]
        grid = [g for g in grid if g[1]]
    if args.ps:
        wanted_ps = set(args.ps)
        grid = [(fam, ns, [p for p in ps if p in wanted_ps]) for fam, ns, ps in grid]
        grid = [g for g in grid if g[2]]
    if args.quick:
        grid = [("3regular", [24, 64], [1, 2]), ("weighted_sparse_qubo", [24], [1])]
        args.seeds = min(args.seeds, 2)
        args.inits = min(args.inits, 2)
        args.include_fd = True
    step_rows: list[A2StepRow] = []
    summary_rows: list[A2SummaryRow] = []
    step_csv = args.out_dir / "A2_training_quality_steps.csv"
    summary_csv = args.out_dir / "A2_training_quality_summary.csv"
    for family, ns, ps in grid:
        for n in ns:
            for p in ps:
                for seed_id in range(args.seed_start, args.seed_start + args.seeds):
                    for init_id in range(args.init_start, args.init_start + args.inits):
                        s, r = run_case(family, n, p, seed_id, init_id, args)
                        step_rows.extend(s)
                        summary_rows.extend(r)
                        if summary_rows:
                            with summary_csv.open("w", newline="", encoding="utf-8") as f:
                                writer = csv.DictWriter(f, fieldnames=list(asdict(summary_rows[0]).keys()))
                                writer.writeheader()
                                for row in summary_rows:
                                    writer.writerow(asdict(row))
                        if step_rows:
                            with step_csv.open("w", newline="", encoding="utf-8") as f:
                                writer = csv.DictWriter(f, fieldnames=list(asdict(step_rows[0]).keys()))
                                writer.writeheader()
                                for row in step_rows:
                                    writer.writerow(asdict(row))
    write_md(summary_rows, args.out_dir / "A2_training_quality.md")
    print(f"WROTE {summary_csv}")
    print(f"WROTE {step_csv}")


if __name__ == "__main__":
    main()
