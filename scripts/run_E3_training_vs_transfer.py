#!/usr/bin/env python
"""E3: does exact bounded-cone training buy anything over angle transfer?

At sizes where global-state training is infeasible a reviewer can reasonably ask why
one should train at all instead of reusing angles. This answers that with the exact
QAOA objective, which LC can evaluate at these sizes and a global-state route cannot.

Arms, all scored by the same complex128 exact objective and paired by graph instance:
  lc_trained        Adam ascent on exact LC gradients, 100 steps (101 value+gradient queries)
  transfer_small_n  angles trained the same way on one small instance of the family
  transfer_same_n   angles trained on a *different instance of the same size*
  random_best       best of 101 random angle draws (equal query budget, no gradient)
  init              the shared deterministic initialization (floor)

The two transfer arms separate "angles do not transfer across sizes" from the stronger
"angles do not transfer across instances". Instances are screened on cone size before
any run, so eligibility never depends on the measured outcome.

No timing is reported: this is a solution-quality experiment and is therefore unaffected
by GPU contention on a shared host.
"""
from __future__ import annotations

import argparse, csv, json, math, os, platform, subprocess, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lcqaoa import (  # noqa: E402
    extract_lightcones, lightcone_expectation, lightcone_gradient_adjoint,
)
from lcqaoa.graphs import (  # noqa: E402
    random_regular_graph, weighted_qubo_graph, weighted_modular_qubo_graph,
)

P = 2
STEPS = 100
LR = 0.02
TRANSFER_N = 24
SCORE_KW = dict(p=P, prefer_gpu=True, complex_dtype=np.complex128, float_dtype=np.float64)
FAST_KW = dict(p=P, prefer_gpu=True, complex_dtype=np.complex64, float_dtype=np.float32)

FAMILIES = {
    "3regular": dict(gen=lambda n, s: random_regular_graph(n, 3, seed=s),
                     sizes=[128, 256, 512], kmax_cap=16),
    "weighted_er2": dict(gen=lambda n, s: weighted_qubo_graph(n, 2.0 / (n - 1), seed=s),
                         sizes=[64, 96, 128, 192], kmax_cap=20),
    "weighted_modular": dict(
        gen=lambda n, s: weighted_modular_qubo_graph(n, modules=4, p_in=0.03, p_out=0.002, seed=s),
        sizes=[64, 96, 128], kmax_cap=20),
}


def wrap(x):
    return ((x + math.pi) % (2.0 * math.pi)) - math.pi


def initial_angles(p: int) -> np.ndarray:
    return np.asarray([0.4, 0.8][:p] + [0.8, 0.4][:p], dtype=np.float64)


def adam(x, grad, step, state, lr=LR):
    b1, b2, eps = 0.9, 0.999, 1e-8
    state.setdefault("m", np.zeros_like(x)); state.setdefault("v", np.zeros_like(x))
    state["m"] = b1 * state["m"] + (1 - b1) * grad
    state["v"] = b2 * state["v"] + (1 - b2) * grad * grad
    mh = state["m"] / (1 - b1 ** step); vh = state["v"] / (1 - b2 ** step)
    return wrap(x + lr * mh / (np.sqrt(vh) + eps))          # ascent: the objective is maximized


def evaluate(graph, x, kw):
    st = lightcone_expectation(graph, x[:P].tolist(), x[P:].tolist(), **kw)
    return float(st.value) if st.status == "ok" else float("nan")


def score(graph, x):
    return evaluate(graph, x, SCORE_KW)


def train(graph, x0, steps=STEPS):
    x, state, best_x, best_v, calls = x0.copy(), {}, x0.copy(), -math.inf, 0
    for step in range(steps + 1):
        adj = lightcone_gradient_adjoint(graph, x[:P].tolist(), x[P:].tolist(), **FAST_KW)
        calls += 1
        if adj.status != "ok" or adj.gradient is None:
            return best_x, best_v, calls, adj.status
        if float(adj.value) > best_v:
            best_v, best_x = float(adj.value), x.copy()
        if step < steps:
            x = adam(x, np.asarray(adj.gradient, dtype=np.float64), step + 1, state)
    return best_x, best_v, calls, "ok"


def random_search(graph, rng, budget):
    """Equal query budget. Winner is picked in the throughput dtype, then scored exactly."""
    best_x, best_v = None, -math.inf
    for _ in range(budget):
        x = rng.uniform(-math.pi, math.pi, size=2 * P)
        v = evaluate(graph, x, FAST_KW)
        if math.isfinite(v) and v > best_v:
            best_v, best_x = v, x
    return best_x


def eligible_seeds(gen, n, cap, want, scan=200):
    """Screen instances on cone size before any run; record what was screened out."""
    keep, scanned, rejected = [], 0, 0
    for s in range(20000, 20000 + scan):
        scanned += 1
        kmax = max(c.k for c in extract_lightcones(gen(n, s), P))
        if kmax <= cap:
            keep.append((s, kmax))
            if len(keep) == want:
                break
        else:
            rejected += 1
    return keep, scanned, rejected


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--family", default=None, help="run a single family")
    ap.add_argument("--out", default="results/E3_training_vs_transfer")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    rows, screening, t0 = [], [], time.perf_counter()

    selected = {args.family: FAMILIES[args.family]} if args.family else FAMILIES
    for fam, cfg in selected.items():
        gen, cap = cfg["gen"], cfg["kmax_cap"]
        x_small, v_small, _, st_small = train(gen(TRANSFER_N, 9000), initial_angles(P))
        print(f"[{fam}] small-n angles (n={TRANSFER_N}): {np.round(x_small,4).tolist()} "
              f"value={v_small:.6f} ({st_small})", flush=True)

        for n in cfg["sizes"]:
            keep, scanned, rejected = eligible_seeds(gen, n, cap, args.seeds)
            screening.append(dict(family=fam, n=n, kmax_cap=cap, scanned=scanned,
                                  eligible=len(keep), rejected=rejected))
            print(f"[{fam} n={n}] screened {scanned} instances, kept {len(keep)} with kmax<={cap}", flush=True)
            # a donor instance of the same size, disjoint from the evaluated instances
            donor_seed = 20000 + scanned + 500
            while max(c.k for c in extract_lightcones(gen(n, donor_seed), P)) > cap:
                donor_seed += 1
            x_same, v_same, _, _ = train(gen(n, donor_seed), initial_angles(P))
            print(f"[{fam} n={n}] same-n donor seed={donor_seed} value={v_same:.6f}", flush=True)

            for seed, kmax in keep:
                g = gen(n, seed)
                x_lc, _, calls, st = train(g, initial_angles(P))
                x_rd = random_search(g, np.random.default_rng(777000 + seed), calls)
                vals = {
                    "lc_trained": score(g, x_lc),
                    "transfer_small_n": score(g, x_small),
                    "transfer_same_n": score(g, x_same),
                    "random_best": score(g, x_rd) if x_rd is not None else float("nan"),
                    "init": score(g, initial_angles(P)),
                }
                rows.append(dict(family=fam, n=n, p=P, seed=seed, kmax=kmax,
                                 edges=len(g.edges), fields=len(g.fields), status=st,
                                 queries=calls, donor_seed=donor_seed,
                                 **{f"obj_{k}": v for k, v in vals.items()}))
                print(f"  {fam} n={n} seed={seed} kmax={kmax} "
                      + " ".join(f"{k}={v:.6f}" for k, v in vals.items()), flush=True)

            with (out / "E3_raw.csv").open("w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
                w.writeheader(); w.writerows(rows)

    with (out / "E3_screening.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(screening[0].keys()))
        w.writeheader(); w.writerows(screening)

    env = dict(python=platform.python_version(), numpy=np.__version__, host=platform.node(),
               cuda_visible=os.environ.get("CUDA_VISIBLE_DEVICES", "unset"),
               steps=STEPS, lr=LR, transfer_n=TRANSFER_N, p=P,
               scoring="complex128/float64 exact light-cone objective",
               training="complex64/float32 adjoint",
               seconds_total=round(time.perf_counter() - t0, 2),
               note="solution-quality experiment; no timing claim, so shared-host GPU contention is irrelevant")
    try:
        env["cupy"] = __import__("cupy").__version__
        env["gpu_name"] = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                                         capture_output=True, text=True).stdout.strip().splitlines()[0]
    except Exception:
        pass
    (out / "environment.json").write_text(json.dumps(env, indent=2), encoding="utf-8")
    print(f"\nwrote {out/'E3_raw.csv'} ({len(rows)} rows) in {env['seconds_total']}s")


if __name__ == "__main__":
    main()
