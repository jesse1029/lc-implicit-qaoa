"""Stage B: matched LC vs TensorCircuit-NG timing on the paper's Table 7 instances.

The instances are reconstructed with the paper's own recipe
(scripts/run_official_regime_matrix.py):
    seed = 73000 + 97*n + 13*p + FAMILY_SEED_OFFSET[family]
    gammas = [0.20 + 0.05*i], betas = [0.32 - 0.035*i]
so the graphs are bit-identical to the ones behind tab:peer.

What is NOT matched: the hardware. Table 7 ran on an RTX 3070; this box has an
RTX 4060 Ti. Absolute seconds are therefore not comparable to the printed table.
The LC/TC-NG ratio measured here on one device is the quantity of interest, and
LC is re-measured on this box so the ratio never mixes machines.

Both methods:
  - see the identical graph object
  - use the identical light-cone decomposition (prior work shared by both)
  - run complex64/float32, matching the Table 7 protocol
  - run on the same physical GPU (LC via CuPy, TC-NG via PyTorch)

Cold = first query on a fresh process state, including cone extraction and any
contraction pathfinding. Steady = median of REPEATS later queries, which is what
an optimizer loop actually pays.
"""
import os, sys, pathlib, time, json, argparse, statistics as st
import numpy as np

CODE = os.environ.get("LCQAOA_CODE") or str(pathlib.Path(__file__).resolve().parents[2])
sys.path.insert(0, CODE)

ap = argparse.ArgumentParser()
ap.add_argument("--repeats", type=int, default=5)
ap.add_argument("--gradient", action="store_true", help="time objective+all gradients instead of objective only")
ap.add_argument("--cases", default="all")
ap.add_argument("--out", default="")
args = ap.parse_args()

import torch
import tensorcircuit as tc

DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device(DEV)
tc.set_backend("pytorch")
tc.set_dtype("complex64")

from lcqaoa.graphs import random_regular_graph, erdos_renyi_graph
from lcqaoa.lightcone import (extract_lightcones, lightcone_expectation,
                              lightcone_gradient_adjoint)

FAMILY_SEED_OFFSET = {"3regular": 101, "er_deg2": 202, "er_deg3": 303}


def graph_for(family, n, seed):
    if family == "3regular":
        return random_regular_graph(n, 3, seed=seed)
    if family == "er_deg2":
        return erdos_renyi_graph(n, min(0.45, 2.0 / max(2, n)), seed=seed)
    if family == "er_deg3":
        return erdos_renyi_graph(n, min(0.45, 3.0 / max(2, n)), seed=seed)
    raise ValueError(family)


def params_for(p):
    return [0.20 + 0.05 * i for i in range(p)], [0.32 - 0.035 * i for i in range(p)]


def make_case(family, n, p):
    seed = 73000 + 97 * n + 13 * p + FAMILY_SEED_OFFSET[family]
    return graph_for(family, n, seed), params_for(p)


# --------------------------------------------------------------------------- LC
def lc_query(graph, gammas, betas, p, gradient):
    fn = lightcone_gradient_adjoint if gradient else lightcone_expectation
    r = fn(graph, gammas, betas, p=p, prefer_gpu=True,
           complex_dtype=np.complex64, float_dtype=np.float32)
    return r


# ------------------------------------------------------------------------ TC-NG
def tc_cone_value(problem, gammas, betas):
    k = problem.k
    c = tc.Circuit(k)
    for q in range(k):
        c.h(q)                      # idiomatic product-state input keeps the TN factorized
    for layer in range(len(gammas)):
        for i, j, w in problem.edges:
            c.exp1(i, j, unitary=tc.gates._zz_matrix, theta=-gammas[layer] * w / 2.0)
        for q in range(k):
            c.rx(q, theta=2.0 * betas[layer])
    i, j = problem.local_term_nodes
    zz = c.expectation((tc.gates.z(), [i]), (tc.gates.z(), [j]))
    return problem.weight * 0.5 * (1.0 - tc.backend.real(zz))


def tc_query(graph, gammas, betas, p, gradient, problems=None):
    if problems is None:
        problems = extract_lightcones(graph, p)
    par = torch.tensor(list(gammas) + list(betas), dtype=torch.float32,
                       device=DEV, requires_grad=gradient)
    g_, b_ = par[:p], par[p:]
    total = None
    for pr in problems:
        v = tc_cone_value(pr, g_, b_)
        total = v if total is None else total + v
    if gradient:
        total.backward()
        return float(total.detach().cpu()), par.grad.detach().cpu().numpy()
    return float(total.detach().cpu()), None


def timed(fn, repeats):
    """Returns (cold_seconds, steady_median_seconds, result)."""
    torch.cuda.synchronize() if DEV == "cuda" else None
    t0 = time.perf_counter()
    res = fn()
    torch.cuda.synchronize() if DEV == "cuda" else None
    cold = time.perf_counter() - t0
    steady = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize() if DEV == "cuda" else None
        steady.append(time.perf_counter() - t0)
    return cold, st.median(steady), res


CASES = [
    ("3regular", 24, 2), ("3regular", 26, 2), ("3regular", 28, 2), ("3regular", 128, 2),
    ("er_deg2", 128, 2), ("er_deg3", 24, 2),
]
if args.cases != "all":
    want = set(args.cases.split(","))
    CASES = [c for c in CASES if "%s_%d" % (c[0], c[1]) in want]

print("device: %s | torch %s | tensorcircuit-ng %s | dtype complex64" %
      (torch.cuda.get_device_name(0) if DEV == "cuda" else "cpu", torch.__version__, tc.__version__))
print("query: %s | repeats %d\n" % ("objective + all 2p gradients" if args.gradient else "objective only", args.repeats))
hdr = ("%-22s %-6s %-6s %-8s %-11s %-11s %-11s %-11s %-9s %s" %
       ("case", "cones", "k_max", "agree", "LC cold", "LC steady", "TC cold", "TC steady", "cold x", "steady x"))
print(hdr)
print("-" * len(hdr))

rows = []
for family, n, p in CASES:
    graph, (gammas, betas) = make_case(family, n, p)
    problems = extract_lightcones(graph, p)
    kmax = max(pr.k for pr in problems)

    try:
        lc_cold, lc_steady, lc_res = timed(
            lambda: lc_query(graph, gammas, betas, p, args.gradient), args.repeats)
        lc_val, lc_status = lc_res.value, lc_res.status
    except Exception as exc:
        lc_cold = lc_steady = float("nan"); lc_val = float("nan"); lc_status = "ERR:" + str(exc)[:40]

    try:
        tc_cold, tc_steady, tc_res = timed(
            lambda: tc_query(graph, gammas, betas, p, args.gradient), args.repeats)
        tc_val = tc_res[0]
    except Exception as exc:
        tc_cold = tc_steady = float("nan"); tc_val = float("nan")
        print("   TC-NG error on %s n=%d: %s" % (family, n, str(exc)[:160]))

    rel = abs(tc_val - lc_val) / max(1.0, abs(lc_val)) if lc_val == lc_val and tc_val == tc_val else float("nan")
    agree = "%.1e" % rel if rel == rel else "n/a"
    cx = tc_cold / lc_cold if lc_cold and lc_cold == lc_cold and tc_cold == tc_cold else float("nan")
    sx = tc_steady / lc_steady if lc_steady and lc_steady == lc_steady and tc_steady == tc_steady else float("nan")
    print("%-22s %-6d %-6d %-8s %-11.4f %-11.4f %-11.4f %-11.4f %-9.2f %.2f" %
          ("%s n=%d p=%d" % (family, n, p), len(problems), kmax, agree,
           lc_cold, lc_steady, tc_cold, tc_steady, cx, sx))
    rows.append(dict(family=family, n=n, p=p, cones=len(problems), kmax=kmax,
                     lc_status=lc_status, lc_value=lc_val, tc_value=tc_val, rel_diff=rel,
                     lc_cold=lc_cold, lc_steady=lc_steady, tc_cold=tc_cold, tc_steady=tc_steady,
                     tc_over_lc_cold=cx, tc_over_lc_steady=sx,
                     query="gradient" if args.gradient else "objective"))

print()
fin = [r for r in rows if r["tc_over_lc_steady"] == r["tc_over_lc_steady"]]
if fin:
    print("LC faster in %d of %d completed cells (steady state)" %
          (sum(1 for r in fin if r["tc_over_lc_steady"] > 1.0), len(fin)))
    print("steady-state ratio range: %.2fx to %.2fx" %
          (min(r["tc_over_lc_steady"] for r in fin), max(r["tc_over_lc_steady"] for r in fin)))
if args.out:
    open(args.out, "w", encoding="utf-8").write(json.dumps(rows, indent=2))
    print("wrote", args.out)
