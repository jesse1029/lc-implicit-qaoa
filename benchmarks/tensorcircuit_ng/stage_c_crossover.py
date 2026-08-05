"""Stage C: where does LC stop winning, and where does it stop working at all?

Stage B measures the regime the paper claims. This measures the regime the paper
concedes -- "large low-width cones remain their regime" -- and turns that
sentence into a number.

Two distinct outcomes are tracked, and they must not be conflated:
  SLOWER  - both methods answer, TC-NG is faster (a speed crossover)
  REJECT  - k_max exceeds LC's guardrail, so LC returns no value at all
            (a capability gap; there is no ratio to report)

Instances come from the paper's own generators; the families are chosen from the
Stage 0 k-vs-treewidth sweep as the ones that produce large, low-width cones.
"""
import os, sys, pathlib, time, json, argparse, statistics as st
import numpy as np

CODE = os.environ.get("LCQAOA_CODE") or str(pathlib.Path(__file__).resolve().parents[2])
sys.path.insert(0, CODE)

ap = argparse.ArgumentParser()
ap.add_argument("--repeats", type=int, default=3)
ap.add_argument("--out", default="")
ap.add_argument("--tc-timeout", type=float, default=300.0)
args = ap.parse_args()

import torch, tensorcircuit as tc
import networkx as nx
from networkx.algorithms.approximation import treewidth_min_fill_in

DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device(DEV)
tc.set_backend("pytorch")
tc.set_dtype("complex64")

from lcqaoa.graphs import (erdos_renyi_graph, modular_graph, random_regular_graph,
                           scale_free_graph)
from lcqaoa.lightcone import extract_lightcones, lightcone_expectation

GUARDRAIL = 24


def make(family, n, seed):
    if family == "3regular":
        return random_regular_graph(n, 3, seed=seed)
    if family == "er_deg3":
        return erdos_renyi_graph(n, min(0.45, 3.0 / max(2, n)), seed=seed)
    if family == "modular_sparse":
        return modular_graph(n, modules=max(2, min(4, n // 4)), p_in=0.28, p_out=0.015, seed=seed)
    if family == "scale_free":
        return scale_free_graph(n, attachment=2, seed=seed)
    raise ValueError(family)


def cone_width(pr):
    g = nx.Graph()
    g.add_nodes_from(range(pr.k))
    g.add_edges_from((i, j) for i, j, _ in pr.edges)
    return treewidth_min_fill_in(g)[0]


def tc_cone_value(pr, g_, b_, p):
    c = tc.Circuit(pr.k)
    for q in range(pr.k):
        c.h(q)
    for layer in range(p):
        for i, j, w in pr.edges:
            c.exp1(i, j, unitary=tc.gates._zz_matrix, theta=-g_[layer] * w / 2.0)
        for q in range(pr.k):
            c.rx(q, theta=2.0 * b_[layer])
    a, b = pr.local_term_nodes
    zz = c.expectation((tc.gates.z(), [a]), (tc.gates.z(), [b]))
    return pr.weight * 0.5 * (1.0 - tc.backend.real(zz))


def tc_query(problems, gammas, betas, p):
    par = torch.tensor(list(gammas) + list(betas), dtype=torch.float32, device=DEV)
    g_, b_ = par[:p], par[p:]
    total = None
    for pr in problems:
        v = tc_cone_value(pr, g_, b_, p)
        total = v if total is None else total + v
    return float(total.detach().cpu())


def timed(fn, repeats):
    t0 = time.perf_counter(); res = fn()
    if DEV == "cuda":
        torch.cuda.synchronize()
    cold = time.perf_counter() - t0
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter(); fn()
        if DEV == "cuda":
            torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return cold, st.median(ts), res


CASES = [
    ("3regular", 64, 2), ("3regular", 128, 2),
    ("er_deg3", 64, 2), ("er_deg3", 128, 2),
    ("modular_sparse", 64, 1), ("modular_sparse", 128, 1), ("modular_sparse", 64, 2),
    ("scale_free", 24, 1), ("scale_free", 24, 2),
    ("scale_free", 64, 1), ("scale_free", 128, 1),
]

print("device %s | dtype complex64 | LC guardrail k<=%d\n" %
      (torch.cuda.get_device_name(0) if DEV == "cuda" else "cpu", GUARDRAIL))
hdr = ("%-24s %-6s %-6s %-6s %-9s %-11s %-11s %s" %
       ("case", "cones", "k_max", "w_max", "LC", "LC steady", "TC steady", "verdict"))
print(hdr); print("-" * len(hdr))

rows = []
for family, n, p in CASES:
    seed = 73000 + 97 * n + 13 * p
    graph = make(family, n, seed)
    gammas = [0.20 + 0.05 * i for i in range(p)]
    betas = [0.32 - 0.035 * i for i in range(p)]
    problems = extract_lightcones(graph, p)
    kmax = max(pr.k for pr in problems)
    wmax = max((cone_width(pr) for pr in problems if pr.k <= 46), default=-1)

    lc_cold = lc_steady = float("nan"); lc_val = float("nan"); lc_state = "?"
    try:
        lc_cold, lc_steady, r = timed(
            lambda: lightcone_expectation(graph, gammas, betas, p=p, prefer_gpu=True,
                                          complex_dtype=np.complex64, float_dtype=np.float32),
            args.repeats)
        lc_val = r.value
        lc_state = "ok" if r.status == "ok" else "REJECT"
        if lc_state == "REJECT":
            lc_cold = lc_steady = float("nan")
    except Exception as exc:
        lc_state = "ERR"

    tc_cold = tc_steady = float("nan"); tc_val = float("nan")
    t_start = time.perf_counter()
    try:
        tc_cold, tc_steady, tc_val = timed(lambda: tc_query(problems, gammas, betas, p), args.repeats)
    except Exception as exc:
        print("   TC-NG failed on %s n=%d p=%d: %s" % (family, n, p, str(exc)[:140]))

    if lc_state == "REJECT":
        verdict = "LC REJECTS, TC-NG answers" if tc_steady == tc_steady else "both unavailable"
        ratio = float("nan")
    elif tc_steady == tc_steady and lc_steady == lc_steady:
        ratio = tc_steady / lc_steady
        verdict = ("LC %.1fx faster" % ratio) if ratio > 1 else ("TC-NG %.1fx faster" % (1 / ratio))
    else:
        ratio = float("nan"); verdict = "incomplete"

    print("%-24s %-6d %-6d %-6s %-9s %-11s %-11s %s" %
          ("%s n=%d p=%d" % (family, n, p), len(problems), kmax,
           wmax if wmax >= 0 else ">46", lc_state,
           "%.4f" % lc_steady if lc_steady == lc_steady else "--",
           "%.4f" % tc_steady if tc_steady == tc_steady else "--", verdict))
    rows.append(dict(family=family, n=n, p=p, cones=len(problems), kmax=kmax, wmax=wmax,
                     lc_state=lc_state, lc_value=lc_val, tc_value=tc_val,
                     lc_cold=lc_cold, lc_steady=lc_steady,
                     tc_cold=tc_cold, tc_steady=tc_steady, tc_over_lc=ratio, verdict=verdict))

print()
rej = [r for r in rows if r["lc_state"] == "REJECT" and r["tc_steady"] == r["tc_steady"]]
both = [r for r in rows if r["lc_state"] == "ok" and r["tc_over_lc"] == r["tc_over_lc"]]
print("cells where LC rejects but TC-NG completes: %d" % len(rej))
for r in rej:
    print("   %-22s k_max=%-4d w_max=%-3s TC-NG %.3f s" %
          ("%s n=%d p=%d" % (r["family"], r["n"], r["p"]), r["kmax"], r["wmax"], r["tc_steady"]))
if both:
    print("cells where both answer: LC faster in %d of %d" %
          (sum(1 for r in both if r["tc_over_lc"] > 1), len(both)))
if args.out:
    open(args.out, "w", encoding="utf-8").write(json.dumps(rows, indent=2))
    print("wrote", args.out)
