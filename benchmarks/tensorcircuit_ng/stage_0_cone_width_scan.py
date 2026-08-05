"""Stage 0 go/no-go: does contraction width w sit far enough below cone size k
to make a per-cone hybrid worth building?

Dense local state costs ~2^k. Contraction costs ~2^w. w <= k always.
If w ~ k everywhere, contraction can never repay its overhead and the hybrid dies.
The decisive count is cones with k > 24 (LC must reject) but small w (contractible).

Caveat recorded up front: w here is the min-fill treewidth of the cone's induced
interaction graph, the same measure the paper's dense-vs-TN study used. For a
depth-p QAOA tensor network the true contraction width can exceed it, so this is
an OPTIMISTIC bound for contraction. That is the right direction for a kill-test:
if even the optimistic bound says w ~ k, the idea is dead.
"""
import os, sys, pathlib, statistics as st

CODE = os.environ.get("LCQAOA_CODE") or str(pathlib.Path(__file__).resolve().parents[2])
sys.path.insert(0, CODE)

import networkx as nx
from networkx.algorithms.approximation import treewidth_min_fill_in
from lcqaoa.graphs import (erdos_renyi_graph, modular_graph, random_regular_graph,
                           weighted_qubo_graph, scale_free_graph)
from lcqaoa.lightcone import extract_lightcones

GUARDRAIL = 24
SKIP_ABOVE = 46          # min-fill gets expensive; report these separately


def make(family, n, seed):
    if family == "3regular":
        return random_regular_graph(n, 3, seed=seed)
    if family == "er2":
        return erdos_renyi_graph(n, min(0.45, 2.0 / max(2, n)), seed=seed)
    if family == "er3":
        return erdos_renyi_graph(n, min(0.45, 3.0 / max(2, n)), seed=seed)
    if family == "modular_sparse":
        return modular_graph(n, modules=max(2, min(4, n // 4)), p_in=0.28, p_out=0.015, seed=seed)
    if family == "weighted":
        return weighted_qubo_graph(n, min(0.45, 2.0 / max(2, n)), field_prob=1.0, seed=seed,
                                   weight_scale=1.0, field_scale=0.7)
    if family == "scale_free":
        return scale_free_graph(n, attachment=2, seed=seed)
    raise ValueError(family)


def cone_width(problem):
    g = nx.Graph()
    g.add_nodes_from(range(problem.k))
    g.add_edges_from((i, j) for i, j, _ in problem.edges)
    if problem.k > SKIP_ABOVE:
        return None
    w, _ = treewidth_min_fill_in(g)
    return w


rows = []
print("%-14s %-5s %-3s %-7s %-7s %-7s %-9s %s" %
      ("family", "n", "p", "cones", "k_max", "w_max", "med k/w", "cones k>24 with small w"))
print("-" * 96)
for family in ["3regular", "er2", "er3", "modular_sparse", "weighted", "scale_free"]:
    for n in [24, 64, 128]:
        for p in [1, 2]:
            try:
                g = make(family, n, 20260805)
                cones = extract_lightcones(g, p=p)
            except Exception as e:
                print("%-14s %-5d %-3d  ERROR %s" % (family, n, p, str(e)[:40]))
                continue
            ks, ws, big_ok, skipped = [], [], 0, 0
            for c in cones:
                w = cone_width(c)
                ks.append(c.k)
                if w is None:
                    skipped += 1
                    continue
                ws.append(w)
                if c.k > GUARDRAIL and w <= GUARDRAIL:
                    big_ok += 1
            if not ws:
                print("%-14s %-5d %-3d %-7d %-7d  all cones too large to profile (k_max=%d)" %
                      (family, n, p, len(cones), max(ks), max(ks)))
                continue
            note = "%d" % big_ok
            if skipped:
                note += " (+%d cones k>%d unprofiled)" % (skipped, SKIP_ABOVE)
            print("%-14s %-5d %-3d %-7d %-7d %-7d %-9s %s" %
                  (family, n, p, len(cones), max(ks), max(ws),
                   "%d/%d" % (st.median(ks), st.median(ws)), note))
            rows.append((family, n, p, max(ks), max(ws), big_ok))

print()
print("=== verdict inputs ===")
gap = [(f, n, p, km, wm) for f, n, p, km, wm, _ in rows if km - wm >= 4]
print("cells where k_max - w_max >= 4 (contraction has real headroom): %d of %d" % (len(gap), len(rows)))
for f, n, p, km, wm in gap[:12]:
    print("   %-14s n=%-4d p=%d   k_max=%-3d w_max=%-3d   dense 2^%d vs contract 2^%d  = %.3g x memory"
          % (f, n, p, km, wm, km, wm, 2.0 ** (km - wm)))
tot_big = sum(b for *_, b in rows)
print()
print("total cones that LC must REJECT (k>24) but contraction could take (w<=24): %d" % tot_big)
