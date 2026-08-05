"""Which side carries the 1e-7 error: LC or TC-NG?

Compare each against a brute-force dense reference built here from scratch in
float64, on a graph small enough to hold the full 2^n state.
"""
import os, sys, pathlib, math, numpy as np

CODE = os.environ.get("LCQAOA_CODE") or str(pathlib.Path(__file__).resolve().parents[2])
sys.path.insert(0, CODE)

import tensorcircuit as tc
import torch
tc.set_backend("pytorch")
tc.set_dtype("complex128")

from lcqaoa.graphs import random_regular_graph
from lcqaoa.lightcone import extract_lightcones, lightcone_expectation

g = random_regular_graph(10, 3, seed=7)
p = 1
gammas, betas = [0.31], [0.22]
n = g.n

# ---- brute force dense reference, written independently in float64 ----------
idx = np.arange(1 << n, dtype=np.uint64)
cost = np.zeros(1 << n, dtype=np.float64)
for i, j, w in g.edges:
    cost += w * (((idx >> np.uint64(i)) ^ (idx >> np.uint64(j))) & np.uint64(1)).astype(np.float64)
psi = np.full(1 << n, (1 << n) ** -0.5, dtype=np.complex128)
for gam, bet in zip(gammas, betas):
    psi = np.exp(-1j * gam * cost) * psi
    c, s = math.cos(bet), math.sin(bet)
    for q in range(n):
        low = 1 << q
        v = psi.reshape(-1, 2, low)
        a, b = v[:, 0, :].copy(), v[:, 1, :].copy()
        psi = np.stack((c * a - 1j * s * b, c * b - 1j * s * a), axis=1).reshape(-1)
ref = float(np.sum((np.abs(psi) ** 2) * cost))

lc64 = lightcone_expectation(g, gammas, betas, p=p, prefer_gpu=False,
                             complex_dtype=np.complex64, float_dtype=np.float32).value
lc128 = lightcone_expectation(g, gammas, betas, p=p, prefer_gpu=False,
                              complex_dtype=np.complex128, float_dtype=np.float64).value

problems = extract_lightcones(g, p)
par = torch.tensor(gammas + betas, dtype=torch.float64, requires_grad=True)
USE_EXACT_INPUT = "--hgate" not in sys.argv
tot = None
for pr in problems:
    if USE_EXACT_INPUT:
        # tc's cached H constant carries only float32 accuracy (see gate probe),
        # so feed the exact uniform state instead of building it from H gates.
        c_ = tc.Circuit(pr.k, inputs=torch.full((1 << pr.k,),
                                                (1 << pr.k) ** -0.5, dtype=torch.complex128))
    else:
        c_ = tc.Circuit(pr.k)
        for q in range(pr.k):
            c_.h(q)
    for layer in range(p):
        for i, j, w in pr.edges:
            c_.exp1(i, j, unitary=tc.gates._zz_matrix, theta=-par[layer] * w / 2.0)
        for q in range(pr.k):
            c_.rx(q, theta=2.0 * par[p + layer])
    a, b = pr.local_term_nodes
    zz = c_.expectation((tc.gates.z(), [a]), (tc.gates.z(), [b]))
    v = pr.weight * 0.5 * (1.0 - tc.backend.real(zz))
    tot = v if tot is None else tot + v
tcv = float(tot.detach())

print("dense float64 reference   %+.14f" % ref)
print("LC complex128             %+.14f   |diff| %.3e" % (lc128, abs(lc128 - ref)))
print("LC complex64              %+.14f   |diff| %.3e" % (lc64, abs(lc64 - ref)))
print("TC-NG complex128          %+.14f   |diff| %.3e" % (tcv, abs(tcv - ref)))
