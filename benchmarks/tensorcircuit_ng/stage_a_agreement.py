"""Stage A: does TensorCircuit-NG compute the SAME number as LC?

A timing race against a peer that computes a different quantity is worthless, so
agreement on both the objective and all 2p shared-parameter gradients is the gate
that must pass before any stopwatch is started.

TC-NG route implemented the way a TC-NG user would write it:
  - same light-cone decomposition (prior work, shared by both methods)
  - per-cone tc.Circuit with ZZ cost rotations and RX mixer
  - native c.expectation() for the term operator
  - torch autograd over the shared (gamma, beta) vector

Convention bridge (LC -> TC):
  LC maxcut cost per edge  w * (b_i XOR b_j)  ==  w * (1 - Z_i Z_j) / 2
  exp(-i*gamma*C) per edge ==  global phase * exp(+i*gamma*w/2 * Z_i Z_j)
  tc.exp1(i,j,unitary=ZZ,theta) == exp(-i*theta*ZZ)   ->  theta = -gamma*w/2
  LC mixer exp(-i*beta*X)  ==  tc.rx(theta=2*beta)
"""
import os, sys, pathlib, math, numpy as np

CODE = os.environ.get("LCQAOA_CODE") or str(pathlib.Path(__file__).resolve().parents[2])
sys.path.insert(0, CODE)

import tensorcircuit as tc
import torch

tc.set_backend("pytorch")
tc.set_dtype("complex128")

from lcqaoa.graphs import random_regular_graph, erdos_renyi_graph
from lcqaoa.lightcone import (extract_lightcones, lightcone_expectation,
                              lightcone_gradient_adjoint)

# Correctness is device-independent; run the gate on CPU so tc's internally
# constructed gate tensors and our parameter tensor cannot land on different
# devices. Device placement is a timing-stage concern, handled there.
DEV = "cpu"


def tc_cone_value(problem, gammas, betas):
    """<C_t> on one light cone, via TensorCircuit-NG. torch tensors in/out."""
    k = problem.k
    # TensorCircuit-NG 1.8.0 caches its Hadamard constant at float32 accuracy and
    # widens it to complex128, which alone costs ~3e-7 in the objective. Feeding
    # the exact uniform state removes that artifact so the comparison measures the
    # method, not a library constant.
    c = tc.Circuit(k, inputs=torch.full((1 << k,), (1 << k) ** -0.5,
                                        dtype=torch.complex128, device=DEV))
    for layer in range(len(gammas)):
        for i, j, w in problem.edges:
            c.exp1(i, j, unitary=tc.gates._zz_matrix, theta=-gammas[layer] * w / 2.0)
        for q in range(k):
            c.rx(q, theta=2.0 * betas[layer])
    i, j = problem.local_term_nodes
    zz = c.expectation((tc.gates.z(), [i]), (tc.gates.z(), [j]))
    return problem.weight * 0.5 * (1.0 - tc.backend.real(zz))


def tc_objective(problems, params):
    gammas, betas = params[: len(params) // 2], params[len(params) // 2:]
    total = tc_cone_value(problems[0], gammas, betas)
    for pr in problems[1:]:
        total = total + tc_cone_value(pr, gammas, betas)
    return total


def check(label, graph, p, gammas, betas):
    problems = extract_lightcones(graph, p)
    kmax = max(pr.k for pr in problems)

    lc_v = lightcone_expectation(graph, gammas, betas, p=p, prefer_gpu=False,
                                 complex_dtype=np.complex128, float_dtype=np.float64)
    lc_g = lightcone_gradient_adjoint(graph, gammas, betas, p=p, prefer_gpu=False,
                                      complex_dtype=np.complex128, float_dtype=np.float64)

    params = torch.tensor(list(gammas) + list(betas), dtype=torch.float64,
                          device=DEV, requires_grad=True)
    val = tc_objective(problems, params)
    val.backward()
    tc_v = float(val.detach().cpu())
    tc_g = params.grad.detach().cpu().numpy()

    dv = abs(tc_v - lc_v.value) / max(1.0, abs(lc_v.value))
    dg = np.max(np.abs(tc_g - lc_g.gradient)) / max(1.0, np.max(np.abs(lc_g.gradient)))
    print("%-28s n=%-4d p=%d cones=%-4d k_max=%-3d" % (label, graph.n, p, len(problems), kmax))
    print("      objective  LC %+.12f   TC-NG %+.12f   rel diff %.3e" % (lc_v.value, tc_v, dv))
    print("      gradient   LC %s" % np.array2string(lc_g.gradient, precision=6, max_line_width=200))
    print("                 TC %s" % np.array2string(tc_g, precision=6, max_line_width=200))
    print("      worst relative gradient diff %.3e   -> %s" % (dg, "AGREE" if max(dv, dg) < 1e-9 else "MISMATCH"))
    print()
    return max(dv, dg) < 1e-9


print("tensorcircuit-ng %s | torch %s | device %s\n" % (tc.__version__, torch.__version__, DEV))

ok = []
ok.append(check("3-regular n=10 p=1", random_regular_graph(10, 3, seed=7), 1,
                [0.31], [0.22]))
ok.append(check("3-regular n=10 p=2", random_regular_graph(10, 3, seed=7), 2,
                [0.31, -0.17], [0.22, 0.41]))
ok.append(check("3-regular n=14 p=2", random_regular_graph(14, 3, seed=11), 2,
                [0.63, 0.09], [-0.28, 0.55]))
ok.append(check("ER deg-2 n=16 p=2", erdos_renyi_graph(16, 2.0 / 16, seed=3), 2,
                [0.41, -0.33], [0.19, 0.62]))

print("=== Stage A gate: %s ===" % ("PASS - TC-NG computes the same quantity" if all(ok)
                                    else "FAIL - do not time a peer that disagrees"))
