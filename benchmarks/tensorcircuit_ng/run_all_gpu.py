"""LC vs TensorCircuit-NG on the paper's own RTX 3090 box.

Runs three things, in the order that makes the timings meaningful:

  A. agreement gate  - LC and TC-NG must compute the same objective and the same
                       2p shared-parameter gradients before any stopwatch starts
  B. Table 7 timing  - the paper's exact instances, both methods on one GPU,
                       TC-NG under BOTH backends:
                         pytorch          (no working jit)
                         jax + jit        (TC-NG's flagship path)
  C. crossover       - large low-width cones: where LC loses, and where LC's
                       k<=24 guardrail forces an outright rejection

Instance recipe is the paper's own (scripts/run_official_regime_matrix.py):
    seed = 73000 + 97*n + 13*p + FAMILY_SEED_OFFSET[family]
    gammas = [0.20 + 0.05*i], betas = [0.32 - 0.035*i]
"""
import argparse, importlib, json, math, os, statistics as st, sys, time

ap = argparse.ArgumentParser()
ap.add_argument("--code", required=True, help="path to the released lcqaoa code root")
ap.add_argument("--stage", default="ABC")
ap.add_argument("--backend", default="pytorch", choices=["pytorch", "jax"])
ap.add_argument("--repeats", type=int, default=5)
ap.add_argument("--gpu", default="0")
ap.add_argument("--out", default="")
args = ap.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
# JAX defaults to float32 and silently downcasts complex128, which makes the
# Stage A agreement gate fail for a reason that has nothing to do with either
# method. This must be set before jax is imported.
if args.backend == "jax":
    os.environ["JAX_ENABLE_X64"] = "1"
sys.path.insert(0, args.code)

import numpy as np
import tensorcircuit as tc

tc.set_backend(args.backend)
tc.set_dtype("complex64")

if args.backend == "jax":
    import jax
    def sync(x):
        return jax.block_until_ready(x)
    def to_param(v):
        return tc.backend.convert_to_tensor(np.asarray(v, dtype=np.float32))
    DEVNAME = str(jax.devices()[0])
else:
    import torch
    torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")
    def sync(x):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return x
    def to_param(v):
        return torch.tensor(list(v), dtype=torch.float32,
                            device="cuda" if torch.cuda.is_available() else "cpu")
    DEVNAME = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"

from lcqaoa.graphs import (erdos_renyi_graph, modular_graph, random_regular_graph,
                           scale_free_graph)
from lcqaoa.lightcone import (extract_lightcones, lightcone_expectation,
                              lightcone_gradient_adjoint)

FAMILY_SEED_OFFSET = {"3regular": 101, "er_deg2": 202, "er_deg3": 303, "modular_sparse": 404}
GUARDRAIL = 24


def make(family, n, seed):
    if family == "3regular":
        return random_regular_graph(n, 3, seed=seed)
    if family == "er_deg2":
        return erdos_renyi_graph(n, min(0.45, 2.0 / max(2, n)), seed=seed)
    if family == "er_deg3":
        return erdos_renyi_graph(n, min(0.45, 3.0 / max(2, n)), seed=seed)
    if family == "modular_sparse":
        return modular_graph(n, modules=max(2, min(4, n // 4)), p_in=0.28, p_out=0.015, seed=seed)
    if family == "scale_free":
        return scale_free_graph(n, attachment=2, seed=seed)
    raise ValueError(family)


def case(family, n, p):
    seed = 73000 + 97 * n + 13 * p + FAMILY_SEED_OFFSET.get(family, 0)
    g = make(family, n, seed)
    return g, [0.20 + 0.05 * i for i in range(p)], [0.32 - 0.035 * i for i in range(p)]


def build_energy(problems, p, exact_input=False):
    def energy(params):
        g_, b_ = params[:p], params[p:]
        total = 0.0
        for pr in problems:
            if exact_input:
                amp = np.full(1 << pr.k, (1 << pr.k) ** -0.5, dtype=np.complex64)
                c = tc.Circuit(pr.k, inputs=tc.backend.convert_to_tensor(amp))
            else:
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
            total = total + pr.weight * 0.5 * (1.0 - tc.backend.real(zz))
        return tc.backend.real(total)
    return energy


def timed(fn, repeats):
    t0 = time.perf_counter(); out = sync(fn()); cold = time.perf_counter() - t0
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter(); sync(fn()); ts.append(time.perf_counter() - t0)
    return cold, st.median(ts), out


results = {"host_gpu": DEVNAME, "tc_backend": args.backend, "tc_version": tc.__version__,
           "stageA": [], "stageB": [], "stageC": []}
print("GPU %s | tc %s | backend %s | complex64\n" % (DEVNAME, tc.__version__, args.backend))

# ----------------------------------------------------------------- Stage A
if "A" in args.stage:
    print("=== Stage A: agreement gate (complex128) ===")
    tc.set_dtype("complex128")
    for family, n, p in [("3regular", 10, 1), ("3regular", 14, 2), ("er_deg2", 16, 2)]:
        g, gam, bet = case(family, n, p)
        problems = extract_lightcones(g, p)
        lc_v = lightcone_expectation(g, gam, bet, p=p, prefer_gpu=False,
                                     complex_dtype=np.complex128, float_dtype=np.float64)
        lc_g = lightcone_gradient_adjoint(g, gam, bet, p=p, prefer_gpu=False,
                                          complex_dtype=np.complex128, float_dtype=np.float64)
        en = build_energy(problems, p, exact_input=True)
        par = to_param(np.array(gam + bet, dtype=np.float64))
        vg = tc.backend.value_and_grad(en)
        v, gr = vg(par)
        v = float(np.asarray(sync(v))); gr = np.asarray(sync(gr), dtype=np.float64)
        dv = abs(v - lc_v.value) / max(1.0, abs(lc_v.value))
        dg = float(np.max(np.abs(gr - lc_g.gradient))) / max(1.0, float(np.max(np.abs(lc_g.gradient))))
        ok = max(dv, dg) < 1e-9
        print("  %-18s cones=%-3d k_max=%-3d  obj rel %.2e  grad rel %.2e  -> %s" %
              ("%s n=%d p=%d" % (family, n, p), len(problems),
               max(pr.k for pr in problems), dv, dg, "AGREE" if ok else "MISMATCH"))
        results["stageA"].append(dict(family=family, n=n, p=p, obj_rel=dv, grad_rel=dg, agree=ok))
    tc.set_dtype("complex64")
    if not all(r["agree"] for r in results["stageA"]):
        print("\nSTAGE A FAILED - refusing to report timings against a peer that disagrees")
        sys.exit(1)
    print()

# ----------------------------------------------------------------- Stage B
STAGE_B = [("3regular", 24, 2), ("3regular", 26, 2), ("3regular", 28, 2),
           ("3regular", 128, 2), ("er_deg2", 128, 2), ("er_deg3", 24, 2)]
if "B" in args.stage:
    print("=== Stage B: Table 7 instances, objective and objective+gradient ===")
    hdr = "%-20s %-6s %-6s %-9s %-11s %-11s %-9s %-11s %-11s %s" % (
        "case", "cones", "k_max", "agree", "LC obj", "TC obj", "obj x", "LC grad", "TC grad", "grad x")
    print(hdr); print("-" * len(hdr))
    for family, n, p in STAGE_B:
        g, gam, bet = case(family, n, p)
        problems = extract_lightcones(g, p)
        kmax = max(pr.k for pr in problems)
        _, lc_o, r_o = timed(lambda: lightcone_expectation(
            g, gam, bet, p=p, prefer_gpu=True, complex_dtype=np.complex64,
            float_dtype=np.float32), args.repeats)
        _, lc_gr, _ = timed(lambda: lightcone_gradient_adjoint(
            g, gam, bet, p=p, prefer_gpu=True, complex_dtype=np.complex64,
            float_dtype=np.float32), args.repeats)
        en = build_energy(problems, p)
        par = to_param(gam + bet)
        f_obj = tc.backend.jit(en) if args.backend == "jax" else en
        f_vg = (tc.backend.jit(tc.backend.value_and_grad(en)) if args.backend == "jax"
                else tc.backend.value_and_grad(en))
        try:
            _, tc_o, v = timed(lambda: f_obj(par), args.repeats)
            tcval = float(np.asarray(sync(v)))
        except Exception as exc:
            tc_o, tcval = float("nan"), float("nan"); print("   TC obj failed:", str(exc)[:110])
        try:
            _, tc_gr, _ = timed(lambda: f_vg(par), args.repeats)
        except Exception as exc:
            tc_gr = float("nan"); print("   TC grad failed:", str(exc)[:110])
        rel = abs(tcval - r_o.value) / max(1.0, abs(r_o.value))
        print("%-20s %-6d %-6d %-9.1e %-11.4f %-11.4f %-9.2f %-11.4f %-11.4f %.2f" %
              ("%s n=%d p=%d" % (family, n, p), len(problems), kmax, rel,
               lc_o, tc_o, tc_o / lc_o, lc_gr, tc_gr, tc_gr / lc_gr))
        results["stageB"].append(dict(family=family, n=n, p=p, cones=len(problems), kmax=kmax,
                                      rel_diff=rel, lc_obj=lc_o, tc_obj=tc_o,
                                      lc_grad=lc_gr, tc_grad=tc_gr,
                                      obj_ratio=tc_o / lc_o, grad_ratio=tc_gr / lc_gr))
    print()

# ----------------------------------------------------------------- Stage C
STAGE_C = [("3regular", 64, 2), ("er_deg3", 64, 2), ("er_deg3", 128, 2),
           ("modular_sparse", 64, 2), ("modular_sparse", 128, 1),
           ("scale_free", 24, 2), ("scale_free", 64, 1), ("scale_free", 64, 2),
           ("scale_free", 128, 1)]
if "C" in args.stage:
    print("=== Stage C: crossover and the cells LC rejects ===")
    hdr = "%-24s %-6s %-6s %-9s %-11s %-11s %s" % ("case", "cones", "k_max", "LC", "LC s", "TC s", "verdict")
    print(hdr); print("-" * len(hdr))
    for family, n, p in STAGE_C:
        g, gam, bet = case(family, n, p)
        problems = extract_lightcones(g, p)
        kmax = max(pr.k for pr in problems)
        lc_s, lc_state = float("nan"), "?"
        try:
            _, lc_s, r = timed(lambda: lightcone_expectation(
                g, gam, bet, p=p, prefer_gpu=True, complex_dtype=np.complex64,
                float_dtype=np.float32), args.repeats)
            lc_state = "ok" if r.status == "ok" else "REJECT"
            if lc_state == "REJECT":
                lc_s = float("nan")
        except Exception as exc:
            lc_state = "ERR"
        en = build_energy(problems, p)
        par = to_param(gam + bet)
        f_obj = tc.backend.jit(en) if args.backend == "jax" else en
        tc_s = float("nan")
        try:
            _, tc_s, _ = timed(lambda: f_obj(par), max(1, args.repeats // 2))
        except Exception as exc:
            print("   TC failed on %s n=%d p=%d: %s" % (family, n, p, str(exc)[:110]))
        if lc_state == "REJECT":
            verdict = "LC REJECTS (k>%d); TC-NG answers" % GUARDRAIL if tc_s == tc_s else "both unavailable"
            ratio = float("nan")
        elif tc_s == tc_s and lc_s == lc_s:
            ratio = tc_s / lc_s
            verdict = "LC %.1fx faster" % ratio if ratio > 1 else "TC-NG %.1fx faster" % (1 / ratio)
        else:
            ratio, verdict = float("nan"), "incomplete"
        print("%-24s %-6d %-6d %-9s %-11s %-11s %s" %
              ("%s n=%d p=%d" % (family, n, p), len(problems), kmax, lc_state,
               "%.4f" % lc_s if lc_s == lc_s else "--",
               "%.4f" % tc_s if tc_s == tc_s else "--", verdict))
        results["stageC"].append(dict(family=family, n=n, p=p, cones=len(problems), kmax=kmax,
                                      lc_state=lc_state, lc_s=lc_s, tc_s=tc_s,
                                      ratio=ratio, verdict=verdict))

if args.out:
    open(args.out, "w").write(json.dumps(results, indent=2, default=str))
    print("\nwrote", args.out)
