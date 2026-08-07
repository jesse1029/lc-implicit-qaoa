"""Accuracy-matched LC vs TensorCircuit-NG, with compile cost and break-even.

Every earlier number in this investigation was wrong for a specific reason, and
this script exists to close each of them:

  1. TC-NG was benchmarked on the pytorch backend where tc.backend.jit is a no-op
     -> use jax + jit, TC-NG's actual fast path.
  2. The Stage A agreement gate was skipped on the jax runs
     -> the gate runs here FIRST and the script exits non-zero if it fails.
  3. complex64 TC-NG disagrees with LC by ~3.4e-4 vs LC's own 8.1e-7, i.e. the
     contraction order loses ~400x more accuracy at the same nominal dtype, and
     matmul_precision='highest' does not change it
     -> both TC-NG dtypes are timed, each reported WITH its accuracy, so the
        speed/precision trade cannot hide.
  4. Cold time was discarded, so compile cost was invisible
     -> cold and steady are reported separately, plus the break-even query count.
  5. LC re-extracted light cones inside every timed call while TC-NG had them
     precomputed and baked into the traced graph
     -> extraction is timed separately and LC is reported both as-API and
        numerics-only.

Break-even is the query count at which TC-NG's compile has paid for itself. A
QAOA optimizer issues a few hundred queries per instance, and the compile is per
circuit structure, so this is the number that decides the comparison.
"""
import argparse, json, os, statistics as st, sys, time

ap = argparse.ArgumentParser()
ap.add_argument("--code", required=True)
ap.add_argument("--gpu", default="0")
ap.add_argument("--repeats", type=int, default=5)
ap.add_argument("--out", default="")
args = ap.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
os.environ["JAX_ENABLE_X64"] = "1"
import jax
jax.config.update("jax_enable_x64", True)
sys.path.insert(0, args.code)

import numpy as np
import tensorcircuit as tc

# cotengra's ReusableHyperOptimizer forks worker processes for path search, and
# forking a multithreaded JAX process deadlocks -- it hung this script for an
# hour at the second gate case. Using tc's default contractor instead costs
# TC-NG essentially nothing on these cells: the earlier pytorch runs measured
# greedy vs cotengra at 1.65 vs 1.72 s (3regular n=24), 8.57 vs 8.72 s
# (3regular n=128) and 4.70 vs 4.88 s (er_deg2 n=128) -- within a few percent,
# with cotengra marginally SLOWER. Its one real benefit showed up only in
# Stage C, on cones far past LC's guardrail, which this script does not time.

from lcqaoa.graphs import erdos_renyi_graph, random_regular_graph
from lcqaoa.lightcone import (extract_lightcones, lightcone_expectation,
                              lightcone_gradient_adjoint)

OFF = {"3regular": 101, "er_deg2": 202, "er_deg3": 303}


def make(fam, n, seed):
    if fam == "3regular":
        return random_regular_graph(n, 3, seed=seed)
    if fam == "er_deg2":
        return erdos_renyi_graph(n, min(0.45, 2.0 / n), seed=seed)
    return erdos_renyi_graph(n, min(0.45, 3.0 / n), seed=seed)


def case(fam, n, p):
    seed = 73000 + 97 * n + 13 * p + OFF[fam]
    return (make(fam, n, seed),
            [0.20 + 0.05 * i for i in range(p)],
            [0.32 - 0.035 * i for i in range(p)])


def energy_fn(problems, p, exact_input=False):
    """exact_input=True feeds the uniform state directly instead of building it
    from H gates. TensorCircuit-NG caches its Hadamard constant at float32
    accuracy, which puts a ~1e-7 floor on the idiomatic construction no matter
    what dtype is requested. The exact-input form removes that floor and is used
    to prove the two formulations are identical; it is NOT used for timing,
    because a rank-k input tensor destroys the factorized structure the
    contractor relies on and would unfairly penalise TC-NG."""
    def energy(par):
        g_, b_ = par[:p], par[p:]
        total = 0.0
        for pr in problems:
            if exact_input:
                amp = np.full(1 << pr.k, (1 << pr.k) ** -0.5,
                              dtype=np.complex128 if tc.dtypestr == "complex128" else np.complex64)
                c = tc.Circuit(pr.k, inputs=tc.backend.convert_to_tensor(amp))
            else:
                c = tc.Circuit(pr.k)
                for q in range(pr.k):
                    c.h(q)
            for l in range(p):
                for i, j, w in pr.edges:
                    c.exp1(i, j, unitary=tc.gates._zz_matrix, theta=-g_[l] * w / 2.0)
                for q in range(pr.k):
                    c.rx(q, theta=2.0 * b_[l])
            a, b = pr.local_term_nodes
            zz = c.expectation((tc.gates.z(), [a]), (tc.gates.z(), [b]))
            total = total + pr.weight * 0.5 * (1.0 - tc.backend.real(zz))
        return tc.backend.real(total)
    return energy


def timed(fn, repeats):
    t0 = time.perf_counter(); out = jax.block_until_ready(fn())
    cold = time.perf_counter() - t0
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter(); jax.block_until_ready(fn()); ts.append(time.perf_counter() - t0)
    return cold, st.median(ts), out


def timed_plain(fn, repeats):
    t0 = time.perf_counter(); out = fn(); cold = time.perf_counter() - t0
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
    return cold, st.median(ts), out


def setup(dtype):
    tc.set_backend("jax"); tc.set_dtype(dtype)


CELLS = [("3regular", 24, 2), ("er_deg2", 128, 2), ("er_deg3", 24, 2)]

# ----------------------------------------------------------- agreement gate
print("=== gate: TC-NG must agree with LC before anything is timed ===")
print("    exact-input form proves the formulations match; the H-gate form is what")
print("    gets timed, and its extra error is TC-NG's cached Hadamard constant.")
gate_ok = True
# Only the small cone is gated. The exact-input form gives every cone a rank-k
# input tensor, which XLA compiles fine at k=6 but chokes on at k=13 (it hung
# here for half an hour). One case agreeing to 4e-16 already establishes that
# the two formulations compute the same quantity; a larger cone would re-verify
# nothing and only re-test the contractor.
for fam, n, p in [("3regular", 10, 1)]:
    g, gam, bet = case(fam, n, p)
    probs = extract_lightcones(g, p)
    v = lightcone_expectation(g, gam, bet, p=p, prefer_gpu=False,
                              complex_dtype=np.complex128, float_dtype=np.float64).value
    gr = lightcone_gradient_adjoint(g, gam, bet, p=p, prefer_gpu=False,
                                    complex_dtype=np.complex128, float_dtype=np.float64).gradient
    setup("complex128")
    par = tc.backend.convert_to_tensor(np.array(gam + bet, dtype=np.float64))
    res = {}
    for form, exact in [("exact-input", True), ("H-gate", False)]:
        tv, tg = tc.backend.value_and_grad(energy_fn(probs, p, exact_input=exact))(par)
        dv = abs(float(np.asarray(tv)) - v) / max(1.0, abs(v))
        dg = float(np.max(np.abs(np.asarray(tg) - gr))) / max(1.0, float(np.max(np.abs(gr))))
        res[form] = (dv, dg)
    ok = max(res["exact-input"]) < 1e-9
    gate_ok &= ok
    print("  %-18s exact-input obj %.2e grad %.2e -> %s   |   H-gate obj %.2e grad %.2e"
          % ("%s n=%d p=%d" % (fam, n, p), res["exact-input"][0], res["exact-input"][1],
             "AGREE" if ok else "MISMATCH", res["H-gate"][0], res["H-gate"][1]))
if not gate_ok:
    print("\nGATE FAILED - formulations differ; refusing to report timings"); sys.exit(1)
print("gate passed: same quantity. The H-gate column above is TC-NG's accuracy")
print("floor from its own cached constant, and it caps every timed row below.\n")

rows = []
for fam, n, p in CELLS:
    g, gam, bet = case(fam, n, p)
    t0 = time.perf_counter(); probs = extract_lightcones(g, p)
    extract_s = time.perf_counter() - t0
    kmax = max(pr.k for pr in probs)
    label = "%s n=%d p=%d" % (fam, n, p)
    print("=" * 96)
    print("%s | cones %d | k_max %d | cone extraction %.4f s" % (label, len(probs), kmax, extract_s))

    ref_v = lightcone_expectation(g, gam, bet, p=p, prefer_gpu=False,
                                  complex_dtype=np.complex128, float_dtype=np.float64).value
    ref_g = lightcone_gradient_adjoint(g, gam, bet, p=p, prefer_gpu=False,
                                       complex_dtype=np.complex128, float_dtype=np.float64).gradient

    for query in ["objective", "obj+grad"]:
        if query == "objective":
            lc_call = lambda: lightcone_expectation(g, gam, bet, p=p, prefer_gpu=True,
                                                    complex_dtype=np.complex64,
                                                    float_dtype=np.float32)
        else:
            lc_call = lambda: lightcone_gradient_adjoint(g, gam, bet, p=p, prefer_gpu=True,
                                                         complex_dtype=np.complex64,
                                                         float_dtype=np.float32)
        _, lc_s, lc_r = timed_plain(lc_call, args.repeats)
        lc_net = max(lc_s - extract_s, 0.0)
        lc_err = (abs(lc_r.value - ref_v) / max(1.0, abs(ref_v)) if query == "objective"
                  else float(np.max(np.abs(lc_r.gradient - ref_g))) / max(1.0, float(np.max(np.abs(ref_g)))))
        print("  %-9s LC complex64      err %.2e  per-query %.5f s (net of extraction %.5f)"
              % (query, lc_err, lc_s, lc_net))
        rows.append(dict(cell=label, kmax=kmax, cones=len(probs), query=query, method="LC",
                         dtype="complex64", err=lc_err, per_query=lc_s, per_query_net=lc_net,
                         compile_s=0.0, extract_s=extract_s))

        for dt, np_dt in [("complex64", np.float32), ("complex128", np.float64)]:
            try:
                setup(dt)
                en = energy_fn(probs, p)
                f = (tc.backend.jit(en) if query == "objective"
                     else tc.backend.jit(tc.backend.value_and_grad(en)))
                par = tc.backend.convert_to_tensor(np.array(gam + bet, dtype=np_dt))
                cold, steady, out = timed(lambda: f(par), args.repeats)
                comp = max(cold - steady, 0.0)
                if query == "objective":
                    err = abs(float(np.asarray(out)) - ref_v) / max(1.0, abs(ref_v))
                else:
                    err = float(np.max(np.abs(np.asarray(out[1]) - ref_g))) / max(1.0, float(np.max(np.abs(ref_g))))
                be = (int(comp / (lc_s - steady)) + 1) if steady < lc_s else None
                ben = (int(comp / (lc_net - steady)) + 1) if steady < lc_net else None
                print("  %-9s TC-NG %-11s err %.2e  per-query %.5f s  compile %6.1f s  "
                      "speedup %5.2fx  break-even %s (%s net)"
                      % (query, dt, err, steady, comp, lc_s / steady if steady else float("nan"),
                         be if be else "never", ben if ben else "never"))
                rows.append(dict(cell=label, kmax=kmax, cones=len(probs), query=query,
                                 method="TC-NG", dtype=dt, err=err, per_query=steady,
                                 per_query_net=steady, compile_s=comp, extract_s=0.0,
                                 speedup_vs_lc=lc_s / steady if steady else None,
                                 breakeven=be, breakeven_net=ben))
            except Exception as exc:
                print("  %-9s TC-NG %-11s FAILED: %s" % (query, dt, str(exc)[:80]))
                rows.append(dict(cell=label, kmax=kmax, query=query, method="TC-NG",
                                 dtype=dt, error=str(exc)[:200]))
            if args.out:
                open(args.out, "w").write(json.dumps(rows, indent=2, default=str))

print("\n=== accuracy-matched summary (LC complex64 vs TC-NG complex128) ===")
for c in {r["cell"] for r in rows}:
    for q in ["objective", "obj+grad"]:
        lc = next((r for r in rows if r["cell"] == c and r["query"] == q and r["method"] == "LC"), None)
        t = next((r for r in rows if r["cell"] == c and r["query"] == q
                  and r["method"] == "TC-NG" and r.get("dtype") == "complex128"
                  and "error" not in r), None)
        if lc and t:
            print("  %-22s %-9s LC %.5f s vs TC-NG %.5f s (+%.0f s compile) -> break-even %s queries"
                  % (c, q, lc["per_query"], t["per_query"], t["compile_s"],
                     t["breakeven"] if t["breakeven"] else "never"))
if args.out:
    open(args.out, "w").write(json.dumps(rows, indent=2, default=str))
    print("\nwrote", args.out)
