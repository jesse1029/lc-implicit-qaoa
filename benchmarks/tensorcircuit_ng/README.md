# LC-Implicit-QAOA vs TensorCircuit-NG

The paper positions TensorCircuit-NG as *"the closest complementary route, not a
baseline LC supersedes"* and does not time it. This directory does time it, on
one RTX 3090, with both methods on the same graphs.

**Headline: per query TC-NG is faster; including its compile, LC is faster on
the query that matters.** Which one wins depends entirely on how many queries
you issue per instance, and the crossover is far past what a QAOA optimizer runs.

## What had to be fixed before any number meant anything

Five earlier attempts produced wrong numbers, each for a specific reason. They
are listed because each one is a way to accidentally rig this comparison:

1. **Wrong backend.** `tc.backend.jit` is a no-op on TensorCircuit-NG's PyTorch
   backend (measured: 1.245 s → 1.232 s). Benchmarking there understates TC-NG
   by ~100×. JAX + jit is its real fast path.
2. **Skipped agreement gate.** Timings were once reported for a configuration
   whose correctness check had been skipped. The gate now runs first and the
   script exits non-zero if it fails.
3. **Unequal precision.** At the same nominal `complex64`, TC-NG's contraction
   carries 9e-05 to 1.5e-03 relative error against LC's 7e-07 — 100× to 2000×
   worse. Comparing them at equal dtype is not comparing them at equal accuracy.
   (`jax_default_matmul_precision='highest'` changes nothing; it is contraction
   order, not TF32.)
4. **Hidden compile cost.** Only steady-state time was recorded, which is the
   half of the story that favours TC-NG.
5. **Asymmetric setup.** LC re-extracted light cones inside every timed call
   while TC-NG had them precomputed and traced into its graph. Extraction is now
   timed separately.

A sixth issue is TC-NG's own: it caches its Hadamard constant at float32
accuracy, so the idiomatic `c.h(q)` construction has a **~1e-7 accuracy floor
regardless of dtype**. Feeding the exact uniform state removes it and agrees
with LC to 4e-16 — that is the agreement gate — but it destroys the factorized
structure the contractor needs, so timing uses the idiomatic form.

## Accuracy

| method | dtype | relative error |
|---|---|---|
| LC | complex64 | 6.8e-07 – 9.8e-07 |
| **TC-NG** | **complex64** | **9.2e-05 – 1.5e-03** |
| TC-NG | complex128 | 9.5e-08 – 4.6e-07 |

TC-NG cannot buy speed with `complex64` here — the error is 100× to 2000× worse
than LC's. Everything below therefore compares **LC complex64 against TC-NG
complex128**, the cheapest configuration at comparable accuracy.

## Per-query time, accuracy-matched

| case | k_max | query | LC | TC-NG | TC-NG faster by |
|---|---:|---|---:|---:|---:|
| 3-regular n=24 | 14 | objective | 0.0773 s | 0.0253 s | 3.1× |
| 3-regular n=24 | 14 | obj+grad | 0.1639 s | 0.1095 s | 1.5× |
| ER deg-2 n=128 | 22 | objective | 0.4461 s | 0.1474 s | 3.0× |
| ER deg-2 n=128 | 22 | obj+grad | 1.0193 s | 0.3692 s | 2.8× |
| ER deg-3 n=24 | 23 | objective | 0.3919 s | 0.2754 s | 1.4× |

In steady state TC-NG wins every cell.

## Compile cost, and where the crossover actually is

TC-NG's steady-state speed is bought with an XLA compile that must be paid once
per circuit structure — that is, **once per QUBO instance**, not once per run.

| case | query | TC-NG compile | break-even |
|---|---|---:|---:|
| 3-regular n=24 | objective | 24.8 s | 477 queries |
| ER deg-2 n=128 | objective | 67.6 s | 227 queries |
| ER deg-3 n=24 | objective | 57.7 s | 496 queries |
| 3-regular n=24 | **obj+grad** | **143.4 s** | **2636 queries** |
| ER deg-2 n=128 | **obj+grad** | **1008.4 s** | **1552 queries** |

End-to-end on ER deg-2 n=128, objective **and gradient** — the query this paper
is about:

| queries issued | LC | TC-NG | |
|---:|---:|---:|---|
| 200 | 204 s | 1082 s | **LC 5.3× faster** |
| 500 | 510 s | 1193 s | **LC 2.3× faster** |
| 1552 | 1582 s | 1581 s | crossover |

## Reading this honestly

**LC does not compute faster than TC-NG.** It computes *sooner*, because it has
nothing to compile, and it stays accurate at half the precision.

The advantage is real only where instance count is high relative to query count
— which is the normal case for QAOA parameter search, where each new graph is a
new circuit structure and a new 2-to-17-minute compile.

Where LC has no advantage at all: a single instance queried thousands of times.
There TC-NG is simply better, and the paper's positioning of it as a
complementary route rather than a superseded baseline is the correct one.

**Neither method covers k > 24.** LC rejects before allocating; TC-NG fails by
OOM mid-run, requesting up to 131,072 GiB on scale-free cones, and enabling
cotengra with an 8 GiB slice target rescued exactly one of nine such cells.

## Reproducing

`final_bench.py` in the repository root of the benchmark run; raw output in
[`results_rtx3090.json`](results_rtx3090.json). Hardware: RTX 3090 24 GB,
TensorCircuit-NG 1.8.0, JAX 0.10.2 (CUDA 12), CuPy 14.1.1, `JAX_ENABLE_X64=1`.

One row (ER deg-3 n=24, obj+grad) is still compiling at the time of writing and
is absent from the JSON.
