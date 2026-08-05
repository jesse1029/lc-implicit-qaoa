# LC vs TensorCircuit-NG

The paper positions TensorCircuit-NG as the closest complementary route and
deliberately does **not** claim to supersede it: large low-width cones are stated
to remain its regime. These scripts turn that sentence into measurements.

TensorCircuit-NG is not a baseline in the paper's runtime table, so nothing here
revises a published number. It is an independent comparison added after the fact.

## Order matters

`stage_a_agreement.py` must pass before any timing is quoted. A stopwatch race
against a peer that computes a different quantity is meaningless, so Stage A
checks that LC and TC-NG agree on the objective **and** on all `2p` shared
parameter gradients, in complex128, against the released light-cone
decomposition.

Expected result: agreement at `1e-15`.

## The scripts

| script | what it answers |
|---|---|
| `stage_0_cone_width_scan.py` | For every cone, how far does treewidth `w` sit below cone size `k`? Contraction costs ~`2^w`, dense local state ~`2^k`. If `w ~ k` everywhere, a hybrid is pointless. |
| `stage_a_agreement.py` | Do LC and TC-NG compute the same objective and the same gradients? Gate for everything below. |
| `stage_b_table7_timing.py` | On the paper's Table 7 instances, on one GPU, how do the two compare on objective and on objective-plus-gradient? |
| `stage_c_crossover.py` | Where does LC stop winning, and where does the `k <= 24` guardrail force an outright rejection? |
| `run_all_gpu.py` | All of the above on a CUDA box, with TC-NG under either the `pytorch` or the `jax` backend. |
| `tc_hadamard_precision_probe.py` | Isolates the Hadamard-constant precision issue described below. |

## Instances

Reconstructed with the paper's own recipe, from
`scripts/run_official_regime_matrix.py`:

```
seed   = 73000 + 97*n + 13*p + FAMILY_SEED_OFFSET[family]
gammas = [0.20 + 0.05*i for i in range(p)]
betas  = [0.32 - 0.035*i for i in range(p)]
```

so the graphs are the same ones behind the published table.

## Two things that will bite you

**TensorCircuit-NG caches its Hadamard constant at float32 accuracy** and widens
it to complex128. `|H|00|^2` evaluates to `0.4999999828857291` rather than `0.5`,
which alone costs about `3e-7` in the objective — enough to fail an agreement
check for a reason unrelated to either method. Stage A therefore feeds the exact
uniform input state. Timing runs use the idiomatic `c.h(q)` construction instead,
because an explicit dense input tensor would destroy the factorized structure the
contractor needs. `tc_hadamard_precision_probe.py` demonstrates both sides.

**JAX defaults to float32** and silently truncates complex128, so the agreement
gate fails unless x64 is enabled before JAX is imported. `run_all_gpu.py` sets it
and verifies it took effect rather than trusting the environment variable.

## Running

```bash
pip install tensorcircuit-ng
# plus ONE backend:
pip install "jax[cuda12]"          # TC-NG's fastest path; Linux only
pip install torch                  # portable, but tc.backend.jit is a no-op here
pip install cupy-cuda12x           # LC's own GPU backend

python benchmarks/tensorcircuit_ng/run_all_gpu.py \
    --code . --stage ABC --backend jax --gpu 0 --repeats 5 --out results.json
```

Keep JAX and PyTorch in **separate environments**. Installing torch on top of a
JAX CUDA environment replaces `nvidia-cudnn-cu12` with an incompatible version
and breaks JAX.

## Reporting

Two outcomes must not be conflated:

- **slower** — both methods answer and one is faster; a ratio is meaningful
- **rejected** — `k_max` exceeds LC's guardrail, so LC returns no value at all;
  there is no ratio, only a capability gap

`stage_c_crossover.py` keeps them in separate columns.
