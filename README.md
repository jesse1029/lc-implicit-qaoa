# LC-Implicit-QAOA

Exact objective **and** adjoint-gradient evaluation for QUBO-QAOA over bounded
light cones, with a byte-budgeted fit-or-reject execution contract.

Reference implementation for *LC-Implicit-QAOA: Active-Workspace-Capped Exact
Objective-and-Gradient Evaluation for Training over Bounded QUBO Light Cones*
(Chih-Chung Hsu, National Yang Ming Chiao Tung University).

## What this is

A QAOA optimizer issues hundreds of exact value and gradient queries before it
returns useful angles, so the evaluator sits inside the optimization loop and its
memory footprint decides whether an instance is trainable at all. At fixed depth
`p`, each cost term depends only on its `p`-hop neighborhood, so the cost of an
exact query is governed by the **cone size** `k` rather than by the qubit count
`n`.

`lcqaoa` never materializes a global state or a global cost table. It profiles
cone structure first, then plans equal-`k` microbatches and a checkpoint schedule
against a declared workspace budget `M`, and **rejects the request before
allocation** when no plan fits. "Implicit" means omitting the global state and
global cost table — not implicit differentiation.

Scope, stated plainly: fixed-depth one- and two-local diagonal QUBO costs with a
transverse-field mixer. It provides neither global states, nor sampling, nor a
hardware-independent fastest-backend rule.

## Install

```bash
python -m pip install -e .
python -m pip install cupy-cuda12x        # optional GPU backend
```

```bash
python tests/test_api_contract.py
python scripts/run_smoke.py
```

The public objective and adjoint APIs validate the requested QAOA depth against
the gamma and beta layers, include the graph's constant offset in objective
values, and expose byte-budgeted checkpoint planning through
`checkpoint_policy="budgeted"` and `memory_budget_bytes=M`.

## Quick use

```python
import numpy as np
from lcqaoa.graphs import random_regular_graph
from lcqaoa.lightcone import lightcone_expectation, lightcone_gradient_adjoint

g = random_regular_graph(128, 3, seed=0)
gammas, betas = [0.20, 0.25], [0.32, 0.285]

val  = lightcone_expectation(g, gammas, betas, p=2, prefer_gpu=True)
grad = lightcone_gradient_adjoint(g, gammas, betas, p=2, prefer_gpu=True)

print(val.status, val.value)      # "ok", or a rejection naming the offending k
print(grad.gradient)              # all 2p shared-parameter derivatives
```

A request whose largest cone exceeds the guardrail returns a rejection status
rather than a number. That is the contract, not a failure.

## Reproduction entry points

- `scripts/run_P0_1_microbatch_memory.py` — microbatch time/memory sweep
- `scripts/run_P0_2_float64_gradient_correctness.py` — independent float64 correctness study
- `scripts/run_P0_2_official_cuaoa_gradient.py` — matched CUAOA value-plus-gradient comparison
- `scripts/run_P0_2_lightning_gpu_adjoint.py` — matched PennyLane-Lightning-GPU complex64/float32 comparison
- `scripts/run_comparator_regime.py` — multi-backend objective/status regime measurements
- `scripts/run_official_regime_matrix.py` — the instance generator behind the published tables
- `scripts/make_P0_4_p2_regime_summary.py` — family-separated `p=2` regime summary
- `scripts/make_P0_5_optimizer_stats.py` — paired optimizer analysis
- `scripts/run_P0_1_heldout_predictor.py` — held-out cost-model validation
- `scripts/run_P1_1_qtensor_precision_diagnostic.py` — QTensor precision and plan-reuse diagnosis
- `scripts/run_cuaoa_clean_provenance.sh` — clean upstream build attempt and CUDA compatibility log
- `scripts/run_cuaoa_documented_patch_build.sh` — isolated compatibility build with wheel and source hashes
- `scripts/analyze_official_crossover_followup.py` — paired crossover and repeated-query break-even analysis
- `scripts/validate_crossover_provenance_package.py` — comparator provenance validation
- `figure_generation/gen_evidence_figure.py` — Figure 2 from released CSV measurements

Every driver records completed, allocation-failed, timed-out, unsupported, and
policy-skipped statuses explicitly. Comparator-specific dependencies and
environment records ship with their raw runs.

## TensorCircuit-NG comparison

`benchmarks/tensorcircuit_ng/` holds an independent comparison against
TensorCircuit-NG, added after the paper was frozen. TensorCircuit-NG is **not** a
baseline in the paper's runtime table — the paper treats it as the closest
complementary route and concedes large low-width cones as its regime — so nothing
there revises a published number.

The scripts are ordered so that an agreement gate on the objective and on all
`2p` gradients must pass before any timing is quoted. See that directory's README
for the two precision traps involved.

## Citation

```bibtex
@misc{hsu2026lcimplicitqaoa,
  title  = {LC-Implicit-QAOA: Active-Workspace-Capped Exact Objective-and-Gradient
            Evaluation for Training over Bounded QUBO Light Cones},
  author = {Hsu, Chih-Chung},
  year   = {2026},
  eprint = {ARXIV_ID_PENDING},
  archivePrefix = {arXiv},
  primaryClass  = {quant-ph}
}
```

## License

MIT. See [LICENSE](LICENSE).
