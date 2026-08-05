from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


THEOREM_TEX = r"""
\paragraph{Theorem 1 (local light-cone exactness).}
Consider depth-$p$ QAOA with a diagonal Hamiltonian composed only of
one- and two-body $Z$-basis QUBO terms and the standard transverse-field
mixer $B=\sum_i X_i$. Let $C_t$ be a nonconstant local term with support
$S_t$, and let $L_p(t)$ be the $p$-hop neighborhood of $S_t$ in the QUBO
interaction graph. Then the expectation of $C_t$ after $p$ QAOA layers is
determined by the induced subgraph on $L_p(t)$:
\[
  \langle C_t\rangle_G = \langle C_t\rangle_{G[L_p(t)]}.
\]
The local circuit includes all one-body fields and two-body interactions
fully supported inside the induced cone. Constant offsets from the
binary-to-Ising transform are added to the reported absolute objective
separately and do not affect gradients or operator support.

\paragraph{Proof sketch.}
Work in the Heisenberg picture and set $R_0=S_t$. A mixer factor acting on a
single qubit does not move the support of an operator to another qubit. A
two-body diagonal cost factor can enlarge support only across an edge incident
to the current support. Thus after the $r$-th backward cost layer the support
is contained in $R_r=N(R_{r-1})$, and after $p$ layers it is contained in
$L_p(t)$. Cost factors disjoint from the current support commute through the
observable and cancel with their adjoints; boundary-crossing factors outside
the induced cone never become incident before the support reaches the boundary
and therefore cancel as well.

\paragraph{Theorem 2 (work and active memory).}
Let $k_t=|L_p(t)|$. Up to batching overhead, exact LC objective evaluation
requires
\[
  \sum_{t\in T} O\!\left(p\bigl(|E[L_p(t)]|+k_t\bigr)2^{k_t}\right)
\]
work. With a size-$k$ active batch of $b_k$ cones, the objective workspace is
$O(b_k2^k)$ amplitudes plus local cost and term tables. The adjoint-gradient
workspace is $O(b_km_k2^k)$ if all forward states for $m_k$ local operations
are cached, or lower memory with recomputation at increased time.

\paragraph{Theorem 3 (adjoint-gradient exactness).}
For exact arithmetic, the local adjoint pass returns
$\partial\langle C_t\rangle_{G[L_p(t)]}/\partial\theta$ for the same LC
objective in Theorem 1. Summing over all terms gives the exact gradient of the
LC-decomposed QUBO-QAOA objective. Numerical implementations retain the exact
decomposition but incur floating-point error; finite-difference checks also
include step-size and cancellation error.

\paragraph{Boundary.}
These statements do not cover non-standard mixers, higher-order terms before
quadratization, dense cardinality penalties unless represented as local
terms, or final sampling distributions. Sparsity alone is insufficient: a
star graph has $n-1$ edges, but a hub-edge term has $L_1(t)=V$.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "benchmark_suite_20260704" / "A10_method_theorems")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "A10_theorem_blocks.tex").write_text(THEOREM_TEX.strip() + "\n", encoding="utf-8")
    (args.out_dir / "A10_method_notes.md").write_text(
        "# A10 Method-Level Blocks\n\n"
        "This file contains manuscript-ready theorem text for the main paper or supplement.\n\n"
        "- Theorem 1 states exact local light-cone expectation under one-/two-body diagonal QUBO terms and standard transverse-field mixer.\n"
        "- Theorem 2 states work and active-memory bounds using k_t and active batch size.\n"
        "- Theorem 3 states adjoint-gradient exactness in exact arithmetic.\n"
        "- The boundary paragraph explicitly excludes non-standard mixers, higher-order terms before quadratization, dense penalties, and final sampling.\n",
        encoding="utf-8",
    )
    print(f"WROTE {args.out_dir / 'A10_theorem_blocks.tex'}")


if __name__ == "__main__":
    main()
