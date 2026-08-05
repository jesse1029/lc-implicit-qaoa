from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Sequence
import math
import time

import numpy as np

from .backend import get_backend
from .graphs import Edge, Field, WeightedGraph
from .qaoa import (
    EvalStats,
    apply_mixer_batched_inplace,
    cost_table,
    term_table,
)


@dataclass(frozen=True)
class LightConeProblem:
    term_kind: str
    global_term_nodes: tuple[int, ...]
    local_term_nodes: tuple[int, ...]
    weight: float
    nodes: tuple[int, ...]
    edges: tuple[Edge, ...]
    fields: tuple[Field, ...]

    @property
    def k(self) -> int:
        return len(self.nodes)


@dataclass
class GradientStats(EvalStats):
    gradient: np.ndarray | None = None
    peak_allocated_bytes: int = 0
    memory_budget_bytes: int | None = None
    checkpoint_policy: str = "cache_all"
    checkpoint_plans: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class CheckpointPlan:
    policy: str
    batch_size: int
    checkpoint_layers: tuple[int, ...]
    predicted_active_bytes: int
    predicted_components: dict[str, int]
    memory_budget_bytes: int | None
    recompute_layer_steps: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "policy": self.policy,
            "batch_size": self.batch_size,
            "checkpoint_layers": list(self.checkpoint_layers),
            "predicted_active_bytes": self.predicted_active_bytes,
            "predicted_components": dict(self.predicted_components),
            "memory_budget_bytes": self.memory_budget_bytes,
            "recompute_layer_steps": self.recompute_layer_steps,
        }


@dataclass
class _MemoryTracker:
    backend: Any
    peak_allocated_bytes: int = 0
    peak_reserved_bytes: int = 0

    def sample(self) -> None:
        self.backend.sync()
        self.peak_allocated_bytes = max(
            self.peak_allocated_bytes, self.backend.allocated_memory_bytes()
        )
        self.peak_reserved_bytes = max(
            self.peak_reserved_bytes, self.backend.memory_pool_bytes()
        )


def lightcone_topology_signature(problem: LightConeProblem) -> tuple:
    edges = tuple(sorted((min(i, j), max(i, j)) for i, j, _ in problem.edges))
    fields = tuple(sorted(i for i, _ in problem.fields))
    return (
        problem.k,
        problem.term_kind,
        problem.local_term_nodes,
        edges,
        fields,
    )


def extract_lightcones(graph: WeightedGraph, p: int) -> list[LightConeProblem]:
    if p < 0:
        raise ValueError("p must be non-negative")
    problems: list[LightConeProblem] = []
    for i, j, w in graph.edges:
        nodes = graph.cone_nodes((i, j), p)
        mapping, edges, fields = graph.relabel_subgraph(nodes)
        problems.append(
            LightConeProblem(
                term_kind="edge",
                global_term_nodes=(i, j),
                local_term_nodes=(mapping[i], mapping[j]),
                weight=w,
                nodes=nodes,
                edges=edges,
                fields=fields,
            )
        )
    for i, w in graph.fields:
        nodes = graph.cone_nodes((i,), p)
        mapping, edges, fields = graph.relabel_subgraph(nodes)
        problems.append(
            LightConeProblem(
                term_kind="field",
                global_term_nodes=(i,),
                local_term_nodes=(mapping[i],),
                weight=w,
                nodes=nodes,
                edges=edges,
                fields=fields,
            )
        )
    return problems


def _split_batches(items: list[LightConeProblem], max_batch_states: int) -> list[list[LightConeProblem]]:
    batches: list[list[LightConeProblem]] = []
    current: list[LightConeProblem] = []
    current_states = 0
    for item in items:
        states = 1 << item.k
        if current and current_states + states > max_batch_states:
            batches.append(current)
            current = []
            current_states = 0
        current.append(item)
        current_states += states
    if current:
        batches.append(current)
    return batches


def _split_batches_by_count(
    items: list[LightConeProblem], batch_size: int
) -> list[list[LightConeProblem]]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    return [items[start : start + batch_size] for start in range(0, len(items), batch_size)]


def _dtype_bytes(dtype) -> int:
    return int(np.dtype(dtype).itemsize)


def _recompute_layer_steps(p: int, checkpoint_layers: tuple[int, ...]) -> int:
    steps = 0
    for target in range(p):
        previous = [layer for layer in checkpoint_layers if layer < target]
        start = max(previous) if previous else -1
        steps += target - start
    return steps


def _predicted_gradient_components(
    *,
    p: int,
    k: int,
    batch_size: int,
    policy: str,
    checkpoint_layers: tuple[int, ...],
    complex_dtype,
    float_dtype,
) -> dict[str, int]:
    elements = batch_size * (1 << k)
    cbytes = _dtype_bytes(complex_dtype)
    fbytes = _dtype_bytes(float_dtype)
    table_bytes = 2 * elements * fbytes
    final_state_bytes = elements * cbytes
    adjoint_bytes = elements * cbytes
    observable_bytes = 2 * elements * fbytes
    derivative_workspace_bytes = 3 * elements * cbytes
    mixer_workspace_bytes = elements * cbytes
    if policy == "cache_all":
        checkpoint_bytes = 2 * p * elements * cbytes
        reconstruction_bytes = 0
    else:
        checkpoint_bytes = len(checkpoint_layers) * elements * cbytes
        reconstruction_bytes = 2 * elements * cbytes
    small_buffers_bytes = 2 * p * np.dtype(np.float64).itemsize
    return {
        "cost_and_term_tables": table_bytes,
        "final_state": final_state_bytes,
        "adjoint_state": adjoint_bytes,
        "checkpoint_states": checkpoint_bytes,
        "reconstruction_states": reconstruction_bytes,
        "derivative_workspace": derivative_workspace_bytes,
        "mixer_workspace": mixer_workspace_bytes,
        "observable_workspace": observable_bytes,
        "small_gradient_buffers": small_buffers_bytes,
    }


def _make_checkpoint_plan(
    *,
    policy: str,
    p: int,
    k: int,
    group_size: int,
    max_batch_states: int,
    complex_dtype,
    float_dtype,
    memory_budget_bytes: int | None,
    checkpoint_interval: int,
) -> CheckpointPlan:
    if p < 1:
        raise ValueError("adjoint evaluation requires p >= 1")
    if policy not in {"cache_all", "recompute_all", "fixed_interval", "budgeted"}:
        raise ValueError(
            "checkpoint_policy must be cache_all, recompute_all, fixed_interval, or budgeted"
        )
    if checkpoint_interval < 1:
        raise ValueError("checkpoint_interval must be positive")
    state_count = 1 << k
    max_batch_size = min(group_size, max(1, max_batch_states // state_count))

    if policy == "cache_all":
        checkpoints: tuple[int, ...] = tuple(range(p))
    elif policy == "recompute_all":
        checkpoints = ()
    elif policy == "fixed_interval":
        checkpoints = tuple(
            layer
            for layer in range(p - 1)
            if (layer + 1) % checkpoint_interval == 0
        )
    else:
        if memory_budget_bytes is None or memory_budget_bytes <= 0:
            raise ValueError("budgeted checkpointing requires memory_budget_bytes > 0")
        candidates = tuple(range(max(0, p - 1)))
        feasible: list[tuple[tuple[int, int, int, int], CheckpointPlan]] = []
        subsets = [
            subset
            for count in range(len(candidates) + 1)
            for subset in combinations(candidates, count)
        ]
        for batch_size in range(1, max_batch_size + 1):
            batch_count = math.ceil(group_size / batch_size)
            for subset in subsets:
                components = _predicted_gradient_components(
                    p=p,
                    k=k,
                    batch_size=batch_size,
                    policy=policy,
                    checkpoint_layers=subset,
                    complex_dtype=complex_dtype,
                    float_dtype=float_dtype,
                )
                predicted = sum(components.values())
                if predicted > memory_budget_bytes:
                    continue
                recompute = _recompute_layer_steps(p, subset)
                work_score = batch_count * (p + recompute)
                plan = CheckpointPlan(
                    policy=policy,
                    batch_size=batch_size,
                    checkpoint_layers=tuple(subset),
                    predicted_active_bytes=predicted,
                    predicted_components=components,
                    memory_budget_bytes=memory_budget_bytes,
                    recompute_layer_steps=recompute,
                )
                feasible.append(
                    ((work_score, batch_count, -batch_size, predicted), plan)
                )
        if not feasible:
            minimum = _predicted_gradient_components(
                p=p,
                k=k,
                batch_size=1,
                policy=policy,
                checkpoint_layers=(),
                complex_dtype=complex_dtype,
                float_dtype=float_dtype,
            )
            raise MemoryError(
                f"memory budget {memory_budget_bytes} is below the minimum predicted "
                f"active requirement {sum(minimum.values())} for k={k}"
            )
        return min(feasible, key=lambda item: item[0])[1]

    components = _predicted_gradient_components(
        p=p,
        k=k,
        batch_size=max_batch_size,
        policy=policy,
        checkpoint_layers=checkpoints,
        complex_dtype=complex_dtype,
        float_dtype=float_dtype,
    )
    return CheckpointPlan(
        policy=policy,
        batch_size=max_batch_size,
        checkpoint_layers=checkpoints,
        predicted_active_bytes=sum(components.values()),
        predicted_components=components,
        memory_budget_bytes=memory_budget_bytes,
        recompute_layer_steps=0
        if policy == "cache_all"
        else _recompute_layer_steps(p, checkpoints),
    )


def plan_checkpoint_schedule(
    *,
    policy: str,
    p: int,
    k: int,
    group_size: int,
    max_batch_states: int = 1 << 22,
    complex_dtype=np.complex64,
    float_dtype=np.float32,
    memory_budget_bytes: int | None = None,
    checkpoint_interval: int = 2,
) -> CheckpointPlan:
    """Plan the active batch and checkpoint set before device allocation."""
    return _make_checkpoint_plan(
        policy=policy,
        p=p,
        k=k,
        group_size=group_size,
        max_batch_states=max_batch_states,
        complex_dtype=complex_dtype,
        float_dtype=float_dtype,
        memory_budget_bytes=memory_budget_bytes,
        checkpoint_interval=checkpoint_interval,
    )


def _x_sum_batched(psi, k: int, xp):
    out = xp.zeros_like(psi)
    batch = psi.shape[0]
    for q in range(k):
        step = 1 << q
        block = step << 1
        src = psi.reshape(batch, -1, block)
        dst = out.reshape(batch, -1, block)
        dst[:, :, :step] += src[:, :, step:block]
        dst[:, :, step:block] += src[:, :, :step]
    return out


def _evaluate_batch(
    batch: list[LightConeProblem],
    gammas: Sequence[float],
    betas: Sequence[float],
    objective: str,
    xp,
    complex_dtype=np.complex64,
    float_dtype=np.float32,
) -> float:
    k = batch[0].k
    nstates = 1 << k
    bsz = len(batch)
    psi = xp.empty((bsz, nstates), dtype=complex_dtype)
    psi.fill(1.0 / math.sqrt(nstates))
    cost = xp.empty((bsz, nstates), dtype=float_dtype)
    terms = xp.empty((bsz, nstates), dtype=float_dtype)
    for row, problem in enumerate(batch):
        cost[row, :] = cost_table(k, problem.edges, problem.fields, objective, xp, float_dtype)
        terms[row, :] = term_table(
            k,
            problem.term_kind,
            problem.local_term_nodes,
            problem.weight,
            objective,
            xp,
            float_dtype,
        )
    for gamma, beta in zip(gammas, betas):
        psi *= xp.exp((-1j * float(gamma)) * cost)
        apply_mixer_batched_inplace(psi, k, beta, xp)
    values = xp.sum((xp.abs(psi) ** 2) * terms, axis=1)
    total = xp.sum(values)
    return float(total.get() if hasattr(total, "get") else total)


def _evaluate_batch_with_adjoint_gradient(
    batch: list[LightConeProblem],
    gammas: Sequence[float],
    betas: Sequence[float],
    objective: str,
    xp,
    complex_dtype=np.complex64,
    float_dtype=np.float32,
) -> tuple[float, np.ndarray]:
    p = len(gammas)
    k = batch[0].k
    nstates = 1 << k
    bsz = len(batch)
    psi = xp.empty((bsz, nstates), dtype=complex_dtype)
    psi.fill(1.0 / math.sqrt(nstates))
    cost = xp.empty((bsz, nstates), dtype=float_dtype)
    terms = xp.empty((bsz, nstates), dtype=float_dtype)
    for row, problem in enumerate(batch):
        cost[row, :] = cost_table(k, problem.edges, problem.fields, objective, xp, float_dtype)
        terms[row, :] = term_table(
            k,
            problem.term_kind,
            problem.local_term_nodes,
            problem.weight,
            objective,
            xp,
            float_dtype,
        )

    after_costs = []
    after_layers = []
    for gamma, beta in zip(gammas, betas):
        psi *= xp.exp((-1j * float(gamma)) * cost)
        after_costs.append(psi.copy())
        apply_mixer_batched_inplace(psi, k, beta, xp)
        after_layers.append(psi.copy())

    values = xp.sum((xp.abs(psi) ** 2) * terms, axis=1)
    total = xp.sum(values)

    grad_gamma = xp.zeros(p, dtype=xp.float64)
    grad_beta = xp.zeros(p, dtype=xp.float64)
    adjoint = terms * psi
    for layer in range(p - 1, -1, -1):
        x_sum = _x_sum_batched(after_layers[layer], k, xp)
        d_beta_state = -1j * x_sum
        grad_beta[layer] = 2.0 * xp.real(xp.sum(xp.conj(adjoint) * d_beta_state))

        apply_mixer_batched_inplace(adjoint, k, -float(betas[layer]), xp)

        d_gamma_state = -1j * cost * after_costs[layer]
        grad_gamma[layer] = 2.0 * xp.real(xp.sum(xp.conj(adjoint) * d_gamma_state))
        adjoint *= xp.exp((1j * float(gammas[layer])) * cost)

    grad = xp.concatenate([grad_gamma, grad_beta])
    if hasattr(total, "get"):
        return float(total.get()), grad.get()
    return float(total), np.asarray(grad)


def _uniform_state(batch_size: int, nstates: int, xp, complex_dtype):
    psi = xp.empty((batch_size, nstates), dtype=complex_dtype)
    psi.fill(1.0 / math.sqrt(nstates))
    return psi


def _reconstruct_layer_states(
    *,
    target_layer: int,
    checkpoints: dict[int, Any],
    batch_size: int,
    nstates: int,
    k: int,
    cost,
    gammas: Sequence[float],
    betas: Sequence[float],
    xp,
    complex_dtype,
    tracker: _MemoryTracker,
):
    previous = [layer for layer in checkpoints if layer < target_layer]
    if previous:
        start = max(previous)
        psi = checkpoints[start].copy()
        first_layer = start + 1
    else:
        psi = _uniform_state(batch_size, nstates, xp, complex_dtype)
        first_layer = 0
    after_cost = None
    for layer in range(first_layer, target_layer + 1):
        psi *= xp.exp((-1j * float(gammas[layer])) * cost)
        if layer == target_layer:
            after_cost = psi.copy()
        apply_mixer_batched_inplace(psi, k, betas[layer], xp)
        tracker.sample()
    if after_cost is None:
        raise RuntimeError("failed to reconstruct target layer")
    return after_cost, psi


def _evaluate_batch_with_checkpoint_gradient(
    batch: list[LightConeProblem],
    gammas: Sequence[float],
    betas: Sequence[float],
    objective: str,
    xp,
    plan: CheckpointPlan,
    tracker: _MemoryTracker,
    complex_dtype=np.complex64,
    float_dtype=np.float32,
) -> tuple[float, np.ndarray]:
    p = len(gammas)
    k = batch[0].k
    nstates = 1 << k
    bsz = len(batch)
    psi = _uniform_state(bsz, nstates, xp, complex_dtype)
    cost = xp.empty((bsz, nstates), dtype=float_dtype)
    terms = xp.empty((bsz, nstates), dtype=float_dtype)
    for row, problem in enumerate(batch):
        cost[row, :] = cost_table(k, problem.edges, problem.fields, objective, xp, float_dtype)
        terms[row, :] = term_table(
            k,
            problem.term_kind,
            problem.local_term_nodes,
            problem.weight,
            objective,
            xp,
            float_dtype,
        )
    tracker.sample()

    after_costs: list[Any] = []
    after_layers: list[Any] = []
    checkpoints: dict[int, Any] = {}
    checkpoint_set = set(plan.checkpoint_layers)
    for layer, (gamma, beta) in enumerate(zip(gammas, betas)):
        psi *= xp.exp((-1j * float(gamma)) * cost)
        if plan.policy == "cache_all":
            after_costs.append(psi.copy())
        apply_mixer_batched_inplace(psi, k, beta, xp)
        if plan.policy == "cache_all":
            after_layers.append(psi.copy())
        elif layer in checkpoint_set:
            checkpoints[layer] = psi.copy()
        tracker.sample()

    values = xp.sum((xp.abs(psi) ** 2) * terms, axis=1)
    total = xp.sum(values)
    grad_gamma = xp.zeros(p, dtype=xp.float64)
    grad_beta = xp.zeros(p, dtype=xp.float64)
    adjoint = terms * psi
    tracker.sample()

    for layer in range(p - 1, -1, -1):
        reconstructed = plan.policy != "cache_all"
        if reconstructed:
            after_cost, after_layer = _reconstruct_layer_states(
                target_layer=layer,
                checkpoints=checkpoints,
                batch_size=bsz,
                nstates=nstates,
                k=k,
                cost=cost,
                gammas=gammas,
                betas=betas,
                xp=xp,
                complex_dtype=complex_dtype,
                tracker=tracker,
            )
        else:
            after_cost = after_costs[layer]
            after_layer = after_layers[layer]

        x_sum = _x_sum_batched(after_layer, k, xp)
        d_beta_state = -1j * x_sum
        grad_beta[layer] = 2.0 * xp.real(xp.sum(xp.conj(adjoint) * d_beta_state))
        tracker.sample()

        apply_mixer_batched_inplace(adjoint, k, -float(betas[layer]), xp)
        d_gamma_state = -1j * cost * after_cost
        grad_gamma[layer] = 2.0 * xp.real(xp.sum(xp.conj(adjoint) * d_gamma_state))
        adjoint *= xp.exp((1j * float(gammas[layer])) * cost)
        tracker.sample()
        if reconstructed:
            del after_cost, after_layer

    grad = xp.concatenate([grad_gamma, grad_beta])
    tracker.sample()
    if hasattr(total, "get"):
        return float(total.get()), grad.get()
    return float(total), np.asarray(grad)


def lightcone_expectation(
    graph: WeightedGraph,
    gammas: Sequence[float],
    betas: Sequence[float],
    *,
    p: int | None = None,
    prefer_gpu: bool = True,
    max_k: int = 24,
    max_batch_states: int = 1 << 22,
    naive: bool = False,
    group_by: str = "size",
    complex_dtype=np.complex64,
    float_dtype=np.float32,
) -> EvalStats:
    if len(gammas) != len(betas):
        raise ValueError("gammas and betas must have the same length")
    if p is None:
        p = len(gammas)
    if p != len(gammas):
        raise ValueError(
            "p must equal the number of gamma/beta layers; "
            f"got p={p} and layers={len(gammas)}"
        )
    backend = get_backend(prefer_gpu)
    xp = backend.xp
    backend.free_memory_pool()
    t0 = time.perf_counter()
    problems = extract_lightcones(graph, p)
    if not problems:
        return EvalStats(graph.constant_offset, 0.0, backend.name, 0, 0, "ok")
    max_observed_k = max(problem.k for problem in problems)
    if max_observed_k > max_k:
        return EvalStats(
            value=float("nan"),
            seconds=0.0,
            backend=backend.name,
            peak_pool_bytes=0,
            state_qubits=max_observed_k,
            status=f"skipped_kmax_{max_observed_k}_over_{max_k}",
        )

    if group_by not in {"size", "topology"}:
        raise ValueError("group_by must be 'size' or 'topology'")
    groups: dict[tuple, list[LightConeProblem]] = {}
    for problem in problems:
        key = (problem.k,) if group_by == "size" else lightcone_topology_signature(problem)
        groups.setdefault(key, []).append(problem)

    total = 0.0
    for key in sorted(groups, key=repr):
        group = groups[key]
        if naive:
            batches = [[problem] for problem in group]
        else:
            batches = _split_batches(group, max_batch_states)
        for batch in batches:
            total += _evaluate_batch(
                batch,
                gammas,
                betas,
                graph.objective,
                xp,
                complex_dtype,
                float_dtype,
            )
    backend.sync()
    return EvalStats(
        value=total + graph.constant_offset,
        seconds=time.perf_counter() - t0,
        backend=backend.name,
        peak_pool_bytes=backend.memory_pool_bytes(),
        state_qubits=max_observed_k,
        status="ok",
    )


def lightcone_gradient_adjoint(
    graph: WeightedGraph,
    gammas: Sequence[float],
    betas: Sequence[float],
    *,
    p: int | None = None,
    prefer_gpu: bool = True,
    max_k: int = 24,
    max_batch_states: int = 1 << 22,
    group_by: str = "size",
    complex_dtype=np.complex64,
    float_dtype=np.float32,
    checkpoint_policy: str = "cache_all",
    memory_budget_bytes: int | None = None,
    checkpoint_interval: int = 2,
) -> GradientStats:
    if len(gammas) != len(betas):
        raise ValueError("gammas and betas must have the same length")
    if p is None:
        p = len(gammas)
    if p != len(gammas):
        raise ValueError(
            "p must equal the number of gamma/beta layers; "
            f"got p={p} and layers={len(gammas)}"
        )
    if group_by not in {"size", "topology"}:
        raise ValueError("group_by must be 'size' or 'topology'")
    backend = get_backend(prefer_gpu)
    xp = backend.xp
    backend.free_memory_pool()
    tracker = _MemoryTracker(backend)
    t0 = time.perf_counter()
    problems = extract_lightcones(graph, p)
    if not problems:
        return GradientStats(
            graph.constant_offset,
            0.0,
            backend.name,
            0,
            0,
            "ok",
            np.zeros(2 * len(gammas)),
            0,
            memory_budget_bytes,
            checkpoint_policy,
            [],
        )
    max_observed_k = max(problem.k for problem in problems)
    if max_observed_k > max_k:
        return GradientStats(
            value=float("nan"),
            seconds=0.0,
            backend=backend.name,
            peak_pool_bytes=0,
            state_qubits=max_observed_k,
            status=f"skipped_kmax_{max_observed_k}_over_{max_k}",
            gradient=None,
        )

    groups: dict[tuple, list[LightConeProblem]] = {}
    for problem in problems:
        key = (problem.k,) if group_by == "size" else lightcone_topology_signature(problem)
        groups.setdefault(key, []).append(problem)

    planned_groups: list[
        tuple[list[LightConeProblem], CheckpointPlan, dict[str, Any]]
    ] = []
    for key in sorted(groups, key=repr):
        group = groups[key]
        plan = _make_checkpoint_plan(
            policy=checkpoint_policy,
            p=p,
            k=group[0].k,
            group_size=len(group),
            max_batch_states=max_batch_states,
            complex_dtype=complex_dtype,
            float_dtype=float_dtype,
            memory_budget_bytes=memory_budget_bytes,
            checkpoint_interval=checkpoint_interval,
        )
        plan_record = plan.as_dict()
        plan_record.update(
            {
                "group_key": repr(key),
                "k": group[0].k,
                "group_size": len(group),
                "batch_count": math.ceil(len(group) / plan.batch_size),
            }
        )
        planned_groups.append((group, plan, plan_record))

    total = 0.0
    gradient = np.zeros(2 * len(gammas), dtype=np.float64)
    checkpoint_plans: list[dict[str, Any]] = []
    for group, plan, plan_record in planned_groups:
        checkpoint_plans.append(plan_record)
        for batch in _split_batches_by_count(group, plan.batch_size):
            value, grad = _evaluate_batch_with_checkpoint_gradient(
                batch,
                gammas,
                betas,
                graph.objective,
                xp,
                plan,
                tracker,
                complex_dtype,
                float_dtype,
            )
            total += value
            gradient += grad
    backend.sync()
    return GradientStats(
        value=total + graph.constant_offset,
        seconds=time.perf_counter() - t0,
        backend=backend.name,
        peak_pool_bytes=max(tracker.peak_reserved_bytes, backend.memory_pool_bytes()),
        state_qubits=max_observed_k,
        status="ok",
        gradient=gradient,
        peak_allocated_bytes=tracker.peak_allocated_bytes,
        memory_budget_bytes=memory_budget_bytes,
        checkpoint_policy=checkpoint_policy,
        checkpoint_plans=checkpoint_plans,
    )
