from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from .graphs import (
    WeightedGraph,
    erdos_renyi_graph,
    modular_graph,
    random_regular_graph,
    scale_free_graph,
    weighted_modular_qubo_graph,
    weighted_qubo_graph,
)
from .lightcone import extract_lightcones, lightcone_expectation, lightcone_gradient_adjoint
from .qaoa import full_state_expectation


def default_params(p: int) -> tuple[list[float], list[float]]:
    return [0.21 + 0.08 * i for i in range(p)], [0.34 - 0.035 * i for i in range(p)]


def parse_float_list(text: str | None, p: int, name: str) -> list[float]:
    if text is None:
        gammas, betas = default_params(p)
        return gammas if name == "gammas" else betas
    values = [float(x) for x in text.split(",") if x.strip()]
    if len(values) != p:
        raise SystemExit(f"--{name} must contain exactly p={p} comma-separated values")
    return values


def graph_from_args(args: argparse.Namespace) -> WeightedGraph:
    if args.graph_json:
        payload = json.loads(Path(args.graph_json).read_text(encoding="utf-8"))
        return WeightedGraph(
            n=int(payload["n"]),
            edges=tuple((int(i), int(j), float(w)) for i, j, w in payload.get("edges", [])),
            fields=tuple((int(i), float(w)) for i, w in payload.get("fields", [])),
            objective=payload.get("objective", args.objective),
        )
    if args.objective == "maxcut":
        if args.family == "3regular":
            return random_regular_graph(args.n, 3, seed=args.seed)
        if args.family == "er":
            return erdos_renyi_graph(args.n, args.edge_prob or min(1.0, 3.0 / max(1, args.n - 1)), seed=args.seed)
        if args.family == "modular":
            return modular_graph(args.n, modules=args.modules, p_in=args.p_in, p_out=args.p_out, seed=args.seed)
        if args.family == "scale_free":
            return scale_free_graph(args.n, attachment=args.attachment, seed=args.seed)
    if args.objective == "qubo":
        if args.family == "er":
            return weighted_qubo_graph(
                args.n,
                args.edge_prob or min(1.0, 2.0 / max(1, args.n - 1)),
                seed=args.seed,
                field_prob=args.field_prob,
            )
        if args.family == "modular":
            return weighted_modular_qubo_graph(
                args.n,
                modules=args.modules,
                p_in=args.p_in,
                p_out=args.p_out,
                field_prob=args.field_prob,
                seed=args.seed,
            )
    raise SystemExit(f"unsupported objective/family combination: {args.objective}/{args.family}")


def cone_summary(graph: WeightedGraph, p: int) -> dict[str, int]:
    cones = extract_lightcones(graph, p)
    if not cones:
        return {"n_cones": 0, "kmax": 0, "total_cone_states": 0}
    return {
        "n_cones": len(cones),
        "kmax": max(c.k for c in cones),
        "total_cone_states": int(sum(1 << c.k for c in cones)),
    }


def result_payload(graph: WeightedGraph, p: int, method: str, stat, extra: dict | None = None) -> dict:
    payload = {
        "method": method,
        "objective": graph.objective,
        "n": graph.n,
        "m": graph.m,
        "p": p,
        "status": stat.status,
        "value": stat.value,
        "seconds": stat.seconds,
        "backend": stat.backend,
        "peak_pool_mb": stat.peak_pool_bytes / 1024**2,
        **cone_summary(graph, p),
    }
    if extra:
        payload.update(extra)
    return payload


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--objective", choices=["maxcut", "qubo"], default="maxcut")
    parser.add_argument("--family", choices=["3regular", "er", "modular", "scale_free"], default="3regular")
    parser.add_argument("--n", type=int, default=24)
    parser.add_argument("--p", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--gammas")
    parser.add_argument("--betas")
    parser.add_argument("--graph-json", help="JSON file with n, edges, fields, objective")
    parser.add_argument("--edge-prob", type=float)
    parser.add_argument("--field-prob", type=float, default=1.0)
    parser.add_argument("--modules", type=int, default=4)
    parser.add_argument("--p-in", type=float, default=0.08)
    parser.add_argument("--p-out", type=float, default=0.002)
    parser.add_argument("--attachment", type=int, default=1)
    parser.add_argument("--max-k", type=int, default=24)
    parser.add_argument("--max-batch-states", type=int, default=1 << 21)
    parser.add_argument("--cpu", action="store_true", help="disable CuPy GPU backend")
    parser.add_argument("--pretty", action="store_true")


def cmd_evaluate(args: argparse.Namespace) -> None:
    graph = graph_from_args(args)
    gammas = parse_float_list(args.gammas, args.p, "gammas")
    betas = parse_float_list(args.betas, args.p, "betas")
    stat = lightcone_expectation(
        graph,
        gammas,
        betas,
        p=args.p,
        prefer_gpu=not args.cpu,
        max_k=args.max_k,
        max_batch_states=args.max_batch_states,
    )
    emit(result_payload(graph, args.p, "lc_batched", stat), args.pretty)


def cmd_gradient(args: argparse.Namespace) -> None:
    graph = graph_from_args(args)
    gammas = parse_float_list(args.gammas, args.p, "gammas")
    betas = parse_float_list(args.betas, args.p, "betas")
    stat = lightcone_gradient_adjoint(
        graph,
        gammas,
        betas,
        p=args.p,
        prefer_gpu=not args.cpu,
        max_k=args.max_k,
        max_batch_states=args.max_batch_states,
    )
    extra = {}
    if stat.gradient is not None:
        extra["gradient"] = np.asarray(stat.gradient, dtype=float).tolist()
        extra["gradient_norm"] = float(np.linalg.norm(stat.gradient))
    emit(result_payload(graph, args.p, "lc_adjoint_gradient", stat, extra), args.pretty)


def cmd_compare_full(args: argparse.Namespace) -> None:
    graph = graph_from_args(args)
    gammas = parse_float_list(args.gammas, args.p, "gammas")
    betas = parse_float_list(args.betas, args.p, "betas")
    full = full_state_expectation(
        graph,
        gammas,
        betas,
        method="precompute",
        prefer_gpu=not args.cpu,
        max_qubits=args.full_cap,
    )
    lc = lightcone_expectation(
        graph,
        gammas,
        betas,
        p=args.p,
        prefer_gpu=not args.cpu,
        max_k=args.max_k,
        max_batch_states=args.max_batch_states,
    )
    payload = {
        "full": result_payload(graph, args.p, "full_precompute", full),
        "lc": result_payload(graph, args.p, "lc_batched", lc),
        "abs_error": abs(lc.value - full.value) if full.status == "ok" and lc.status == "ok" else None,
        "speedup_vs_full": full.seconds / lc.seconds if full.status == "ok" and lc.status == "ok" and lc.seconds > 0 else None,
        "memory_reduction_vs_full": (full.peak_pool_bytes / lc.peak_pool_bytes) if full.peak_pool_bytes and lc.peak_pool_bytes else None,
    }
    emit(payload, args.pretty)


def emit(payload: dict, pretty: bool) -> None:
    print(json.dumps(payload, indent=2 if pretty else None, sort_keys=pretty))


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="lcqaoa")
    sub = parser.add_subparsers(dest="command", required=True)
    eval_parser = sub.add_parser("evaluate", help="evaluate exact LC objective")
    add_common_args(eval_parser)
    eval_parser.set_defaults(func=cmd_evaluate)

    grad_parser = sub.add_parser("gradient", help="evaluate exact LC objective and adjoint gradient")
    add_common_args(grad_parser)
    grad_parser.set_defaults(func=cmd_gradient)

    cmp_parser = sub.add_parser("compare-full", help="compare LC against full-state precompute baseline")
    add_common_args(cmp_parser)
    cmp_parser.add_argument("--full-cap", type=int, default=24)
    cmp_parser.set_defaults(func=cmd_compare_full)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
