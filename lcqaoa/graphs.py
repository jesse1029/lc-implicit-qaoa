from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
import random


Edge = tuple[int, int, float]
Field = tuple[int, float]


@dataclass(frozen=True)
class WeightedGraph:
    n: int
    edges: tuple[Edge, ...]
    fields: tuple[Field, ...] = ()
    objective: str = "maxcut"
    constant_offset: float = 0.0

    def __post_init__(self) -> None:
        if self.objective not in {"maxcut", "qubo"}:
            raise ValueError("objective must be 'maxcut' or 'qubo'")
        for i, j, _ in self.edges:
            if i == j:
                raise ValueError("self loops are not supported")
            if not (0 <= i < self.n and 0 <= j < self.n):
                raise ValueError("edge endpoint out of range")
        for i, _ in self.fields:
            if not (0 <= i < self.n):
                raise ValueError("field endpoint out of range")

    @property
    def m(self) -> int:
        return len(self.edges)

    def neighbors(self) -> list[list[int]]:
        adj = [[] for _ in range(self.n)]
        for i, j, _ in self.edges:
            adj[i].append(j)
            adj[j].append(i)
        return adj

    def cone_nodes(self, seeds: Iterable[int], radius: int) -> tuple[int, ...]:
        if radius < 0:
            raise ValueError("radius must be non-negative")
        adj = self.neighbors()
        seen = set(seeds)
        frontier = set(seeds)
        for _ in range(radius):
            nxt: set[int] = set()
            for node in frontier:
                for nb in adj[node]:
                    if nb not in seen:
                        seen.add(nb)
                        nxt.add(nb)
            frontier = nxt
            if not frontier:
                break
        return tuple(sorted(seen))

    def induced_edges(self, nodes: Iterable[int]) -> tuple[Edge, ...]:
        node_set = set(nodes)
        return tuple((i, j, w) for i, j, w in self.edges if i in node_set and j in node_set)

    def induced_fields(self, nodes: Iterable[int]) -> tuple[Field, ...]:
        node_set = set(nodes)
        return tuple((i, w) for i, w in self.fields if i in node_set)

    def relabel_subgraph(
        self, nodes: tuple[int, ...]
    ) -> tuple[dict[int, int], tuple[Edge, ...], tuple[Field, ...]]:
        mapping = {node: idx for idx, node in enumerate(nodes)}
        edges = tuple((mapping[i], mapping[j], w) for i, j, w in self.induced_edges(nodes))
        fields = tuple((mapping[i], w) for i, w in self.induced_fields(nodes))
        return mapping, edges, fields


def _rng(seed: int | None) -> random.Random:
    return random.Random(seed)


def random_regular_graph(n: int, degree: int, seed: int | None = None) -> WeightedGraph:
    try:
        import networkx as nx

        g = nx.random_regular_graph(degree, n, seed=seed)
        edges = tuple((int(i), int(j), 1.0) for i, j in g.edges())
        return WeightedGraph(n=n, edges=edges, objective="maxcut")
    except Exception:
        pass

    if n * degree % 2 != 0:
        raise ValueError("n * degree must be even")
    r = _rng(seed)
    for _ in range(2000):
        stubs = [i for i in range(n) for _ in range(degree)]
        r.shuffle(stubs)
        seen: set[tuple[int, int]] = set()
        ok = True
        for a, b in zip(stubs[0::2], stubs[1::2]):
            if a == b:
                ok = False
                break
            e = (min(a, b), max(a, b))
            if e in seen:
                ok = False
                break
            seen.add(e)
        if ok:
            return WeightedGraph(n=n, edges=tuple((i, j, 1.0) for i, j in sorted(seen)))
    raise RuntimeError("failed to generate random regular graph")


def erdos_renyi_graph(n: int, edge_prob: float, seed: int | None = None) -> WeightedGraph:
    r = _rng(seed)
    edges: list[Edge] = []
    for i in range(n):
        for j in range(i + 1, n):
            if r.random() < edge_prob:
                edges.append((i, j, 1.0))
    if not edges and n >= 2:
        edges.append((0, 1, 1.0))
    return WeightedGraph(n=n, edges=tuple(edges), objective="maxcut")


def modular_graph(
    n: int,
    modules: int = 4,
    p_in: float = 0.35,
    p_out: float = 0.02,
    seed: int | None = None,
) -> WeightedGraph:
    r = _rng(seed)
    edges: list[Edge] = []
    block = max(1, n // modules)
    labels = [min(modules - 1, i // block) for i in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            p = p_in if labels[i] == labels[j] else p_out
            if r.random() < p:
                edges.append((i, j, 1.0))
    if not edges and n >= 2:
        edges.append((0, 1, 1.0))
    return WeightedGraph(n=n, edges=tuple(edges), objective="maxcut")


def scale_free_graph(n: int, attachment: int = 2, seed: int | None = None) -> WeightedGraph:
    try:
        import networkx as nx

        g = nx.barabasi_albert_graph(n, attachment, seed=seed)
        edges = tuple((int(i), int(j), 1.0) for i, j in g.edges())
        return WeightedGraph(n=n, edges=edges, objective="maxcut")
    except Exception:
        pass

    r = _rng(seed)
    edges_set: set[tuple[int, int]] = set()
    degrees = [0] * n
    for i in range(1, min(n, attachment + 1)):
        edges_set.add((0, i))
        degrees[0] += 1
        degrees[i] += 1
    for new in range(attachment + 1, n):
        weights = [degrees[i] + 1 for i in range(new)]
        total = sum(weights)
        targets: set[int] = set()
        while len(targets) < min(attachment, new):
            pick = r.uniform(0, total)
            acc = 0.0
            for i, w in enumerate(weights):
                acc += w
                if acc >= pick:
                    targets.add(i)
                    break
        for t in targets:
            e = (min(new, t), max(new, t))
            edges_set.add(e)
            degrees[new] += 1
            degrees[t] += 1
    return WeightedGraph(n=n, edges=tuple((i, j, 1.0) for i, j in sorted(edges_set)), objective="maxcut")


def weighted_qubo_graph(
    n: int,
    edge_prob: float,
    *,
    field_prob: float = 1.0,
    seed: int | None = None,
    weight_scale: float = 1.0,
    field_scale: float = 0.5,
) -> WeightedGraph:
    r = _rng(seed)
    edges: list[Edge] = []
    for i in range(n):
        for j in range(i + 1, n):
            if r.random() < edge_prob:
                w = r.uniform(-weight_scale, weight_scale)
                if abs(w) < 1e-6:
                    w = weight_scale
                edges.append((i, j, w))
    if not edges and n >= 2:
        edges.append((0, 1, weight_scale))
    fields: list[Field] = []
    for i in range(n):
        if r.random() < field_prob:
            w = r.uniform(-field_scale, field_scale)
            fields.append((i, w))
    return WeightedGraph(n=n, edges=tuple(edges), fields=tuple(fields), objective="qubo")


def weighted_modular_qubo_graph(
    n: int,
    *,
    modules: int = 4,
    p_in: float = 0.20,
    p_out: float = 0.004,
    field_prob: float = 1.0,
    seed: int | None = None,
) -> WeightedGraph:
    r = _rng(seed)
    edges: list[Edge] = []
    block = max(1, n // modules)
    labels = [min(modules - 1, i // block) for i in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            p = p_in if labels[i] == labels[j] else p_out
            if r.random() < p:
                scale = 0.8 if labels[i] == labels[j] else 0.4
                edges.append((i, j, r.uniform(-scale, scale)))
    if not edges and n >= 2:
        edges.append((0, 1, 0.5))
    fields = tuple((i, r.uniform(-0.5, 0.5)) for i in range(n) if r.random() < field_prob)
    return WeightedGraph(n=n, edges=tuple(edges), fields=fields, objective="qubo")
