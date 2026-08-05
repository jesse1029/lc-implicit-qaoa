from __future__ import annotations

import argparse
import importlib.util
import json
import math
import tempfile
import time
from pathlib import Path
import subprocess


def load_case(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def make_nx_graph(case: dict):
    import networkx as nx

    graph = nx.Graph()
    graph.add_nodes_from(range(case["n"]))
    for i, j, w in case["edges"]:
        graph.add_edge(int(i), int(j), weight=float(w))
    return graph


def run_qokit_cpu(case: dict) -> dict:
    import networkx as nx
    from qokit.maxcut import get_maxcut_terms
    from qokit.fur.python.qaoa_simulator import QAOAFURXSimulator

    graph = make_nx_graph(case)
    terms = get_maxcut_terms(graph)
    sim = QAOAFURXSimulator(case["n"], terms=terms)
    gammas = [2.0 * float(x) for x in case["gammas"]]
    betas = [float(x) for x in case["betas"]]
    t0 = time.perf_counter()
    result = sim.simulate_qaoa(gammas, betas)
    value = float(sim.get_expectation(result))
    return {
        "method": "qokit_cpu_external",
        "status": "ok",
        "value": value,
        "seconds": time.perf_counter() - t0,
        "backend": "qokit_python",
        "notes": "gammas doubled to match qokit exp(-0.5i gamma Hc) convention",
    }


def qtensor_transform(gammas: list[float], betas: list[float], transform: str) -> tuple[list[float], list[float]]:
    if transform == "identity":
        return gammas, betas
    if transform == "neg_gamma":
        return [-g for g in gammas], betas
    if transform == "paper_qiskit_inverse":
        # QTensor tests compare QTensor gamma,beta to Qiskit gamma=-2*pi*gamma, beta=pi*beta.
        # Invert that mapping from our exp(-i gamma C), exp(-i beta X) convention.
        return [-g / (2.0 * math.pi) for g in gammas], [b / math.pi for b in betas]
    if transform == "paper_qiskit_inverse_pos":
        return [g / (2.0 * math.pi) for g in gammas], [b / math.pi for b in betas]
    if transform == "half_gamma":
        return [0.5 * g for g in gammas], betas
    if transform == "double_gamma":
        return [2.0 * g for g in gammas], betas
    raise ValueError(f"unknown transform {transform}")


def run_qtensor_cpu(case: dict, transform: str) -> dict:
    import qtensor

    graph = make_nx_graph(case)
    gammas, betas = qtensor_transform(
        [float(x) for x in case["gammas"]],
        [float(x) for x in case["betas"]],
        transform,
    )
    composer = getattr(
        qtensor,
        "DefaultQAOAComposer",
        getattr(qtensor, "QtreeQAOAComposer", getattr(qtensor, "QAOAComposer")),
    )
    sim = qtensor.QAOAQtreeSimulator(composer)
    t0 = time.perf_counter()
    value = float(sim.energy_expectation(graph, gamma=gammas, beta=betas))
    return {
        "method": "qtensor_cpu_external",
        "status": "ok",
        "value": value,
        "seconds": time.perf_counter() - t0,
        "backend": "qtensor_qtree_numpy",
        "notes": f"transform={transform}",
    }


def run_qtensor_gpu(case: dict, transform: str) -> dict:
    import os
    import sys

    source = Path(os.environ.get("QTENSOR_GPU_SOURCE", Path.home() / "lc_implicit_qaoa_peers" / "src" / "QTensor-cupybackend"))
    if source.exists():
        sys.path.insert(0, str(source))

    import cupy as cp
    import qtensor
    from qtensor.contraction_backends import CuPyBackend

    graph = make_nx_graph(case)
    gammas, betas = qtensor_transform(
        [float(x) for x in case["gammas"]],
        [float(x) for x in case["betas"]],
        transform,
    )
    cp.get_default_memory_pool().free_all_blocks()
    composer = getattr(
        qtensor,
        "DefaultQAOAComposer",
        getattr(qtensor, "QtreeQAOAComposer", getattr(qtensor, "QAOAComposer")),
    )
    sim = qtensor.QAOAQtreeSimulator(composer, backend=CuPyBackend())
    cp.cuda.Stream.null.synchronize()
    t0 = time.perf_counter()
    value = float(sim.energy_expectation(graph, gamma=gammas, beta=betas))
    cp.cuda.Stream.null.synchronize()
    seconds = time.perf_counter() - t0
    peak_mb = cp.get_default_memory_pool().total_bytes() / 1024**2
    return {
        "method": "qtensor_gpu_external",
        "status": "ok",
        "value": value,
        "seconds": seconds,
        "backend": "qtensor_cupybackend_branch_cupy",
        "peak_pool_mb": peak_mb,
        "notes": f"transform={transform}; source={source}",
    }


def run_cudaq_observe(case: dict) -> dict:
    import cudaq
    from cudaq import spin

    # Keep this path conservative. If the installed CUDA-Q cannot JIT a dynamic
    # Python kernel on this host, the caller records the failure instead of
    # comparing non-equivalent numbers.
    if cudaq.has_target("nvidia"):
        cudaq.set_target("nvidia")

    n = int(case["n"])
    edges = [(int(i), int(j), float(w)) for i, j, w in case["edges"]]
    gammas = [float(x) for x in case["gammas"]]
    betas = [float(x) for x in case["betas"]]

    # CUDA-Q kernels cannot capture arbitrary Python lists from parent scope.
    # Generate a tiny constant-expanded kernel for the current benchmark case.
    lines = [
        "import cudaq",
        "@cudaq.kernel",
        "def kernel():",
        f"    q = cudaq.qvector({n})",
    ]
    for i in range(n):
        lines.append(f"    h(q[{i}])")
    for layer, (gamma, beta) in enumerate(zip(gammas, betas)):
        lines.append(f"    # layer {layer}")
        for i, j, w in edges:
            # exp(-i gamma*w*(1-ZZ)/2) differs by a global phase from
            # exp(+i gamma*w*ZZ/2). Implement RZZ through cx-rz-cx.
            angle = -float(gamma) * float(w)
            lines.append(f"    x.ctrl(q[{i}], q[{j}])")
            lines.append(f"    rz({angle!r}, q[{j}])")
            lines.append(f"    x.ctrl(q[{i}], q[{j}])")
        for i in range(n):
            lines.append(f"    rx({(2.0 * float(beta))!r}, q[{i}])")
    tmp_dir = Path(tempfile.gettempdir()) / "lcqaoa_cudaq_kernels"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    module_path = tmp_dir / f"cudaq_case_{abs(hash(json.dumps(case, sort_keys=True)))}.py"
    module_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    spec = importlib.util.spec_from_file_location(module_path.stem, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to import generated CUDA-Q kernel {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    kernel = module.kernel

    hamiltonian = 0
    for i, j, w in edges:
        hamiltonian += 0.5 * w * (1.0 - spin.z(i) * spin.z(j))

    t0 = time.perf_counter()
    result = cudaq.observe(kernel, hamiltonian)
    value = float(result.expectation())
    return {
        "method": "cudaq_observe_external",
        "status": "ok",
        "value": value,
        "seconds": time.perf_counter() - t0,
        "backend": str(cudaq.get_target()),
        "notes": "CUDA-Q observe over MaxCut Hamiltonian",
    }


def run_cuaoa_gpu(case: dict) -> dict:
    import numpy as np
    import pycuaoa

    n = int(case["n"])
    if case.get("objective", "maxcut") != "maxcut":
        raise ValueError("CUAOA adapter is currently wired for MaxCut only")
    adjacency = np.zeros((n, n), dtype=np.float64)
    for i, j, w in case["edges"]:
        adjacency[int(i), int(j)] += float(w)
        adjacency[int(j), int(i)] += float(w)
    gammas = np.asarray([float(x) for x in case["gammas"]], dtype=np.float64)
    betas = np.asarray([float(x) for x in case["betas"]], dtype=np.float64)
    params = pycuaoa.Parameters(betas, gammas)
    sim = pycuaoa.CUAOA(adjacency, depth=len(gammas), parameters=params)
    # pycuaoa 0.1.0 segfaulted with create_handle(..., exact=True) on the
    # target CUDA 13.3 host. The default handle path is stable and matches
    # upstream examples.
    handle = pycuaoa.create_handle(n)
    t0 = time.perf_counter()
    try:
        raw_value = float(sim.expectation_value(handle))
    finally:
        try:
            handle.destroy()
        except Exception:
            pass
    return {
        "method": "cuaoa_gpu_external",
        "status": "ok",
        "value": -raw_value,
        "seconds": time.perf_counter() - t0,
        "backend": "pycuaoa_0.1.0_cuda",
        "notes": "official CUAOA pycuaoa; reports negative MaxCut energy, so value=-raw; exact=True handle path segfaulted on this host",
    }


def _cost_from_state_index(idx: int, edges: list[tuple[int, int, float]], objective: str) -> float:
    total = 0.0
    if objective == "maxcut":
        for i, j, w in edges:
            total += float(w) * (((idx >> i) ^ (idx >> j)) & 1)
    elif objective == "qubo":
        for i, j, w in edges:
            total += float(w) * (((idx >> i) & 1) & ((idx >> j) & 1))
    else:
        raise ValueError(f"unsupported objective: {objective}")
    return total


def run_qblaze_cpu(case: dict) -> dict:
    import numpy as np
    import qblaze

    n = int(case["n"])
    edges = [(int(i), int(j), float(w)) for i, j, w in case["edges"]]
    objective = case.get("objective", "maxcut")
    sim = qblaze.Simulator(qubit_count=n)
    t0 = time.perf_counter()
    for i in range(n):
        sim.h(i)
    for gamma, beta in zip(case["gammas"], case["betas"]):
        gamma = float(gamma)
        beta = float(beta)
        if objective == "maxcut":
            for i, j, w in edges:
                phase = -gamma * float(w)
                sim.mcphase([(i, False), (j, True)], phase)
                sim.mcphase([(i, True), (j, False)], phase)
        elif objective == "qubo":
            for i, j, w in edges:
                sim.mcphase([(i, True), (j, True)], -gamma * float(w))
        else:
            raise ValueError(f"unsupported objective: {objective}")
        for i in range(n):
            sim.rx(i, 2.0 * beta)
    sim.flush()
    state = np.zeros(1 << n, dtype=np.complex128)
    sim.copy_amplitudes(state)
    probs = np.abs(state) ** 2
    value = 0.0
    for idx, prob in enumerate(probs):
        if prob != 0.0:
            value += float(prob) * _cost_from_state_index(idx, edges, objective)
    return {
        "method": "qblaze_cpu_external",
        "status": "ok",
        "value": float(value),
        "seconds": time.perf_counter() - t0,
        "backend": "qblaze_sparse_cpu",
        "notes": "official qblaze Simulator API; QAOA starts in |+> and quickly becomes dense, so this is small-n only",
    }


def _julia_vector(values: list[float]) -> str:
    return "[" + ", ".join(repr(float(v)) for v in values) + "]"


def _julia_edges(edges: list[tuple[int, int, float]], one_indexed: bool = True) -> str:
    items = []
    for i, j, w in edges:
        ii = int(i) + 1 if one_indexed else int(i)
        jj = int(j) + 1 if one_indexed else int(j)
        items.append(f"({ii}, {jj}, {float(w)!r})")
    return "[" + ", ".join(items) + "]"


def _run_julia_script(script: str, *, project: str | None = None, timeout: int = 180) -> tuple[float, float, str]:
    tmp_dir = Path(tempfile.gettempdir()) / "lcqaoa_julia_runners"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    path = tmp_dir / f"runner_{abs(hash(script))}.jl"
    path.write_text(script, encoding="utf-8")
    julia = Path.home() / ".juliaup" / "bin" / "julia"
    cmd = [str(julia)]
    if project:
        cmd.append(f"--project={project}")
    cmd.append(str(path))
    cp = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=timeout)
    marker = "LCQAOA_RESULT "
    for line in reversed(cp.stdout.splitlines()):
        if line.startswith(marker):
            _, value, seconds = line.split()
            return float(value), float(seconds), cp.stdout[-1000:]
    raise RuntimeError(f"Julia runner failed rc={cp.returncode}: {cp.stdout[-1000:]}")


def run_juliqaoa_cpu(case: dict) -> dict:
    n = int(case["n"])
    if case.get("objective", "maxcut") != "maxcut":
        raise ValueError("JuliQAOA adapter is currently wired for MaxCut only")
    edges = [(int(i), int(j), float(w)) for i, j, w in case["edges"]]
    script = f"""
using JuliQAOA
n = {n}
edges = {_julia_edges(edges)}
betas = {_julia_vector([float(x) for x in case["betas"]])}
gammas = {_julia_vector([float(x) for x in case["gammas"]])}
function cost_index(idx)
    total = 0.0
    for (i, j, w) in edges
        bi = (idx >> (i - 1)) & 1
        bj = (idx >> (j - 1)) & 1
        total += w * (bi ⊻ bj)
    end
    return total
end
obj_vals = [cost_index(idx) for idx in 0:(2^n - 1)]
mixer = mixer_x(n)
angles = vcat(betas, gammas)
t0 = time()
value = exp_value(angles, mixer, obj_vals)
println("LCQAOA_RESULT ", value, " ", time() - t0)
"""
    value, seconds, out = _run_julia_script(script)
    return {
        "method": "juliqaoa_cpu_external",
        "status": "ok",
        "value": value,
        "seconds": seconds,
        "backend": "julia_JuliQAOA_statevector",
        "notes": "official JuliQAOA main branch statevector API",
    }


def run_mps_juliqaoa(case: dict) -> dict:
    n = int(case["n"])
    if case.get("objective", "maxcut") != "maxcut":
        raise ValueError("MPS-JuliQAOA adapter is currently wired for MaxCut only")
    edges = [(int(i), int(j), float(w)) for i, j, w in case["edges"]]
    total_weight = sum(float(w) for _, _, w in edges)
    project = str(Path.home() / "lc_implicit_qaoa_peers" / "julia-mps")
    script = f"""
using JuliQAOA
n = {n}
edges = {_julia_edges(edges)}
betas = {_julia_vector([float(x) for x in case["betas"]])}
gammas = {_julia_vector([float(x) for x in case["gammas"]])}
total_weight = {float(total_weight)!r}
interactions = ZInteractions[]
for (i, j, w) in edges
    push!(interactions, ZInteractions([i, j], -0.5 * w))
end
problem = QAOAProblem(interactions; nqubits=n)
angles = vcat(betas, gammas)
t0 = time()
raw = run_qaoa_mps(angles, problem; cutoff=1e-10, maxdim=1024)
value = 0.5 * total_weight + raw
println("LCQAOA_RESULT ", value, " ", time() - t0)
"""
    value, seconds, out = _run_julia_script(script, project=project, timeout=300)
    return {
        "method": "mps_juliqaoa_external",
        "status": "ok",
        "value": value,
        "seconds": seconds,
        "backend": "julia_JuliQAOA_mps_branch_ITensorMPS",
        "notes": "MPS-JuliQAOA branch run_qaoa_mps with MaxCut ZZ mapping; cutoff=1e-10,maxdim=1024",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        required=True,
        choices=[
            "qokit_cpu",
            "qtensor_cpu",
            "qtensor_gpu",
            "cudaq_observe",
            "cuaoa_gpu",
            "qblaze_cpu",
            "juliqaoa_cpu",
            "mps_juliqaoa",
        ],
    )
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--qtensor-transform", default="identity")
    args = parser.parse_args()

    case = load_case(args.case)
    try:
        if args.method == "qokit_cpu":
            result = run_qokit_cpu(case)
        elif args.method == "qtensor_cpu":
            result = run_qtensor_cpu(case, args.qtensor_transform)
        elif args.method == "qtensor_gpu":
            result = run_qtensor_gpu(case, args.qtensor_transform)
        elif args.method == "cudaq_observe":
            result = run_cudaq_observe(case)
        elif args.method == "cuaoa_gpu":
            result = run_cuaoa_gpu(case)
        elif args.method == "qblaze_cpu":
            result = run_qblaze_cpu(case)
        elif args.method == "juliqaoa_cpu":
            result = run_juliqaoa_cpu(case)
        elif args.method == "mps_juliqaoa":
            result = run_mps_juliqaoa(case)
        else:
            raise ValueError(args.method)
    except Exception as exc:
        result = {
            "method": args.method,
            "status": f"failed:{type(exc).__name__}",
            "value": float("nan"),
            "seconds": 0.0,
            "backend": "",
            "notes": str(exc)[:1000],
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
