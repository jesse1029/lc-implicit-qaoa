from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CUAOA_PY = Path.home() / "lc_implicit_qaoa_peers" / "venvs" / "cuaoa-py312" / "bin" / "python"


class MemorySampler:
    def __init__(self, xp=None, interval: float = 0.01):
        self.xp = xp
        self.interval = interval
        self.pid = os.getpid()
        self.stop_event = threading.Event()
        self.peak_device_mb = 0.0
        self.peak_allocated_mb = 0.0
        self.peak_reserved_mb = 0.0
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _device_memory(self) -> float:
        try:
            cp = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory", "--format=csv,noheader,nounits"],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, timeout=2,
            )
            for line in cp.stdout.splitlines():
                parts = [x.strip() for x in line.split(",")]
                if len(parts) >= 2 and int(parts[0]) == self.pid:
                    return float(parts[1])
        except Exception:
            pass
        return 0.0

    def _run(self):
        while not self.stop_event.is_set():
            self.peak_device_mb = max(self.peak_device_mb, self._device_memory())
            if self.xp is not None:
                try:
                    pool = self.xp.get_default_memory_pool()
                    self.peak_allocated_mb = max(self.peak_allocated_mb, pool.used_bytes() / 1024**2)
                    self.peak_reserved_mb = max(self.peak_reserved_mb, pool.total_bytes() / 1024**2)
                except Exception:
                    pass
            self.stop_event.wait(self.interval)

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=3)
        self.peak_device_mb = max(self.peak_device_mb, self._device_memory())


def load_case(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def worker_cuaoa(case_path: Path, out_path: Path, repeats: int) -> None:
    process_start = time.perf_counter()
    import numpy as np
    import pycuaoa

    case = load_case(case_path)
    sampler = MemorySampler()
    sampler.start()
    prep_start = time.perf_counter()
    polynomial: dict[tuple[int, ...], float] = {}
    if case["objective"] == "maxcut":
        for i, j, w in case["edges"]:
            i, j, w = int(i), int(j), float(w)
            polynomial[(i,)] = polynomial.get((i,), 0.0) + w
            polynomial[(j,)] = polynomial.get((j,), 0.0) + w
            key = tuple(sorted((i, j)))
            polynomial[key] = polynomial.get(key, 0.0) - 2.0 * w
    else:
        for i, j, w in case["edges"]:
            key = tuple(sorted((int(i), int(j))))
            polynomial[key] = polynomial.get(key, 0.0) + float(w)
        for i, w in case["fields"]:
            key = (int(i),)
            polynomial[key] = polynomial.get(key, 0.0) + float(w)
    preprocess_seconds = time.perf_counter() - prep_start
    setup_start = time.perf_counter()
    sim = pycuaoa.CUAOA.from_map(int(case["n"]), polynomial, depth=int(case["p"]))
    handle = pycuaoa.create_handle(int(case["n"]))
    setup_seconds = time.perf_counter() - setup_start
    betas = np.asarray(case["betas"], dtype=np.float64)
    # CUAOA applies exp(+i gamma C), whereas LC uses exp(-i gamma C).
    # Negate gamma at the adapter boundary and apply the chain rule to d/dgamma.
    gammas = -np.asarray(case["gammas"], dtype=np.float64)

    def call():
        start = time.perf_counter()
        grads, value = sim.gradients(handle, betas=betas, gammas=gammas)
        seconds = time.perf_counter() - start
        gradient = np.concatenate(
            [-np.asarray(grads.gammas, dtype=np.float64), np.asarray(grads.betas, dtype=np.float64)]
        )
        return seconds, float(value), gradient

    try:
        cold, value, gradient = call()
        warm, value, gradient = call()
        steady = []
        for _ in range(repeats):
            seconds, value, gradient = call()
            steady.append(seconds)
        status = "ok"
        notes = "official pycuaoa CUAOA.from_map gradients(); native complex128/float64; gamma negated to map CUAOA exp(+i gamma C) to LC exp(-i gamma C), with chain-rule gradient sign; allocated/reserved allocator counters are not exposed"
    except Exception as exc:
        cold = warm = float("nan")
        steady = []
        value = float("nan")
        gradient = np.asarray([], dtype=np.float64)
        status = f"failed:{type(exc).__name__}"
        notes = str(exc)[:1000]
    finally:
        try:
            handle.destroy()
        except Exception:
            pass
        sampler.stop()
    payload = {
        "backend": "CUAOA official pycuaoa 0.1.0",
        "status": status,
        "precision": "complex128/float64 native",
        "preprocess_seconds": preprocess_seconds,
        "setup_seconds": setup_seconds,
        "cold_seconds": cold,
        "warm_seconds": warm,
        "steady_seconds": steady,
        "steady_median_seconds": float(np.median(steady)) if steady else float("nan"),
        "peak_device_mb": sampler.peak_device_mb,
        "peak_allocated_mb": float("nan"),
        "peak_reserved_mb": float("nan"),
        "value": value,
        "gradient": gradient.tolist(),
        "notes": notes,
        "worker_internal_seconds": time.perf_counter() - process_start,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def worker_lc(case_path: Path, out_path: Path, repeats: int) -> None:
    process_start = time.perf_counter()
    import numpy as np

    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "scripts"))
    from lcqaoa.backend import get_backend
    from lcqaoa.graphs import WeightedGraph
    from lcqaoa.lightcone import lightcone_gradient_adjoint
    from benchmark_common import cone_metrics

    case = load_case(case_path)
    prep_start = time.perf_counter()
    graph = WeightedGraph(
        int(case["n"]),
        tuple((int(i), int(j), float(w)) for i, j, w in case["edges"]),
        tuple((int(i), float(w)) for i, w in case["fields"]),
        str(case["objective"]),
    )
    cone = cone_metrics(graph, int(case["p"]))
    preprocess_seconds = time.perf_counter() - prep_start
    backend = get_backend(prefer_gpu=True)
    backend.free_memory_pool()
    sampler = MemorySampler(backend.xp)
    sampler.start()
    gammas = np.asarray(case["gammas"], dtype=np.float64)
    betas = np.asarray(case["betas"], dtype=np.float64)

    def call():
        start = time.perf_counter()
        stats = lightcone_gradient_adjoint(
            graph, gammas, betas, p=int(case["p"]), prefer_gpu=True,
            max_k=24, max_batch_states=1 << 20,
            complex_dtype=np.complex128, float_dtype=np.float64,
        )
        seconds = time.perf_counter() - start
        return seconds, float(stats.value), np.asarray(stats.gradient, dtype=np.float64), stats.status

    setup_seconds = 0.0
    try:
        cold, value, gradient, status = call()
        warm, value, gradient, status = call()
        steady = []
        for _ in range(repeats):
            seconds, value, gradient, status = call()
            steady.append(seconds)
        notes = f"LC local adjoint complex128/float64; kmax={cone['kmax']}; total_cone_states={cone['total_cone_states']}"
    except Exception as exc:
        cold = warm = float("nan")
        steady = []
        value = float("nan")
        gradient = np.asarray([], dtype=np.float64)
        status = f"failed:{type(exc).__name__}"
        notes = str(exc)[:1000]
    sampler.stop()
    payload = {
        "backend": "LC local adjoint",
        "status": status,
        "precision": "complex128/float64 matched",
        "preprocess_seconds": preprocess_seconds,
        "setup_seconds": setup_seconds,
        "cold_seconds": cold,
        "warm_seconds": warm,
        "steady_seconds": steady,
        "steady_median_seconds": float(np.median(steady)) if steady else float("nan"),
        "peak_device_mb": sampler.peak_device_mb,
        "peak_allocated_mb": sampler.peak_allocated_mb,
        "peak_reserved_mb": sampler.peak_reserved_mb,
        "value": value,
        "gradient": gradient.tolist(),
        "kmax": int(cone["kmax"]),
        "total_cone_states": int(cone["total_cone_states"]),
        "notes": notes,
        "worker_internal_seconds": time.perf_counter() - process_start,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


@dataclass
class ResultRow:
    host_label: str
    gpu_name: str
    family: str
    n: int
    p: int
    seed: int
    backend: str
    status: str
    precision: str
    preprocess_seconds: float
    setup_seconds: float
    process_wall_seconds: float
    process_startup_seconds: float
    cold_seconds: float
    warm_seconds: float
    steady_median_seconds: float
    peak_device_mb: float
    peak_allocated_mb: float
    peak_reserved_mb: float
    value: float
    gradient_norm: float
    value_abs_error_vs_lc: float
    gradient_relative_l2_error_vs_lc: float
    gradient_cosine_vs_lc: float
    kmax: float
    total_cone_states: float
    notes: str


def random_regular_edges(n: int, degree: int, seed: int):
    try:
        sys.path.insert(0, str(ROOT))
        from lcqaoa.graphs import random_regular_graph

        return list(random_regular_graph(n, degree, seed=seed).edges)
    except Exception:
        rng = random.Random(seed)
        for _ in range(5000):
            stubs = [i for i in range(n) for _ in range(degree)]
            rng.shuffle(stubs)
            edges = set()
            ok = True
            for a, b in zip(stubs[0::2], stubs[1::2]):
                if a == b or tuple(sorted((a, b))) in edges:
                    ok = False
                    break
                edges.add(tuple(sorted((a, b))))
            if ok:
                return [(i, j, 1.0) for i, j in sorted(edges)]
        raise RuntimeError("regular graph generation failed")


def make_case(family: str, n: int, p: int, seed_id: int):
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "scripts"))
    from lcqaoa.graphs import weighted_qubo_graph
    from benchmark_common import params_for_depth

    seed = 550000 + seed_id * 997 + n * 37 + p * 131
    if family == "3regular":
        edges = random_regular_edges(n, 3, seed)
        fields = []
        objective = "maxcut"
    elif family == "weighted_qubo_er2":
        graph = weighted_qubo_graph(n, min(0.45, 2.0 / max(2, n)), seed=seed, field_scale=0.7)
        edges = list(graph.edges)
        fields = list(graph.fields)
        objective = "qubo"
    else:
        raise ValueError(family)
    gammas, betas = params_for_depth(p, seed=seed_id)
    return {
        "family": family, "n": n, "p": p, "seed": seed_id,
        "objective": objective, "edges": edges, "fields": fields,
        "gammas": list(map(float, gammas)), "betas": list(map(float, betas)),
    }


def run_process(python: Path, mode: str, case: Path, out: Path, repeats: int, env: dict, timeout: int):
    cmd = [str(python), str(Path(__file__).resolve()), "--worker", mode, "--case", str(case), "--worker-out", str(out), "--repeats", str(repeats)]
    start = time.perf_counter()
    cp = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        encoding="utf-8", errors="replace", timeout=timeout, env=env,
    )
    wall = time.perf_counter() - start
    if cp.returncode != 0 or not out.exists():
        return {"backend": mode, "status": f"process_failed_rc{cp.returncode}", "notes": cp.stdout[-1500:]}, wall
    return json.loads(out.read_text(encoding="utf-8")), wall


def main_driver(args):
    import numpy as np

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cases_dir = args.out_dir / "cases"
    logs_dir = args.out_dir / "logs"
    cases_dir.mkdir(exist_ok=True)
    logs_dir.mkdir(exist_ok=True)
    rows = []
    csv_path = args.out_dir / "P0_2_official_cuaoa_gradient.csv"
    base_env = os.environ.copy()
    lc_env = base_env.copy()
    lc_env["PYTHONPATH"] = os.pathsep.join([str(ROOT), str(ROOT / "scripts"), lc_env.get("PYTHONPATH", "")])
    cuaoa_env = base_env.copy()
    cuaoa_env.pop("PYTHONPATH", None)
    if args.cuaoa_library_path:
        cuaoa_env["LD_LIBRARY_PATH"] = str(args.cuaoa_library_path)

    for family in args.families:
        for n in args.ns:
            for seed_id in range(args.seeds):
                case = make_case(family, n, 2, seed_id)
                stem = f"{family}_n{n}_p2_seed{seed_id}"
                case_path = cases_dir / f"{stem}.json"
                case_path.write_text(json.dumps(case, indent=2), encoding="utf-8")
                lc_out = cases_dir / f"{stem}_lc.json"
                peer_out = cases_dir / f"{stem}_cuaoa.json"
                lc, lc_wall = run_process(Path(sys.executable), "lc", case_path, lc_out, args.repeats, lc_env, args.timeout)
                peer, peer_wall = run_process(args.cuaoa_python, "cuaoa", case_path, peer_out, args.repeats, cuaoa_env, args.timeout)
                lg = np.asarray(lc.get("gradient", []), dtype=float)
                pg = np.asarray(peer.get("gradient", []), dtype=float)
                value_err = abs(float(peer.get("value", np.nan)) - float(lc.get("value", np.nan)))
                if len(lg) and len(pg) == len(lg):
                    grad_err = float(np.linalg.norm(pg - lg) / max(np.linalg.norm(lg), 1e-15))
                    cosine = float(np.dot(pg, lg) / max(np.linalg.norm(pg) * np.linalg.norm(lg), 1e-15))
                else:
                    grad_err = cosine = float("nan")
                for payload, wall in [(lc, lc_wall), (peer, peer_wall)]:
                    grad = np.asarray(payload.get("gradient", []), dtype=float)
                    rows.append(
                        ResultRow(
                            args.host_label, args.gpu_name, family, n, 2, seed_id,
                            str(payload.get("backend", "")), str(payload.get("status", "")), str(payload.get("precision", "")),
                            float(payload.get("preprocess_seconds", np.nan)), float(payload.get("setup_seconds", np.nan)),
                            wall, max(0.0, wall - float(payload.get("worker_internal_seconds", wall))),
                            float(payload.get("cold_seconds", np.nan)), float(payload.get("warm_seconds", np.nan)),
                            float(payload.get("steady_median_seconds", np.nan)), float(payload.get("peak_device_mb", np.nan)),
                            float(payload.get("peak_allocated_mb", np.nan)), float(payload.get("peak_reserved_mb", np.nan)),
                            float(payload.get("value", np.nan)), float(np.linalg.norm(grad)) if len(grad) else float("nan"),
                            0.0 if payload is lc else value_err, 0.0 if payload is lc else grad_err,
                            1.0 if payload is lc else cosine, float(lc.get("kmax", np.nan)),
                            float(lc.get("total_cone_states", np.nan)), str(payload.get("notes", ""))[:1000],
                        )
                    )
                with csv_path.open("w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
                    writer.writeheader()
                    for row in rows:
                        writer.writerow(asdict(row))
                print(f"{args.host_label} {stem} LC={lc.get('status')} CUAOA={peer.get('status')} err={value_err:.3e} grad={grad_err:.3e}", flush=True)
    summary = []
    for family in args.families:
        for n in args.ns:
            subset = [r for r in rows if r.family == family and r.n == n]
            item = {"family": family, "n": n}
            for backend_key, token in [("lc", "LC local"), ("cuaoa", "CUAOA official")]:
                values = [r.steady_median_seconds for r in subset if token in r.backend and r.status == "ok"]
                item[f"{backend_key}_success"] = len(values)
                item[f"{backend_key}_median_seconds"] = float(np.median(values)) if values else float("nan")
            summary.append(item)
    (args.out_dir / "P0_2_official_cuaoa_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", choices=["lc", "cuaoa"])
    parser.add_argument("--case", type=Path)
    parser.add_argument("--worker-out", type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--families", nargs="+", default=["3regular", "weighted_qubo_er2"])
    parser.add_argument("--ns", nargs="+", type=int, default=[18, 20, 22, 24, 26])
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--cuaoa-python", type=Path, default=DEFAULT_CUAOA_PY)
    parser.add_argument("--cuaoa-library-path", type=Path)
    parser.add_argument("--host-label", default="rtx3070")
    parser.add_argument("--gpu-name", default="NVIDIA GeForce RTX 3070 8GB")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "aaai27_followup_20260711" / "P0_2_official_cuaoa_gradient_rtx3070")
    args = parser.parse_args()
    if args.worker == "cuaoa":
        worker_cuaoa(args.case, args.worker_out, args.repeats)
    elif args.worker == "lc":
        worker_lc(args.case, args.worker_out, args.repeats)
    else:
        main_driver(args)


if __name__ == "__main__":
    main()
