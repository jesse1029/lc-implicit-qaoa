from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


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
            proc = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-compute-apps=pid,used_gpu_memory",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=2,
                check=False,
            )
            for line in proc.stdout.splitlines():
                pid, memory, *_ = [part.strip() for part in line.split(",")]
                if int(pid) == self.pid:
                    return float(memory)
        except Exception:
            pass
        return 0.0

    def _run(self) -> None:
        while not self.stop_event.is_set():
            self.peak_device_mb = max(self.peak_device_mb, self._device_memory())
            if self.xp is not None:
                try:
                    pool = self.xp.get_default_memory_pool()
                    self.peak_allocated_mb = max(self.peak_allocated_mb, pool.used_bytes() / 2**20)
                    self.peak_reserved_mb = max(self.peak_reserved_mb, pool.total_bytes() / 2**20)
                except Exception:
                    pass
            self.stop_event.wait(self.interval)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=3)
        self.peak_device_mb = max(self.peak_device_mb, self._device_memory())


def load_case(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def cost_hamiltonian(qml, case: dict):
    coeffs: list[float] = []
    ops = []
    if case["objective"] == "maxcut":
        for i, j, weight in case["edges"]:
            weight = float(weight)
            coeffs.extend([weight / 2.0, -weight / 2.0])
            ops.extend([qml.Identity(int(i)), qml.PauliZ(int(i)) @ qml.PauliZ(int(j))])
    else:
        for i, j, weight in case["edges"]:
            i, j, weight = int(i), int(j), float(weight)
            coeffs.extend([weight / 4.0, -weight / 4.0, -weight / 4.0, weight / 4.0])
            ops.extend(
                [
                    qml.Identity(i),
                    qml.PauliZ(i),
                    qml.PauliZ(j),
                    qml.PauliZ(i) @ qml.PauliZ(j),
                ]
            )
        for i, weight in case["fields"]:
            i, weight = int(i), float(weight)
            coeffs.extend([weight / 2.0, -weight / 2.0])
            ops.extend([qml.Identity(i), qml.PauliZ(i)])
    return qml.Hamiltonian(coeffs, ops)


def apply_cost_layer(qml, case: dict, gamma) -> None:
    if case["objective"] == "maxcut":
        for i, j, weight in case["edges"]:
            qml.MultiRZ(-gamma * float(weight), wires=[int(i), int(j)])
    else:
        for i, j, weight in case["edges"]:
            i, j, weight = int(i), int(j), float(weight)
            qml.RZ(-gamma * weight / 2.0, wires=i)
            qml.RZ(-gamma * weight / 2.0, wires=j)
            qml.MultiRZ(gamma * weight / 2.0, wires=[i, j])
        for i, weight in case["fields"]:
            qml.RZ(-gamma * float(weight), wires=int(i))


def worker_lightning(case_path: Path, output_path: Path, repeats: int) -> None:
    process_start = time.perf_counter()
    import autograd
    import pennylane as qml
    from pennylane import numpy as pnp

    case = load_case(case_path)
    sampler = MemorySampler()
    sampler.start()
    prep_start = time.perf_counter()
    hamiltonian = cost_hamiltonian(qml, case)
    preprocess_seconds = time.perf_counter() - prep_start
    setup_start = time.perf_counter()
    device = qml.device(
        "lightning.gpu",
        wires=int(case["n"]),
        c_dtype=np.complex64,
        batch_obs=True,
    )

    @qml.qnode(device, interface="autograd", diff_method="adjoint")
    def qnode(theta):
        for wire in range(int(case["n"])):
            qml.Hadamard(wires=wire)
        p = int(case["p"])
        for layer in range(p):
            apply_cost_layer(qml, case, theta[layer])
            for wire in range(int(case["n"])):
                qml.RX(2.0 * theta[p + layer], wires=wire)
        return qml.expval(hamiltonian)

    value_and_grad = autograd.value_and_grad(qnode)
    theta = pnp.array(
        list(case["gammas"]) + list(case["betas"]),
        dtype=np.float32,
        requires_grad=True,
    )
    setup_seconds = time.perf_counter() - setup_start

    def call():
        start = time.perf_counter()
        value, gradient = value_and_grad(theta)
        seconds = time.perf_counter() - start
        return seconds, float(value), np.asarray(gradient, dtype=np.float64)

    try:
        cold, value, gradient = call()
        warm, value, gradient = call()
        steady = []
        for _ in range(repeats):
            seconds, value, gradient = call()
            steady.append(seconds)
        status = "ok"
        notes = (
            "official PennyLane-Lightning-GPU adjoint; lightning.gpu c_dtype=complex64; "
            "float32 trainable angles; allocator allocated/reserved counters are not exposed"
        )
    except Exception as exc:
        cold = warm = float("nan")
        steady = []
        value = float("nan")
        gradient = np.asarray([], dtype=np.float64)
        status = f"failed:{type(exc).__name__}"
        notes = repr(exc)[:1200]
    finally:
        sampler.stop()

    payload = {
        "backend": "PennyLane-Lightning-GPU 0.45.0 official adjoint",
        "status": status,
        "precision": "complex64/float32 native",
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
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def worker_lc(case_path: Path, output_path: Path, repeats: int) -> None:
    process_start = time.perf_counter()
    sys.path.insert(0, str(ROOT))
    from lcqaoa.backend import get_backend
    from lcqaoa.graphs import WeightedGraph
    from lcqaoa.lightcone import lightcone_gradient_adjoint

    case = load_case(case_path)
    prep_start = time.perf_counter()
    graph = WeightedGraph(
        n=int(case["n"]),
        edges=tuple((int(i), int(j), float(w)) for i, j, w in case["edges"]),
        fields=tuple((int(i), float(w)) for i, w in case["fields"]),
        objective=str(case["objective"]),
    )
    preprocess_seconds = time.perf_counter() - prep_start
    backend = get_backend(prefer_gpu=True)
    backend.free_memory_pool()
    sampler = MemorySampler(backend.xp)
    sampler.start()

    def call():
        start = time.perf_counter()
        stats = lightcone_gradient_adjoint(
            graph,
            case["gammas"],
            case["betas"],
            p=int(case["p"]),
            prefer_gpu=True,
            max_k=24,
            max_batch_states=1 << 20,
            complex_dtype=np.complex64,
            float_dtype=np.float32,
        )
        seconds = time.perf_counter() - start
        gradient = np.asarray(stats.gradient if stats.gradient is not None else [], dtype=np.float64)
        return seconds, float(stats.value), gradient, stats.status

    try:
        cold, value, gradient, status = call()
        warm, value, gradient, status = call()
        steady = []
        for _ in range(repeats):
            seconds, value, gradient, status = call()
            steady.append(seconds)
        notes = "LC local adjoint; complex64 states, float32 local costs, float64 host reduction"
    except Exception as exc:
        cold = warm = float("nan")
        steady = []
        value = float("nan")
        gradient = np.asarray([], dtype=np.float64)
        status = f"failed:{type(exc).__name__}"
        notes = repr(exc)[:1200]
    finally:
        sampler.stop()

    payload = {
        "backend": "LC local adjoint",
        "status": status,
        "precision": "complex64/float32 matched",
        "preprocess_seconds": preprocess_seconds,
        "setup_seconds": 0.0,
        "cold_seconds": cold,
        "warm_seconds": warm,
        "steady_seconds": steady,
        "steady_median_seconds": float(np.median(steady)) if steady else float("nan"),
        "peak_device_mb": sampler.peak_device_mb,
        "peak_allocated_mb": sampler.peak_allocated_mb,
        "peak_reserved_mb": sampler.peak_reserved_mb,
        "value": value,
        "gradient": gradient.tolist(),
        "notes": notes,
        "worker_internal_seconds": time.perf_counter() - process_start,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def selected_case(path: Path) -> bool:
    if path.stem.endswith(("_lc", "_cuaoa", "_lightning")):
        return False
    try:
        case = load_case(path)
    except Exception:
        return False
    n = int(case.get("n", -1))
    family = case.get("family")
    if int(case.get("p", -1)) != 2 or int(case.get("seed", -1)) not in range(5):
        return False
    return (family == "3regular" and n in {18, 20, 22, 24, 26}) or (
        family == "weighted_qubo_er2" and n in {18, 20, 22, 24}
    )


def run_worker(script: Path, backend: str, case: Path, output: Path, repeats: int, env: dict) -> float:
    start = time.perf_counter()
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--worker",
            backend,
            "--case",
            str(case),
            "--worker-output",
            str(output),
            "--repeats",
            str(repeats),
        ],
        check=False,
        env=env,
    )
    return time.perf_counter() - start


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--host-label", default="rtx3090_gpu1")
    parser.add_argument("--gpu", default="1")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--worker", choices=["lc", "lightning"])
    parser.add_argument("--case", type=Path)
    parser.add_argument("--worker-output", type=Path)
    args = parser.parse_args()

    if args.worker:
        if args.worker == "lc":
            worker_lc(args.case, args.worker_output, args.repeats)
        else:
            worker_lightning(args.case, args.worker_output, args.repeats)
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    worker_dir = args.output_dir / "worker_json"
    worker_dir.mkdir(exist_ok=True)
    cases = sorted(path for path in args.cases_dir.glob("*.json") if selected_case(path))
    if args.limit:
        cases = cases[: args.limit]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    rows = []
    for case_path in cases:
        case = load_case(case_path)
        payloads = {}
        for backend in ("lc", "lightning"):
            output = worker_dir / f"{case_path.stem}_{backend}.json"
            wall = run_worker(Path(__file__), backend, case_path, output, args.repeats, env)
            if output.exists():
                payload = json.loads(output.read_text(encoding="utf-8"))
            else:
                payload = {
                    "backend": backend,
                    "status": "failed:no_worker_output",
                    "precision": "unknown",
                    "notes": "worker did not write output",
                }
            payload["process_wall_seconds"] = wall
            payload["process_startup_seconds"] = wall - float(payload.get("worker_internal_seconds", wall))
            payloads[backend] = payload

        lc = payloads["lc"]
        peer = payloads["lightning"]
        lc_grad = np.asarray(lc.get("gradient", []), dtype=np.float64)
        peer_grad = np.asarray(peer.get("gradient", []), dtype=np.float64)
        value_error = abs(float(peer.get("value", math.nan)) - float(lc.get("value", math.nan)))
        if lc_grad.size and lc_grad.shape == peer_grad.shape:
            gradient_error = float(np.linalg.norm(peer_grad - lc_grad) / max(np.linalg.norm(lc_grad), 1e-30))
            cosine = float(np.dot(peer_grad, lc_grad) / max(np.linalg.norm(peer_grad) * np.linalg.norm(lc_grad), 1e-30))
        else:
            gradient_error = cosine = math.nan

        for key, payload in payloads.items():
            rows.append(
                {
                    "host_label": args.host_label,
                    "family": case["family"],
                    "n": case["n"],
                    "p": case["p"],
                    "seed": case["seed"],
                    "backend": payload.get("backend"),
                    "status": payload.get("status"),
                    "precision": payload.get("precision"),
                    "preprocess_seconds": payload.get("preprocess_seconds", math.nan),
                    "setup_seconds": payload.get("setup_seconds", math.nan),
                    "process_wall_seconds": payload.get("process_wall_seconds", math.nan),
                    "process_startup_seconds": payload.get("process_startup_seconds", math.nan),
                    "cold_seconds": payload.get("cold_seconds", math.nan),
                    "warm_seconds": payload.get("warm_seconds", math.nan),
                    "steady_median_seconds": payload.get("steady_median_seconds", math.nan),
                    "peak_device_mb": payload.get("peak_device_mb", math.nan),
                    "peak_allocated_mb": payload.get("peak_allocated_mb", math.nan),
                    "peak_reserved_mb": payload.get("peak_reserved_mb", math.nan),
                    "value": payload.get("value", math.nan),
                    "gradient_norm": float(np.linalg.norm(np.asarray(payload.get("gradient", []), dtype=np.float64))),
                    "value_abs_error_vs_lc": 0.0 if key == "lc" else value_error,
                    "gradient_relative_l2_error_vs_lc": 0.0 if key == "lc" else gradient_error,
                    "gradient_cosine_vs_lc": 1.0 if key == "lc" else cosine,
                    "notes": payload.get("notes", ""),
                }
            )
        print(
            f"{case_path.stem}: LC={lc.get('status')} Lightning={peer.get('status')} "
            f"value_err={value_error:.3e} grad_err={gradient_error:.3e}",
            flush=True,
        )

    output_csv = args.output_dir / "P0_2_lightning_gpu_adjoint_c64.csv"
    if rows:
        with output_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    summary = {
        "requested_cases": len(cases),
        "rows": len(rows),
        "successful_lc": sum(row["backend"] == "LC local adjoint" and row["status"] == "ok" for row in rows),
        "successful_lightning": sum("Lightning" in str(row["backend"]) and row["status"] == "ok" for row in rows),
        "max_value_abs_error": max((float(row["value_abs_error_vs_lc"]) for row in rows if "Lightning" in str(row["backend"]) and math.isfinite(float(row["value_abs_error_vs_lc"]))), default=math.nan),
        "max_gradient_relative_l2_error": max((float(row["gradient_relative_l2_error_vs_lc"]) for row in rows if "Lightning" in str(row["backend"]) and math.isfinite(float(row["gradient_relative_l2_error_vs_lc"]))), default=math.nan),
        "minimum_gradient_cosine": min((float(row["gradient_cosine_vs_lc"]) for row in rows if "Lightning" in str(row["backend"]) and math.isfinite(float(row["gradient_cosine_vs_lc"]))), default=math.nan),
    }
    (args.output_dir / "P0_2_lightning_gpu_adjoint_c64_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
