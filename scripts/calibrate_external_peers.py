from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lcqaoa.graphs import random_regular_graph
from lcqaoa.qaoa import full_state_expectation


PREFIX = Path.home() / "lc_implicit_qaoa_peers"
CORE_PY = PREFIX / "venvs" / "lcqaoa-core" / "bin" / "python"
QTENSOR_PY = PREFIX / "venvs" / "qtensor-py310" / "bin" / "python"


def run_external(py: Path, args: list[str], out: Path) -> dict:
    cmd = [str(py), str(ROOT / "scripts" / "external_peer_runner.py"), *args, "--out", str(out)]
    cp = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=180)
    if out.exists():
        data = json.loads(out.read_text(encoding="utf-8"))
        data["process_returncode"] = cp.returncode
        data["process_output_tail"] = cp.stdout[-1000:]
        return data
    return {
        "method": args[1] if len(args) > 1 else "unknown",
        "status": f"process_failed:{cp.returncode}",
        "value": float("nan"),
        "seconds": 0.0,
        "backend": "",
        "notes": cp.stdout[-1000:],
        "process_returncode": cp.returncode,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "external_peer_calibration.json")
    parser.add_argument("--markdown", type=Path, default=ROOT / "results" / "external_peer_calibration.md")
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    graph = random_regular_graph(10, 3, seed=314159)
    gammas = [0.21, 0.26]
    betas = [0.31, 0.27]
    ref = full_state_expectation(graph, gammas, betas, method="precompute", prefer_gpu=True, max_qubits=20)
    case = {
        "n": graph.n,
        "edges": [list(edge) for edge in graph.edges],
        "gammas": gammas,
        "betas": betas,
        "objective": graph.objective,
    }
    case_path = args.out.parent / "external_peer_calibration_case.json"
    case_path.write_text(json.dumps(case, indent=2), encoding="utf-8")

    results: list[dict] = []
    if CORE_PY.exists():
        results.append(
            run_external(
                CORE_PY,
                ["--method", "qokit_cpu", "--case", str(case_path)],
                args.out.parent / "cal_qokit.json",
            )
        )
        results.append(
            run_external(
                CORE_PY,
                ["--method", "cudaq_observe", "--case", str(case_path)],
                args.out.parent / "cal_cudaq.json",
            )
        )
    else:
        results.append({"method": "qokit_cpu", "status": "missing_core_python", "value": float("nan"), "seconds": 0.0})
        results.append({"method": "cudaq_observe", "status": "missing_core_python", "value": float("nan"), "seconds": 0.0})

    transforms = [
        "identity",
        "neg_gamma",
        "paper_qiskit_inverse",
        "paper_qiskit_inverse_pos",
        "half_gamma",
        "double_gamma",
    ]
    for transform in transforms:
        if QTENSOR_PY.exists():
            results.append(
                run_external(
                    QTENSOR_PY,
                    ["--method", "qtensor_cpu", "--case", str(case_path), "--qtensor-transform", transform],
                    args.out.parent / f"cal_qtensor_{transform}.json",
                )
            )
        else:
            results.append(
                {
                    "method": "qtensor_cpu_external",
                    "status": "missing_qtensor_python",
                    "value": float("nan"),
                    "seconds": 0.0,
                    "notes": transform,
                }
            )

    for item in results:
        try:
            item["abs_error_vs_full"] = abs(float(item["value"]) - ref.value)
        except Exception:
            item["abs_error_vs_full"] = float("nan")

    ok_qtensor = [
        item
        for item in results
        if item.get("method") == "qtensor_cpu_external" and item.get("status") == "ok" and math.isfinite(item["abs_error_vs_full"])
    ]
    best_qtensor = min(ok_qtensor, key=lambda x: x["abs_error_vs_full"]) if ok_qtensor else None

    report = {
        "reference": {
            "method": "full_precompute_gpu",
            "value": ref.value,
            "seconds": ref.seconds,
            "backend": ref.backend,
        },
        "case": case,
        "best_qtensor": best_qtensor,
        "results": results,
    }
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = [
        "# External Peer Calibration",
        "",
        f"Reference full-state value: `{ref.value:.10f}`",
        "",
        "| Method | Status | Value | Error | Seconds | Backend | Notes |",
        "|---|---|---:|---:|---:|---|---|",
    ]
    for item in results:
        lines.append(
            f"| {item.get('method')} | {item.get('status')} | {float(item.get('value', float('nan'))):.8g} | "
            f"{float(item.get('abs_error_vs_full', float('nan'))):.3g} | {float(item.get('seconds', 0.0)):.4g} | "
            f"{item.get('backend', '')} | {str(item.get('notes', ''))[:160]} |"
        )
    args.markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"WROTE {args.out}")
    print(f"WROTE {args.markdown}")


if __name__ == "__main__":
    main()

