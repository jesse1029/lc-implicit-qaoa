from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path


PEERS = [
    {
        "name": "CUDA-Q/cuStateVec",
        "kind": "general full-state GPU",
        "imports": ["cudaq"],
        "executables": [],
        "url": "https://nvidia.github.io/cuda-quantum/latest/using/backends/sims/svsims.html",
    },
    {
        "name": "QOKit",
        "kind": "QAOA diagonal precompute full-state",
        "imports": ["qokit"],
        "executables": [],
        "url": "https://github.com/jpmorganchase/QOKit",
    },
    {
        "name": "CUAOA",
        "kind": "CUDA-native QAOA full-state",
        "imports": ["pycuaoa"],
        "executables": ["nvcc"],
        "url": "https://github.com/JFLXB/cuaoa",
    },
    {
        "name": "QTensor",
        "kind": "tensor-network light-cone baseline",
        "imports": ["qtensor", "qtensor_ai"],
        "executables": [],
        "url": "https://github.com/danlkv/QTensor",
    },
    {
        "name": "JuliQAOA",
        "kind": "Julia QAOA simulator",
        "imports": [],
        "executables": ["julia"],
        "url": "https://arxiv.org/html/2312.06451v1",
    },
    {
        "name": "MPS-JuliQAOA",
        "kind": "MPS QAOA simulator",
        "imports": [],
        "executables": ["julia"],
        "url": "https://arxiv.org/html/2508.05883",
    },
    {
        "name": "BMQSim",
        "kind": "lossy compression simulator",
        "imports": ["bmqsim"],
        "executables": [],
        "url": "https://arxiv.org/abs/2410.14088",
    },
    {
        "name": "qblaze",
        "kind": "sparse-state simulator",
        "imports": ["qblaze"],
        "executables": [],
        "url": "https://github.com/insait-institute/qblaze",
    },
    {
        "name": "Queen/QueenV2",
        "kind": "general full-state simulator",
        "imports": ["queen"],
        "executables": [],
        "url": "https://arxiv.org/abs/2406.14084",
    },
]


@dataclass
class PeerStatus:
    name: str
    kind: str
    status: str
    runnable: bool
    reason: str
    url: str


def has_import(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def check_peer(peer: dict) -> PeerStatus:
    imports = peer["imports"]
    import_hits = [name for name in imports if has_import(name)]
    exe_hits = [name for name in peer["executables"] if shutil.which(name)]

    if peer["name"] == "QOKit" and import_hits:
        try:
            import qokit

            qokit_root = Path(qokit.__file__).resolve().parent
            furx = qokit_root / "fur" / "nbcuda" / "furx.cu"
            if furx.exists():
                reason = "python import: qokit; GPU source file furx.cu present"
                status = "installed"
            else:
                reason = "python import: qokit; CPU simulator usable, but GPU FUR path is missing qokit/fur/nbcuda/furx.cu"
                status = "installed-cpu-only"
            return PeerStatus(peer["name"], peer["kind"], status, True, reason, peer["url"])
        except Exception as exc:
            return PeerStatus(peer["name"], peer["kind"], "import-probe-failed", False, str(exc), peer["url"])

    if imports and import_hits:
        return PeerStatus(peer["name"], peer["kind"], "installed", True, f"python import: {', '.join(import_hits)}", peer["url"])
    if peer["executables"] and exe_hits:
        return PeerStatus(peer["name"], peer["kind"], "toolchain-present", False, f"found executable: {', '.join(exe_hits)}; adapter not wired", peer["url"])
    if peer["name"] == "CUAOA" and not shutil.which("nvcc"):
        return PeerStatus(peer["name"], peer["kind"], "unavailable", False, "nvcc is absent, so the CUDA source baseline cannot be built on this host", peer["url"])
    if peer["name"] in {"JuliQAOA", "MPS-JuliQAOA"} and not shutil.which("julia"):
        return PeerStatus(peer["name"], peer["kind"], "unavailable", False, "Julia is absent on this host", peer["url"])
    if peer["name"] == "QTensor":
        return PeerStatus(
            peer["name"],
            peer["kind"],
            "unavailable",
            False,
            "not installed; PyPI qtensor 0.1.2 pins qiskit==0.17.0 and conflicts with the active QOKit/Qiskit environment",
            peer["url"],
        )
    return PeerStatus(peer["name"], peer["kind"], "unavailable", False, "package/module not installed in the active environment", peer["url"])


def collect_environment() -> dict:
    env = {"python": sys.version.split()[0]}
    for cmd in [
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version,compute_cap", "--format=csv,noheader"],
        ["which", "nvcc"],
        ["which", "julia"],
    ]:
        try:
            out = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT, timeout=8).strip()
        except Exception as exc:
            out = f"ERROR: {exc}"
        env[" ".join(cmd)] = out
    return env


def write_markdown(path: Path, statuses: list[PeerStatus], env: dict) -> None:
    lines = [
        "# Peer Method Status",
        "",
        "## Environment",
        "",
    ]
    for k, v in env.items():
        lines.append(f"- `{k}`: `{v}`")
    lines.extend(
        [
            "",
            "## Peer Methods",
            "",
            "| Method | Kind | Runnable here | Status | Reason | Source |",
            "|---|---|---:|---|---|---|",
        ]
    )
    for st in statuses:
        runnable = "yes" if st.runnable else "no"
        lines.append(f"| {st.name} | {st.kind} | {runnable} | {st.status} | {st.reason} | {st.url} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("results/peer_status.json"))
    parser.add_argument("--markdown", type=Path, default=Path("results/peer_status.md"))
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    statuses = [check_peer(peer) for peer in PEERS]
    env = collect_environment()
    args.out.write_text(
        json.dumps({"environment": env, "peers": [asdict(st) for st in statuses]}, indent=2),
        encoding="utf-8",
    )
    write_markdown(args.markdown, statuses, env)
    for st in statuses:
        print(f"{st.name}: {st.status} ({st.reason})")


if __name__ == "__main__":
    main()
