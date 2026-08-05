from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREFIX = Path.home() / "lc_implicit_qaoa_peers"


@dataclass
class ToolStatus:
    name: str
    path: str
    version: str
    status: str


@dataclass
class PythonEnvStatus:
    name: str
    python: str
    version: str
    modules: dict[str, str]
    status: str


@dataclass
class RepoStatus:
    name: str
    path: str
    present: bool
    revision: str
    files: list[str]
    status: str


def run(cmd: list[str], timeout: int = 30, env: dict[str, str] | None = None) -> tuple[int, str]:
    try:
        cp = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
            env=env,
        )
        return cp.returncode, cp.stdout.strip()
    except Exception as exc:
        return 999, f"{type(exc).__name__}: {exc}"


def first_existing(paths: list[Path]) -> str:
    for path in paths:
        if path.exists():
            return str(path)
    return ""


def tool_status(name: str, candidates: list[str], version_args: list[str]) -> ToolStatus:
    path = ""
    for candidate in candidates:
        found = shutil.which(candidate)
        if found:
            path = found
            break
        if Path(candidate).exists():
            path = candidate
            break
    if not path:
        return ToolStatus(name, "", "", "missing")
    code, out = run([path, *version_args], timeout=30)
    return ToolStatus(name, path, out.splitlines()[0] if out else "", "ok" if code == 0 else f"version_failed:{code}")


def python_env_status(name: str, python_path: Path, modules: list[str]) -> PythonEnvStatus:
    if not python_path.exists():
        return PythonEnvStatus(name, str(python_path), "", {}, "missing")
    code, out = run(
        [
            str(python_path),
            "-c",
            (
                "import importlib.metadata as md, importlib.util, json, sys;"
                "mods=" + repr(modules) + ";"
                "res={};"
                "\nfor m in mods:\n"
                "    spec=importlib.util.find_spec(m)\n"
                "    if not spec:\n"
                "        res[m]='MISSING'\n"
                "    else:\n"
                "        try: res[m]='OK '+md.version(m)\n"
                "        except Exception: res[m]='OK'\n"
                "print(json.dumps({'version': sys.version.split()[0], 'executable': sys.executable, 'modules': res}))"
            ),
        ],
        timeout=60,
    )
    if code != 0:
        return PythonEnvStatus(name, str(python_path), "", {}, f"probe_failed:{code}:{out[:180]}")
    data = json.loads(out)
    status = "ok"
    return PythonEnvStatus(name, data["executable"], data["version"], data["modules"], status)


def repo_status(name: str, path: Path, file_globs: list[str]) -> RepoStatus:
    if not path.exists():
        return RepoStatus(name, str(path), False, "", [], "missing")
    code, rev = run(["git", "-C", str(path), "rev-parse", "--short", "HEAD"], timeout=20)
    files: list[str] = []
    for pattern in file_globs:
        files.extend(str(p.relative_to(path)) for p in path.glob(pattern))
    return RepoStatus(name, str(path), True, rev if code == 0 else "", sorted(files), "ok")


def qokit_gpu_source_status(core_python: Path) -> str:
    code, out = run(
        [
            str(core_python),
            "-c",
            (
                "from pathlib import Path; import qokit; "
                "root=Path(qokit.__file__).resolve().parent; "
                "p=root/'fur'/'nbcuda'/'furx.cu'; "
                "print(str(p)); print('PRESENT' if p.exists() else 'MISSING')"
            ),
        ],
        timeout=30,
    )
    return out if code == 0 else f"failed:{out}"


def julia_status(julia_path: str) -> str:
    if not julia_path:
        return "missing"
    code, out = run([julia_path, "-e", "using Pkg; Pkg.status()"], timeout=120)
    return out if code == 0 else f"failed:{code}:{out[:500]}"


def julia_project_status(julia_path: str, project: Path) -> str:
    if not julia_path:
        return "missing"
    if not project.exists():
        return f"missing project: {project}"
    code, out = run([julia_path, f"--project={project}", "-e", "using Pkg; Pkg.status()"], timeout=120)
    return out if code == 0 else f"failed:{code}:{out[:500]}"


def write_markdown(path: Path, report: dict) -> None:
    lines: list[str] = ["# Peer Installation Probe", ""]
    lines.append("## Tools")
    lines.append("")
    lines.append("| Tool | Status | Path | Version |")
    lines.append("|---|---|---|---|")
    for item in report["tools"]:
        lines.append(f"| {item['name']} | {item['status']} | `{item['path']}` | `{item['version']}` |")

    lines.extend(["", "## Python Environments", ""])
    for env in report["python_envs"]:
        lines.append(f"### {env['name']}")
        lines.append("")
        lines.append(f"- Python: `{env['python']}`")
        lines.append(f"- Version: `{env['version']}`")
        lines.append(f"- Status: `{env['status']}`")
        lines.append("")
        lines.append("| Module | Status |")
        lines.append("|---|---|")
        for mod, st in env["modules"].items():
            lines.append(f"| {mod} | {st} |")
        lines.append("")

    lines.extend(["## Repositories", "", "| Repo | Status | Path | Revision | Files |", "|---|---|---|---|---|"])
    for repo in report["repos"]:
        files = "<br>".join(repo["files"][:12])
        lines.append(f"| {repo['name']} | {repo['status']} | `{repo['path']}` | `{repo['revision']}` | {files} |")

    lines.extend(["", "## Special Checks", ""])
    for key, value in report["special"].items():
        lines.append(f"### {key}")
        lines.append("")
        lines.append("```text")
        lines.append(str(value))
        lines.append("```")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", type=Path, default=DEFAULT_PREFIX)
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "peer_probe.json")
    parser.add_argument("--markdown", type=Path, default=ROOT / "results" / "peer_probe.md")
    args = parser.parse_args()

    prefix = args.prefix
    path_env = os.environ.copy()
    path_env["PATH"] = (
        "/usr/local/cuda/bin:"
        f"{Path.home()}/.juliaup/bin:"
        f"{Path.home()}/.cargo/bin:"
        + path_env.get("PATH", "")
    )

    tools = [
        tool_status("nvcc", ["/usr/local/cuda/bin/nvcc", "nvcc"], ["--version"]),
        tool_status("julia", [str(Path.home() / ".juliaup" / "bin" / "julia"), "julia"], ["--version"]),
        tool_status("cargo", [str(Path.home() / ".cargo" / "bin" / "cargo"), "cargo"], ["--version"]),
        tool_status("git", ["git"], ["--version"]),
    ]

    modules = [
        "cupy",
        "cudaq",
        "qokit",
        "qblaze",
        "qtensor",
        "qtree",
        "cirq",
        "qiskit",
        "numba",
        "networkx",
        "numpy",
        "pycuaoa",
        "cuaoa",
    ]
    core_python = prefix / "venvs" / "lcqaoa-core" / "bin" / "python"
    qtensor_python = prefix / "venvs" / "qtensor-py310" / "bin" / "python"
    cuaoa_python = prefix / "venvs" / "cuaoa-py312" / "bin" / "python"
    python_envs = [
        python_env_status("core", core_python, modules),
        python_env_status("qtensor", qtensor_python, modules),
        python_env_status("cuaoa", cuaoa_python, modules),
        python_env_status("system", Path("/usr/bin/python3"), modules),
    ]

    repos = [
        repo_status("CUAOA", prefix / "src" / "cuaoa", ["pyproject.toml", "Cargo.toml", "target/release/**/*.so"]),
        repo_status("QTensor", prefix / "src" / "QTensor", ["setup.py", "qtree/setup.py", "qtensor/**/*.py"]),
        repo_status("BMQSim", prefix / "src" / "bmqsim", ["**/*"]),
        repo_status("QueenV2", prefix / "src" / "queenv2", ["**/*"]),
    ]

    julia_path = next((t.path for t in tools if t.name == "julia" and t.path), "")
    special = {
        "qokit_gpu_source": qokit_gpu_source_status(core_python) if core_python.exists() else "core python missing",
        "julia_pkg_status": julia_status(julia_path),
        "mps_julia_pkg_status": julia_project_status(julia_path, prefix / "julia-mps"),
        "liblbfgs_search": "\n".join(
            str(p)
            for root in [
                prefix,
                Path.home() / ".local",
                Path("/usr/local"),
                Path("/usr/lib"),
            ]
            if root.exists()
            for p in root.rglob("liblbfgs*")
        )
        or "MISSING",
    }

    report = {
        "prefix": str(prefix),
        "tools": [asdict(x) for x in tools],
        "python_envs": [asdict(x) for x in python_envs],
        "repos": [asdict(x) for x in repos],
        "special": special,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(args.markdown, report)
    print(f"WROTE {args.out}")
    print(f"WROTE {args.markdown}")


if __name__ == "__main__":
    main()
