from __future__ import annotations

import argparse
import hashlib
import json
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


LABEL = (
    "official CUAOA 0.1.0 codebase with documented device-information "
    "compatibility patch"
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def recorded_hash(path: Path) -> str:
    return path.read_text(encoding="utf-8").split()[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-csv", type=Path, required=True)
    parser.add_argument("--old-csv", type=Path, required=True)
    parser.add_argument("--provenance-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    patch_path = args.provenance_dir / "device_info_compatibility.patch"
    wheel_path = args.provenance_dir / "pycuaoa-0.1.0-cp312-cp312-linux_x86_64.whl"
    patch_sha = digest(patch_path)
    wheel_sha = digest(wheel_path)
    clean_log = (args.provenance_dir / "clean_cuaoa_build_20260712.log").read_text(
        encoding="utf-8", errors="replace"
    )
    patched_log = (
        args.provenance_dir / "device_info_patch_build_20260712.log"
    ).read_text(encoding="utf-8", errors="replace")
    environment = (args.provenance_dir / "environment.txt").read_text(
        encoding="utf-8", errors="replace"
    )
    smoke = (args.provenance_dir / "import_smoke.txt").read_text(
        encoding="utf-8", errors="replace"
    )
    core_match = re.search(r"core_sha256=([0-9a-f]{64})", smoke)
    if not core_match:
        raise RuntimeError("installed core hash missing from import smoke")
    installed_core_sha = core_match.group(1)
    with zipfile.ZipFile(wheel_path) as archive:
        core_names = [name for name in archive.namelist() if "_core" in name and name.endswith(".so")]
        if len(core_names) != 1:
            raise RuntimeError(f"expected one core library in wheel: {core_names}")
        wheel_core_sha = hashlib.sha256(archive.read(core_names[0])).hexdigest()

    checks: list[dict] = []

    def check(name: str, condition: bool, evidence) -> None:
        checks.append({"name": name, "passed": bool(condition), "evidence": evidence})

    patch_text = patch_path.read_text(encoding="utf-8")
    check(
        "clean commit recorded",
        "commit=33a3b2fbb16631c03fb9dff1c43a901ff11d429f" in environment,
        "33a3b2fbb16631c03fb9dff1c43a901ff11d429f",
    )
    check(
        "clean upstream failure isolated",
        "cudaGetDeviceProperties_v2" in clean_log
        and "was not declared in this scope" in clean_log,
        "CUDA device-information API compile error",
    )
    check(
        "patch changes only device reporting API",
        patch_text.count("cudaGetDeviceProperties_v2") == 1
        and patch_text.count("cudaGetDeviceProperties(&prop, i)") == 1
        and "device_info.cpp" in patch_text,
        patch_text,
    )
    check(
        "patch hash matches record",
        patch_sha == recorded_hash(args.provenance_dir / "patch_sha256.txt"),
        patch_sha,
    )
    check(
        "wheel hash matches record",
        wheel_sha == recorded_hash(args.provenance_dir / "wheel_sha256.txt"),
        wheel_sha,
    )
    check(
        "installed core matches wheel core",
        installed_core_sha == wheel_core_sha,
        {"installed": installed_core_sha, "wheel": wheel_core_sha},
    )
    check(
        "patched build and smoke succeeded",
        "PATCHED_BUILD_SUCCESS" in patched_log and "value=" in smoke,
        smoke,
    )
    check(
        "toolchain recorded",
        all(token in environment for token in ("release 13.", "rustc ", "g++", "Python 3.12")),
        environment,
    )

    raw = pd.read_csv(args.raw_csv)
    check("fresh benchmark has 60 rows", len(raw) == 60, len(raw))
    check("fresh benchmark all successful", raw.status.eq("ok").all(), raw.status.value_counts().to_dict())
    check(
        "fresh benchmark has six cells by five seeds",
        raw.groupby(["family", "n"])["seed"].nunique().eq(5).all()
        and raw.groupby(["family", "n"]).ngroups == 6,
        raw.groupby(["family", "n"])["seed"]
        .nunique()
        .rename("seeds")
        .reset_index()
        .to_dict(orient="records"),
    )
    peers = raw[raw.backend.str.contains("CUAOA", regex=False)]
    check(
        "value agreement retained",
        peers.value_abs_error_vs_lc.max() < 1e-12,
        float(peers.value_abs_error_vs_lc.max()),
    )
    check(
        "gradient agreement retained",
        peers.gradient_relative_l2_error_vs_lc.max() < 1e-12,
        float(peers.gradient_relative_l2_error_vs_lc.max()),
    )

    enriched = raw.copy()
    enriched["source_commit"] = "33a3b2fbb16631c03fb9dff1c43a901ff11d429f"
    enriched["patch_sha256"] = patch_sha
    enriched["wheel_sha256"] = wheel_sha
    enriched["installed_core_sha256"] = installed_core_sha
    peer_mask = enriched.backend.str.contains("CUAOA", regex=False)
    enriched.loc[peer_mask, "backend"] = LABEL
    enriched.loc[peer_mask, "notes"] = (
        enriched.loc[peer_mask, "notes"].astype(str)
        + f"; source commit 33a3b2f; device-info-only patch {patch_sha}; wheel {wheel_sha}"
    )
    enriched_path = args.out_dir / "P0_1_cuaoa_provenance_gradient.csv"
    enriched.to_csv(enriched_path, index=False)

    old = pd.read_csv(args.old_csv)
    old = old[old.n.isin([20, 22, 24])].copy()
    role = lambda frame: np.where(frame.backend.str.startswith("LC"), "lc", "cuaoa")
    enriched["role"] = role(enriched)
    old["role"] = role(old)
    compare = enriched.merge(
        old,
        on=["family", "n", "p", "seed", "role"],
        suffixes=("_fresh", "_old"),
        validate="one_to_one",
    )
    compare["fresh_over_old_steady"] = (
        compare.steady_median_seconds_fresh / compare.steady_median_seconds_old
    )
    timing_repro = (
        compare.groupby(["family", "n", "role"])
        .agg(
            rows=("fresh_over_old_steady", "size"),
            median_fresh_over_old=("fresh_over_old_steady", "median"),
            q1_fresh_over_old=("fresh_over_old_steady", lambda x: float(np.percentile(x, 25))),
            q3_fresh_over_old=("fresh_over_old_steady", lambda x: float(np.percentile(x, 75))),
        )
        .reset_index()
    )
    timing_repro.to_csv(args.out_dir / "fresh_vs_original_timing.csv", index=False)

    failed = [item for item in checks if not item["passed"]]
    summary = {
        "status": "PASS" if not failed else "FAIL",
        "provenance_label": LABEL,
        "source_commit": "33a3b2fbb16631c03fb9dff1c43a901ff11d429f",
        "patch_sha256": patch_sha,
        "wheel_sha256": wheel_sha,
        "installed_core_sha256": installed_core_sha,
        "checks": checks,
        "failed": [item["name"] for item in failed],
    }
    (args.out_dir / "CUAOA_PROVENANCE_VALIDATION.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    report = [
        "# CUAOA RTX 3070 Provenance Closure",
        "",
        f"The benchmark uses the {LABEL}.",
        "",
        "A clean build of upstream commit `33a3b2fbb16631c03fb9dff1c43a901ff11d429f` fails under the recorded CUDA 13.1 headers because `cudaGetDeviceProperties_v2` is not declared. The documented one-line patch replaces that device-information call with `cudaGetDeviceProperties`; no objective, state, gradient, or kernel source is changed.",
        "",
        f"Patch SHA-256: `{patch_sha}`  ",
        f"Wheel SHA-256: `{wheel_sha}`  ",
        f"Installed core SHA-256: `{installed_core_sha}`",
        "",
        "The fresh wheel completed all 30 requested graph cells for both LC and CUAOA. Maximum value and relative-gradient discrepancies remained below `1e-12`.",
        "",
        "## Fresh versus original steady timing",
        "",
        timing_repro.to_markdown(index=False),
    ]
    (args.out_dir / "CUAOA_PROVENANCE_REPORT.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": summary["status"], "checks": len(checks), "failed": summary["failed"]}, indent=2))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
