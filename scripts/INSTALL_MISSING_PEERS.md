# Installing Missing Peer Baselines

Use `scripts/install_all_peers_once.sh` on the Ubuntu GPU host for the normal
one-shot install path.

```bash
cd $HOME/lc_implicit_qaoa_20260630
bash scripts/install_all_peers_once.sh
```

It asks for `sudo` once, keeps the sudo credential alive, installs all configured
peer environments, then writes:

- log: `results/install_logs/install_all_peers.latest.log`
- status: `results/install_logs/install_all_peers.status`
- probe: `results/peer_probe_latest.md`

`scripts/install_missing_peers.sh` is kept as the lower-level helper used by the
one-shot script.

The script deliberately separates environments because the peer methods are not
mutually compatible in one Python environment:

- core env: CuPy, CUDA-Q `cudaq`, QOKit, qblaze, and LC-Implicit-QAOA dependencies.
- QTensor env: separate Python 3.10 env, because QTensor's published install path
  is old and conflicts with QOKit/Qiskit.
- CUAOA: requires `nvcc` and Rust/Cargo.
- JuliQAOA/MPS-JuliQAOA: requires Julia.
- BMQSim/QueenV2: no installable public repository was confirmed by the script;
  set `BMQSIM_REPO_URL` or `QUEENV2_REPO_URL` if you have one.

## Recommended commands on an Ubuntu GPU host

```bash
cd $HOME/lc_implicit_qaoa_20260630

# User-space Python peers only.
bash scripts/install_missing_peers.sh --core

# If you can sudo, install system prerequisites.
bash scripts/install_missing_peers.sh --system

# If you can sudo and want CUAOA, install nvcc first.
bash scripts/install_missing_peers.sh --cuda-toolkit
export PATH=/usr/local/cuda/bin:$PATH

# Then build CUAOA.
bash scripts/install_missing_peers.sh --cuaoa

# Isolated QTensor install.
bash scripts/install_missing_peers.sh --qtensor

# Julia-based peers.
bash scripts/install_missing_peers.sh --julia --juliqaoa
```

## Dry run

```bash
bash scripts/install_missing_peers.sh --all --dry-run
```
