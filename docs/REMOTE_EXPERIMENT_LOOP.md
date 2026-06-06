# Remote Experiment Loop

This project can run a local Codex -> GitHub -> remote server -> local results
loop through SSH key authentication.

## Defaults

- Remote: `lbz@192.168.70.60`
- Remote repo: `~/mac_agent/sdr-lutattn-qkformer-lut`
- Local repo: `/Users/cvue/Documents/github_zr/sdr-lutattn-qkformer-lut`
- Branch: `codex/qkformer-lut-hybrid`
- Server-side Git remote: `origin` (points to the working GitHub fork)
- Conda environment: `sdr`
- Small diagnostics/calibration GPU: `2`
- Large training jobs: auto-pick an idle GPU, then fall back to GPU `2`

Do not store server passwords in scripts, docs, or commits. Use SSH key
authentication for unattended runs.

## One-Time SSH Check

```bash
ssh -o BatchMode=yes lbz@192.168.70.60 'hostname && whoami'
```

This must complete without a password prompt before unattended local scripts can
run.

## Default T=1 E3 Seed Sweep

```bash
cd /Users/cvue/Documents/github_zr/sdr-lutattn-qkformer-lut
bash scripts/local/run_remote_qk_lut_loop.sh
```

The default local loop:

1. SSHes into the server.
2. Runs `git fetch`, `git checkout`, and `git pull --ff-only` for the branch.
3. Activates conda environment `sdr`.
4. Runs `scripts/server/run_qkformer_lut_t1_e3_seed_sweep.sh --gpu 2`.
5. Runs `scripts/server/package_qkformer_lut_t1_e3_seed_sweep.sh`.
6. Downloads the artifact into local `results/`.
7. Extracts the artifact and removes the local tarball.
8. Runs `scripts/local/analyze_qk_lut_results.py`.

## Large Training Jobs

For expensive training jobs, use `--kind train` so the remote command checks
`nvidia-smi` and chooses the first GPU with memory usage below 1000 MiB and GPU
utilization below 20%. If no such GPU is found, it falls back to `--gpu`.

Example:

```bash
bash scripts/local/run_remote_qk_lut_loop.sh \
  --kind train \
  --server-script scripts/server/run_qkformer_cifar10_t1_train.sh \
  --package-script "" \
  --no-download
```

## Useful Overrides

```bash
bash scripts/local/run_remote_qk_lut_loop.sh --gpu 1
bash scripts/local/run_remote_qk_lut_loop.sh --no-download
bash scripts/local/run_remote_qk_lut_loop.sh --skip-run
bash scripts/local/run_remote_qk_lut_loop.sh --keep-archive
python3 scripts/local/analyze_qk_lut_results.py --root .
python3 scripts/local/analyze_qk_lut_results.py --brief
bash scripts/local/qk_lut_quick_state.sh
```
