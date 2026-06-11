# Remote Experiment Loop

This project can run a local Codex -> GitHub -> remote server -> local results
loop through SSH key authentication.

## Defaults

- Remote: `lbz@192.168.70.60`
- Remote repo: `~/mac_agent/sdr-lutattn-qkformer-lut`
- Local repo: `/Users/cvue/Documents/github_zr/sdr-lutattn-qkformer-lut`
- Branch: current local branch by default. The active paper branch is currently
  `codex/qkformer-lut-5way-controls`.
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

For expensive training jobs, prefer detached jobs. They need SSH only long
enough to launch `nohup` on the server, then the job survives local network
dropouts. Detached `--kind train` checks `nvidia-smi` and chooses the first GPU
with memory usage below 1000 MiB and GPU utilization below 20%. If no such GPU
is found, it falls back to `--gpu`.

Example:

```bash
bash scripts/local/run_remote_detached_qk_lut_job.sh \
  --kind train \
  --job-name c100_t4_seed42 \
  --env QKFORMER_LUT_TIME_STEP=4 \
  --env QKFORMER_CIFAR100_TRAIN_SEED=42 \
  --env QKFORMER_CIFAR100_TRAIN_EXPERIMENT=qkformer_cifar100_t4_seed42 \
  --server-script scripts/server/run_qkformer_cifar100_train.sh
```

Check status and tail the remote log:

```bash
bash scripts/local/check_remote_detached_qk_lut_job.sh \
  --job-name c100_t4_seed42 \
  --tail 120
```

After the CIFAR-100 T=4 checkpoint finishes, run the registered LUT
post-training workflow:

```bash
bash scripts/local/run_remote_detached_qk_lut_job.sh \
  --job-name c100_t4_lut_after_train \
  --gpu 2 \
  --server-script scripts/server/run_qkformer_lut_cifar100_t4_after_latest_train.sh
```

Download, extract, and analyze the T=4 artifact:

```bash
bash scripts/local/check_remote_detached_qk_lut_job.sh \
  --job-name c100_t4_lut_after_train \
  --artifact qk_lutformer_cifar100_t4_artifacts.tar.gz \
  --extract \
  --analyze
```

## Fixed-Budget Hierarchy

Run the compressed hierarchy gate as a detached small diagnostic:

```bash
bash scripts/local/run_remote_detached_qk_lut_job.sh \
  --job-name e6_budgetfrac_c100_t1 \
  --gpu 2 \
  --server-script scripts/server/run_qkformer_lut_e6_budget_fraction_sweep.sh
```

Download and analyze:

```bash
bash scripts/local/check_remote_detached_qk_lut_job.sh \
  --job-name e6_budgetfrac_c100_t1 \
  --artifact qk_lutformer_e6_budget_fraction_artifacts.tar.gz \
  --extract \
  --analyze
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
