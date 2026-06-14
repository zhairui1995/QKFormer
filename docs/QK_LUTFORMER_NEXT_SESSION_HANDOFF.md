# QK-LUTFormer Next-Session Handoff

Updated: 2026-06-14

## Start Here

Read, in order:

1. `/Users/cvue/.codex/memory/personal-codex-rules.md`
2. `AGENTS.md`
3. `docs/QK_LUTFORMER_QUICK_STATE.md`
4. this file
5. `paper/PAPER_STATUS_AND_EXPERIMENT_PLAN.md` only before changing claims

Read the Markdown-only local result summary:

```bash
cat results/EXPERIMENT_RESULTS_SUMMARY.md
```

Do not run the JSON analyzer by default. Expanded local result JSON files were
removed after being verified into a cold archive outside the repository. The
normal continuation path is Markdown-only; restore raw JSON only for a disputed
numerical audit.

Branch: `codex/qkformer-lut-5way-controls`

Remote: `lbz@192.168.70.60`

Remote repository: `~/mac_agent/sdr-lutattn-qkformer-lut`

Conda environment: `sdr`

Datasets: `/home/datasets`

## Scientific State

- Overall verdict remains `GO-AUDIT`, not `GO-METHOD`.
- E1 supports Q/K-address-specific reconstruction across datasets, time steps,
  calibration subsets, stages, and checkpoints.
- E5 exposes misses through hierarchical backoff but misses the compact-entry
  budget.
- E7 is the main compactness result: aligned subspace-decoupled LUTs beat
  token/channel and shuffled controls below the 25% supported-entry budget.
- E3/E4 accuracy changes are limits evidence unless a preregistered matched
  control gate passes independently across backbones.
- The seed-42 CIFAR-100 T=4 one-epoch deterministic gate is a checkpoint-level
  `PASS`, not a backbone-stable method claim.

## Latest Backbone Audit

Independent CIFAR-100 T=4 training completed:

| Backbone seed | Best Acc@1 | Epoch |
|---|---:|---:|
| 43 | 81.35% | 380 |
| 44 | 81.05% | 384 |

The two trainings started in the same second and both wrote under
`results/qkformer_cifar100_train_20260613_134245`. Their output subdirectories
are separate, but the parent `checkpoint_manifest.json` was overwritten by
seed 44. The training runner is now patched to include time step, seed, GPU,
and PID in each parent result directory.

Do not use the existing report named `backbone43` as seed-43 evidence. Both
existing two-epoch `backbone43` and `backbone44` JSON files record this seed-44
checkpoint:

```text
results/qkformer_cifar100_train_20260613_134245/output/
qkformer_cifar100_t4_seed44/model_best.pth.tar
```

The valid seed-44 two-epoch result is `FAIL`:

- paired baseline: 81.04%
- aligned gated mean: 81.23% (+0.19)
- global gated mean: 81.3433% (+0.3033)
- shuffled gated mean: 81.39% (+0.35)

This blocks a backbone-stable accuracy claim at present.

## Runs In Progress

Two explicit one-epoch matched-control gates were launched on 2026-06-14:

- seed-43 checkpoint on GPU 0, launcher PID 2727532
- seed-44 checkpoint on GPU 1, launcher PID 2727533

Monitor:

```bash
ssh lbz@192.168.70.60 'cd ~/mac_agent/sdr-lutattn-qkformer-lut && \
pgrep -af "run_qkformer_lut_cifar100_t4_deterministic_gate|qkformer_lut_e3_trainable_lut.py" && \
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits'
```

Logs:

```bash
ssh lbz@192.168.70.60 'tail -f ~/mac_agent/sdr-lutattn-qkformer-lut/results/qk_lutformer_cifar100_t4_deterministic_gate_backbone43_1epoch_launcher.log'
ssh lbz@192.168.70.60 'tail -f ~/mac_agent/sdr-lutattn-qkformer-lut/results/qk_lutformer_cifar100_t4_deterministic_gate_backbone44_1epoch_launcher.log'
```

Expected artifacts:

```text
qk_lutformer_cifar100_t4_deterministic_gate_backbone43_1epoch_artifacts.tar.gz
qk_lutformer_cifar100_t4_deterministic_gate_backbone44_1epoch_artifacts.tar.gz
```

## Completion Workflow

1. Confirm both launcher processes have exited and both tarballs exist.
2. Inspect each report JSON on the server before downloading. The `checkpoint`
   field must contain the matching `seed43` or `seed44` path.
3. Download to local `results/`, extract, then remove only the downloaded
   tarballs.
4. Report per backbone: baseline, aligned/global/shuffled means, minimum aligned
   delta, control gaps, and gate decision.
5. Update `docs/QK_LUTFORMER_QUICK_STATE.md` and paper planning/audit files.
6. Change paper claims only after the cross-backbone verdict is known.

Download template:

```bash
cd /Users/cvue/Documents/github_zr/sdr-lutattn-qkformer-lut
for s in 43 44; do
  scp lbz@192.168.70.60:~/mac_agent/sdr-lutattn-qkformer-lut/qk_lutformer_cifar100_t4_deterministic_gate_backbone${s}_1epoch_artifacts.tar.gz results/
  tar -xzf results/qk_lutformer_cifar100_t4_deterministic_gate_backbone${s}_1epoch_artifacts.tar.gz
  rm results/qk_lutformer_cifar100_t4_deterministic_gate_backbone${s}_1epoch_artifacts.tar.gz
done
```

## Working Habits To Preserve

- Use `results/EXPERIMENT_RESULTS_SUMMARY.md` as the only default local result
  entry point. Do not recursively read `results/` or raw archives.
- Read real metrics and checkpoint paths before interpreting filenames.
- Check `nvidia-smi` before large jobs. Use genuinely idle GPUs for parallel
  work; small diagnostics may default to GPU 2 only when it is available.
- Use SSH keys, GitHub sync, `nohup`, explicit launcher logs, and timestamped
  result directories.
- For parallel experiments, include seed/GPU/PID in parent result paths.
- Keep epochs, data partitions, adapter seeds, and parameter budgets matched
  across aligned, shuffled, token/channel, and global controls.
- Tricks are acceptable only when preregistered and applied fairly to all
  matched controls. Never perform alpha, seed, stage, or checkpoint shopping.
- Do not promote a best single run. Report means, minima, control gaps, and
  failed replications.
- Require machine-readable JSON/CSV plus a concise Markdown report.
- Keep datasets, checkpoints, logs, result directories, and tarballs out of
  Git. Never commit credentials or private paths beyond the established local
  workflow documentation.
- Preserve unrelated dirty and untracked files. Do not clean or reset the
  worktree.
- Compile LaTeX and check page count, undefined references, and overfull boxes
  before committing paper changes.
- Commit narrowly and push to `fork/codex/qkformer-lut-5way-controls` after
  verification.

## Claim Rule For The Next Session

- If both explicit one-epoch backbones pass the registered aligned-vs-control
  gate, consider a carefully scoped backbone-stable selective-utility claim.
- If either fails, retain the seed-42 result as checkpoint-conditional evidence
  and keep the paper's core contribution on E1/E7 reconstruction and compact
  address structure.
- Never claim energy, latency, SRAM compression, ImageNet performance, or
  stable classification improvement without direct evidence.
