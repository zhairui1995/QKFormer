# CIFAR10-DVS 84% Route Handoff

> 新对话启动指令：
>
> ```text
> 请读取 AGENTS.md、个人规则、结果摘要、Quick State，以及
> docs/CIFAR10_DVS_84PCT_ROUTE_HANDOFF.md，然后检查远端状态并按 handoff
> 继续执行；任一 LUT 部署模型 full-validation Acc@1 达到 84.0% 后立即停止。
> ```

Updated: 2026-06-22

Status: completed below gate

## User-Authorized Route-3 Extension (2026-06-22)

After the original fixed queue completed, the user explicitly authorized the
previously out-of-scope SpiLiFormer-2-256 clean training. Route 3 is therefore
now scientifically complete:

- The released SpiLiFormer CIFAR10-DVS recipe ran from scratch for 130
  effective epochs (the `epochs=120` argument plus 10 scheduler cooldown
  epochs).
- Best clean full-validation Acc@1 was `81.2%` at epoch 103, below the
  published `86.7%`.
- The fixed 12-epoch first-FF-LiDiff LUT transfer completed.
- Best valid LUT-only deployment was `81.7%` at LUT epoch 7, a paired
  `+0.5`-point change from its `81.2%` clean teacher.
- Evaluation used all 1,000 validation samples, current normalization was
  enabled, and the replaced `proj_conv` was not retained or executed.
- Route 3 did not reach the `84.0%` gate. No additional hyperparameter,
  checkpoint, stage, or seed search was launched.

Authoritative summary:

- `results/spiliformer_cifar10dvs_route3_20260622.md`
- `results/spiliformer_cifar10dvs_route3_clean_20260621_232837/`
- `results/spiliformer_cifar10dvs_route3_lut_run_20260622_012835/`

## Completion Record (2026-06-21)

The original fixed route queue is complete. No registered route reached the valid
full-validation `84.0%` LUT-deployment gate, so no additional search was
launched.

- Route 2 completed its fixed 20-epoch budget. Best valid LUT-only deployment:
  `83.6%` Acc@1 at epoch 8 from a clean `83.5%` teacher.
- Route 3 was not scientifically launched because the official SpiLiFormer
  release contains the CIFAR10-DVS code and reported `86.7%` result but no
  CIFAR10-DVS checkpoint. The official GitHub revision inspected was
  `23c6bf9fa927ae1edc593df56193ad46a43d7034`; the official Hugging Face model
  repository listed only ImageNet checkpoints. Training a new clean backbone
  for the official 120-epoch recipe would have exceeded this fixed protocol.
- Route 1 backbone continuation completed its fixed 30-epoch budget. Its best
  clean checkpoint was `82.3%` at epoch 28.
- Route 1 LUT adaptation completed its fixed 12-epoch budget. Best valid
  LUT-only deployment: `83.9%` Acc@1 at epoch 12, from the `82.3%` Route-1
  backbone.
- The first Route-1 LUT launch failed during calibration because an unrelated
  process occupied roughly 18.9 GiB on the assigned GPU. The identical
  checkpoint and protocol were rerun on an idle GPU; no parameter changed.

The final `83.9%` result uses all 1,000 fixed validation samples, trainable
per-time-step/per-channel current normalization, and a true deployment path
that does not retain or execute the replaced `proj_conv`.

Authoritative local artifacts:

- `results/cifar10dvs_84pct_queue_20260621_183929/route_summary.csv`
- `results/cifar10dvs_84pct_queue_20260621_183929/stop_reason.json`
- `results/qkformer_cifar10dvs_route1_lut_retry_20260621_193009/best_metrics.json`
- `results/qkformer_cifar10dvs_route1_lut_retry_20260621_193009/checkpoint_manifest.json`
- `results/spiliformer_cifar10dvs_route3_20260622.md`
- `results/spiliformer_cifar10dvs_route3_clean_20260621_232837/best_metrics.json`
- `results/spiliformer_cifar10dvs_route3_lut_run_20260622_012835/best_metrics.json`

## Authority And Stop Rule

This document is the controlling protocol for the next CIFAR10-DVS experiment
session. Before editing code or launching a job, the new session must inspect
the remote server for active processes and newer result artifacts. Do not
duplicate an experiment that is already running or complete.

Execute the routes strictly in this order:

1. Route 2: LUT-aware QKFormer training
2. Route 3: SpiLiFormer-2-256 LUT transfer
3. Route 1: one fixed QKFormer backbone improvement run, followed by LUT
   adaptation

The global stop condition is:

- evaluation uses the complete fixed CIFAR10-DVS validation split of 1,000
  samples;
- the evaluated deployment path has LUT replacement and current normalization
  enabled;
- Acc@1 is at least `84.0%`.

When any route satisfies all three conditions, finish the current evaluation,
save its checkpoint and machine-readable artifacts, stop the route queue, and
wait for the user's decision. Clean-backbone accuracy, training accuracy,
partial-validation accuracy, or an evaluation that still uses the original
projection output does not trigger the stop condition.

If all three routes finish below 84.0%, stop and report the best valid LUT
deployment result. Do not start an unregistered search.

## Current Evidence

- Fixed QKFormer CIFAR10-DVS backbone:
  `83.5%` best full-validation Acc@1 at epoch 104.
- Backbone result:
  `results/qkformer_cifar10dvs_train_20260620_232449`.
- Current best LUT result:
  `82.7%` full-validation Acc@1.
- Best LUT configuration:
  `stage1.0.tssa`, `global_plus_address_lut`, `alpha=1`,
  `shrinkage_tau=64`, projection-current injection, and trainable
  per-time-step/per-channel current normalization.
- Best LUT result:
  `results/qkformer_cifar10dvs_e3_currentnorm_tau64_controls_e5_20260621_163716`.
- Paired controls from the same result:
  shuffled address `82.2%`, token/channel `82.1%`, global mean `81.6%`.
- At the last status check there was no active CIFAR10-DVS training process.
  This is historical status only; recheck before launch.

## Fixed Experimental Constraints

- Dataset: CIFAR10-DVS, T=16 event-count frames, two polarity channels.
- Split: deterministic class-wise first 90% train and last 10% validation.
- New paper-mainline seed: 42, except the inherited baseline training recipe
  whose fixed seed is 2021.
- No SEWResNet or other convolutional replacement backbone.
- No alpha search. Use full replacement with `alpha=1`.
- No stage, tau, checkpoint, or random-seed sweep.
- QKFormer target is fixed to `stage1.0.tssa`.
- QKFormer LUT is fixed to global plus aligned address residual,
  `shrinkage_tau=64`, support-aware initialization/backoff, and
  per-time-step/per-channel current normalization.
- Do not count a hook path that executes and consumes the original projection
  result as a deployment-success result. The final accepted checkpoint must
  provide a LUT-only output path for the replaced projection.
- Any deviation needed to fix a software defect must be documented as a defect
  correction, not silently treated as hyperparameter selection.

## Route 2: LUT-Aware QKFormer

Hypothesis: joint optimization of the established global plus address-residual
LUT and current normalization can recover the remaining 0.8-point post-hoc
replacement loss.

Initialization and structure:

- Initialize from the existing `83.5%` best QKFormer checkpoint.
- Keep the stage-1 Q/K address contract fixed.
- Train the LUT tables, current-normalization parameters, modules after the
  replaced stage-1 projection, and classifier.
- Keep a frozen copy of the original checkpoint as the teacher.
- At inference, use the LUT replacement output and current normalization;
  remove the original `proj_conv` output from the prediction path.

Fixed training budget:

- Complete 9,000-sample training split.
- Batch size 16.
- Maximum 20 epochs.
- LUT and current-normalization learning rate: `3e-3`.
- Unfrozen downstream-network learning rate: `3e-4`.
- Cosine learning-rate decay.
- Loss:
  `CE + 2.0 * KL(student, frozen teacher) + 0.2 * local current MSE`.
- Run full 1,000-sample validation after every epoch.
- Save the best valid LUT deployment checkpoint.

Transition:

- If full-validation LUT Acc@1 reaches 84.0%, stop the complete queue.
- Otherwise, finish the fixed budget, retain the best checkpoint and metrics,
  then begin Route 3.

## Route 3: SpiLiFormer-2-256 Transfer

Run only if Route 2 does not reach 84.0%.

Hypothesis: the stronger SpiLiFormer-2-256 event backbone can provide enough
accuracy margin for a bounded Q/K-address LUT transfer.

Preparation and structure:

- Use the official SpiLiFormer implementation and the CIFAR10-DVS
  `SpiLiFormer-2-256` configuration associated with the reported `86.7%`
  result.
- Record the repository revision, checkpoint provenance, model parameter
  count, input representation, split, and clean full-validation accuracy.
- Adapt only the first FF-LiDiff block in this route.
- Construct the Q/K address, global plus aligned address-residual LUT,
  support-aware fallback, and per-time-step/per-channel current normalization.
- Freeze the address-producing block.
- Train LUT/current-normalization parameters, downstream modules, and
  classifier, using the clean model as a frozen teacher.
- The accepted deployment evaluation must not consume the original replaced
  projection response.

Fixed training budget:

- Complete training split, batch size 16.
- Maximum 12 epochs.
- LUT and current-normalization learning rate: `3e-3`.
- Unfrozen downstream-network learning rate: `3e-4`.
- Use the same CE/KL/local-current-MSE weights as Route 2.
- Run full validation after every epoch.
- Test only the aligned LUT candidate before the 84% gate.

Transition:

- If the valid LUT deployment reaches 84.0%, stop the complete queue.
- Otherwise, finish the fixed budget and begin Route 1.
- Do not expand to FB-LiDiff, additional blocks, or a hyperparameter sweep
  within this protocol.

## Route 1: Fixed Backbone Improvement

Run only if Routes 2 and 3 do not reach 84.0%.

Hypothesis: one conservative backbone fine-tuning run can raise the QKFormer
ceiling enough for the already successful LUT calibration mechanism to cross
84.0%.

Backbone phase:

- Initialize from the existing `83.5%` QKFormer checkpoint.
- Fine-tune once for at most 30 epochs.
- Complete training split, batch size 16.
- Learning rate `5e-4`, weight decay `0.06`, cosine schedule.
- No recipe grid or checkpoint selection beyond the best checkpoint produced
  by this single run.
- Clean backbone accuracy does not trigger the global stop condition.

LUT phase:

- Initialize from the best checkpoint produced by the fixed backbone phase.
- Apply the same Route-2 QKFormer LUT structure and losses.
- Train for at most 12 epochs with the Route-2 learning rates.
- Evaluate the valid LUT-only deployment path on the full validation split
  after each epoch.

Transition:

- Stop if LUT Acc@1 reaches 84.0%.
- Otherwise end the protocol and report the best valid result across all
  routes.

## Remote Environment

- Server: `lbz@192.168.70.60`
- Remote repository:
  `/home/lbz/mac_agent/sdr-lutattn-qkformer-lut`
- Conda environment: `sdr`
- Python: `/home/lbz/miniconda/envs/sdr/bin/python`
- CIFAR10-DVS data:
  `/mnt/data_a/datasets/CIFAR10-DVS`
- Scientific workloads must run remotely.
- Store logs, checkpoints, metrics, and route state under the remote
  repository's `results/` tree.

Default remote command prefix:

```bash
ssh lbz@192.168.70.60 \
  'cd /home/lbz/mac_agent/sdr-lutattn-qkformer-lut && \
   PYTHONPATH=$PWD /home/lbz/miniconda/envs/sdr/bin/python ...'
```

Use `nohup` or a server-side runner for long jobs. Do not store credentials in
scripts or result artifacts.

## Existing Code Entrypoints

The following components already exist and must be inspected before adding
parallel implementations:

- Baseline training:
  `scripts/server/run_qkformer_cifar10dvs_train.sh`
- Existing baseline/download pipeline:
  `scripts/server/run_qkformer_cifar10dvs_pipeline.sh`
- Existing post-hoc E3 accuracy runner:
  `scripts/server/run_qkformer_cifar10dvs_e3_accuracy.sh`
- GPU-wait wrapper:
  `scripts/server/run_qkformer_cifar10dvs_e3_accuracy_wait.sh`
- E3 implementation:
  `tools/qkformer_lut_e3_trainable_lut.py`
- DVS E3 configuration:
  `configs/qkformer_lut_cifar10dvs_e3_trainable_lut.yaml`
- Dataset-specific model and training:
  `cifar10-dvs/model.py`, `cifar10-dvs/train.py`
- Earlier protocol:
  `docs/CIFAR10_DVS_QKLUT_PROTOCOL.md`

The current E3 implementation already supports:

- global plus address-residual LUT;
- aligned, shuffled, token/channel, and global controls;
- fixed `alpha=1`;
- shrinkage initialization;
- per-time-step/per-channel trainable current normalization initialized from
  calibration moments;
- frozen-backbone CE, teacher KL, and local-MSE adapter training.

## Required Implementation Work

Do not assume the following items are already complete:

1. Extend E3 from a frozen-backbone adapter into the Route-2 selective
   joint-training policy with separate parameter groups and learning rates.
2. Add a true LUT-only deployment path that bypasses the replaced original
   projection instead of relying only on a forward hook after executing it.
3. Add full-validation evaluation after each epoch, best-checkpoint saving,
   and an exact `84.0%` early-stop signal.
4. Implement SpiLiFormer-2-256 acquisition/integration and the first-FF-LiDiff
   LUT adapter for Route 3.
5. Implement the one-run Route-1 backbone continuation and its subsequent LUT
   phase.
6. Add one serial remote orchestrator that:
   - checks for existing jobs and newer completed results;
   - selects an idle GPU;
   - executes Route 2, then Route 3, then Route 1;
   - reads machine-readable full-validation metrics;
   - stops the queue immediately after a valid LUT result reaches 84.0%;
   - never launches forbidden searches.

Before substantial training, run a one-batch remote forward/backward smoke test
for every newly introduced model path.

## Required Artifacts

Each route must write:

- `protocol.json`
- `epoch_metrics.csv`
- `best_metrics.json`
- `checkpoint_manifest.json`
- training/evaluation log
- best valid LUT deployment checkpoint

The serial orchestrator must write:

- `route_summary.csv`
- `stop_reason.json`
- a route-state file that records pending, running, completed, failed, or
  stopped-after-gate status.

At minimum, each accepted evaluation row must record:

- route and epoch;
- evaluated sample count;
- clean teacher/backbone Acc@1;
- LUT deployment Acc@1;
- paired delta;
- KL to teacher;
- local current MSE;
- address coverage and fallback-level counts;
- confirmation that current normalization is enabled;
- confirmation that original projection output is bypassed.

A process crash is not permission to silently retry with changed parameters.
Record the failure and wait for diagnosis unless the correction is an
unambiguous software/environment repair.

## New-Session Startup Checklist

Read in order:

1. `/Users/cvue/.codex/memory/personal-codex-rules.md`
2. `AGENTS.md`
3. `results/EXPERIMENT_RESULTS_SUMMARY.md`
4. `docs/QK_LUTFORMER_QUICK_STATE.md`
5. `docs/QK_LUTFORMER_NEXT_SESSION_HANDOFF.md`
6. this document

Then perform these non-negotiable checks:

1. Inspect local `git status`; preserve all unrelated and uncommitted changes.
2. Inspect the remote branch and working tree before synchronizing code.
3. Check active server processes and GPU state.
4. Search for result directories newer than the results listed above.
5. Verify that `/mnt/data_a/datasets/CIFAR10-DVS` is readable and its prepared
   T=16 data is available.
6. If a valid LUT full-validation result at or above 84.0% already exists,
   do not launch anything; report it and wait for the user.
7. If a route is already running, monitor it rather than launching a duplicate.
8. Otherwise continue from the first incomplete route in the fixed order.

## Paper Claim Boundary

Crossing 84.0% would support a CIFAR10-DVS event-data result for the specific
LUT-adapted architecture and protocol. It would not by itself establish
unrestricted Spiking Transformer generalization, measured hardware speed,
energy reduction, or full-network LUT replacement.

Route 3 may support bounded cross-architecture transfer only if its clean
protocol and LUT deployment path are traceable and comparable. Existing
Spikformer boundary failures and matched controls must not be hidden.
