# Codex Handoff: QK-LUTFormer / QKFormer-LUT Hybrid

Last updated: 2026-06-06

## One-Line State

QKFormer CIFAR-10 T=4 reached 96.08% best Acc@1. T=4 E2/E3 showed stable but
weak LUT signals because global smoothing nearly matched or beat address LUT.
The T=1 stress test is now the strongest paper-relevant lead: T=1 training
reached 95.20% best Acc@1, T=1 E0 lowered conditional variance while preserving
coverage, and T=1 conservative E3 made `address_lut` the only positive adapter
against `global_mean` and `token_channel_lut` controls. Current judgment:
`CONDITIONAL GO` to a T=1 E3 multi-seed control sweep before claiming
address-specific LUT advantage.

## Project Identity

- Local repo: this repository checkout
- Server repo: `~/mac_agent/sdr-lutattn-qkformer-lut`
- Branch: `codex/qkformer-lut-hybrid`
- Push remote: `fork` / `git@github.com:zhairui1995/QKFormer.git`
- Server Git remote: `origin` / `git@github.com:zhairui1995/QKFormer.git`
- Upstream reference: `https://github.com/zhouchenlin2096/QKFormer`
- Core status doc: `docs/QKFORMER_LUT_BRANCH_STATUS.md`

## Why This Branch Exists

CCS-LUTAttn was stopped after E0-E2. Row context and background corrections were
diagnostically useful, but current pairwise response prototype LUTs did not
justify a converted Transformer wrapper:

- context coarse variance reduction around 20.99%;
- candidate-specific tables degraded split-aware reconstruction;
- background tables helped, but aggregate head-output improvement was only
  around 1.75%.

The new hypothesis is to avoid DeiT softmax row-normalization entirely and test
QKFormer's spike-form Q-K attention as a binary, LUT-friendly address source.

## Implemented Commits

- `5cc7e50` Add QKFormer LUT E0 diagnostics
- `e2c0ba7` Add QKFormer server dependency installer
- `d913da1` Bypass timm factory for QKFormer E0 model build
- `27d3c5f` Handle pre-reshape QKFormer projection hooks
- `dfab09b` Add CIFAR10 data link and training scripts
- `03cdcf4` Make CIFAR10 train splitbn import optional
- `1b2c6ab` Filter timm factory kwargs for CIFAR10 QKFormer
- `cc1f18a` Add QKFormer LUT E1 reconstruction diagnostic
- `202835c` Add server GPU selection logging
- `f1722a9` Record E1 GPU repeat result
- `592b96e` Add QKFormer LUT E2 replacement diagnostic
- `ba3f483` Add E2 full validation target sweep
- `69f7c3c` Add E2 calibration sweep
- `06239e0` Add E2 blend sweep
- Add E2 random calibration sweep
- Add E2 address-vs-global control sweep
- Add E2 shuffled-address control mode
- Add E3 trainable residual LUT adapter
- Add CIFAR-10 T=1 stress-test scripts

## Implemented Files

- `AGENTS.md`: project rules for future Codex sessions.
- `docs/QKFORMER_LUT_BRANCH_STATUS.md`: branch-level method/status note.
- `configs/qkformer_lut_e0_diag.yaml`: E0 config.
- `configs/qkformer_lut_e1_recon.yaml`: E1 split-aware reconstruction config.
- `configs/qkformer_lut_e2_replace.yaml`: E2 stage-wise replacement config.
- `configs/qkformer_lut_e3_trainable_lut.yaml`: E3 trainable LUT adapter config.
- `qkformer_lut/hooks.py`: non-invasive hooks for Q/K/gate/proj capture.
- `qkformer_lut/stats.py`: bucket occupancy and conditional variance stats.
- `tools/qkformer_lut_e0_diag.py`: E0 runner.
- `tools/qkformer_lut_e1_recon.py`: E1 calibration/evaluation reconstruction runner.
- `tools/qkformer_lut_e2_replace.py`: E2 stage-wise replacement runner.
- `tools/qkformer_lut_e3_trainable_lut.py`: E3 trainable residual LUT adapter runner.
- `scripts/server/install_qkformer_lut_deps.sh`: installs/checks server deps.
- `scripts/server/link_cifar10_data.sh`: symlinks CIFAR-10 into repo-local data path.
- `scripts/server/qkformer_lut_common.sh`: common Python/GPU selection helpers.
- `scripts/server/run_qkformer_lut_e0_diag.sh`: zero-arg E0 run.
- `scripts/server/run_qkformer_cifar10_train.sh`: zero-arg CIFAR-10 training run.
- `scripts/server/run_qkformer_cifar10_t1_train.sh`: CIFAR-10 T=1 training run.
- `scripts/server/run_qkformer_lut_e0_after_latest_train.sh`: runs E0 using the latest
  training result checkpoint.
- `scripts/server/run_qkformer_lut_e1_recon.sh`: runs E1 using the latest
  training result checkpoint.
- `scripts/server/run_qkformer_lut_e2_replace.sh`: runs E2 stage-wise
  replacement using the latest training result checkpoint.
- `scripts/server/run_qkformer_lut_e2_sweep.sh`: runs full-validation E2 target
  sweep for stage1-only, stage2-only, and stage1+stage2.
- `scripts/server/run_qkformer_lut_e2_calib_sweep.sh`: runs stage1-only E2
  across calibration sizes, with full validation by default.
- `scripts/server/run_qkformer_lut_e2_blend_sweep.sh`: runs stage1-only E2
  full-calibration/full-validation blend sweep.
- `scripts/server/run_qkformer_lut_e2_random_calib_sweep.sh`: runs stage1-only
  randomized calibration subset stability sweep.
- `scripts/server/run_qkformer_lut_e2_control_sweep.sh`: runs stage1-only
  `address_lut` vs `global_mean` vs `shuffled_address_lut` control sweep.
- `scripts/server/run_qkformer_lut_e3_trainable_lut.sh`: runs one E3 trainable
  LUT adapter experiment.
- `scripts/server/run_qkformer_lut_e3_adapter_sweep.sh`: runs E3 pilot control
  sweep across `address_lut`, `global_mean`, and `token_channel_lut`.
- `scripts/server/run_qkformer_lut_e3_conservative_sweep.sh`: runs E3
  conservative control sweep with fixed alpha 0.1, fewer epochs, lower LR, and
  stronger KL/local-MSE constraints.
- `scripts/server/run_qkformer_lut_t1_e0_after_latest_train.sh`: runs E0 with
  `QKFORMER_LUT_TIME_STEP=1` after latest T=1 training.
- `scripts/server/run_qkformer_lut_t1_e3_conservative_sweep.sh`: runs E3
  conservative control sweep with `QKFORMER_LUT_TIME_STEP=1`.
- `scripts/server/run_qkformer_lut_t1_e3_seed_sweep.sh`: runs T=1 E3
  conservative controls across seeds 42/43/44 by default.
- `scripts/server/package_qkformer_lut_t1_e3_seed_sweep.sh`: packages latest
  T=1 train/E0/E3 artifacts for upload.
- `scripts/local/run_remote_qk_lut_loop.sh`: local SSH/GitHub/server/download
  loop for unattended remote experiments.
- `scripts/local/analyze_qk_lut_results.py`: local result summarizer for T=1
  train/E0/E3 metrics.
- `scripts/local/qk_lut_quick_state.sh`: low-token quick-state command.
- `docs/QK_LUTFORMER_QUICK_STATE.md`: compact current state and next actions.
- `docs/REMOTE_EXPERIMENT_LOOP.md`: remote automation workflow and defaults.

## Server Results So Far

Dependency setup:

- `spikingjelly==0.0.0.0.12` installed.
- CUDA detected as 12.1.
- `cupy-cuda12x` installed and saw 4 CUDA devices.
- Server has `torch 2.5.1+cu121`, `timm 1.0.27`, `torchvision 0.20.1+cu121`.

Data:

- Server source dataset root is configurable with `CIFAR10_SOURCE_ROOT`.
- The observed server dataset root contains `cifar-10-batches-py` and
  `cifar-10-python.tar.gz`.
- `scripts/server/link_cifar10_data.sh` links this into
  `data/cifar10/cifar-10-batches-py`.

Uploaded E0 smoke result:

- Archive uploaded locally:
  `qkformer_lut_e0_diag_20260602_092338.tar.gz`
- Extracted/inspected files:
  - `metrics.json`
  - `train_log.txt`
- Commit used on server: `dfab09b`
- Data source in metrics: `cifar10`
- Batches: 8, batch size 32
- Checkpoint loaded: false
- Overall address coverage: 0.1328125
- Singleton fraction: 0.0
- Conditional variance: 0.0
- Candidate/background variance: 0.0 / 0.0
- Module spike rates: all Q/K/gate rates 0.0

Interpretation:

- The real-data diagnostic pipeline works.
- Random-init model is silent, so the variance/occupancy values do not prove
  LUT feasibility.
- Current verdict: `PENDING_REAL_DATA_CHECKPOINT`.

Trained CIFAR-10 result:

- Result: `results/qkformer_cifar10_train_20260604_171317`
- Best checkpoint: `output/qkformer_cifar10/model_best.pth.tar`
- Best validation Acc@1: 96.08% at epoch 384.
- Final epoch 409 Acc@1: 95.73%.

Formal trained-checkpoint E0 result:

- Result: `results/qkformer_lut_e0_diag_20260605_110814`
- Data source: real CIFAR-10
- Checkpoint loaded: true
- Overall address coverage: 0.5992965698242188
- Singleton fraction: 0.08653305969439885
- Conditional variance: 0.07126443506367477
- Stage1/stage2 address coverage: about 1.0 / 0.999
- Stage3 address coverage: about 0.199 average
- Current phase judgment: `CONDITIONAL GO`

Interpretation:

- The trained model is not silent; Q/K/gate spike rates are meaningful.
- Stage1/stage2 Token-QK addresses are well occupied.
- Conditional response variance remains high, so proceed to E1 reconstruction
  before any full LUT replacement.

E1 split-aware reconstruction result:

- Result: `results/qkformer_lut_e1_recon_20260605_133910`
- Commit used on server: `cc1f18a`
- Calibration: real CIFAR-10 train, 32 batches.
- Evaluation: real CIFAR-10 validation, 16 batches.
- Checkpoint loaded: true
- Overall global mean MSE: 0.0714635615743191
- Overall address LUT MSE: 0.06887314827955697
- Overall address relative MSE reduction: 3.582838808068491%
- Candidate/background relative MSE reduction: 0.033154317038117744%
- Stage1 / stage2 / stage3 address reductions: 6.147753618879997% /
  2.554365246580402% / 2.814618183406784%
- Eval address hit rate: 0.9994738101959229
- Current phase judgment: `CONDITIONAL GO` to E2 stage-wise replacement
  diagnostics.

Interpretation:

- Address-based prototypes beat global and candidate/background baselines on
  held-out batches.
- Candidate/background mean is effectively not useful here.
- Gains are modest; do not implement a full wrapper yet.

GPU-selection E1 repeat:

- Result: `results/qkformer_lut_e1_recon_20260605_134638`
- Commit used on server: `202835c`
- Command used `--gpu 2`; log confirms `CUDA_VISIBLE_DEVICES=2`,
  torch visible device count 1, current device 0, device name RTX 4090.
- Reconstruction metrics are identical to
  `results/qkformer_lut_e1_recon_20260605_133910`, confirming the GPU-selection
  wrapper did not change E1 behavior.

E2 stage1+stage2 replacement result:

- Result: `results/qkformer_lut_e2_replace_20260605_150017`
- Commit used on server: `592b96e`
- Calibration: real CIFAR-10 train, 32 batches.
- Evaluation: real CIFAR-10 validation, 16 batches.
- Targets: `stage1.0.tssa`, `stage2.0.tssa`
- Baseline loss / Acc@1 / Acc@5: 0.37332091107964516 / 95.1171875% / 100.0%
- Replacement loss / Acc@1 / Acc@5: 0.3554967865347862 / 96.09375% / 99.609375%
- Delta Acc@1 / Acc@5: +0.9765625% / -0.390625%
- Logit MSE: 0.06788053233176469
- KL to baseline: 0.02834802505094558
- Local replacement MSE: stage1 0.07325678256650765, stage2
  0.04382377505923311

Interpretation:

- E2 hook replacement works and does not collapse classification.
- The apparent Acc@1 improvement is based on only 512 validation images, so it
  is not yet a claim.
- Next action is a full-validation target sweep: stage1-only, stage2-only,
  stage1+stage2.

E2 full-validation target sweep:

- Stage1-only:
  - Result: `results/qkformer_lut_e2_replace_20260605_151339`
  - Baseline Acc@1 / loss: 95.73% / 0.3581527572154999
  - Replacement Acc@1 / loss: 95.88% / 0.3540114137172699
  - Delta Acc@1: +0.15%
- Stage2-only:
  - Result: `results/qkformer_lut_e2_replace_20260605_151502`
  - Replacement Acc@1 / loss: 95.59% / 0.3633519110202789
  - Delta Acc@1: -0.14%
- Stage1+stage2:
  - Result: `results/qkformer_lut_e2_replace_20260605_151608`
  - Replacement Acc@1 / loss: 95.54% / 0.3601838129043579
  - Delta Acc@1: -0.19%

Interpretation:

- Full-validation sweep weakens the 512-image smoke result.
- Stage1-only is the only slightly positive replacement.
- Stage2 and combined replacement are negative.
- Next action is a calibration-size sweep for stage1-only to test whether the
  small positive effect is robust or just calibration noise.

E2 stage1-only calibration-size sweep:

- 32 calibration batches:
  - Result: `results/qkformer_lut_e2_replace_20260605_152749`
  - Acc@1 95.73 -> 95.88, delta +0.15; loss 0.35815 -> 0.35401.
- 128 calibration batches:
  - Result: `results/qkformer_lut_e2_replace_20260605_152910`
  - Acc@1 95.73 -> 95.76, delta +0.03; loss 0.35815 -> 0.35620.
- 512 calibration batches:
  - Result: `results/qkformer_lut_e2_replace_20260605_153131`
  - Acc@1 95.73 -> 95.81, delta +0.08; loss 0.35815 -> 0.35428.
- Full train calibration:
  - Result: `results/qkformer_lut_e2_replace_20260605_153751`
  - Acc@1 95.73 -> 96.02, delta +0.29; loss 0.35815 -> 0.35419.

Interpretation:

- Stage1-only remains positive across calibration sizes.
- Full train calibration gives the largest small gain.
- The effect is still small; next action is a blend sweep to test whether
  partial replacement controls logit drift while preserving Acc@1.

E2 stage1-only blend sweep with full train calibration/full validation:

- Blend 0.0:
  - Result: `results/qkformer_lut_e2_replace_20260605_161043`
  - Acc@1 95.73 -> 95.73, delta +0.00; loss unchanged.
- Blend 0.25:
  - Result: `results/qkformer_lut_e2_replace_20260605_162745`
  - Acc@1 95.73 -> 96.00, delta +0.27; loss 0.35815 -> 0.35323.
  - Logit MSE / KL: 0.04099666884899139 / 0.01876032644510269.
- Blend 0.5:
  - Result: `results/qkformer_lut_e2_replace_20260605_164452`
  - Acc@1 95.73 -> 95.90, delta +0.17; loss 0.35815 -> 0.35499.
- Blend 0.75:
  - Result: `results/qkformer_lut_e2_replace_20260605_170218`
  - Acc@1 95.73 -> 95.85, delta +0.12; loss 0.35815 -> 0.35629.
- Blend 1.0:
  - Result: `results/qkformer_lut_e2_replace_20260605_171921`
  - Acc@1 95.73 -> 96.02, delta +0.29; loss 0.35815 -> 0.35419.
  - Logit MSE / KL: 0.053548186321258545 / 0.023518988977372646.

Interpretation:

- Full replacement has the highest Acc@1.
- Blend 0.25 has the best loss, preserves Acc@5, and has lower logit drift.
- The signal is still small and diagnostic. Next action is randomized
  calibration-subset stability testing for stage1-only, comparing blend 0.25
  and 1.0.

E2 stage1-only randomized calibration stability sweep:

- 128 calibration batches, blend 0.25, seeds 42/43/44:
  - mean Acc@1: 95.9333%; mean delta: +0.2033%.
  - min/max delta: +0.13/+0.24.
  - mean loss: 0.355467; mean logit MSE: 0.040937.
- 128 calibration batches, blend 1.0, seeds 42/43/44:
  - mean Acc@1: 95.7400%; mean delta: +0.0100%.
  - min/max delta: -0.09/+0.09.
  - mean loss: 0.355344; mean logit MSE: 0.054565.
- 512 calibration batches, blend 0.25, seeds 42/43/44:
  - mean Acc@1: 95.8600%; mean delta: +0.1300%.
  - min/max delta: +0.06/+0.19.
  - mean loss: 0.356766; mean logit MSE: 0.041317.
- 512 calibration batches, blend 1.0, seeds 42/43/44:
  - mean Acc@1: 95.8367%; mean delta: +0.1067%.
  - min/max delta: +0.02/+0.19.
  - mean loss: 0.355430; mean logit MSE: 0.054583.

Interpretation:

- Blend 0.25 is more stable under random calibration subsets, especially at
  128 batches.
- Full replacement has higher drift and one negative 128-batch seed.
- Next action is a control sweep: compare `address_lut` with `global_mean`
  smoothing at stage1-only, 128 calibration batches, blend 0.25, seeds
  42/43/44.

E2 stage1-only address-vs-global control sweep:

- Address LUT, 128 calibration batches, blend 0.25, seeds 42/43/44:
  - mean Acc@1: 95.9333%; mean delta: +0.2033%.
  - min/max delta: +0.13/+0.24.
  - mean loss: 0.355467; mean logit MSE: 0.040937.
- Global mean, same setting:
  - mean Acc@1: 95.9233%; mean delta: +0.1933%.
  - min/max delta: +0.07/+0.30.
  - mean loss: 0.354103; mean logit MSE: 0.041799.

Interpretation:

- Global-mean smoothing nearly matches address LUT in Acc@1 and is better in
  loss.
- This weakens the address-specific LUT hypothesis. Do not start wrapper
  design.
- Next action is shuffled-address LUT control. It keeps the prototype
  distribution but breaks address/prototype alignment. If it also matches
  address LUT, mark the current forward-only E2 replacement route `NO-GO` for
  an address-specific wrapper.

E3 trainable residual LUT adapter implementation:

- Freezes the trained QKFormer backbone.
- Calibrates LUT initialization from Q/K address prototypes with count-aware
  shrinkage.
- Trains only tiny LUT adapter tables plus optional learnable blend alpha.
- Uses CE + KL-to-baseline + local MSE loss.
- Default target: `stage1.0.tssa`.
- Default pilot: 128 calibration batches, 5 epochs over 128 train batches,
  full validation.
- Default control modes:
  - `address_lut`: full Q/K-derived address table.
  - `global_mean`: non-address trainable smoothing control.
  - `token_channel_lut`: weaker-address control that drops Q/K bit detail.

Interpretation:

- E3 is the paper-oriented method test. It does not claim a full hardware
  wrapper yet.
- A paper claim requires `address_lut` to beat both controls under the same
  frozen-backbone budget.

First E3 pilot result:

- Address LUT:
  - 2049 trainable parameters.
  - Learned alpha: 0.5625.
  - Acc@1 95.67 -> 95.51, delta -0.16.
  - Loss 0.23467 -> 0.23767.
- Global mean:
  - 2 trainable parameters.
  - Learned alpha: 0.3541.
  - Acc@1 94.97 -> 94.12, delta -0.85.
  - Loss 0.30126 -> 0.33587.
- Token-channel LUT:
  - 513 trainable parameters.
  - Learned alpha: 0.5374.
  - Acc@1 95.76 -> 95.43, delta -0.33.
  - Loss 0.23999 -> 0.24589.

Interpretation:

- E3 code path works and address LUT is least damaging.
- The learned alpha grows too high, causing drift/overfit.
- E3 evaluation now explicitly resets SNN state before baseline/teacher and
  replacement/student passes for cleaner repeatability.
- Next action is conservative E3: fixed alpha 0.1, 2 epochs, LR 0.003,
  lambda_kl 2.0, lambda_local_mse 0.2.

E3 conservative result:

- Address LUT:
  - 2048 trainable parameters.
  - Fixed alpha: 0.1.
  - Acc@1 95.72 -> 95.82, delta +0.10.
  - Loss 0.250749 -> 0.250908.
  - KL/logit MSE/local MSE: 0.022793 / 0.048916 / 0.002057.
- Global mean:
  - 1 trainable parameter.
  - Fixed alpha: 0.1.
  - Acc@1 95.74 -> 95.91, delta +0.17.
  - Loss 0.255298 -> 0.256633.
  - KL/logit MSE/local MSE: 0.025575 / 0.055398 / 0.005486.
- Token-channel LUT:
  - 512 trainable parameters.
  - Fixed alpha: 0.1.
  - Acc@1 95.80 -> 95.88, delta +0.08.
  - Loss 0.253012 -> 0.252757.
  - KL/logit MSE/local MSE: 0.021938 / 0.048714 / 0.002228.

Interpretation:

- Conservative E3 is stable and all modes stay above 95%.
- Address LUT has cleaner loss/drift than global mean, but global mean has the
  best Acc@1 delta.
- Address-specific accuracy advantage is not established on T=4.
- Next action is T=1 CIFAR-10 stress test to remove temporal averaging and
  evaluate the LUT address boundary.

T=1 CIFAR-10 stress result:

- Training:
  - Result: `results/qkformer_cifar10_train_20260606_001151`.
  - Best validation Acc@1: 95.20% at epoch 407.
  - Final epoch 409 Acc@1: 94.64%.
- E0:
  - Result: `results/qkformer_lut_e0_diag_20260606_093127`.
  - Checkpoint loaded: true.
  - Time step: 1.
  - Overall address coverage: 0.5804824829101562.
  - Singleton fraction: 0.07936228803294151.
  - Conditional variance: 0.05487808446146928.
  - Candidate/background variance: 0.06090743407254731 /
    0.05607881493643653.
  - Stage1/stage2/stage3 conditional variance:
    0.055049262856841516 / 0.02716715404410884 /
    0.06864796047246338.
- Conservative E3:
  - Address LUT: 2048 trainable parameters, fixed alpha 0.1, Acc@1
    94.88 -> 95.00, delta +0.12; loss 0.29417879979610445 ->
    0.2886085723400116; KL 0.03908095574975014.
  - Global mean: 1 trainable parameter, fixed alpha 0.1, Acc@1
    94.79 -> 94.72, delta -0.07; loss 0.2944240644454956 ->
    0.2944105110883713.
  - Token-channel LUT: 512 trainable parameters, fixed alpha 0.1, Acc@1
    94.91 -> 94.85, delta -0.06; loss 0.2899289110660553 ->
    0.2893794428348541.

Interpretation:

- T=1 is useful as a LUT boundary test because it removes temporal averaging
  while preserving a competitive CIFAR-10 checkpoint.
- T=1 E0 lowers conditional variance compared with T=4
  (0.054878 vs 0.071264) at similar address coverage
  (0.580482 vs 0.599297).
- T=1 conservative E3 gives the first control-separated address signal:
  `address_lut` is positive while `global_mean` and `token_channel_lut` are
  negative in the first run.
- This is not enough for a final paper claim because it is single-seed. The
  next action is T=1 E3 multi-seed control repeat.

T=1 E3 multi-seed conservative control sweep:

- Address LUT:
  - Seed 42: Acc@1 94.88 -> 95.00, delta +0.12.
  - Seed 43: Acc@1 94.58 -> 94.65, delta +0.07.
  - Seed 44: Acc@1 94.54 -> 94.53, delta -0.01.
  - Mean Acc@1 delta: +0.06; mean loss delta: -0.002594.
- Global mean:
  - Seed 42: Acc@1 94.79 -> 94.72, delta -0.07.
  - Seed 43: Acc@1 94.81 -> 94.64, delta -0.17.
  - Seed 44: Acc@1 94.52 -> 94.48, delta -0.04.
  - Mean Acc@1 delta: -0.0933; mean loss delta: +0.000182.
- Token-channel LUT:
  - Seed 42: Acc@1 94.91 -> 94.85, delta -0.06.
  - Seed 43: Acc@1 94.60 -> 94.66, delta +0.06.
  - Seed 44: Acc@1 94.71 -> 94.64, delta -0.07.
  - Mean Acc@1 delta: -0.0233; mean loss delta: -0.001102.

Interpretation:

- Address LUT is the only mode with positive mean Acc@1 delta and the best mean
  loss delta in this T=1 conservative control sweep.
- Current phase judgment: `CONDITIONAL GO`.
- This result supports the paper-oriented claim that Q/K binary addresses are
  more useful than global smoothing or token/channel-only addressing in the T=1
  boundary setting. It still does not justify energy/latency, ImageNet, or
  production-wrapper claims.
- Next action: either repeat with another T=1 checkpoint/training seed, or run a
  larger-dataset/stage stress test before escalating the claim.

## Known Compatibility Fixes

The server uses newer `timm` than upstream QKFormer expected.

- E0 bypasses `timm.create_model()` and directly instantiates
  `cifar10/model.py::spiking_transformer`.
- CIFAR-10 training keeps `create_model()`, but `cifar10/model.py::QKFormer`
  now filters timm factory kwargs against the real constructor signature.
- `convert_splitbn_model` import is optional because `timm 1.x` may not export
  it from `timm.models`.

## Next Action

Run the T=1 E3 multi-seed conservative control sweep on server:

```bash
cd ~/mac_agent/sdr-lutattn-qkformer-lut && git pull && bash scripts/server/run_qkformer_lut_t1_e3_seed_sweep.sh --gpu 2
```

Or run the full loop from local Codex:

```bash
cd /Users/cvue/Documents/github_zr/sdr-lutattn-qkformer-lut && bash scripts/local/run_remote_qk_lut_loop.sh
```

The local loop SSHes to `lbz@192.168.70.60`, pulls the latest GitHub branch,
activates conda env `sdr`, runs the server script, packages artifacts, downloads
them to local `results/`, extracts them, and runs local metric analysis.
Small diagnostics default to GPU 2. Expensive training jobs should use
`--kind train`, which checks `nvidia-smi` for an idle GPU before falling back to
GPU 2.

To specify a GPU, pass `--gpu N`:

```bash
cd ~/mac_agent/sdr-lutattn-qkformer-lut && bash scripts/server/run_qkformer_lut_e2_replace.sh --gpu 2
```

The scripts also accept `QKFORMER_LUT_GPU=2`. They set
`CUDA_VISIBLE_DEVICES` and print the requested GPU, visible CUDA devices,
current torch device, device name, and visible device count in the log.

Then package the latest T=1 train/E0/E3 artifacts:

```bash
cd ~/mac_agent/sdr-lutattn-qkformer-lut && bash scripts/server/package_qkformer_lut_t1_e3_seed_sweep.sh
```

Then upload or inspect:

- latest T=1 train `train_log.txt`, `checkpoint_manifest.json`, and
  `summary.csv`
- latest T=1 E0 `metrics.json` and `train_log.txt`
- latest nine T=1 E3 `metrics.json` and `train_log.txt`

Suggested artifact package:

```bash
qk_lutformer_t1_e3_seed_sweep_artifacts.tar.gz
```

## What To Avoid

- Do not infer GO/NO-GO from random-init E0.
- Do not return to CCS wrapper engineering unless explicitly requested.
- Do not claim QK-LUTFormer improves accuracy or energy before trained
  checkpoint E0 plus downstream experiments.
- Do not implement a full LUT wrapper until E1 shows split-aware reconstruction
  MSE improvement over global and candidate/background baselines.
- After E1, the next implementation target is E2 stage-wise replacement
  diagnostics, prioritizing stage1/stage2.
- E2 is a diagnostic hook replacement of module outputs. Do not describe it as
  a production LUTFormer implementation.
- Do not commit datasets, checkpoints, tarballs, or server logs.
