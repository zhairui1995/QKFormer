# Codex Handoff: QK-LUTFormer / QKFormer-LUT Hybrid

Last updated: 2026-06-05

## One-Line State

QKFormer CIFAR-10 training reached 96.08% best Acc@1. Formal E0 and E1 remain
CONDITIONAL GO: Q/K/gate addresses are active and reconstruct slightly better
than baselines, but response variance is still high. E2 stage1-only replacement
is the only target that stays slightly positive on full validation. Randomized
calibration subsets show blend 0.25 is more stable than full replacement. Next
action is an E2 shuffled-address control after `global_mean` smoothing nearly
matched `address_lut`, weakening the address-specific wrapper hypothesis.

## Project Identity

- Local repo: this repository checkout
- Server repo: `~/mac_agent/sdr-lutattn-qkformer-lut`
- Branch: `codex/qkformer-lut-hybrid`
- Push remote: `fork` / `git@github.com:zhairui1995/QKFormer.git`
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

## Implemented Files

- `AGENTS.md`: project rules for future Codex sessions.
- `docs/QKFORMER_LUT_BRANCH_STATUS.md`: branch-level method/status note.
- `configs/qkformer_lut_e0_diag.yaml`: E0 config.
- `configs/qkformer_lut_e1_recon.yaml`: E1 split-aware reconstruction config.
- `configs/qkformer_lut_e2_replace.yaml`: E2 stage-wise replacement config.
- `qkformer_lut/hooks.py`: non-invasive hooks for Q/K/gate/proj capture.
- `qkformer_lut/stats.py`: bucket occupancy and conditional variance stats.
- `tools/qkformer_lut_e0_diag.py`: E0 runner.
- `tools/qkformer_lut_e1_recon.py`: E1 calibration/evaluation reconstruction runner.
- `tools/qkformer_lut_e2_replace.py`: E2 stage-wise replacement runner.
- `scripts/server/install_qkformer_lut_deps.sh`: installs/checks server deps.
- `scripts/server/link_cifar10_data.sh`: symlinks CIFAR-10 into repo-local data path.
- `scripts/server/qkformer_lut_common.sh`: common Python/GPU selection helpers.
- `scripts/server/run_qkformer_lut_e0_diag.sh`: zero-arg E0 run.
- `scripts/server/run_qkformer_cifar10_train.sh`: zero-arg CIFAR-10 training run.
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

## Known Compatibility Fixes

The server uses newer `timm` than upstream QKFormer expected.

- E0 bypasses `timm.create_model()` and directly instantiates
  `cifar10/model.py::spiking_transformer`.
- CIFAR-10 training keeps `create_model()`, but `cifar10/model.py::QKFormer`
  now filters timm factory kwargs against the real constructor signature.
- `convert_splitbn_model` import is optional because `timm 1.x` may not export
  it from `timm.models`.

## Next Action

Run E2 shuffled-address control sweep on server:

```bash
cd ~/mac_agent/sdr-lutattn-qkformer-lut && git pull && bash scripts/server/run_qkformer_lut_e2_control_sweep.sh --gpu 2
```

To specify a GPU, pass `--gpu N`:

```bash
cd ~/mac_agent/sdr-lutattn-qkformer-lut && bash scripts/server/run_qkformer_lut_e2_replace.sh --gpu 2
```

The scripts also accept `QKFORMER_LUT_GPU=2`. They set
`CUDA_VISIBLE_DEVICES` and print the requested GPU, visible CUDA devices,
current torch device, device name, and visible device count in the log.

Then upload or inspect:

- latest nine `results/qkformer_lut_e2_replace_*/metrics.json`
- latest nine `results/qkformer_lut_e2_replace_*/train_log.txt`

Suggested artifact package:

```bash
cd ~/mac_agent/sdr-lutattn-qkformer-lut
E2_DIRS=$(ls -td results/qkformer_lut_e2_replace_* | head -9)
tar -czf qk_lutformer_e2_shuffled_control_sweep_artifacts.tar.gz \
  $(for d in $E2_DIRS; do echo "$d/metrics.json" "$d/train_log.txt"; done)
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
