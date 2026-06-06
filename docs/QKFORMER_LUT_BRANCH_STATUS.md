# QK-LUTFormer Branch Status

Branch: `codex/qkformer-lut-hybrid`

Working scope: **QK-LUTFormer / QKFormer-LUT hybrid**. This branch is
independent from the stopped CCS-LUTAttn wrapper line and does not touch the
manifesto-lut series.

## Repository Baseline

- Upstream code: `https://github.com/zhouchenlin2096/QKFormer`
- Paper: `QKFormer: Hierarchical Spiking Transformer using Q-K Attention`,
  NeurIPS 2024, arXiv `2403.16552`
- Local reference project: `sdr-lutattn-ccs-lutattn`
- License check: the upstream repository currently has no root `LICENSE` file.
  Some ImageNet files contain inherited Meta/MAE license headers that refer to a
  missing root license file. Treat redistribution and paper artifact packaging
  as license-risk until clarified upstream.
- Dependencies from upstream README: `torch==1.12.1`, `timm==0.6.12`,
  `cupy==11.4.0`, `spikingjelly==0.0.0.0.12`, `pyyaml`, `tensorboard`.
- Training/eval entries:
  - CIFAR-10: `cifar10/train.py`, config `cifar10/cifar10.yml`
  - CIFAR-100: `cifar100/train.py`, config `cifar100/cifar100.yml`
  - ImageNet: `imagenet/train.py`, `imagenet/test.py`
  - DVS: `dvs128-gesture/train.py`, `cifar10-dvs/train.py`
- Q-K attention code:
  - CIFAR static image path: `cifar10/model.py`
  - CIFAR-100 path: `cifar100/model.py`
  - ImageNet path: `imagenet/qkformer.py`
  - Main modules: `Token_QK_Attention` and `Spiking_Self_Attention`

## Why CCS-LUTAttn Stopped

CCS-LUTAttn E0-E2 showed useful diagnostics but did not justify building the
converted wrapper:

- Coarse context reduced conditional variance by about **20.99%**, so row
  context contains real information.
- Local candidate masks identify high-variance salient pair regimes, but
  candidate-specific prototype tables degraded split-aware reconstruction.
- Background-specific tables were consistently useful, yet the aggregate
  head-output gain was small: about **1.75%** MSE reduction for straight
  two-tier and **0.60%** for the hybrid background-table metric.
- The surviving signal was diagnostic, not an implementation mandate. The
  current pairwise response prototype LUT is therefore **NO-GO** for a full
  converted Transformer attention wrapper.

## Why Pivot To QKFormer

The prior DeiT route fought standard softmax attention's row normalization:
pairwise addresses could not uniquely determine `A_ij * v_j` without row-level
context. QKFormer starts from a different premise: spike-form Q-K attention
already operates through binary Q/K spike vectors and token/channel gating.

This branch tests whether those binary spike vectors form a more natural LUT
address space:

- Binary Q/K vectors can be used directly as address bits instead of anchor
  order codes over floating-point Q/K.
- Token/channel gates are already part of the forward computation, so LUT
  calibration can target local gating or correction responses.
- QKFormer is hierarchical, so diagnostics can be stage-wise instead of a
  global 197x197 pair table.

## Claim Boundary

Current claims are limited to diagnostics and feasibility:

- We do **not** claim a full pure-spike Transformer block beyond what QKFormer
  already implements.
- We do **not** revive scalar softmax-score buckets.
- We do **not** use `exp(center)` plus row normalization.
- We may use direct training, distillation, or surrogate gradients in later
  phases, but that would be a trained hybrid method, not forward-only
  ANN-to-SNN conversion.
- E0 only measures address coverage, occupancy, sparsity, and conditional
  response variance. It does not claim accuracy, energy, latency, or downstream
  model improvement.
- E1 measures split-aware reconstruction only: calibration split prototypes
  are evaluated on held-out batches against global and candidate/background
  baselines. E1 still does not claim downstream accuracy, energy, or latency.

## QK-LUTFormer E0

Name: **QK-LUTFormer E0 = QKFormer Binary Address LUT Feasibility Diagnostic**

Goal:

- Capture binary Q/K spike vectors from QKFormer attention modules.
- Capture token/channel gating or attention-like outputs.
- Measure whether binary Q/K-derived addresses have practical bucket occupancy
  and lower conditional response variance than the CCS coarse-context baseline.

Implementation:

- Diagnostic code is external to the model:
  - `qkformer_lut/hooks.py`
  - `qkformer_lut/stats.py`
  - `tools/qkformer_lut_e0_diag.py`
- Server entry:
  - `scripts/server/run_qkformer_lut_e0_diag.sh`
- Dependency installer:
  - `scripts/server/install_qkformer_lut_deps.sh`
- CIFAR-10 data linker:
  - `scripts/server/link_cifar10_data.sh`
- CIFAR-10 checkpoint training entry:
  - `scripts/server/run_qkformer_cifar10_train.sh`
- Latest-checkpoint E0 entry:
  - `scripts/server/run_qkformer_lut_e0_after_latest_train.sh`
- Config:
  - `configs/qkformer_lut_e0_diag.yaml`

Default behavior:

- Uses CIFAR-10 QKFormer architecture because it is the smallest static-image
  path and avoids a large ImageNet run.
- Loads a checkpoint only if `checkpoint` in the YAML or
  `QKFORMER_LUT_CKPT` is set.
- Uses real CIFAR-10 validation data when available at the configured path.
  If data is missing, it falls back to synthetic input and marks
  `data.source=synthetic` in `metrics.json`. Synthetic/random-init results are
  smoke tests only and must not be used for a GO verdict.

Metrics written to `metrics.json`:

- `address_coverage`
- `bucket_occupancy`
- `singleton_fraction`
- `conditional_variance`
- `candidate_background_variance`
- `per_stage_summary`
- `per_block_summary`
- `module_summary`

## QK-LUTFormer E1

Name: **QK-LUTFormer E1 = Split-Aware Address Reconstruction Diagnostic**

Goal:

- Calibrate address-wise response prototypes on CIFAR-10 train batches.
- Evaluate held-out validation reconstruction MSE.
- Compare `address_lut` against `global_mean` and
  `candidate_background_mean` baselines.
- Decide whether stage-wise LUT replacement is worth implementing.

Implementation:

- E1 runner:
  - `tools/qkformer_lut_e1_recon.py`
- Config:
  - `configs/qkformer_lut_e1_recon.yaml`
- Server entry:
  - `scripts/server/run_qkformer_lut_e1_recon.sh`

Default behavior:

- Uses the latest trained CIFAR-10 checkpoint when
  `QKFORMER_LUT_CKPT` is unset and training results exist.
- Calibrates on 32 train batches and evaluates on 16 validation batches.
- Resets SNN state after every batch to match the training/evaluation loop.

Metrics written to `metrics.json`:

- `overall_reconstruction`
- `module_reconstruction`
- `per_stage_reconstruction`
- `per_block_reconstruction`
- `calibration_prototypes`
- `calibration_hook_summary`
- `evaluation_hook_summary`

## QK-LUTFormer E2

Name: **QK-LUTFormer E2 = Stage-Wise Replacement Diagnostic**

Goal:

- Calibrate address-wise response prototypes as in E1.
- Replace selected attention modules' `proj_lif` output with address-LUT
  predictions through forward hooks.
- Compare baseline and replacement validation loss/Acc@1/Acc@5 on the same
  batches.
- Measure logit MSE/KL to baseline and local replacement MSE.

Implementation:

- E2 runner:
  - `tools/qkformer_lut_e2_replace.py`
- Config:
  - `configs/qkformer_lut_e2_replace.yaml`
- Server entry:
  - `scripts/server/run_qkformer_lut_e2_replace.sh`

Default behavior:

- Uses latest trained CIFAR-10 checkpoint when `QKFORMER_LUT_CKPT` is unset.
- Calibrates on 32 train batches and evaluates on 16 validation batches.
- Default replacement targets are `stage1.0.tssa` and `stage2.0.tssa`.
- This is diagnostic module-output replacement, not a full optimized wrapper.
- `QKFORMER_LUT_E2_TARGETS` can override comma-separated target modules.
- `QKFORMER_LUT_E2_MODE` can override the replacement mode. Supported modes
  are `address_lut`, `global_mean`, and `shuffled_address_lut`.
- `QKFORMER_LUT_E2_CALIB_BATCHES` and `QKFORMER_LUT_E2_EVAL_BATCHES` can
  override batch counts; `QKFORMER_LUT_E2_EVAL_BATCHES=0` evaluates the full
  split.
- `scripts/server/run_qkformer_lut_e2_sweep.sh` runs stage1-only, stage2-only,
  and stage1+stage2 with full validation by default.
- `scripts/server/run_qkformer_lut_e2_calib_sweep.sh` runs stage1-only across
  calibration sizes 32, 128, 512, and full train by default, with full
  validation evaluation.
- `scripts/server/run_qkformer_lut_e2_blend_sweep.sh` runs stage1-only
  full-calibration/full-validation blend ratios 0, 0.25, 0.5, 0.75, and 1.0.
- `QKFORMER_LUT_E2_CALIB_SHUFFLE=1` and
  `QKFORMER_LUT_E2_CALIB_SEED=N` can randomize calibration subset order for
  stability checks.
- `scripts/server/run_qkformer_lut_e2_random_calib_sweep.sh` runs randomized
  stage1-only calibration subset sweeps across calibration sizes, blend ratios,
  and seeds.
- `scripts/server/run_qkformer_lut_e2_control_sweep.sh` runs stage1-only
  `address_lut` vs `global_mean` vs `shuffled_address_lut` replacement
  controls.

Metrics written to `metrics.json`:

- `classification`
- `local_replacement`
- `calibration_prototypes`
- `calibration_hook_summary`
- `target_modules`

First E2 result:

- Result: `results/qkformer_lut_e2_replace_20260605_150017`
- Targets: `stage1.0.tssa`, `stage2.0.tssa`
- Evaluation: 16 validation batches.
- Baseline Acc@1: 95.1171875%
- Replacement Acc@1: 96.09375%
- Delta Acc@1: +0.9765625%
- Replacement loss improved from 0.37332091107964516 to 0.3554967865347862.

Interpretation: diagnostic replacement ran successfully and did not collapse
accuracy, but the evaluation set was only 512 images. Treat this as a smoke
result; run the full-validation E2 target sweep before making a stage decision.

Full-validation E2 target sweep:

- Stage1-only: Acc@1 95.73 -> 95.88, delta +0.15; loss 0.35815 -> 0.35401.
- Stage2-only: Acc@1 95.73 -> 95.59, delta -0.14; loss 0.35815 -> 0.36335.
- Stage1+stage2: Acc@1 95.73 -> 95.54, delta -0.19; loss 0.35815 -> 0.36018.

Interpretation: only stage1-only is slightly positive on full validation.
Stage2 and combined replacement are negative. Run calibration-size sweep for
stage1-only before making a forward-only replacement decision.

Stage1-only calibration-size sweep:

- 32 calibration batches: Acc@1 95.73 -> 95.88, delta +0.15.
- 128 calibration batches: Acc@1 95.73 -> 95.76, delta +0.03.
- 512 calibration batches: Acc@1 95.73 -> 95.81, delta +0.08.
- Full train calibration: Acc@1 95.73 -> 96.02, delta +0.29.

Interpretation: stage1-only stays positive, and full calibration is best so
far. The effect is small; run a blend sweep before deciding whether this is a
usable hybrid replacement signal.

Stage1-only blend sweep with full train calibration/full validation:

- Blend 0.0: Acc@1 95.73 -> 95.73, delta +0.00; loss unchanged.
- Blend 0.25: Acc@1 95.73 -> 96.00, delta +0.27; loss 0.35815 -> 0.35323;
  logit MSE 0.04099666884899139, KL 0.01876032644510269.
- Blend 0.5: Acc@1 95.73 -> 95.90, delta +0.17; loss 0.35815 -> 0.35499.
- Blend 0.75: Acc@1 95.73 -> 95.85, delta +0.12; loss 0.35815 -> 0.35629.
- Blend 1.0: Acc@1 95.73 -> 96.02, delta +0.29; loss 0.35815 -> 0.35419;
  logit MSE 0.053548186321258545, KL 0.023518988977372646.

Interpretation: full replacement gives the highest Acc@1, while blend 0.25
gives the best loss, preserves Acc@5, and has lower logit drift. Treat this as
a small diagnostic signal, not a production claim. Next step is a randomized
calibration-subset stability sweep for stage1-only, comparing blend 0.25 and
1.0 before any wrapper/prototype design.

Stage1-only randomized calibration stability sweep:

- 128 calibration batches, blend 0.25, seeds 42/43/44: mean Acc@1 95.9333%,
  mean delta +0.2033%, min/max delta +0.13/+0.24, mean loss 0.355467,
  mean logit MSE 0.040937.
- 128 calibration batches, blend 1.0, seeds 42/43/44: mean Acc@1 95.7400%,
  mean delta +0.0100%, min/max delta -0.09/+0.09, mean loss 0.355344,
  mean logit MSE 0.054565.
- 512 calibration batches, blend 0.25, seeds 42/43/44: mean Acc@1 95.8600%,
  mean delta +0.1300%, min/max delta +0.06/+0.19, mean loss 0.356766,
  mean logit MSE 0.041317.
- 512 calibration batches, blend 1.0, seeds 42/43/44: mean Acc@1 95.8367%,
  mean delta +0.1067%, min/max delta +0.02/+0.19, mean loss 0.355430,
  mean logit MSE 0.054583.

Interpretation: blend 0.25 is more stable than full replacement under random
calibration subsets. The next control should compare `address_lut` against
`global_mean` smoothing at stage1-only, 128 calibration batches, blend 0.25,
and the same random seeds. If global smoothing matches the address LUT, the
current accuracy signal is not address-specific enough to justify wrapper
design.

Stage1-only address-vs-global control sweep:

- Address LUT, 128 calibration batches, blend 0.25, seeds 42/43/44: mean Acc@1
  95.9333%, mean delta +0.2033%, min/max delta +0.13/+0.24, mean loss
  0.355467, mean logit MSE 0.040937.
- Global mean, same setting: mean Acc@1 95.9233%, mean delta +0.1933%,
  min/max delta +0.07/+0.30, mean loss 0.354103, mean logit MSE 0.041799.

Interpretation: global-mean smoothing nearly matches address LUT in Acc@1 and
is better in loss. This weakens the address-specific LUT hypothesis; do not
start wrapper design. Next step is a shuffled-address LUT control that keeps
the prototype distribution but breaks address/prototype alignment. If shuffled
address LUT also matches address LUT, treat the current forward-only E2
replacement route as `NO-GO` for an address-specific wrapper and pivot to either
trained hybrid regularization or stop at diagnostic reporting.

## QK-LUTFormer E3

Name: **QK-LUTFormer E3 = Trainable Residual LUT Adapter**

Goal:

- Move from forward-only statistical prototypes to a paper-relevant trainable
  LUT adapter.
- Freeze the trained QKFormer backbone.
- Calibrate LUT initialization from Q/K address prototypes with count-aware
  shrinkage.
- Train only small LUT tables and optional learnable blend alpha.
- Compare address-specific LUTs against non-address and weaker-address
  controls under the same training budget.

Implementation:

- E3 runner:
  - `tools/qkformer_lut_e3_trainable_lut.py`
- Config:
  - `configs/qkformer_lut_e3_trainable_lut.yaml`
- Server entries:
  - `scripts/server/run_qkformer_lut_e3_trainable_lut.sh`
  - `scripts/server/run_qkformer_lut_e3_adapter_sweep.sh`

Default behavior:

- Uses latest trained CIFAR-10 checkpoint when `QKFORMER_LUT_CKPT` is unset.
- Default target is `stage1.0.tssa`.
- Default calibration is 128 shuffled train batches.
- Default training is 5 epochs over 128 shuffled train batches.
- Default evaluation is full CIFAR-10 validation.
- Default sweep compares `address_lut`, `global_mean`, and
  `token_channel_lut`.

Training loss:

- CE to ground-truth labels.
- KL to frozen baseline logits.
- Local MSE to constrain adapter output drift.

Metrics written to `metrics.json`:

- `classification`
- `train_history`
- `adapter_summary`
- `calibration_prototypes`
- `calibration_hook_summary`

Interpretation:

- E2 showed that naive forward-only address prototypes are not address-specific
  enough, because global-mean smoothing nearly matched address LUT.
- E3 tests the stronger paper claim: a small trainable residual address LUT
  can exploit Q/K address semantics better than controls while keeping the
  backbone frozen.
- If `address_lut` does not beat `global_mean` and `token_channel_lut`, the
  paper story should not claim address-specific LUT advantage.

First E3 pilot sweep:

- Address LUT: 2049 trainable parameters, alpha learned to 0.5625, Acc@1
  95.67 -> 95.51, delta -0.16, loss 0.23467 -> 0.23767.
- Global mean: 2 trainable parameters, alpha learned to 0.3541, Acc@1
  94.97 -> 94.12, delta -0.85, loss 0.30126 -> 0.33587.
- Token-channel LUT: 513 trainable parameters, alpha learned to 0.5374, Acc@1
  95.76 -> 95.43, delta -0.33, loss 0.23999 -> 0.24589.

Interpretation:

- The E3 path runs end-to-end and address LUT is the least damaging adapter,
  but all modes degrade validation accuracy in the first pilot.
- Alpha grows too high under 5 epochs, suggesting overfitting/drift rather than
  stable residual correction.
- Next step is a conservative E3 sweep: fixed alpha 0.1, 2 epochs, lower LR,
  stronger KL and local-MSE constraints, same three controls.

Conservative E3 sweep:

- Address LUT: 2048 trainable parameters, fixed alpha 0.1, Acc@1
  95.72 -> 95.82, delta +0.10; loss 0.250749 -> 0.250908; KL 0.022793.
- Global mean: 1 trainable parameter, fixed alpha 0.1, Acc@1 95.74 -> 95.91,
  delta +0.17; loss 0.255298 -> 0.256633; KL 0.025575.
- Token-channel LUT: 512 trainable parameters, fixed alpha 0.1, Acc@1
  95.80 -> 95.88, delta +0.08; loss 0.253012 -> 0.252757; KL 0.021938.

Interpretation:

- Conservative E3 stabilizes the adapters and all three remain above 95% with
  small positive Acc@1 deltas.
- Address LUT does not beat global mean in Acc@1, so address-specific accuracy
  advantage is not yet established.
- Address LUT has cleaner loss/drift than global mean, but the paper claim
  still needs a sharper setting.
- Next step is a CIFAR-10 T=1 stress test. T=1 removes temporal averaging and
  should expose whether address LUTs retain useful structure in the single-step
  boundary case.

T=1 CIFAR-10 stress test:

- Training:
  - Result: `results/qkformer_cifar10_train_20260606_001151`.
  - Best Acc@1: 95.20% at epoch 407.
  - Final epoch 409 Acc@1: 94.64%.
- E0:
  - Result: `results/qkformer_lut_e0_diag_20260606_093127`.
  - Checkpoint loaded: true; time step: 1.
  - Overall address coverage: 0.5804824829101562.
  - Singleton fraction: 0.07936228803294151.
  - Conditional variance: 0.05487808446146928.
  - Stage1/stage2/stage3 conditional variance:
    0.055049262856841516 / 0.02716715404410884 /
    0.06864796047246338.
- Conservative E3:
  - Address LUT: 2048 trainable parameters, fixed alpha 0.1, Acc@1
    94.88 -> 95.00, delta +0.12; loss 0.294179 -> 0.288609; KL 0.039081.
  - Global mean: 1 trainable parameter, fixed alpha 0.1, Acc@1
    94.79 -> 94.72, delta -0.07; loss nearly unchanged.
  - Token-channel LUT: 512 trainable parameters, fixed alpha 0.1, Acc@1
    94.91 -> 94.85, delta -0.06; loss 0.289929 -> 0.289379.

Interpretation:

- T=1 lowers conditional variance compared with T=4
  (0.054878 vs 0.071264) while maintaining similar coverage
  (0.580482 vs 0.599297).
- In the first T=1 conservative E3 run, address LUT is the only adapter with a
  positive Acc@1 delta and reaches 95.00%.
- This is the strongest address-specific signal so far, but it is still
  single-seed. The next step is a T=1 E3 multi-seed sweep across
  `address_lut`, `global_mean`, and `token_channel_lut`.

## Phase Gate

- **GO**: QKFormer binary Q/K addresses have materially better bucket occupancy
  than CCS and conditional variance lower than the CCS coarse context baseline,
  using real data and a trained checkpoint.
- **CONDITIONAL GO**: addresses are stable and better occupied, but response
  variance remains high; next experiment should try background correction LUT or
  stage-wise LUT.
- **NO-GO**: binary Q/K addresses are still highly sparse or do not explain
  the response; stop LUT-ization and switch to ordinary QKFormer direct
  training or hybrid ANN-SNN baseline.

Current verdict: **CONDITIONAL GO after trained-checkpoint E0**. The next gate
is E1 split-aware reconstruction. E1 result
`results/qkformer_lut_e1_recon_20260605_133910` showed held-out address LUT
MSE improvement over global and candidate/background baselines:

- overall global mean MSE: 0.0714635615743191
- overall address LUT MSE: 0.06887314827955697
- overall address relative MSE reduction: 3.582838808068491%
- candidate/background relative MSE reduction: 0.033154317038117744%
- stage1 / stage2 / stage3 address reductions: 6.147753618879997% /
  2.554365246580402% / 2.814618183406784%

Current verdict: **CONDITIONAL GO to E2 stage-wise replacement diagnostics**.
The E1 gain is real but modest; prioritize stage1/stage2 and do not build a
full wrapper yet.

GPU-selection repeat result `results/qkformer_lut_e1_recon_20260605_134638`
matches the original E1 metrics exactly and confirms `--gpu 2` logging works:
the server log records `CUDA_VISIBLE_DEVICES=2`, one visible torch device, and
RTX 4090 as the active CUDA device.

## Server Workflow Update

The CIFAR-10 training script writes both `checkpoint_manifest.txt` and
`checkpoint_manifest.json`. The JSON manifest records `best_checkpoint` and
`latest_checkpoint` so formal E0 can run without manually copying paths.
Set `QKFORMER_LUT_TIME_STEP=1` or use the T=1 wrapper scripts to train and
evaluate single-step QKFormer checkpoints.

Formal E0 after a completed training run:

```bash
cd ~/mac_agent/sdr-lutattn-qkformer-lut && bash scripts/server/run_qkformer_lut_e0_after_latest_train.sh
```

This is equivalent to:

```bash
cd ~/mac_agent/sdr-lutattn-qkformer-lut && QKFORMER_LUT_CKPT=auto bash scripts/server/run_qkformer_lut_e0_diag.sh
```

CIFAR-10 T=1 checkpoint training:

```bash
cd ~/mac_agent/sdr-lutattn-qkformer-lut && bash scripts/server/run_qkformer_cifar10_t1_train.sh --gpu 2
```

T=1 E0 after a completed T=1 training run:

```bash
cd ~/mac_agent/sdr-lutattn-qkformer-lut && bash scripts/server/run_qkformer_lut_t1_e0_after_latest_train.sh --gpu 2
```

Formal E1 after a completed training run:

```bash
cd ~/mac_agent/sdr-lutattn-qkformer-lut && bash scripts/server/run_qkformer_lut_e1_recon.sh
```

Formal E2 after a completed training run:

```bash
cd ~/mac_agent/sdr-lutattn-qkformer-lut && bash scripts/server/run_qkformer_lut_e2_replace.sh --gpu 2
```

Full-validation E2 target sweep:

```bash
cd ~/mac_agent/sdr-lutattn-qkformer-lut && bash scripts/server/run_qkformer_lut_e2_sweep.sh --gpu 2
```

Stage1-only calibration-size sweep:

```bash
cd ~/mac_agent/sdr-lutattn-qkformer-lut && bash scripts/server/run_qkformer_lut_e2_calib_sweep.sh --gpu 2
```

Stage1-only blend sweep:

```bash
cd ~/mac_agent/sdr-lutattn-qkformer-lut && bash scripts/server/run_qkformer_lut_e2_blend_sweep.sh --gpu 2
```

Stage1-only randomized calibration stability sweep:

```bash
cd ~/mac_agent/sdr-lutattn-qkformer-lut && bash scripts/server/run_qkformer_lut_e2_random_calib_sweep.sh --gpu 2
```

Stage1-only address-vs-global control sweep:

```bash
cd ~/mac_agent/sdr-lutattn-qkformer-lut && bash scripts/server/run_qkformer_lut_e2_control_sweep.sh --gpu 2
```

E3 trainable LUT adapter pilot sweep:

```bash
cd ~/mac_agent/sdr-lutattn-qkformer-lut && bash scripts/server/run_qkformer_lut_e3_adapter_sweep.sh --gpu 2
```

E3 conservative trainable LUT adapter sweep:

```bash
cd ~/mac_agent/sdr-lutattn-qkformer-lut && bash scripts/server/run_qkformer_lut_e3_conservative_sweep.sh --gpu 2
```

T=1 E3 conservative trainable LUT adapter sweep:

```bash
cd ~/mac_agent/sdr-lutattn-qkformer-lut && bash scripts/server/run_qkformer_lut_t1_e3_conservative_sweep.sh --gpu 2
```

T=1 E3 multi-seed conservative control sweep:

```bash
cd ~/mac_agent/sdr-lutattn-qkformer-lut && git pull && bash scripts/server/run_qkformer_lut_t1_e3_seed_sweep.sh --gpu 2
cd ~/mac_agent/sdr-lutattn-qkformer-lut && bash scripts/server/package_qkformer_lut_t1_e3_seed_sweep.sh
```

Local SSH/GitHub/server/download/analyze loop:

```bash
cd /Users/cvue/Documents/github_zr/sdr-lutattn-qkformer-lut && bash scripts/local/run_remote_qk_lut_loop.sh
```

Use `--kind train` for expensive training jobs so the remote command checks
GPU availability before falling back to GPU 2. Small calibration and diagnostic
jobs default to GPU 2 without probing every time.

Specify GPU for train/E0/E1/E2:

```bash
cd ~/mac_agent/sdr-lutattn-qkformer-lut && bash scripts/server/run_qkformer_lut_e1_recon.sh --gpu 2
```

Equivalent:

```bash
cd ~/mac_agent/sdr-lutattn-qkformer-lut && QKFORMER_LUT_GPU=2 bash scripts/server/run_qkformer_lut_e1_recon.sh
```
