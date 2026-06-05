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
is E1 split-aware reconstruction. Proceed to wrapper implementation only if E1
shows held-out `address_lut_mse` improvement over global and
candidate/background baselines, especially in stage1/stage2.

## Server Workflow Update

The CIFAR-10 training script writes both `checkpoint_manifest.txt` and
`checkpoint_manifest.json`. The JSON manifest records `best_checkpoint` and
`latest_checkpoint` so formal E0 can run without manually copying paths.

Formal E0 after a completed training run:

```bash
cd ~/mac_agent/sdr-lutattn-qkformer-lut && bash scripts/server/run_qkformer_lut_e0_after_latest_train.sh
```

This is equivalent to:

```bash
cd ~/mac_agent/sdr-lutattn-qkformer-lut && QKFORMER_LUT_CKPT=auto bash scripts/server/run_qkformer_lut_e0_diag.sh
```

Formal E1 after a completed training run:

```bash
cd ~/mac_agent/sdr-lutattn-qkformer-lut && bash scripts/server/run_qkformer_lut_e1_recon.sh
```
