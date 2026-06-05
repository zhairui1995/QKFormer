# Codex Handoff: QK-LUTFormer / QKFormer-LUT Hybrid

Last updated: 2026-06-05

## One-Line State

QKFormer CIFAR-10 training reached 96.08% best Acc@1, and formal E0 with the
trained checkpoint is a CONDITIONAL GO: Q/K/gate spikes are active and address
occupancy is useful, but conditional response variance remains high. E1
split-aware reconstruction code is implemented and awaits a server run.

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
- pending local changes: add E1 split-aware reconstruction diagnostic and
  latest-checkpoint server runner.

## Implemented Files

- `AGENTS.md`: project rules for future Codex sessions.
- `docs/QKFORMER_LUT_BRANCH_STATUS.md`: branch-level method/status note.
- `configs/qkformer_lut_e0_diag.yaml`: E0 config.
- `configs/qkformer_lut_e1_recon.yaml`: E1 split-aware reconstruction config.
- `qkformer_lut/hooks.py`: non-invasive hooks for Q/K/gate/proj capture.
- `qkformer_lut/stats.py`: bucket occupancy and conditional variance stats.
- `tools/qkformer_lut_e0_diag.py`: E0 runner.
- `tools/qkformer_lut_e1_recon.py`: E1 calibration/evaluation reconstruction runner.
- `scripts/server/install_qkformer_lut_deps.sh`: installs/checks server deps.
- `scripts/server/link_cifar10_data.sh`: symlinks CIFAR-10 into repo-local data path.
- `scripts/server/run_qkformer_lut_e0_diag.sh`: zero-arg E0 run.
- `scripts/server/run_qkformer_cifar10_train.sh`: zero-arg CIFAR-10 training run.
- `scripts/server/run_qkformer_lut_e0_after_latest_train.sh`: runs E0 using the latest
  training result checkpoint.
- `scripts/server/run_qkformer_lut_e1_recon.sh`: runs E1 using the latest
  training result checkpoint.

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

## Known Compatibility Fixes

The server uses newer `timm` than upstream QKFormer expected.

- E0 bypasses `timm.create_model()` and directly instantiates
  `cifar10/model.py::spiking_transformer`.
- CIFAR-10 training keeps `create_model()`, but `cifar10/model.py::QKFormer`
  now filters timm factory kwargs against the real constructor signature.
- `convert_splitbn_model` import is optional because `timm 1.x` may not export
  it from `timm.models`.

## Next Action

Run E1 split-aware reconstruction on server:

```bash
cd ~/mac_agent/sdr-lutattn-qkformer-lut && git pull && bash scripts/server/run_qkformer_lut_e1_recon.sh
```

Then upload or inspect:

- latest `results/qkformer_lut_e1_recon_*/metrics.json`
- latest `results/qkformer_lut_e1_recon_*/module_reconstruction.csv`
- latest `results/qkformer_lut_e1_recon_*/train_log.txt`

Suggested artifact package:

```bash
cd ~/mac_agent/sdr-lutattn-qkformer-lut
E1_DIR=$(ls -td results/qkformer_lut_e1_recon_* | head -1)
tar -czf qk_lutformer_e1_artifacts.tar.gz \
  "$E1_DIR/metrics.json" \
  "$E1_DIR/module_reconstruction.csv" \
  "$E1_DIR/train_log.txt"
```

## What To Avoid

- Do not infer GO/NO-GO from random-init E0.
- Do not return to CCS wrapper engineering unless explicitly requested.
- Do not claim QK-LUTFormer improves accuracy or energy before trained
  checkpoint E0 plus downstream experiments.
- Do not implement a full LUT wrapper until E1 shows split-aware reconstruction
  MSE improvement over global and candidate/background baselines.
- Do not commit datasets, checkpoints, tarballs, or server logs.
