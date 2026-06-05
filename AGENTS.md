# QK-LUTFormer Codex Project Rules

This repository is the independent **QK-LUTFormer / QKFormer-LUT hybrid**
branch for SDR-LUTAttn.

## Required Context

Before important work, read:

1. the user's global Codex rules file, when available
2. `docs/QKFORMER_LUT_BRANCH_STATUS.md`
3. `docs/CODEX_HANDOFF_QKFORMER_LUT.md`

Use the original CCS project only as historical context. If the sibling CCS
checkout is available, relevant docs there are:
  - `AGENTS.md`
  - `docs/CCS_SERIES_STATUS.md`
  - `docs/decision_log.md`

Do not modify the CCS repository unless the user explicitly asks.

## Scope

- Branch: `codex/qkformer-lut-hybrid`
- Upstream base: `https://github.com/zhouchenlin2096/QKFormer`
- Working remote fork: `git@github.com:zhairui1995/QKFormer.git`
- Main local path: this repository checkout
- Server path used so far: `~/mac_agent/sdr-lutattn-qkformer-lut`

This branch tests whether QKFormer binary Q/K spike vectors and token/channel
gating form a better LUT address space than the stopped DeiT/CCS response
prototype route.

## Strict Boundaries

- Do not continue CCS-LUTAttn wrapper implementation.
- Do not touch the manifesto-lut series.
- Do not revive scalar softmax-score bucket LUTs.
- Do not implement `exp(center)` plus row normalization.
- Do not claim a full pure-spike Transformer block beyond what QKFormer itself
  implements.
- Do not claim accuracy, energy, latency, or downstream gains without real
  metrics.
- Distillation, surrogate gradients, or direct training are allowed later only
  as trained hybrid methods, not forward-only ANN-to-SNN conversion.

## Current Evidence

CCS-LUTAttn E0-E2 ended as NO-GO for the current converted wrapper:

- Coarse context variance reduction: about 20.99%.
- Candidate-specific prototype tables degraded split-aware reconstruction.
- Background tables helped, but aggregate head-output improvement was only
  about 1.75%.

QK-LUTFormer E0 code path is implemented and can run on CIFAR-10:

- Diagnostic hooks: `qkformer_lut/hooks.py`
- Stats helpers: `qkformer_lut/stats.py`
- E0 runner: `tools/qkformer_lut_e0_diag.py`
- E1 runner: `tools/qkformer_lut_e1_recon.py`
- E2 runner: `tools/qkformer_lut_e2_replace.py`
- Config: `configs/qkformer_lut_e0_diag.yaml`
- E1 config: `configs/qkformer_lut_e1_recon.yaml`
- E2 config: `configs/qkformer_lut_e2_replace.yaml`
- Server scripts:
  - `scripts/server/install_qkformer_lut_deps.sh`
  - `scripts/server/link_cifar10_data.sh`
  - `scripts/server/qkformer_lut_common.sh`
  - `scripts/server/run_qkformer_lut_e0_diag.sh`
  - `scripts/server/run_qkformer_cifar10_train.sh`
  - `scripts/server/run_qkformer_lut_e0_after_latest_train.sh`
  - `scripts/server/run_qkformer_lut_e1_recon.sh`
  - `scripts/server/run_qkformer_lut_e2_replace.sh`
  - `scripts/server/run_qkformer_lut_e2_sweep.sh`
  - `scripts/server/run_qkformer_lut_e2_calib_sweep.sh`
  - `scripts/server/run_qkformer_lut_e2_blend_sweep.sh`

Latest uploaded E0 smoke result:

- Result: `results/qkformer_lut_e0_diag_20260602_092338`
- Data source: real CIFAR-10
- Checkpoint loaded: false
- Q/K/gate spike rates: all 0.0
- Conditional variance: 0.0
- Verdict: `PENDING_REAL_DATA_CHECKPOINT`

Interpretation: the diagnostic pipeline and real-data link work, but the result
is not scientifically meaningful because the model is random-init and silent.

Latest trained CIFAR-10 checkpoint evidence:

- Train result: `results/qkformer_cifar10_train_20260604_171317`
- Best validation accuracy: 96.08% Acc@1 at epoch 384.
- Formal E0 result: `results/qkformer_lut_e0_diag_20260605_110814`
- Checkpoint loaded: true
- Data source: real CIFAR-10
- Overall address coverage: 0.5992965698242188
- Singleton fraction: 0.08653305969439885
- Conditional variance: 0.07126443506367477

Interpretation: trained Q/K/gate spikes are active and address occupancy is
useful, but conditional response variance is still high. Current phase judgment
is `CONDITIONAL GO`; next step is E1 split-aware reconstruction, not a full
LUT wrapper.

Latest E1 split-aware reconstruction:

- Result: `results/qkformer_lut_e1_recon_20260605_133910`
- Calibration: real CIFAR-10 train, 32 batches.
- Evaluation: real CIFAR-10 validation, 16 batches.
- Checkpoint loaded: true.
- Overall global mean MSE: 0.0714635615743191.
- Overall address LUT MSE: 0.06887314827955697.
- Overall address LUT relative MSE reduction: 3.582838808068491%.
- Candidate/background mean relative MSE reduction: 0.033154317038117744%.
- Stage relative reductions: stage1 6.147753618879997%, stage2
  2.554365246580402%, stage3 2.814618183406784%.

Interpretation: address LUT reconstruction beats global and
candidate/background baselines, but gains are modest. Current judgment is
`CONDITIONAL GO` to E2 stage-wise replacement diagnostics, not a full wrapper.

E2 stage-wise replacement diagnostic is implemented:

- Replaces selected attention modules' `proj_lif` output with address-LUT
  prototype predictions through hooks.
- Default targets: `stage1.0.tssa` and `stage2.0.tssa`.
- Reports baseline vs replacement loss/Acc@1/Acc@5, logit MSE/KL, and local
  replacement MSE.
- This is still diagnostic replacement, not an optimized LUT wrapper.

Latest E2 stage1+stage2 replacement:

- Result: `results/qkformer_lut_e2_replace_20260605_150017`
- Calibration: real CIFAR-10 train, 32 batches.
- Evaluation: real CIFAR-10 validation, 16 batches.
- Targets: `stage1.0.tssa`, `stage2.0.tssa`.
- Baseline Acc@1/Acc@5: 95.1171875% / 100.0%.
- Replacement Acc@1/Acc@5: 96.09375% / 99.609375%.
- Delta Acc@1/Acc@5: +0.9765625% / -0.390625%.
- Replacement loss improved from 0.37332091107964516 to 0.3554967865347862.

Interpretation: E2 hook replacement ran successfully and did not collapse
accuracy, but evaluation used only 512 validation images. Next step is E2
robustness sweep on the full validation split across stage1-only, stage2-only,
and stage1+stage2 targets.

Latest E2 full-validation target sweep:

- Stage1-only: Acc@1 95.73 -> 95.88, delta +0.15; loss 0.35815 -> 0.35401.
- Stage2-only: Acc@1 95.73 -> 95.59, delta -0.14; loss 0.35815 -> 0.36335.
- Stage1+stage2: Acc@1 95.73 -> 95.54, delta -0.19; loss 0.35815 -> 0.36018.

Interpretation: only stage1-only is slightly positive on full validation.
Stage2 and combined replacement are negative. Next step is calibration-size
sweep for stage1-only before any wrapper/prototype design.

Latest E2 stage1-only calibration-size sweep:

- 32 calibration batches: Acc@1 95.73 -> 95.88, delta +0.15.
- 128 calibration batches: Acc@1 95.73 -> 95.76, delta +0.03.
- 512 calibration batches: Acc@1 95.73 -> 95.81, delta +0.08.
- Full train calibration: Acc@1 95.73 -> 96.02, delta +0.29.

Interpretation: stage1-only remains positive, and full calibration gives the
largest small gain. Next step is a stage1-only blend sweep with full
calibration/full validation to test whether partial replacement reduces logit
drift while preserving or improving Acc@1.

## Server Commands

Set up data link and run E0 diagnostic:

```bash
cd ~/mac_agent/sdr-lutattn-qkformer-lut && git pull && bash scripts/server/link_cifar10_data.sh && bash scripts/server/run_qkformer_lut_e0_diag.sh
```

Train CIFAR-10 checkpoint:

```bash
cd ~/mac_agent/sdr-lutattn-qkformer-lut && git pull && bash scripts/server/run_qkformer_cifar10_train.sh
```

Specify a server GPU when running train/E0/E1/E2:

```bash
cd ~/mac_agent/sdr-lutattn-qkformer-lut && bash scripts/server/run_qkformer_lut_e1_recon.sh --gpu 2
```

Equivalent environment form:

```bash
cd ~/mac_agent/sdr-lutattn-qkformer-lut && QKFORMER_LUT_GPU=2 bash scripts/server/run_qkformer_lut_e1_recon.sh
```

After training, inspect latest outputs:

```bash
cd ~/mac_agent/sdr-lutattn-qkformer-lut
tail -f $(ls -td results/qkformer_cifar10_train_* | head -1)/train_log.txt
find $(ls -td results/qkformer_cifar10_train_* | head -1) -name "*.pth" -o -name "*.pth.tar"
```

Run E0 with a trained checkpoint:

```bash
cd ~/mac_agent/sdr-lutattn-qkformer-lut && QKFORMER_LUT_CKPT=/path/to/checkpoint.pth.tar bash scripts/server/run_qkformer_lut_e0_diag.sh
```

Run E0 against the best checkpoint from the latest training result:

```bash
cd ~/mac_agent/sdr-lutattn-qkformer-lut && bash scripts/server/run_qkformer_lut_e0_after_latest_train.sh
```

Run E1 split-aware reconstruction against the latest trained checkpoint:

```bash
cd ~/mac_agent/sdr-lutattn-qkformer-lut && bash scripts/server/run_qkformer_lut_e1_recon.sh
```

Run E2 stage-wise replacement diagnostic:

```bash
cd ~/mac_agent/sdr-lutattn-qkformer-lut && bash scripts/server/run_qkformer_lut_e2_replace.sh --gpu 2
```

Run E2 full-validation target sweep:

```bash
cd ~/mac_agent/sdr-lutattn-qkformer-lut && bash scripts/server/run_qkformer_lut_e2_sweep.sh --gpu 2
```

Run E2 calibration-size sweep for stage1-only:

```bash
cd ~/mac_agent/sdr-lutattn-qkformer-lut && bash scripts/server/run_qkformer_lut_e2_calib_sweep.sh --gpu 2
```

Run E2 stage1-only blend sweep:

```bash
cd ~/mac_agent/sdr-lutattn-qkformer-lut && bash scripts/server/run_qkformer_lut_e2_blend_sweep.sh --gpu 2
```

## Phase Gate

Use `GO`, `CONDITIONAL GO`, or `NO-GO` only after real-data E0 with a trained
checkpoint.

- `GO`: bucket occupancy is materially better than CCS and conditional variance
  is below the CCS coarse context baseline.
- `CONDITIONAL GO`: addresses are stable and better occupied, but response
  variance remains high; next test should be background correction or stage-wise
  LUT.
- `NO-GO`: binary Q/K addresses remain sparse or do not explain response; stop
  LUT-ization and switch to standard QKFormer training or a hybrid ANN-SNN
  baseline.

## Hygiene

- Keep generated outputs under `results/` on server.
- Do not commit checkpoints, datasets, logs, tarballs, or private paths.
- Keep user/server credentials, tokens, private keys, cookies, and raw private
  conversation out of docs and commits.
- Prefer zero-argument server scripts with dependency checks, tee logs, and
  timestamped result directories.
