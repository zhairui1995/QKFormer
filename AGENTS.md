# QK-LUTFormer Codex Project Rules

This repository is the independent **QK-LUTFormer / QKFormer-LUT hybrid**
branch for SDR-LUTAttn.

## Required Context

Before important work, read:

1. the user's global Codex rules file, when available
2. `docs/QK_LUTFORMER_QUICK_STATE.md`
3. `docs/QKFORMER_LUT_BRANCH_STATUS.md`
4. `docs/CODEX_HANDOFF_QKFORMER_LUT.md`

For low-token continuation or side conversations, start with
`docs/QK_LUTFORMER_QUICK_STATE.md` and
`python3 scripts/local/analyze_qk_lut_results.py --brief`. Open the full
handoff/status docs only when implementation details, claim boundaries, or
historical evidence are needed.

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
- E3 runner: `tools/qkformer_lut_e3_trainable_lut.py`
- Config: `configs/qkformer_lut_e0_diag.yaml`
- E1 config: `configs/qkformer_lut_e1_recon.yaml`
- E2 config: `configs/qkformer_lut_e2_replace.yaml`
- E3 config: `configs/qkformer_lut_e3_trainable_lut.yaml`
- Server scripts:
  - `scripts/server/install_qkformer_lut_deps.sh`
  - `scripts/server/link_cifar10_data.sh`
  - `scripts/server/qkformer_lut_common.sh`
  - `scripts/server/run_qkformer_lut_e0_diag.sh`
  - `scripts/server/run_qkformer_cifar10_train.sh`
  - `scripts/server/run_qkformer_cifar10_t1_train.sh`
  - `scripts/server/run_qkformer_lut_e0_after_latest_train.sh`
  - `scripts/server/run_qkformer_lut_e1_recon.sh`
  - `scripts/server/run_qkformer_lut_e2_replace.sh`
  - `scripts/server/run_qkformer_lut_e2_sweep.sh`
  - `scripts/server/run_qkformer_lut_e2_calib_sweep.sh`
  - `scripts/server/run_qkformer_lut_e2_blend_sweep.sh`
  - `scripts/server/run_qkformer_lut_e2_random_calib_sweep.sh`
  - `scripts/server/run_qkformer_lut_e2_control_sweep.sh`
  - `scripts/server/run_qkformer_lut_e3_trainable_lut.sh`
  - `scripts/server/run_qkformer_lut_e3_adapter_sweep.sh`
  - `scripts/server/run_qkformer_lut_e3_conservative_sweep.sh`
  - `scripts/server/run_qkformer_lut_t1_e0_after_latest_train.sh`
  - `scripts/server/run_qkformer_lut_t1_e3_conservative_sweep.sh`
  - `scripts/server/run_qkformer_lut_t1_e3_seed_sweep.sh`
  - `scripts/server/package_qkformer_lut_t1_e3_seed_sweep.sh`
- Local automation:
  - `scripts/local/run_remote_qk_lut_loop.sh`
  - `scripts/local/analyze_qk_lut_results.py`
  - `scripts/local/qk_lut_quick_state.sh`
  - `docs/QK_LUTFORMER_QUICK_STATE.md`
  - `docs/REMOTE_EXPERIMENT_LOOP.md`

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
- Replacement modes include `address_lut`, `global_mean`, and
  `shuffled_address_lut`.

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
largest small gain.

Latest E2 stage1-only blend sweep with full train calibration/full validation:

- Blend 0.0: Acc@1 95.73 -> 95.73, delta +0.00; loss unchanged.
- Blend 0.25: Acc@1 95.73 -> 96.00, delta +0.27; loss 0.35815 -> 0.35323;
  logit MSE 0.04099666884899139, KL 0.01876032644510269.
- Blend 0.5: Acc@1 95.73 -> 95.90, delta +0.17; loss 0.35815 -> 0.35499.
- Blend 0.75: Acc@1 95.73 -> 95.85, delta +0.12; loss 0.35815 -> 0.35629.
- Blend 1.0: Acc@1 95.73 -> 96.02, delta +0.29; loss 0.35815 -> 0.35419;
  logit MSE 0.053548186321258545, KL 0.023518988977372646.

Interpretation: full replacement has the highest Acc@1, while blend 0.25 has
the best loss, preserves Acc@5, and has lower logit drift. The effect is still
small. Next step is a randomized calibration-subset stability sweep for
stage1-only, comparing blend 0.25 and 1.0 before any wrapper/prototype design.

Latest E2 stage1-only randomized calibration stability sweep:

- 128 calibration batches, blend 0.25, seeds 42/43/44: mean Acc@1
  95.9333%, mean delta +0.2033%, min/max delta +0.13/+0.24, mean loss
  0.355467, mean logit MSE 0.040937.
- 128 calibration batches, blend 1.0, seeds 42/43/44: mean Acc@1 95.7400%,
  mean delta +0.0100%, min/max delta -0.09/+0.09, mean loss 0.355344,
  mean logit MSE 0.054565.
- 512 calibration batches, blend 0.25, seeds 42/43/44: mean Acc@1
  95.8600%, mean delta +0.1300%, min/max delta +0.06/+0.19, mean loss
  0.356766, mean logit MSE 0.041317.
- 512 calibration batches, blend 1.0, seeds 42/43/44: mean Acc@1 95.8367%,
  mean delta +0.1067%, min/max delta +0.02/+0.19, mean loss 0.355430,
  mean logit MSE 0.054583.

Interpretation: blend 0.25 is more stable than full replacement under random
calibration subsets, especially at 128 batches. Next step is an E2 control
sweep comparing `address_lut` against `global_mean` smoothing at the same
stage1-only, 128-batch, blend-0.25, random-seed setting.

Latest E2 stage1-only address-vs-global control sweep:

- Address LUT, 128 calibration batches, blend 0.25, seeds 42/43/44:
  mean Acc@1 95.9333%, mean delta +0.2033%, mean loss 0.355467,
  mean logit MSE 0.040937.
- Global mean, same setting: mean Acc@1 95.9233%, mean delta +0.1933%,
  mean loss 0.354103, mean logit MSE 0.041799.

Interpretation: global-mean smoothing nearly matches address LUT and has
better loss. Current accuracy signal is not address-specific enough to justify
wrapper design. Next step is a shuffled-address LUT control that preserves the
prototype distribution while breaking address/prototype alignment.

E3 trainable residual LUT adapter is implemented for the paper-oriented route:

- Freezes the trained QKFormer backbone.
- Calibrates prototype/shrinkage initialization from Q/K addresses.
- Trains only tiny LUT adapter tables plus optional learnable blend alpha.
- Uses CE + KL-to-baseline + local MSE loss.
- Default target is `stage1.0.tssa`.
- Default control modes are `address_lut`, `global_mean`, and
  `token_channel_lut`.

Interpretation: E2 showed that forward-only address prototypes are not enough.
E3 tests the stronger paper claim: whether a trainable residual address LUT
adapter exploits Q/K address semantics better than non-address or weaker-address
controls under the same frozen-backbone training budget.

Latest E3 trainable residual LUT adapter pilot:

- Address LUT: 2049 trainable parameters, alpha 0.5625, Acc@1 95.67 -> 95.51,
  delta -0.16.
- Global mean: 2 trainable parameters, alpha 0.3541, Acc@1 94.97 -> 94.12,
  delta -0.85.
- Token-channel LUT: 513 trainable parameters, alpha 0.5374, Acc@1
  95.76 -> 95.43, delta -0.33.

Interpretation: E3 runs end-to-end and address LUT is least damaging, but the
first pilot overfits/drifts because learnable alpha grows too high. Next step
is conservative E3: fixed alpha 0.1, 2 epochs, lower LR, stronger KL and local
MSE constraints.

Latest E3 conservative trainable residual LUT adapter sweep:

- Address LUT: 2048 trainable parameters, fixed alpha 0.1, Acc@1
  95.72 -> 95.82, delta +0.10; loss 0.250749 -> 0.250908.
- Global mean: 1 trainable parameter, fixed alpha 0.1, Acc@1 95.74 -> 95.91,
  delta +0.17; loss 0.255298 -> 0.256633.
- Token-channel LUT: 512 trainable parameters, fixed alpha 0.1, Acc@1
  95.80 -> 95.88, delta +0.08; loss 0.253012 -> 0.252757.

Interpretation: conservative E3 is stable and all adapters stay above 95% with
small positive Acc@1 deltas, but address LUT still does not beat the global
control in Acc@1. Address LUT has cleaner loss/drift than global mean. Next
step is a T=1 CIFAR-10 stress test to evaluate LUT address ability under a
single-step setting instead of relying only on T=4 temporal averaging.

Latest T=1 CIFAR-10 stress test:

- Train result: `results/qkformer_cifar10_train_20260606_001151`
- Best validation accuracy: 95.20% Acc@1 at epoch 407.
- Final epoch 409 Acc@1: 94.64%.
- T=1 E0 result: `results/qkformer_lut_e0_diag_20260606_093127`
- Checkpoint loaded: true.
- Overall address coverage: 0.5804824829101562.
- Singleton fraction: 0.07936228803294151.
- Conditional variance: 0.05487808446146928.
- Stage1/stage2/stage3 conditional variance: 0.055049262856841516 /
  0.02716715404410884 / 0.06864796047246338.
- T=1 conservative E3 address LUT: Acc@1 94.88 -> 95.00, delta +0.12; loss
  0.29417879979610445 -> 0.2886085723400116.
- T=1 conservative E3 global mean: Acc@1 94.79 -> 94.72, delta -0.07.
- T=1 conservative E3 token-channel LUT: Acc@1 94.91 -> 94.85, delta -0.06.

Interpretation: T=1 lowers conditional variance versus T=4 while keeping useful
address coverage, and address LUT is the only positive conservative E3 adapter
in the first T=1 run. This is a promising address-specific signal, but it is
single-seed evidence. Next step is a T=1 E3 seed sweep across
`address_lut`, `global_mean`, and `token_channel_lut` before making a paper
claim.

Latest T=1 E3 multi-seed control sweep:

- Result dirs: `results/qkformer_lut_e3_trainable_lut_20260606_132549` through
  `results/qkformer_lut_e3_trainable_lut_20260606_134407`.
- Address LUT, seeds 42/43/44: deltas +0.12 / +0.07 / -0.01; mean +0.06.
- Global mean, seeds 42/43/44: deltas -0.07 / -0.17 / -0.04; mean -0.0933.
- Token-channel LUT, seeds 42/43/44: deltas -0.06 / +0.06 / -0.07; mean
  -0.0233.
- Mean loss delta: address LUT -0.002594, global mean +0.000182,
  token-channel LUT -0.001102.

Interpretation: T=1 address LUT now beats both controls on mean Acc@1 delta and
loss delta under the same conservative adapter budget. Current judgment is
`CONDITIONAL GO`: enough to support a paper-oriented address-specific LUT
ablation, but still not enough for energy/latency or broad dataset claims.
Next step should either repeat on another checkpoint/seed or broaden to a
larger dataset/stage stress test.

## Server Commands

Set up data link and run E0 diagnostic:

```bash
cd ~/mac_agent/sdr-lutattn-qkformer-lut && git pull && bash scripts/server/link_cifar10_data.sh && bash scripts/server/run_qkformer_lut_e0_diag.sh
```

Train CIFAR-10 checkpoint:

```bash
cd ~/mac_agent/sdr-lutattn-qkformer-lut && git pull && bash scripts/server/run_qkformer_cifar10_train.sh
```

Train CIFAR-10 T=1 checkpoint:

```bash
cd ~/mac_agent/sdr-lutattn-qkformer-lut && git pull && bash scripts/server/run_qkformer_cifar10_t1_train.sh --gpu 2
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

Run E2 randomized calibration stability sweep:

```bash
cd ~/mac_agent/sdr-lutattn-qkformer-lut && bash scripts/server/run_qkformer_lut_e2_random_calib_sweep.sh --gpu 2
```

Run E2 address-vs-global control sweep:

```bash
cd ~/mac_agent/sdr-lutattn-qkformer-lut && bash scripts/server/run_qkformer_lut_e2_control_sweep.sh --gpu 2
```

Run E3 trainable LUT adapter pilot sweep:

```bash
cd ~/mac_agent/sdr-lutattn-qkformer-lut && bash scripts/server/run_qkformer_lut_e3_adapter_sweep.sh --gpu 2
```

Run E3 conservative trainable LUT adapter sweep:

```bash
cd ~/mac_agent/sdr-lutattn-qkformer-lut && bash scripts/server/run_qkformer_lut_e3_conservative_sweep.sh --gpu 2
```

Run T=1 E0 and E3 conservative sweeps after T=1 training:

```bash
cd ~/mac_agent/sdr-lutattn-qkformer-lut && bash scripts/server/run_qkformer_lut_t1_e0_after_latest_train.sh --gpu 2
cd ~/mac_agent/sdr-lutattn-qkformer-lut && bash scripts/server/run_qkformer_lut_t1_e3_conservative_sweep.sh --gpu 2
```

Run the T=1 E3 multi-seed control sweep and package artifacts:

```bash
cd ~/mac_agent/sdr-lutattn-qkformer-lut && git pull && bash scripts/server/run_qkformer_lut_t1_e3_seed_sweep.sh --gpu 2
cd ~/mac_agent/sdr-lutattn-qkformer-lut && bash scripts/server/package_qkformer_lut_t1_e3_seed_sweep.sh
```

Run the same loop from local Codex through SSH/GitHub/server/download/analyze:

```bash
cd /Users/cvue/Documents/github_zr/sdr-lutattn-qkformer-lut && bash scripts/local/run_remote_qk_lut_loop.sh
```

Remote automation defaults: server `lbz@192.168.70.60`, remote repo
`~/mac_agent/sdr-lutattn-qkformer-lut`, conda env `sdr`, branch
`codex/qkformer-lut-hybrid`, small diagnostics on GPU 2. For expensive training
jobs, use `--kind train`; it checks `nvidia-smi` for an idle GPU before falling
back to GPU 2. Never store server passwords in scripts, docs, or commits; the
loop requires SSH key authentication.

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
