# QKFormer CIFAR-10 T=4 Semantic Address Controls

Date: 2026-06-18

## Protocol

- Backbone: QKFormer CIFAR-10 T=4, seed 42 checkpoint.
- Target: `stage1.0.tssa` current-unit replacement.
- Calibration: 128 train batches.
- Evaluation: full CIFAR-10 validation.
- Support threshold: minimum support 2.
- Fixed comparison: same checkpoint, target, calibration budget, and alpha 1.0.

## Result Files

- Aligned: `results/tcslu_qkf_c10_t4_full_20260618_161423_t4rerun/summary.csv`
- Shuffled: `results/tcslu_qkf_c10_t4_shuffled_20260618_163112/summary.csv`
- Token/channel: `results/tcslu_qkf_c10_t4_token_channel_mean_20260618_164404/summary.csv`
- Global: `results/tcslu_qkf_c10_t4_global_mean_20260618_164404/summary.csv`
- Random table: `results/tcslu_qkf_c10_t4_random_table_20260618_165105/summary.csv`

## Table C Candidate

| Control | Static hard Top-1 | Static drop | TCSLU/moment Top-1 | TCSLU/moment drop |
|---|---:|---:|---:|---:|
| aligned Q/K address | 95.08 | 0.62 | 95.81 | -0.11 |
| global mean | 91.41 | 4.29 | 95.63 | 0.07 |
| same-size random table | 91.87 | 3.83 | 95.52 | 0.18 |
| token/channel mean | 94.86 | 0.84 | 95.45 | 0.25 |
| shuffled address | 90.57 | 5.13 | 95.31 | 0.39 |

## Interpretation

The semantic-address control passes for QKFormer CIFAR-10 T=4:

- Aligned Q/K address has the best TCSLU/moment fidelity: 95.81 Acc@1.
- Every control is below aligned under TCSLU/moment calibration:
  global 95.63, random 95.52, token/channel 95.45, shuffled 95.31.
- Static hard replacement shows a larger separation: aligned 95.08 versus
  token/channel 94.86, random 91.87, global 91.41, and shuffled 90.57.

Claim unlocked: the current-unit replacement benefit is not explained by
generic smoothing, coarse token/channel state, random same-size capacity, or
prototype-distribution preservation alone. Correct Q/K spike-address alignment
is the strongest control under the fixed protocol.

Claim boundary: this remains a single-layer current replacement diagnostic on
QKFormer CIFAR-10 T=4. It does not establish full model replacement, measured
hardware speedup, or cross-architecture generalization.
