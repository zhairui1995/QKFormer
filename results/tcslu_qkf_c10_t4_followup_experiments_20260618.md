# QKFormer CIFAR-10 T=4 TCSLU Follow-Up Experiments

Date: 2026-06-18

## Fixed Protocol

- Backbone: QKFormer CIFAR-10 T=4, seed 42 checkpoint.
- Default target: `stage1.0.tssa` current-unit replacement.
- Calibration: 128 train batches.
- Evaluation: full CIFAR-10 validation.
- Default support threshold: minimum support 2.
- Default mode: aligned Q/K spike-address LUT.

## Group 1: Semantic Address Controls

Result memo: `results/tcslu_qkf_c10_t4_semantic_controls_20260618.md`

| Control | Static hard Top-1 | TCSLU/moment Top-1 |
|---|---:|---:|
| aligned Q/K address | 95.08 | 95.81 |
| global mean | 91.41 | 95.63 |
| same-size random table | 91.87 | 95.52 |
| token/channel mean | 94.86 | 95.45 |
| shuffled address | 90.57 | 95.31 |

Verdict: passes. Correct Q/K spike-address alignment is the strongest control;
the result is not explained by generic smoothing, coarse token/channel means,
same-size random capacity, or prototype-distribution preservation alone.

## Group 2: Mild Corruption Robustness

| Corruption | Clean Top-1 | Static hard Top-1 | TCSLU/moment Top-1 | TCSLU drop |
|---|---:|---:|---:|---:|
| none | 95.70 | 95.08 | 95.81 | -0.11 |
| noise severity 1 | 94.27 | 92.21 | 94.26 | 0.01 |
| brightness severity 1 | 95.93 | 95.29 | 95.87 | 0.06 |

Result directories:

- `results/tcslu_qkf_c10_t4_aligned_noise_s1_20260618_165646`
- `results/tcslu_qkf_c10_t4_aligned_brightness_s1_20260618_170144`

Verdict: passes. Noise and brightness are near-lossless.

## Group 3: Multi-Layer Current Replacement

Target expansion: `stage1.0.tssa,stage2.0.tssa`.

| Target set | Clean Top-1 | Static hard Top-1 | TCSLU/moment Top-1 | Channel hard Top-1 |
|---|---:|---:|---:|---:|
| stage1 only | 95.70 | 95.08 | 95.81 | 95.71 |
| stage1+stage2 | 95.70 | 88.99 | 95.43 | 95.59 |

Result directory:

- `results/tcslu_qkf_c10_t4_aligned_stage1_stage2_20260618_170728`

Verdict: passes. Static hard replacement collapses when expanding to stage1+2,
but moment/channel/TCSLU calibration recovers the current contract to within
0.27 pp of clean, with channel hard within 0.11 pp.

## Group 4: Support Threshold Ablation

| Min support | Static hard Top-1 | TCSLU/moment Top-1 | Supported entries |
|---:|---:|---:|---:|
| 1 | 95.08 | 95.81 | 2048 |
| 2 | 95.08 | 95.81 | 2048 |
| 4 | 95.08 | 95.81 | 2048 |

Result directories:

- support 1: `results/tcslu_qkf_c10_t4_aligned_support1_20260618_171410`
- support 2: `results/tcslu_qkf_c10_t4_full_20260618_161423_t4rerun`
- support 4: `results/tcslu_qkf_c10_t4_aligned_support4_20260618_171410`

Verdict: passes. Under the 128-batch calibration budget, all 2048 scalar
entries satisfy support thresholds 1/2/4, so the result is insensitive to this
threshold range.

## Overall Interpretation

All four follow-up groups support the current QK-Spike LUT / TCSLU route:

- Q/K spike-address alignment is the strongest semantic control.
- Moment/channel calibration preserves current replacement under mild noise and
  brightness.
- The method extends from a single target to stage1+stage2 current replacement
  when calibrated.
- The main result is not brittle to min-support thresholds 1/2/4 under the
  current calibration budget.

Claim boundary: all rows are QKFormer CIFAR-10 T=4 single- or two-target
current-replacement diagnostics. They do not establish full model replacement,
ImageNet generalization, measured hardware acceleration, or cross-architecture
lossless replacement.
