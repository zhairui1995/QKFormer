# QKFormer CIFAR-10 Current-Unit LUT Shuffled-Address Control

Date: 2026-06-18

## Protocol

- Backbone: QKFormer CIFAR-10, seed 42 checkpoints.
- Target: `stage1.0.tssa` current-unit replacement.
- Calibration: 128 train batches.
- Evaluation: full CIFAR-10 validation.
- Support threshold: minimum support 2.
- Control: `shuffled_address`, preserving the LUT value distribution while
  breaking the Q/K address-to-prototype alignment.

## Result Files

- T=1 aligned: `results/tcslu_qkf_c10_t1_full_20260618_161423/summary.csv`
- T=4 aligned: `results/tcslu_qkf_c10_t4_full_20260618_161423_t4rerun/summary.csv`
- T=1 shuffled: `results/tcslu_qkf_c10_t1_shuffled_20260618_163112/summary.csv`
- T=4 shuffled: `results/tcslu_qkf_c10_t4_shuffled_20260618_163112/summary.csv`

## Key Comparison

| Setting | Clean Top-1 | Row | Aligned Top-1 | Aligned Drop | Shuffled Top-1 | Shuffled Drop | Shuffled Penalty vs Aligned |
|---|---:|---|---:|---:|---:|---:|---:|
| CIFAR-10 T=1 | 94.64 | static hard | 93.64 | 1.00 | 91.48 | 3.16 | 2.16 |
| CIFAR-10 T=1 | 94.64 | TCSLU/moment hard | 94.57 | 0.07 | 94.37 | 0.27 | 0.20 |
| CIFAR-10 T=4 | 95.70 | static hard | 95.08 | 0.62 | 90.57 | 5.13 | 4.51 |
| CIFAR-10 T=4 | 95.70 | TCSLU/moment hard | 95.81 | -0.11 | 95.31 | 0.39 | 0.50 |
| CIFAR-10 T=4 | 95.70 | channel hard | 95.71 | -0.01 | 95.28 | 0.42 | 0.43 |

## Interpretation

The CIFAR-10 shuffled-address control supports the semantic-address claim for
current replacement:

- Static hard replacement is much worse when the Q/K address-to-prototype
  alignment is broken: +2.16 pp extra drop for T=1 and +4.51 pp extra drop for
  T=4.
- Moment/TCSLU calibration reduces the mismatch, but aligned lookup remains
  better: +0.20 pp for T=1 and +0.50 pp for T=4.
- T=4 gives the clearest address-alignment evidence because both static and
  calibrated shuffled controls are worse than the aligned counterparts.

Claim unlocked: for QKFormer CIFAR-10 current-unit LUT replacement, preserving
the Q/K spike-address alignment improves end-to-end replacement fidelity beyond
distribution-matched prototype smoothing.

Claim boundary: this remains a single-layer current replacement diagnostic; it
does not establish complete QKFormer replacement, measured hardware speedup, or
cross-architecture generalization.
