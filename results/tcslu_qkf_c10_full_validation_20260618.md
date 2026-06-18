# QKFormer CIFAR-10 Current-Unit LUT Full-Validation Results

Date: 2026-06-18

## Protocol

- Backbone: QKFormer CIFAR-10, seed 42 checkpoints.
- Target: `stage1.0.tssa` current-unit replacement.
- Calibration: 128 train batches.
- Evaluation: full CIFAR-10 validation.
- Support threshold: minimum support 2.
- Replacement unit: Q/K spike-address LUT at the local current before the
  original BN/LIF path.

## Result Files

- T=1: `results/tcslu_qkf_c10_t1_full_20260618_161423/summary.csv`
- T=4: `results/tcslu_qkf_c10_t4_full_20260618_161423_t4rerun/summary.csv`

## Main Rows

| Setting | Clean Top-1 | Static Hard | Static Drop | Moment/TCSLU Hard | Moment/TCSLU Drop | Channel Hard | Temporal Hard |
|---|---:|---:|---:|---:|---:|---:|---:|
| CIFAR-10 T=1 | 94.64 | 93.64 | 1.00 | 94.57 | 0.07 | 94.57 | 89.30 |
| CIFAR-10 T=4 | 95.70 | 95.08 | 0.62 | 95.81 | -0.11 | 95.71 | 92.23 |

## Interpretation

These full-validation QKFormer rows strengthen the replacement-fidelity
mainline:

- QKFormer CIFAR-10 T=1 is near-lossless after moment/channel calibration:
  94.64 -> 94.57 Acc@1, a 0.07 pp drop.
- QKFormer CIFAR-10 T=4 is also preserved by moment/TCSLU calibration:
  95.70 -> 95.81 Acc@1. Treat the +0.11 pp as within normal evaluation noise,
  not as an accuracy-gain claim.
- Static hard replacement is already much less damaging on CIFAR-10 than on
  CIFAR-100 T=4, but calibration still improves fidelity.
- Temporal-only matching remains unstable in both settings and should not be
  used as a standalone method row.

Claim boundary: this is a single-layer current replacement diagnostic, not a
complete QKFormer replacement, measured hardware speedup, or cross-architecture
generalization result.
