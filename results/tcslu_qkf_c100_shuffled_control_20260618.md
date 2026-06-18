# QKFormer CIFAR-100 Current-Unit LUT Shuffled-Address Control

Date: 2026-06-18

## Protocol

- Backbone: QKFormer CIFAR-100, seed 42 checkpoints.
- Targets: `stage1.0.tssa` current-unit replacement.
- Calibration: 128 train batches.
- Evaluation: full CIFAR-100 validation.
- LUT modes:
  - `aligned_lut`: normal Q/K spike-address to prototype mapping.
  - `shuffled_address`: preserves prototype/value distribution but breaks the address-to-prototype alignment.
- Hard replacement rows use alpha 1.0.

## Result Files

- T=1 aligned: `results/tcslu_qkf_c100_t1_full_20260618_104451/summary.csv`
- T=4 aligned: `results/tcslu_qkf_c100_t4_full_20260618_104512/summary.csv`
- T=1 shuffled: `results/tcslu_qkf_c100_t1_shuffled_20260618_160718/summary.csv`
- T=4 shuffled: `results/tcslu_qkf_c100_t4_shuffled_20260618_160718/summary.csv`

## Key Comparison

| Setting | Clean Top-1 | Row | Aligned Top-1 | Aligned Drop | Shuffled Top-1 | Shuffled Drop | Shuffled Penalty vs Aligned |
|---|---:|---|---:|---:|---:|---:|---:|
| CIFAR-100 T=1 | 77.78 | static hard | 75.55 | 2.23 | 72.79 | 4.99 | 2.76 |
| CIFAR-100 T=1 | 77.78 | TCSLU/moment hard | 77.26 | 0.52 | 75.95 | 1.83 | 1.31 |
| CIFAR-100 T=4 | 81.09 | static hard | 68.02 | 13.07 | 63.32 | 17.77 | 4.70 |
| CIFAR-100 T=4 | 81.09 | TCSLU/moment hard | 80.65 | 0.44 | 79.99 | 1.10 | 0.66 |
| CIFAR-100 T=4 | 81.09 | channel hard | 80.15 | 0.94 | 80.14 | 0.95 | 0.01 |

## Interpretation

The shuffled-address control is a positive semantic-address check for the
current replacement track. It keeps the LUT value distribution and replacement
budget fixed, but breaks the Q/K address alignment. Under this control:

- T=1 hard current replacement loses an additional 2.76 pp without calibration.
- T=1 TCSLU/moment calibration still helps, but remains 1.31 pp worse than the
  aligned address LUT.
- T=4 hard current replacement loses an additional 4.70 pp without calibration.
- T=4 TCSLU/moment calibration remains close to baseline, but is still 0.66 pp
  worse than aligned.
- T=4 channel-only calibration nearly erases the aligned-vs-shuffled difference,
  so the strongest address-specific statement should use static hard and
  TCSLU/moment rows, not the channel-only row.

Claim unlocked: for QKFormer CIFAR-100 current-unit LUT replacement, Q/K
spike-address alignment contributes measurable fidelity beyond distribution-
matched prototype smoothing. This supports an ablation/control table, but does
not by itself establish systems speedup, measured energy, or cross-architecture
generalization.
