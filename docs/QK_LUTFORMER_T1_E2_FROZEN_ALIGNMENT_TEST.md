# QK-LUTFormer T=1 Frozen Prototype Alignment Test

## Purpose

This experiment tests the paper-critical question: whether Q/K binary spike addresses are a meaningful LUT indexing signal, rather than merely a source of trainable residual parameters or generic smoothing.

## Setting

- Backbone: T=1 CIFAR-10 QKFormer best checkpoint from `qkformer_cifar10_train_20260606_001151`.
- Target module: `stage1.0.tssa`.
- LUT construction: frozen calibration prototypes; no LUT table training.
- Calibration: 128 CIFAR-10 train batches.
- Evaluation: full CIFAR-10 validation split.
- Replacement strength: `blend=1.0`.
- Controls: `address_lut`, `shuffled_address_lut`, `global_mean`.
- Artifact policy: only `metrics.json` and `train_log.txt` were downloaded; no checkpoint weights were transferred.

## Results

| Mode | Baseline Acc@1 | Replacement Acc@1 | Delta Acc@1 | Delta Loss | KL to Baseline | Logit MSE | Local MSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| `address_lut` | 94.9000 | 94.9400 | 0.0400 | -0.002451 | 0.038093 | 0.083226 | 0.055959 |
| `shuffled_address_lut` | 94.9000 | 94.8400 | -0.0600 | 0.000452 | 0.039542 | 0.088816 | 0.063405 |
| `global_mean` | 94.9000 | 94.7900 | -0.1100 | 0.006395 | 0.039620 | 0.087185 | 0.059798 |

## Interpretation

- `address_lut` improves Acc@1 by 0.0400, while `shuffled_address_lut` changes Acc@1 by -0.0600 and `global_mean` by -0.1100.
- `address_lut` also has lower loss drift (-0.002451) than shuffled (0.000452) and global (0.006395).
- Behavior preservation is better for `address_lut`: KL 0.038093 and logit MSE 0.083226, compared with shuffled KL 0.039542 / logit MSE 0.088816.
- This is the cleanest current evidence for the paper method: when LUT prototypes are frozen and cannot retrain around a broken address map, correct Q/K address alignment beats shuffled alignment.

## Claim Boundary

- Supported: Q/K binary spike addresses provide a useful frozen LUT indexing signal for stage1 module-output replacement under this T=1 CIFAR-10 setting.
- Not supported yet: energy gain, latency gain, ImageNet-scale gain, or a production LUT wrapper.
- Caveat: full-train and 512-batch calibration were attempted but the current hook aggregation path was too slow for this turn; 128-batch calibration is the completed evidence point.

## Evidence

- Metrics: `results/qkformer_lut_e2_alignment_test_20260606_182021/metrics.json`.
- Remote result directory: `results/qkformer_lut_e2_alignment_test_20260606_182021`.