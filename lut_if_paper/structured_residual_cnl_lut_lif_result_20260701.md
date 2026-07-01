# Structured Residual CNL-LUT-LIF Probe Result

Date: 2026-07-01

Status: `NO-GO` for the structured residual CNL-LUT-LIF module under the fixed
QKFormer CIFAR-100 `T=4` seed-42 protocol.

## Protocol

- Host: remote GPU server, single RTX 4090 GPU.
- Code snapshot: public fork branch `codex/qklut-lif-rl-architecture`, runner
  commit `0f5c9a3`.
- Dataset/model: QKFormer CIFAR-100 `T=4`, seed 42.
- Checkpoint: retained QKFormer CIFAR-100 `T=4` seed-42 checkpoint.
- Calibration: 32 training batches for range and current-moment collection.
- Training: 3 epochs, 128 training batches per epoch, frozen backbone.
- Trainable parameters: per-module threshold plus low-degree state-vector,
  input-vector, and global residual over a fixed CNL-LUT-LIF transition table.
- Final evaluation: complete 10,000-image CIFAR-100 validation set.
- Local artifacts:
  `results/structured_residual_cnl_lut_lif_cifar100_t4_seed42_full_20260701_124750/`.

## Result

| Method | Acc@1 | Drop | Logit MSE/sample | Gate |
|---|---:|---:|---:|---|
| posthoc dense transition LUT | 78.96 | 2.06 | 24.577839 | control |
| quantized arithmetic LIF | 78.90 | 2.12 | 24.559828 | control |
| fixed CNL-LUT-LIF | 78.98 | 2.04 | 24.570013 | best fixed control |
| structured residual CNL-LUT-LIF init | 78.77 | 2.25 | 22.710421 | init |
| structured residual CNL-LUT-LIF final | 78.86 | 2.16 | 89.575663 | `NO-GO` |

Gate:

- beats fixed CNL on Acc@1 or logit MSE: `False`;
- drop below fixed CNL: `False`;
- drop <= 1.25 pp: `False`;
- all requested targets executed: `True`.

Formal verdict: `NO-GO`.

## Diagnosis

The low-degree residual reduces the number of trainable degrees of freedom
relative to the dense CNL-LUT-LIF table, but it does not rescue the QKFormer
post-hoc all-LIF replacement setting. The initialization has lower logit MSE
than fixed CNL but lower accuracy. After training, full-validation accuracy
improves from the initialization but remains below fixed CNL, while logit MSE
becomes much worse. Training accuracy reaches `99.71%`, `99.85%`, and `100.00%`
across the three fixed-budget epochs, so the run still shows train-subset
overfit/drift under hard-address evaluation rather than a robust architecture
gain.

## Claim Impact

This result blocks the claim that the current structured residual
CNL-LUT-LIF module improves QKFormer CIFAR-100 `T=4` under the registered
post-hoc frozen-backbone protocol. Together with the RL selector and dense
trainable CNL `NO-GO` results, it means the third-paper mainline should not
claim that a simple post-hoc trainable LUT-LIF insertion already forms a
positive new QKFormer architecture.

The next defensible routes are narrower:

1. from-start or joint training with QK-LUT-LIF inserted as a native module;
2. an explicitly labeled cross-architecture boundary test on Spikformer or an
   LL-ViT-like implementation after locating a runnable, protocol-matched
   codebase;
3. repositioning the third paper as a negative boundary/diagnostic study
   unless new preregistered evidence changes the architecture story.
