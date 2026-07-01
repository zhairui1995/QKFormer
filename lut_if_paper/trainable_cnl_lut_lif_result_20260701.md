# Trainable CNL-LUT-LIF Probe Result

Date: 2026-07-01

Status: `NO-GO` for the trainable dense CNL-LUT-LIF module under the fixed
QKFormer CIFAR-100 `T=4` seed-42 protocol.

## Protocol

- Host: server-lbz, single RTX 4090 GPU.
- Code snapshot: public fork branch `codex/qklut-lif-rl-architecture`, runner
  commit `db7e78a`.
- Dataset/model: QKFormer CIFAR-100 `T=4`, seed 42.
- Checkpoint: retained QKFormer CIFAR-100 `T=4` seed-42 checkpoint.
- Calibration: 32 training batches.
- Training: 3 epochs, 128 training batches per epoch, frozen backbone.
- Trainable parameters: dense LUT transition tables, per-module thresholds,
  and temporal/channel CNL affine correction.
- Final evaluation: full CIFAR-100 validation set.

## Result

| Method | Acc@1 | Drop | Logit MSE/sample | Gate |
|---|---:|---:|---:|---|
| posthoc dense transition LUT | 78.96 | 2.06 | 24.577839 | control |
| quantized arithmetic LIF | 78.90 | 2.12 | 24.559828 | control |
| fixed CNL-LUT-LIF | 78.98 | 2.04 | 24.570013 | best fixed control in this 32-calib protocol |
| trainable CNL-LUT-LIF init | 78.77 | 2.25 | 22.710421 | init |
| trainable CNL-LUT-LIF final | 78.58 | 2.44 | 82.543650 | `NO-GO` |

Gate:

- beats fixed CNL on Acc@1 or logit MSE: `False`;
- drop below fixed CNL: `False`;
- drop <= 1.25 pp: `False`;
- all requested targets executed: `True`.

Formal verdict: `NO-GO`.

## Diagnosis

The training objective appears to overfit or drift away from hard-address
evaluation. Training accuracy on the fixed 128-batch budget reaches
`99.78%--99.95%`, while full-validation accuracy remains below the fixed CNL
control and paired logit MSE worsens sharply. This is consistent with the
earlier Dense LUT-IF failure: differentiable table training can optimize the
surrogate path without preserving the final hard-address LUT behavior.

## Claim Impact

This result blocks the claim that CNL initialization plus trainable dense
transition tables is sufficient for a new positive QK-LUT-LIF architecture.
The next credible architecture route must avoid dense per-entry table drift,
for example by training a smaller structured residual, training the whole
network with the module inserted from the start, or moving to a backbone where
the module is part of the architecture rather than a post-hoc all-LIF
replacement.
