# CCFA Experiment Design: QK-Contract Spikformer Native QK-LUT-LIF

Date: 2026-07-01

Mode: experiment design and cross-backbone smoke queue.

## Purpose

The primary architecture evidence is still QKFormer native training. This
Spikformer route tests a narrower transfer-substrate question: can the same
native QK-LUT-LIF module execute and train inside a QK-contract Spikformer
block?

The route is deliberately QK-contract rather than vanilla Spikformer. The
patched attention interaction uses `gate(Q) * K`, matching the second paper's
QK-LUT addressability principle more closely than original Spikformer SSA.

## Registered Experiment

Run spec:
`run_specs/native_qklut_lif_spikformer_qk_contract_c10_t4_seed42_20260701.yaml`.

Runner:
`scripts/server/run_native_qklut_lif_spikformer_cifar10.sh`.

Rows:

| Row | Attention contract | Native LUT-LIF | Purpose |
|---|---|---|---|
| `baseline_spikformer_qk_contract` | `qk_sum` | no | matched QK-contract baseline |
| `native_qklut_lif_spikformer_attention` | `qk_sum` | attention scope | transfer-substrate test |

## Gates

Smoke passes only if both rows complete, the native row selects attention LIF
targets, and summary/checkpoint manifests are written.

Pilot passes only if the native row is no worse than 1.50 pp below the matched
QK-contract baseline under the same seed, dataset, attention contract, epoch,
batch, and validation budget.

## Claim Boundary

Allowed after smoke pass:

> QK-contract Spikformer can execute the native QK-LUT-LIF module under a
> bounded CIFAR-10 smoke protocol.

Allowed after pilot pass:

> QK-contract Spikformer is a viable transfer substrate for native QK-LUT-LIF
> pilot evaluation.

Disallowed:

- broad transfer to Spikformer/SpikeFormer;
- superiority over QKFormer, Spikformer, SpikeFormer, or LL-ViT;
- near-SOTA accuracy;
- hardware speed, energy, SRAM, latency, or area.

## Execution Priority

Run only smoke while the two QKFormer native pilots are still active. Do not
launch a Spikformer pilot before interpreting the QKFormer attention/MLP pilot
results unless a new decision memo explains why the cross-backbone result is
needed immediately.
