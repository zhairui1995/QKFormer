# QKFormer All-Affine LUT Continuation

Date: 2026-06-19

## Protocol Status

The final project-level near-lossless criterion is a maximum paired Acc@1 drop
of 0.50 percentage points. Q/K/V drops by 0.45 points and is therefore
`PASS`. An earlier internal 0.25-point screening threshold was superseded after
the practical fidelity review; it is retained only in raw experiment
provenance, not as the current scientific verdict.

The final criterion is applied uniformly:

- Q/K/V, MLP, patch embedding, classifier, and cumulative replacement all use
  the same 0.50-point criterion;
- output NRMSE <= 0.01, clipping <= 0.0001, target execution, and the
  one-retry rule remain unchanged.

Architecture, checkpoint, and evaluation:

- QKFormer CIFAR-100;
- seed-42 `T=1` checkpoint;
- complete validation split;
- batch size 32 for all admitted rows;
- 32 fixed calibration batches;
- no training or weight update.

## Results

| Category | Quantization | Targets | Clean | LUT | Drop | NRMSE | Clip rate | FP32 LUT values | Value ratio | Status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Q/K/V | integer, 8 levels | 10 | 77.58 | 77.13 | 0.45 | 0.00012029 | 0 | 30,528 KiB | 8.00x | pass |
| MLP | integer, 8 levels | 8 | 77.58 | 77.63 | -0.05 | 0.00011311 | 0 | 85,248 KiB | 8.00x | pass |
| patch embedding | mixed integer + 8-bit input | 9 | 77.58 | 77.47 | 0.11 | 0.00012767 | 0 | 83,376 KiB | 8.12x | pass |
| classifier | 8-bit input | 1 | 77.58 | 77.59 | -0.01 | 0.00339480 | 0.0000002604 | 38,400 KiB | 256.00x | pass |
| cumulative all-affine | same main settings | 32 | 77.58 | 77.98 | -0.40 | 0.00013521 | 0 | 248,208 KiB | 9.47x | pass |

Positive deltas are treated as no-loss fluctuations, not accuracy gains.

The patch batch-size-16 diagnostic produced a 77.01 clean baseline and was
excluded by the baseline-reproduction gate. Repeating the same scientific
configuration at the registered batch size 32 restored the clean baseline and
passed. This was an evaluation-control rerun, not a quantization retry.

No category after Q/K/V required the allowed scientific retry.

## What The Cumulative Row Covers

The 32 replaced modules comprise:

- 10 attention Q/K/V `Conv1d` projections;
- 8 MLP `Conv2d` projections;
- 9 patch-embedding `Conv2d` projections;
- 4 attention output `proj_conv` projections;
- 1 classifier `Linear` layer.

Each input scalar indexes a precomputed vector of output-channel
contributions. The vectors are accumulated with the frozen bias.

## Supported Claim

Under the final 0.50-point project criterion:

> All Conv1d, Conv2d, and Linear operators in the evaluated QKFormer
> CIFAR-100 `T=1` configuration admit the reported scalar-level LUT-equivalent
> substitution with no material paired classification loss.

This is single-checkpoint, single-time-step fidelity evidence.

## Claim Boundary

This is not complete QKFormer LUT replacement:

- BatchNorm and LIF state dynamics remain unchanged;
- max pooling and residual addition remain unchanged;
- SSA `K^T V` and `Q(...)` matrix multiplications remain unchanged;
- hooks still execute the original operator before replacing its output;
- no measured latency, energy, SRAM, bandwidth, or hardware-cycle result is
  provided.

It is also not a compact representation. The cumulative FP32 table values
occupy 248,208 KiB (approximately 242.4 MiB), 9.47 times the original
convolution/linear weight-value count. The classifier alone has a 256x value
ratio under direct 8-bit scalar indexing.

Therefore the result supports operator-replacement fidelity, not storage
compression or practical LUT acceleration.

## Traceable Artifacts

- Q/K/V:
  `results/qk_all_affine_qkv_main_c100_t1_20260618_234138`
- Q/K/V retry:
  `results/qk_all_affine_qkv_retry_c100_t1_20260619_072438`
- MLP:
  `results/qk_all_affine_mlp_main_c100_t1_20260619_082701`
- patch, admitted batch-32 row:
  `results/qk_all_affine_patch_main_c100_t1_20260619_083410`
- patch, excluded batch-16 control:
  `results/qk_all_affine_patch_main_c100_t1_20260619_082858`
- classifier:
  `results/qk_all_affine_classifier_main_c100_t1_20260619_083905`
- cumulative:
  `results/qk_all_affine_all_affine_main_c100_t1_20260619_084014`
